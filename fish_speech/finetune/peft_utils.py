"""LoRA targeting, trainable-parameter selection, saving and loading for Drama 3.

Routed MoE experts (256 per layer, FP8 in the release) stay frozen in every mode: a
single voice does not need them, and the FP8 experts cannot take gradient updates.
LoRA goes on the attention / Gated DeltaNet projections and the shared expert of each
backbone layer, and on the codebook-MTP tower, which predicts the nine acoustic codes
that carry most of the timbre. The tower can instead be trained in full.
"""

import json
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file, save_file

from fish_speech.finetune.model import Drama3ForTraining
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel

FULL_WEIGHTS_FILE = "trainable_full.safetensors"
FINETUNE_META_FILE = "drama3_finetune.json"

BACKBONE_ATTENTION = r"backbone\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
BACKBONE_LINEAR_ATTENTION = (
    r"backbone\.model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
)
BACKBONE_SHARED_EXPERT = (
    r"backbone\.model\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
)
TOWER_LINEARS = (
    r"audio_decoder\.(shared_layers|residual_layers)\.\d+\."
    r"(eh_proj|transformer_layer\.attention\.(wqkv|wo)|"
    r"transformer_layer\.feed_forward\.(w1|w2|w3))"
)
LORA_PRESETS = {
    # Attention only: smallest adapter.
    "attention": (BACKBONE_ATTENTION, BACKBONE_LINEAR_ATTENTION),
    # Every dense backbone projection (attention, linear attention, shared expert).
    "backbone": (BACKBONE_ATTENTION, BACKBONE_LINEAR_ATTENTION, BACKBONE_SHARED_EXPERT),
}

# Name patterns for full fine-tuning scopes (matched against Drama3ForTraining names).
FULL_SCOPES = {
    "tower": ("audio_decoder.",),
    # Every backbone weight except routed experts, routers, embeddings and lm_head.
    "backbone_dense": (
        ".self_attn.",
        ".linear_attn.",
        ".shared_expert.",
        ".shared_expert_gate.",
        "layernorm",
        "backbone.model.norm.",
    ),
    "embeddings": ("backbone.model.embed_tokens.", "backbone.lm_head."),
    "routers": (".mlp.gate.",),
}


def lora_target_regex(preset: str, tower: str) -> str:
    """PEFT target_modules regex (full match) for a backbone preset and tower mode."""
    if preset not in LORA_PRESETS:
        raise ValueError(f"unknown LoRA preset {preset!r}; one of {sorted(LORA_PRESETS)}")
    parts = list(LORA_PRESETS[preset])
    if tower == "lora":
        parts.append(TOWER_LINEARS)
    return "(" + "|".join(parts) + ")"


def set_full_trainable(model: torch.nn.Module, scopes: list[str]) -> int:
    """requires_grad on every parameter matching the scopes; returns how many."""
    patterns = [p for scope in scopes for p in FULL_SCOPES[scope]]
    count = 0
    for name, parameter in model.named_parameters():
        plain = strip_peft_prefix(name)
        if "lora_" in plain:
            continue
        if any(p in plain for p in patterns) and ".mlp.experts." not in plain:
            parameter.requires_grad_(True)
            count += parameter.numel()
    return count


def strip_peft_prefix(name: str) -> str:
    for prefix in ("base_model.model.", "module."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace(".base_layer.", ".")


def full_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Fully trained (non-LoRA) parameters under their Drama3ForTraining names."""
    return {
        strip_peft_prefix(name): parameter.detach().to("cpu").contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    }


def save_finetune(model: torch.nn.Module, output_dir: str | Path, meta: dict) -> None:
    """Adapter (if any) via PEFT's save_pretrained, full weights, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "peft_config"):
        model.save_pretrained(str(output_dir))
    full = full_trainable_state(model)
    if full:
        save_file(full, str(output_dir / FULL_WEIGHTS_FILE))
    with open(output_dir / FINETUNE_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@torch.no_grad()
def apply_finetune(model: FishOmniModel, finetune_dir: str | Path) -> FishOmniModel:
    """Load a fine-tune onto a loaded FishOmniModel and merge LoRA into the weights.

    After this the model is a plain FishOmniModel again, so every inference path
    (CLI, API server, chunked generation, CUDA graphs) uses the fine-tuned voice.
    """
    finetune_dir = Path(finetune_dir)
    wrapper = Drama3ForTraining(model, gradient_checkpointing=False)
    full_path = finetune_dir / FULL_WEIGHTS_FILE
    if full_path.is_file():
        state = load_file(str(full_path))
        for name, tensor in state.items():
            target = wrapper.get_parameter(name)  # raises on an unknown name
            if target.shape != tensor.shape:
                raise ValueError(f"{name}: {tuple(tensor.shape)} != {tuple(target.shape)}")
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        logger.info(f"Loaded {len(state)} fully fine-tuned tensors from {full_path}")
    if (finetune_dir / "adapter_config.json").is_file():
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(wrapper, str(finetune_dir))
        peft_model.merge_and_unload()
        logger.info(f"Merged LoRA adapter from {finetune_dir}")
    return model.eval()


def changed_tensors(model: FishOmniModel, finetune_dir: str | Path) -> set[str]:
    """Drama3ForTraining-space names of every tensor a fine-tune changes."""
    finetune_dir = Path(finetune_dir)
    names: set[str] = set()
    full_path = finetune_dir / FULL_WEIGHTS_FILE
    if full_path.is_file():
        names.update(load_file(str(full_path)).keys())
    config_path = finetune_dir / "adapter_config.json"
    if config_path.is_file():
        import re

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        pattern = config["target_modules"]
        wrapper_names = {
            f"backbone.{n}" for n, _ in model.backbone.named_modules()
        } | {f"audio_decoder.{n}" for n, _ in model.audio_decoder.named_modules()}
        for module_name in wrapper_names:
            matched = (
                re.fullmatch(pattern, module_name)
                if isinstance(pattern, str)
                else any(module_name.endswith(t) for t in pattern)
            )
            if matched:
                names.add(f"{module_name}.weight")
    return names
