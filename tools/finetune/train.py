"""Fine-tune Drama 3 so that its default (no-reference) voice is your speaker.

Built on transformers.Trainer + PEFT LoRA. Single GPU:

    python tools/finetune/train.py \
        --checkpoint_path checkpoints/drama-3 \
        --train_manifest data/prepared/manifest.jsonl \
        --output_dir results/drama3-myvoice \
        --mode lora --tower lora \
        --learning_rate 1e-4 --num_train_epochs 3 \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
        --bf16 --logging_steps 10 --save_steps 500

Data-parallel over several GPUs (each holds a full copy, about 42 GB with FP8 experts):

    accelerate launch --num_processes 8 tools/finetune/train.py ...

One copy spread over several smaller GPUs (single process, layer-wise placement):

    python tools/finetune/train.py --devices cuda:0,cuda:1,cuda:2 ...

See docs/en/finetune.md for the data format and the recipe.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers import HfArgumentParser, Trainer, TrainingArguments

from fish_speech.finetune.data import (
    Drama3Collator,
    Drama3FinetuneDataset,
    MixtureConfig,
    read_manifest,
)
from fish_speech.finetune.model import Drama3ForTraining, LossWeights, count_parameters
from fish_speech.finetune.peft_utils import (
    FULL_SCOPES,
    FULL_WEIGHTS_FILE,
    lora_target_regex,
    save_finetune,
    set_full_trainable,
)
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel


@dataclass
class ModelArguments:
    checkpoint_path: str = field(default="checkpoints/drama-3")
    devices: str = field(
        default="auto",
        metadata={
            "help": "auto: this process's GPU (cuda:LOCAL_RANK). Or a comma list; the "
            "backbone is spread over them and the last one also holds the tower."
        },
    )
    fp8_experts: str = field(
        default="auto",
        metadata={
            "help": "auto | on | off. FP8 experts halve memory (SM89+ GPUs: L40S, H100, "
            "H200, B200). off loads them as bf16 (about 70 GB)."
        },
    )


@dataclass
class FinetuneArguments:
    train_manifest: str = field(default=None, metadata={"help": "prepared manifest.jsonl"})
    eval_manifest: str | None = field(default=None)
    mode: str = field(default="lora", metadata={"help": "lora | full"})
    lora_preset: str = field(default="backbone", metadata={"help": "attention | backbone"})
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    tower: str = field(
        default="lora",
        metadata={"help": "lora | full | frozen: how to train the codebook-MTP tower"},
    )
    full_scopes: str = field(
        default="tower,backbone_dense",
        metadata={"help": f"--mode full: comma list of {sorted(FULL_SCOPES)}"},
    )
    text_loss_weight: float = 1.0
    semantic_loss_weight: float = 0.5
    codebook_loss_weight: float = 0.5
    kl_weight: float = field(
        default=0.0,
        metadata={"help": "KL to the frozen base on replay rows (LoRA mode only)"},
    )
    router_aux_weight: float = 0.0
    tower_learning_rate: float | None = field(
        default=None,
        metadata={"help": "learning rate of fully trained tower weights (default: --learning_rate)"},
    )
    model_gradient_checkpointing: bool = True
    # Task mixture for target-speaker rows.
    default_prob: float = 0.55
    history_prob: float = 0.2
    clone_prob: float = 0.25
    max_history_turns: int = 2
    max_references: int = 2
    language_hint_prob: float = 0.2
    style_drop_prob: float = 0.2
    max_rows: int = 4096


class Drama3Trainer(Trainer):
    """Logs the loss parts and saves adapters + fully trained weights."""

    def __init__(self, *args, finetune_meta: dict, tower_learning_rate=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finetune_meta = finetune_meta
        self.tower_learning_rate = tower_learning_rate
        self._parts: dict[str, list[float]] = {}

    def create_optimizer(self, model=None):
        """Default AdamW groups, plus a separate learning rate for the full tower."""
        if self.tower_learning_rate is None or self.optimizer is not None:
            return super().create_optimizer(model)
        model = model or self.model
        decay = set(self.get_decay_parameter_names(model))
        groups: dict[tuple, list] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            tower = "audio_decoder." in name and "lora_" not in name
            key = (tower, name in decay)
            groups.setdefault(key, []).append(parameter)
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
        self.optimizer = optimizer_cls(
            [
                {
                    "params": params,
                    "lr": self.tower_learning_rate if tower else self.args.learning_rate,
                    "weight_decay": self.args.weight_decay if decayed else 0.0,
                }
                for (tower, decayed), params in groups.items()
            ],
            **optimizer_kwargs,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, return_parts=True)
        for key, value in outputs["parts"].items():
            self._parts.setdefault(key, []).append(float(value))
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def log(self, logs: dict, start_time=None) -> None:
        if self._parts and "loss" in logs:
            for key, values in self._parts.items():
                logs[f"loss_{key}"] = round(sum(values) / len(values), 4)
            self._parts = {}
        super().log(logs, start_time)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = model(**inputs)["loss"].detach()
        return loss, None, None

    def _save(self, output_dir=None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        if self.args.should_save:
            model = self.accelerator.unwrap_model(self.model)
            save_finetune(model, output_dir, self.finetune_meta)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        model = model or self.model
        if hasattr(model, "peft_config") and os.path.isfile(
            os.path.join(resume_from_checkpoint, "adapter_config.json")
        ):
            model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
        full = os.path.join(resume_from_checkpoint, FULL_WEIGHTS_FILE)
        if os.path.isfile(full):
            named = {
                n.replace("base_model.model.", "").replace(".base_layer.", "."): p
                for n, p in model.named_parameters()
            }
            with torch.no_grad():
                for name, tensor in load_file(full).items():
                    named[name].copy_(tensor.to(named[name].device, named[name].dtype))
        logger.info(f"Resumed fine-tune weights from {resume_from_checkpoint}")


def resolve_devices(spec: str) -> tuple[list[str], str]:
    if spec == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Drama 3 fine-tuning needs CUDA")
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        return [device], device
    specs = [s.strip() for s in spec.split(",") if s.strip()]
    return (specs, specs[0]) if len(specs) == 1 else (specs[:-1], specs[-1])


def build_model(model_args: ModelArguments, args: FinetuneArguments, dtype) -> tuple:
    if model_args.fp8_experts != "auto":
        os.environ["FISH_DRAMA3_FP8_EXPERTS"] = "1" if model_args.fp8_experts == "on" else "0"
    devices, tower_device = resolve_devices(model_args.devices)
    omni = FishOmniModel.from_pretrained(
        model_args.checkpoint_path, devices=devices, tower_device=tower_device, dtype=dtype
    )
    weights = LossWeights(
        text=args.text_loss_weight,
        semantic=args.semantic_loss_weight,
        codebook=args.codebook_loss_weight,
        kl=args.kl_weight,
        router_aux=args.router_aux_weight,
    )
    wrapper = Drama3ForTraining(omni, weights, args.model_gradient_checkpointing)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    model = wrapper
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_regex(args.lora_preset, args.tower),
            bias="none",
        )
        model = get_peft_model(wrapper, config)
        wrapper._disable_adapter = model.disable_adapter
        if args.tower == "full":
            set_full_trainable(model, ["tower"])
    elif args.mode == "full":
        if args.kl_weight > 0:
            raise ValueError("--kl_weight needs --mode lora (the frozen base is the adapter-off model)")
        scopes = [s.strip() for s in args.full_scopes.split(",") if s.strip()]
        if args.tower == "frozen":
            scopes = [s for s in scopes if s != "tower"]
        set_full_trainable(model, scopes)
    else:
        raise ValueError(f"unknown --mode {args.mode!r}")
    if args.router_aux_weight > 0 and not any(
        p.requires_grad for n, p in model.named_parameters() if ".mlp.gate." in n
    ):
        logger.warning("--router_aux_weight has no effect: no router is trained")

    trainable, total = count_parameters(model)
    logger.info(f"Trainable parameters: {trainable / 1e6:.1f}M of {total / 1e9:.2f}B")
    return model, omni


def main() -> None:
    parser = HfArgumentParser((ModelArguments, FinetuneArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        loader = parser.parse_json_file if sys.argv[1].endswith(".json") else parser.parse_yaml_file
        model_args, ft_args, training_args = loader(os.path.abspath(sys.argv[1]))
    else:
        model_args, ft_args, training_args = parser.parse_args_into_dataclasses()
    if not ft_args.train_manifest:
        parser.error("--train_manifest is required")

    # The collator emits custom keys and the model computes its own loss.
    training_args.remove_unused_columns = False
    training_args.label_names = []
    # Checkpointing is handled inside Drama3ForTraining (--model_gradient_checkpointing).
    training_args.gradient_checkpointing = False
    if training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = False

    dtype = torch.bfloat16 if not training_args.fp16 else torch.float16
    model, omni = build_model(model_args, ft_args, dtype)

    mixture = MixtureConfig(
        default_prob=ft_args.default_prob,
        history_prob=ft_args.history_prob,
        clone_prob=ft_args.clone_prob,
        max_history_turns=ft_args.max_history_turns,
        max_references=ft_args.max_references,
        language_hint_prob=ft_args.language_hint_prob,
        style_drop_prob=ft_args.style_drop_prob,
        max_rows=ft_args.max_rows,
        seed=training_args.seed,
    )
    train_dataset = Drama3FinetuneDataset(
        read_manifest(ft_args.train_manifest), omni.tokenizer, omni.config, mixture
    )
    eval_dataset = None
    if ft_args.eval_manifest:
        eval_dataset = Drama3FinetuneDataset(
            read_manifest(ft_args.eval_manifest), omni.tokenizer, omni.config, mixture
        )

    meta = {
        "base_checkpoint": str(Path(model_args.checkpoint_path).resolve()),
        "model_args": asdict(model_args),
        "finetune_args": asdict(ft_args),
    }
    pad_id = omni.tokenizer.token_to_id("<|pad|>") or omni.config.eos_token_id
    trainer = Drama3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Drama3Collator(pad_token_id=pad_id),
        finetune_meta=meta,
        tower_learning_rate=ft_args.tower_learning_rate,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        with open(Path(training_args.output_dir) / "train_log.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Saved fine-tune to {training_args.output_dir}")


if __name__ == "__main__":
    main()
