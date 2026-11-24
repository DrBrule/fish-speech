"""Write a stand-alone Drama 3 checkpoint with a fine-tune merged in.

The output directory has the same layout and tensor names as the release (config,
tokenizer, codec, safetensors shards, FP8 experts untouched), so the stock tools work
with it unchanged: tools/api_server.py, fish_speech/models/text2semantic/inference.py,
and SGLang-Omni / vLLM-Omni. Only tensors the fine-tune changed are rewritten.

    python tools/finetune/export_merged.py \
        --checkpoint-path checkpoints/drama-3 \
        --finetune results/drama3-myvoice \
        --output checkpoints/drama-3-myvoice --device cuda:0
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from fish_speech.finetune.peft_utils import apply_finetune, changed_tensors
from fish_speech.models.text2semantic.fish_omni.backbone import CHECKPOINT_BACKBONE_PREFIX
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.tower import TOWER_CHECKPOINT_PREFIXES


def checkpoint_key_candidates(name: str) -> list[str]:
    """Release tensor names a Drama3ForTraining parameter name can be stored under."""
    if name.startswith("backbone."):
        inner = name[len("backbone.") :]
        if inner == "lm_head.weight":
            return ["lm_head.weight"]
        assert inner.startswith("model."), name
        rest = inner[len("model.") :]
        return [CHECKPOINT_BACKBONE_PREFIX + rest, inner]
    if name.startswith("audio_decoder."):
        inner = name[len("audio_decoder.") :]
        return [
            old + inner[len(new) :]
            for old, new in TOWER_CHECKPOINT_PREFIXES.items()
            if inner.startswith(new)
        ]
    raise ValueError(f"not a model tensor: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0",
                        help="comma list to spread the model as in inference")
    args = parser.parse_args()

    source, output = Path(args.checkpoint_path), Path(args.output)
    if output.resolve() == source.resolve():
        raise SystemExit("--output must differ from --checkpoint-path")
    output.mkdir(parents=True, exist_ok=True)

    devices = [d.strip() for d in args.device.split(",")]
    backbone_devices, tower_device = (devices, devices[0]) if len(devices) == 1 else (devices[:-1], devices[-1])
    model = FishOmniModel.from_pretrained(
        source, devices=backbone_devices, tower_device=tower_device, dtype=torch.bfloat16
    )
    names = changed_tensors(model, args.finetune)
    apply_finetune(model, args.finetune)
    params = {f"backbone.{n}": p for n, p in model.backbone.named_parameters()}
    params |= {f"audio_decoder.{n}": p for n, p in model.audio_decoder.named_parameters()}

    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        with safe_open(str(source / "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    replacements: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name in sorted(names):
        key = next((k for k in checkpoint_key_candidates(name) if k in weight_map), None)
        if key is None:
            raise KeyError(f"{name}: no matching tensor in {source}")
        replacements[weight_map[key]][key] = params[name].detach().cpu()
    logger.info(f"Rewriting {len(names)} tensors in {len(replacements)} shard(s)")

    for shard in sorted(set(weight_map.values())):
        target = output / shard
        if shard not in replacements:
            if not target.exists():
                shutil.copy2(source / shard, target)
            continue
        tensors, metadata = {}, None
        with safe_open(str(source / shard), "pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in replacements[shard]:
                    new = replacements[shard][key]
                    if new.shape != tensor.shape:
                        raise ValueError(f"{key}: {tuple(new.shape)} != {tuple(tensor.shape)}")
                    tensor = new.to(tensor.dtype)
                tensors[key] = tensor
        save_file(tensors, str(target), metadata=metadata)
        logger.info(f"Wrote {target}")

    for item in source.iterdir():
        if item.name.endswith(".safetensors") or item.name.startswith("."):
            continue
        destination = output / item.name
        if item.is_file() and not destination.exists():
            shutil.copy2(item, destination)
    for extra in ("drama3_finetune.json",):
        path = Path(args.finetune) / extra
        if path.is_file():
            shutil.copy2(path, output / extra)
    logger.info(f"Merged checkpoint ready: {output}")


if __name__ == "__main__":
    main()
