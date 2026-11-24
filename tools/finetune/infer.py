"""Speak with a fine-tuned Drama 3: base checkpoint + fine-tune directory, merged in memory.

With no reference the model now speaks in the fine-tuned default voice; every other
Drama 3 control still applies (speaker tags, inline / scoped tags, style prefixes,
director's notes, references for other voices, long-text chunking).

    python tools/finetune/infer.py \
        --checkpoint-path checkpoints/drama-3 \
        --finetune results/drama3-myvoice \
        --text "[with bright eager excitement] We won! <whisper>Don't tell anyone.</whisper>" \
        --output out.wav

Several lines (one per line of a text file) go to numbered files with --text-file.
"""

import argparse
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from fish_speech.finetune.peft_utils import apply_finetune
from fish_speech.models.text2semantic.inference import (
    decode_to_audio,
    encode_audio,
    generate_long,
    init_model,
    load_codec_model,
    resolve_codec_path,
)


def load_finetuned(checkpoint_path: str, finetune: str | None, device: str, half: bool = False):
    precision = torch.half if half else torch.bfloat16
    model, _ = init_model(checkpoint_path, device, precision)
    if finetune:
        model = apply_finetune(model, finetune)
    codec = load_codec_model(
        resolve_codec_path(checkpoint_path), str(model.device), model.dtype
    )
    return model, codec


def synthesize(
    model,
    codec,
    text: str,
    reference_audio: list[str] = (),
    reference_text: list[str] = (),
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 30,
    max_new_tokens: int = 0,
    single_chunk: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
    prompt_tokens = [encode_audio(p, codec, str(model.device)).cpu() for p in reference_audio]
    pieces = []
    for response in generate_long(
        model=model,
        device=str(model.device),
        decode_one_token=None,
        text=text,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        single_chunk=single_chunk,
        prompt_text=list(reference_text) or None,
        prompt_tokens=prompt_tokens or None,
    ):
        if response.action == "sample":
            pieces.append(response.codes)
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    codes = torch.cat(pieces, dim=1).to(model.device)
    with torch.inference_mode():
        return decode_to_audio(codes, codec).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-path", default="checkpoints/drama-3")
    parser.add_argument("--finetune", help="fine-tune output directory (omit for the base model)")
    parser.add_argument("--text")
    parser.add_argument("--text-file")
    parser.add_argument("--reference-audio", nargs="*", default=[])
    parser.add_argument("--reference-text", nargs="*", default=[])
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--single-chunk", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    texts = [args.text] if args.text else []
    if args.text_file:
        texts += [l.strip() for l in open(args.text_file, encoding="utf-8") if l.strip()]
    if not texts:
        parser.error("give --text or --text-file")

    model, codec = load_finetuned(args.checkpoint_path, args.finetune, args.device, args.half)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(texts):
        audio = synthesize(
            model, codec, text, args.reference_audio, args.reference_text,
            args.temperature, args.top_p, args.top_k, args.max_new_tokens,
            args.single_chunk, args.seed,
        )
        path = output if len(texts) == 1 else output.with_stem(f"{output.stem}_{index:03d}")
        sf.write(path, audio, codec.sample_rate)
        logger.info(f"{path}: {len(audio) / codec.sample_rate:.1f}s  {text!r}")


if __name__ == "__main__":
    main()
