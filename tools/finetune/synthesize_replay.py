"""Self-distilled replay data: the base model clones the target voice on multilingual
and directed lines, and the takes that pass ASR and speaker checks become extra
target-voice training rows.

A real single-speaker dataset is usually one language and mostly neutral. Trained on
it alone, the default voice can drift toward that language's accent and away from
tags it never saw. These rows teach "default voice = target" on exactly the inputs
the base model already handles well (other languages, point / scoped tags, style
prefixes, director's notes), so those abilities carry over to the new default voice.

    python tools/finetune/synthesize_replay.py \
        --checkpoint-path checkpoints/drama-3 \
        --target-manifest data/prepared/manifest.jsonl \
        --output-dir data/prepared --takes 3

Rows are appended to <output-dir>/manifest.jsonl with source "synthetic".
License note: the Fish Audio Community License restricts using outputs to train
*other* speech models; check that your use is covered before training on them.
"""

import argparse
import json
import random
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import soundfile as sf
import torch
import torchaudio
from loguru import logger

from fish_speech.models.text2semantic.inference import (
    decode_to_audio,
    generate_long,
)
from tools.finetune.evaluate import SpeakerEmbedder, error_rate, spoken_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-path", default="checkpoints/drama-3")
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--prompts", default=str(Path(__file__).with_name("replay_prompts.jsonl")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--takes", type=int, default=3, help="takes per prompt; best kept")
    parser.add_argument("--references", type=int, default=2, help="reference clips per take")
    parser.add_argument("--max-wer", type=float, default=0.15)
    parser.add_argument("--min-sim", type=float, default=0.0,
                        help="absolute floor; takes are also ranked by similarity")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--speaker", help="speaker name for the rows (default: the target's)")
    parser.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    args = parser.parse_args()

    from transformers import pipeline

    from tools.finetune.infer import load_finetuned

    manifest_dir = Path(args.target_manifest).parent
    targets = []
    for line in open(args.target_manifest, encoding="utf-8"):
        row = json.loads(line) if line.strip() else None
        if row and row.get("role", "target") == "target" and row.get("source", "real") == "real":
            codes = Path(row["codes"])
            row["codes"] = str(codes if codes.is_absolute() else manifest_dir / codes)
            targets.append(row)
    if not targets:
        raise SystemExit("no real target rows in the manifest")
    # Prefer clips of 4-15 s as references: long enough for timbre, short for context.
    pool = [t for t in targets if 86 <= t["frames"] <= 320] or targets
    speaker = args.speaker or targets[0]["speaker"]
    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]

    model, codec = load_finetuned(args.checkpoint_path, None, args.device)
    embedder = SpeakerEmbedder(args.sv_model, args.device)
    asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                   device=args.device, torch_dtype=torch.float16)
    reference_embedding = []
    for row in pool[:20]:
        if row.get("audio"):
            wav, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
            reference_embedding.append(embedder(wav.mean(1), sr))
    reference_embedding = (
        torch.nn.functional.normalize(torch.stack(reference_embedding).mean(0), dim=-1)
        if reference_embedding else None
    )

    out_dir = Path(args.output_dir)
    (out_dir / "codes").mkdir(parents=True, exist_ok=True)
    (out_dir / "synthetic_audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    kept = 0
    with open(out_dir / "manifest.jsonl", "a", encoding="utf-8") as manifest:
        for index, prompt in enumerate(prompts):
            line = prompt["text"]
            style = prompt.get("style")
            text = f"[{style}] {line}" if style else line
            best = None
            for take in range(args.takes):
                refs = rng.sample(pool, min(args.references, len(pool)))
                torch.manual_seed(10_000 * index + take)
                pieces = [
                    r.codes for r in generate_long(
                        model=model, device=args.device, decode_one_token=None, text=text,
                        temperature=args.temperature, single_chunk=True, max_new_tokens=0,
                        prompt_text=[r["text"] for r in refs],
                        prompt_tokens=[torch.from_numpy(np.load(r["codes"]).astype(np.int64)) for r in refs],
                    ) if r.action == "sample"
                ]
                if not pieces:
                    continue
                codes = torch.cat(pieces, dim=1)
                with torch.inference_mode():
                    audio = decode_to_audio(codes.to(model.device), codec).float().cpu().numpy()
                wav16 = torchaudio.functional.resample(torch.from_numpy(audio), codec.sample_rate, 16000)
                language = prompt.get("language")
                kwargs = {"generate_kwargs": {"language": language}} if language else {}
                transcript = asr({"raw": wav16.numpy(), "sampling_rate": 16000}, **kwargs)["text"]
                wer = error_rate(spoken_text(line), transcript, language or "en")
                sim = (
                    float(embedder(audio, codec.sample_rate) @ reference_embedding)
                    if reference_embedding is not None else 0.0
                )
                logger.info(f"prompt {index} take {take}: wer={wer:.3f} sim={sim:.3f}")
                if wer <= args.max_wer and sim >= args.min_sim and (best is None or sim > best[0]):
                    best = (sim, wer, codes, audio)
            if best is None:
                logger.warning(f"prompt {index}: no take passed; skipped")
                continue
            sim, wer, codes, audio = best
            uid = f"synthetic_{index:04d}"
            np.save(out_dir / "codes" / f"{uid}.npy", codes.numpy().astype(np.int16))
            audio_path = out_dir / "synthetic_audio" / f"{uid}.wav"
            sf.write(audio_path, audio, codec.sample_rate)
            manifest.write(json.dumps({
                "id": uid, "text": line, "codes": f"codes/{uid}.npy",
                "frames": int(codes.shape[1]), "speaker": speaker, "role": "target",
                "language": prompt.get("language"), "style": style, "group": None,
                "order": None, "source": "synthetic", "audio": str(audio_path),
                "sim": round(sim, 4), "wer": round(wer, 4),
            }, ensure_ascii=False) + "\n")
            kept += 1
    logger.info(f"Kept {kept} of {len(prompts)} prompts")


if __name__ == "__main__":
    main()
