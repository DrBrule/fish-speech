"""Encode a speaker dataset into Drama 3 codec codes and write a prepared manifest.

Input is either a directory of audio files with fish-speech style sidecar transcripts
(clip.wav + clip.lab or clip.txt), or a JSONL manifest with one utterance per line:

    {"audio": "clips/0001.wav", "text": "Hello there.", "language": "en",
     "style": "warm, smiling", "group": "session1", "order": 0}

Optional keys (speaker, role, language, style, group, order, source) pass through;
see fish_speech/finetune/data.py. Clips without a transcript are skipped unless
--transcribe is set, which runs a Hugging Face ASR pipeline (Whisper by default).

    python tools/finetune/prepare_data.py \
        --input data/myvoice --output-dir data/prepared \
        --speaker myvoice --language en --consecutive

Run it again with --role replay --speaker <name> --append to add other voices for
cloning regularization (each replay speaker needs at least two clips).
"""

import argparse
import json
import re
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import soundfile as sf
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.generate import validate_drama3_text

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
CODEC_SAMPLE_RATE = 44100
FRAME_SAMPLES = 2048


def natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def read_sidecar(audio: Path) -> str | None:
    for suffix in (".lab", ".txt"):
        sidecar = audio.with_suffix(suffix)
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8").strip()
    return None


def collect_items(args) -> list[dict]:
    source = Path(args.input)
    items = []
    if source.is_file():
        with open(source, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                audio = Path(item["audio"])
                item["audio"] = str(audio if audio.is_absolute() else source.parent / audio)
                items.append(item)
        return items
    files = sorted(
        (p for p in source.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES),
        key=natural_key,
    )
    order_in_group: dict[str, int] = {}
    for path in files:
        rel = path.relative_to(source).with_suffix("")
        item = {"audio": str(path), "text": read_sidecar(path), "id": "__".join(rel.parts)}
        if args.consecutive:
            group = str(path.parent.relative_to(source)) or "."
            item["group"] = group
            item["order"] = order_in_group.get(group, 0)
            order_in_group[group] = item["order"] + 1
        items.append(item)
    return items


def load_audio(path: str, peak_normalize: bool) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    if sample_rate != CODEC_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, CODEC_SAMPLE_RATE)
    if peak_normalize:
        peak = wav.abs().max()
        if peak > 0:
            wav = wav * (0.89 / peak)  # about -1 dBFS
    return wav


@torch.inference_mode()
def encode(codec, wav: torch.Tensor, device: str) -> np.ndarray:
    dtype = next(codec.parameters()).dtype
    audios = wav.to(device=device, dtype=dtype)[None, None]
    lengths = torch.tensor([wav.shape[-1]], device=device)
    indices, feature_lengths = codec.encode(audios, lengths)
    return indices[0, :, : feature_lengths[0]].cpu().numpy().astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="audio directory or JSONL manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speaker", default="target")
    parser.add_argument("--role", choices=("target", "replay"), default="target")
    parser.add_argument("--language", help="ISO code for rows that do not set one")
    parser.add_argument("--source", default="real")
    parser.add_argument("--codec-path", default="checkpoints/drama-3/codec.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-seconds", type=float, default=0.6)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--consecutive", action="store_true",
                        help="files in one folder are consecutive segments of a recording "
                        "(enables multi-turn training)")
    parser.add_argument("--peak-normalize", action="store_true")
    parser.add_argument("--transcribe", action="store_true",
                        help="transcribe clips without a transcript")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    from fish_speech.models.text2semantic.inference import load_codec_model

    out_dir = Path(args.output_dir)
    codes_dir = out_dir / "codes"
    codes_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists() and not args.append:
        raise SystemExit(f"{manifest} exists; pass --append to add to it")

    items = collect_items(args)
    logger.info(f"{len(items)} clips found")
    asr = None
    if args.transcribe and any(not item.get("text") for item in items):
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                       device=args.device, torch_dtype=torch.float16)

    codec = load_codec_model(args.codec_path, args.device, torch.bfloat16)
    written = skipped = 0
    total_seconds = 0.0
    with open(manifest, "a", encoding="utf-8") as out:
        for item in tqdm(items):
            audio_path = item.pop("audio")
            wav = load_audio(audio_path, args.peak_normalize)
            seconds = wav.shape[-1] / CODEC_SAMPLE_RATE
            if not args.min_seconds <= seconds <= args.max_seconds:
                skipped += 1
                continue
            text = (item.get("text") or "").strip()
            if not text and asr is not None:
                kwargs = {"generate_kwargs": {"language": args.language}} if args.language else {}
                text = asr({"raw": wav.numpy(), "sampling_rate": CODEC_SAMPLE_RATE}, **kwargs)["text"].strip()
            if not text:
                logger.warning(f"{audio_path}: no transcript; skipped")
                skipped += 1
                continue
            try:
                validate_drama3_text(text, audio_path)
            except InvalidInputError as error:
                logger.warning(f"skipped: {error}")
                skipped += 1
                continue
            uid = re.sub(r"[^\w.-]+", "_", item.get("id") or Path(audio_path).stem)
            codes = encode(codec, wav, args.device)
            np.save(codes_dir / f"{uid}.npy", codes)
            record = {
                "id": uid,
                "text": text,
                "codes": f"codes/{uid}.npy",
                "frames": int(codes.shape[1]),
                "speaker": item.get("speaker") or args.speaker,
                "role": item.get("role") or args.role,
                "language": item.get("language") or args.language,
                "style": item.get("style"),
                "group": item.get("group"),
                "order": item.get("order"),
                "source": item.get("source") or args.source,
                "audio": audio_path,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            total_seconds += seconds
    logger.info(
        f"Wrote {written} clips ({total_seconds / 3600:.2f} h) to {manifest}; skipped {skipped}"
    )


if __name__ == "__main__":
    main()
