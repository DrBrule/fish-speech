"""Compare the base and fine-tuned default voice on multilingual and directed prompts.

For every prompt in --prompts (default: tools/finetune/eval_prompts.jsonl) it
generates audio with no reference, from the base model and then from the fine-tune,
and scores:

    sim   cosine similarity of a speaker-verification embedding (Hugging Face
          microsoft/wavlm-base-plus-sv) to the mean embedding of the target clips
    wer   ASR error rate (Hugging Face Whisper; CER for zh / ja / ko / th / yue)
          against the prompt with direction tags removed

The fine-tune should raise sim on every kind of prompt while wer stays close to the
base model; a wer jump on the multilingual or tagged rows means those abilities are
being lost (raise the replay / KL share, lower the learning rate or the epochs).
Direction adherence is not scored automatically: listen to the *_tag / style rows.

    python tools/finetune/evaluate.py \
        --checkpoint-path checkpoints/drama-3 --finetune results/drama3-myvoice \
        --target-manifest data/prepared/manifest.jsonl --output-dir results/eval
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import soundfile as sf
import torch
import torchaudio
from loguru import logger

from fish_speech.finetune.peft_utils import apply_finetune

CER_LANGUAGES = {"zh", "ja", "ko", "th", "yue"}
TAG_PATTERN = re.compile(r"\[[^\[\]]*\]|</?[A-Za-z][\w-]*[^>]*>|<\|[^|>]*\|>")


def spoken_text(text: str) -> str:
    if "\nLine:" in text:
        text = text.split("\nLine:", 1)[1]
    return TAG_PATTERN.sub(" ", text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, language: str) -> float:
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if language in CER_LANGUAGES:
        ref, hyp = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
    else:
        ref, hyp = reference.split(), hypothesis.split()
    return edit_distance(ref, hyp) / max(1, len(ref))


class SpeakerEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        wav16 = torchaudio.functional.resample(torch.from_numpy(wav).float(), sample_rate, 16000)
        inputs = self.extractor(wav16.numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
        embedding = self.model(**inputs).embeddings[0]
        return torch.nn.functional.normalize(embedding, dim=-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-path", default="checkpoints/drama-3")
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--target-manifest", required=True,
                        help="prepared manifest; its target rows' audio is the voice reference")
    parser.add_argument("--prompts", default=str(Path(__file__).with_name("eval_prompts.jsonl")))
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2, help="takes per prompt")
    parser.add_argument("--reference-clips", type=int, default=20)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    args = parser.parse_args()

    from transformers import pipeline

    from tools.finetune.infer import load_finetuned, synthesize

    out_dir = Path(args.output_dir)
    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    targets = [json.loads(l) for l in open(args.target_manifest, encoding="utf-8") if l.strip()]
    targets = [t for t in targets if t.get("role", "target") == "target" and t.get("audio")]
    random.Random(0).shuffle(targets)
    if not targets:
        raise SystemExit("no target rows with an 'audio' path in the manifest")

    embedder = SpeakerEmbedder(args.sv_model, args.device)
    reference = []
    for row in targets[: args.reference_clips]:
        wav, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        reference.append(embedder(wav.mean(1), sr))
    reference = torch.nn.functional.normalize(torch.stack(reference).mean(0), dim=-1)
    asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                   device=args.device, torch_dtype=torch.float16)

    model, codec = load_finetuned(args.checkpoint_path, None, args.device)
    results = []
    for variant in ("base", "finetuned"):
        if variant == "base" and args.skip_base:
            continue
        if variant == "finetuned":
            model = apply_finetune(model, args.finetune)
        (out_dir / variant).mkdir(parents=True, exist_ok=True)
        for index, prompt in enumerate(prompts):
            for take in range(args.samples):
                audio = synthesize(model, codec, prompt["text"], seed=1000 * index + take)
                path = out_dir / variant / f"{index:03d}_{prompt['kind']}_{take}.wav"
                sf.write(path, audio, codec.sample_rate)
                row = {"variant": variant, "index": index, "take": take, **prompt,
                       "audio": str(path), "seconds": len(audio) / codec.sample_rate}
                if len(audio) < codec.sample_rate // 4:
                    row.update(sim=None, wer=None, transcript="")
                else:
                    row["sim"] = float(embedder(audio, codec.sample_rate) @ reference)
                    wav16 = torchaudio.functional.resample(torch.from_numpy(audio), codec.sample_rate, 16000)
                    language = prompt.get("language")
                    kwargs = {"generate_kwargs": {"language": language}} if language else {}
                    row["transcript"] = asr({"raw": wav16.numpy(), "sampling_rate": 16000}, **kwargs)["text"]
                    row["wer"] = error_rate(spoken_text(prompt["text"]), row["transcript"], language or "en")
                results.append(row)
                logger.info(f"{variant} {index} take {take}: sim={row['sim']} wer={row['wer']}")

    summary: dict = {}
    for row in results:
        for key in (row["kind"], "all"):
            bucket = summary.setdefault(row["variant"], {}).setdefault(key, {"sim": [], "wer": []})
            for metric in ("sim", "wer"):
                if row[metric] is not None:
                    bucket[metric].append(row[metric])
    table = {
        variant: {
            kind: {m: round(float(np.mean(v)), 4) if v else None for m, v in metrics.items()}
            for kind, metrics in kinds.items()
        }
        for variant, kinds in summary.items()
    }
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    logger.info("Summary (mean sim up is better; wer should stay near base):\n"
                + json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
