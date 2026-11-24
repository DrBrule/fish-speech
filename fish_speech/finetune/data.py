"""Prepared-manifest dataset with a task mixture, and the batch collator.

A prepared manifest is JSONL (written by tools/finetune/prepare_data.py), one
utterance per line:

    id        unique id
    text      transcript, inline direction tags allowed ([laughs], <whisper>..</whisper>)
    codes     .npy file of int codes [10, frames] (relative to the manifest directory)
    frames    number of frames
    speaker   speaker name
    role      "target" (the voice that becomes the default) or "replay" (other voices,
              used only as cloning regularization, never without a reference)
    language  optional ISO code, used for the optional [language: xx] hint
    style     optional delivery description, written as a style prefix or a
              director's note (keeps directorial control tied to the default voice)
    group     optional recording / session id; with order, lets consecutive
    order     utterances form multi-turn (chunk-continuation) sequences
    source    "real" or "synthetic" (self-distilled replay in the target voice)

Each item draws one task:

    default   no reference: the system turn is the plain TTS instruction, so the
              target voice becomes what the model speaks when no voice is given
    history   no reference plus 1..N earlier turns of the same recording, as the
              chunked inference path builds them (keeps long texts consistent)
    clone     indexed references of the same speaker (keeps cloning working, and
              anchors the default voice to how it sounds when cloned)

Replay rows always use "clone" and are flagged for the optional KL regularizer.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from fish_speech.finetune.sequence import TrainingSequence, build_training_sequence
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import (
    DRAMA3_MAX_PROMPT_ROWS,
    Drama3PromptTooLongError,
)

STYLE_TEMPLATES = (
    "[{style}] {text}",
    "Direction: {style}\nLine: {text}",
)


@dataclass
class MixtureConfig:
    default_prob: float = 0.55
    history_prob: float = 0.2
    clone_prob: float = 0.25
    max_history_turns: int = 2
    max_references: int = 2
    max_reference_frames: int = 650  # about 30 s
    language_hint_prob: float = 0.2
    # Probability of dropping a style annotation (the plain line is also trained).
    style_drop_prob: float = 0.2
    supervise_history: bool = True
    max_rows: int = 4096
    seed: int = 42
    weights: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.default_prob + self.history_prob + self.clone_prob
        if total <= 0:
            raise ValueError("task probabilities must not all be zero")
        if self.max_rows > DRAMA3_MAX_PROMPT_ROWS + 2048:
            logger.warning(
                f"max_rows={self.max_rows} is longer than inference ever sees"
            )


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                codes = Path(row["codes"])
                if not codes.is_absolute():
                    row["codes"] = str(path.parent / codes)
                row.setdefault("role", "target")
                row.setdefault("source", "real")
                rows.append(row)
    return rows


def load_codes(path: str) -> torch.Tensor:
    return torch.from_numpy(np.load(path).astype(np.int64))


class Drama3FinetuneDataset(Dataset):
    """Draws a task per item and returns a TrainingSequence as a dict."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        config: FishOmniConfig,
        mixture: MixtureConfig | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.config = config
        self.mixture = mixture or MixtureConfig()
        self.rng = random.Random(self.mixture.seed)
        self.by_speaker: dict[str, list[int]] = defaultdict(list)
        self.by_group: dict[tuple, int] = {}
        for index, record in enumerate(records):
            self.by_speaker[record["speaker"]].append(index)
            if record.get("group") is not None and record.get("order") is not None:
                self.by_group[(record["group"], int(record["order"]))] = index
        single = [s for s, ids in self.by_speaker.items() if len(ids) < 2]
        replay_single = [
            s for s in single if records[self.by_speaker[s][0]]["role"] == "replay"
        ]
        if replay_single:
            logger.warning(
                f"{len(replay_single)} replay speakers have one clip and cannot be "
                "cloned from another clip; their rows are dropped"
            )
            drop = {self.by_speaker[s][0] for s in replay_single}
            self.indices = [i for i in range(len(records)) if i not in drop]
        else:
            self.indices = list(range(len(records)))
        roles = defaultdict(int)
        for i in self.indices:
            roles[records[i]["role"]] += 1
        logger.info(f"Fine-tuning dataset: {dict(roles)} utterances")

    def __len__(self) -> int:
        return len(self.indices)

    # Text --------------------------------------------------------------------------

    def format_text(self, record: dict) -> str:
        text = record["text"].strip()
        style = (record.get("style") or "").strip()
        if style and self.rng.random() >= self.mixture.style_drop_prob:
            text = self.rng.choice(STYLE_TEMPLATES).format(style=style, text=text)
        language = record.get("language")
        if language and self.rng.random() < self.mixture.language_hint_prob:
            text = f"{text} [language: {language}]"
        return text

    # Task pieces ---------------------------------------------------------------------

    def pick_references(self, record_index: int) -> tuple[list[str], list[torch.Tensor]]:
        record = self.records[record_index]
        candidates = [i for i in self.by_speaker[record["speaker"]] if i != record_index]
        self.rng.shuffle(candidates)
        count = self.rng.randint(1, self.mixture.max_references)
        texts, codes, frames = [], [], 0
        for i in candidates:
            other = self.records[i]
            if texts and frames + other["frames"] > self.mixture.max_reference_frames:
                continue
            texts.append(other["text"].strip())
            codes.append(load_codes(other["codes"]))
            frames += other["frames"]
            if len(texts) >= count:
                break
        return texts, codes

    def pick_history(self, record_index: int) -> list[tuple[str, torch.Tensor]]:
        record = self.records[record_index]
        if record.get("group") is None or record.get("order") is None:
            return []
        turns = self.rng.randint(1, self.mixture.max_history_turns)
        history = []
        for back in range(turns, 0, -1):
            index = self.by_group.get((record["group"], int(record["order"]) - back))
            if index is None:
                history = []  # keep only an unbroken run ending right before this turn
                continue
            history.append(
                (self.format_text(self.records[index]), load_codes(self.records[index]["codes"]))
            )
        return history

    def choose_task(self, record: dict) -> str:
        if record["role"] == "replay":
            return "clone"
        m = self.mixture
        return self.rng.choices(
            ("default", "history", "clone"),
            weights=(m.default_prob, m.history_prob, m.clone_prob),
        )[0]

    def build(self, record_index: int, task: str) -> TrainingSequence:
        record = self.records[record_index]
        target = load_codes(record["codes"])
        text = self.format_text(record)
        refs: tuple[list[str], list[torch.Tensor]] = ([], [])
        history: list = []
        if task == "clone":
            refs = self.pick_references(record_index)
        elif task == "history":
            history = self.pick_history(record_index)
        # Drop context until the sequence fits.
        while True:
            try:
                sequence = build_training_sequence(
                    text,
                    target,
                    self.tokenizer,
                    self.config,
                    refs[0],
                    refs[1],
                    history,
                    supervise_history=self.mixture.supervise_history,
                )
            except Drama3PromptTooLongError:
                sequence = None
            if sequence is not None and len(sequence) <= self.mixture.max_rows:
                return sequence
            if history:
                history = history[1:]
            elif len(refs[0]) > 1:
                refs = (refs[0][:-1], refs[1][:-1])
            else:
                raise ValueError(
                    f"{record['id']}: {record['frames']} frames do not fit max_rows="
                    f"{self.mixture.max_rows}; filter long clips in prepare_data"
                )

    def __getitem__(self, item: int) -> dict:
        worker = torch.utils.data.get_worker_info()
        if worker is not None and getattr(self, "_worker_seed", None) != worker.seed:
            # Each DataLoader worker (and epoch) gets its own task / reference draws.
            self._worker_seed = worker.seed
            self.rng = random.Random(worker.seed)
        record_index = self.indices[item]
        record = self.records[record_index]
        task = self.choose_task(record)
        sequence = self.build(record_index, task)
        return {
            "rows": sequence.rows,
            "supervised": sequence.supervised,
            "regularize": record["role"] == "replay",
            "task": task,
        }


@dataclass
class Drama3Collator:
    """Right-pads rows; padding rows are never supervised and never attended to by
    real positions (attention is causal and padding comes last)."""

    pad_token_id: int

    def __call__(self, items: list[dict]) -> dict:
        length = max(len(item["rows"]) for item in items)
        width = items[0]["rows"].shape[1]
        rows = torch.zeros((len(items), length, width), dtype=torch.long)
        rows[:, :, 0] = self.pad_token_id
        supervised = torch.zeros((len(items), length), dtype=torch.bool)
        lengths = torch.zeros(len(items), dtype=torch.long)
        for i, item in enumerate(items):
            n = len(item["rows"])
            rows[i, :n] = item["rows"]
            supervised[i, :n] = item["supervised"]
            lengths[i] = n
        return {
            "input_rows": rows,
            "supervised": supervised,
            "lengths": lengths,
            "regularize": torch.tensor([bool(i["regularize"]) for i in items]),
        }
