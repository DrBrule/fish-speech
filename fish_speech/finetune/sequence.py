"""Training sequences in the Drama 3 conversation format.

The prompt part of every sequence is produced by the inference prompt builder
(build_drama3_prompt), so training and inference see byte-identical prompts. The
assistant audio and its closing <|im_end|> are then appended and marked as supervised,
as in the technical report (Sec. 3, Appendix A/B): loss falls only on assistant audio
frames and the closing <|im_end|>; reference audio is context only.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

HistoryTurn = tuple[str, torch.Tensor]


@dataclass
class TrainingSequence:
    """rows [T, 1 + N] (token, code 0..N-1) and supervised [T] (row is a target)."""

    rows: torch.Tensor
    supervised: torch.Tensor

    def __len__(self) -> int:
        return len(self.rows)


def audio_rows(codes: torch.Tensor, config: FishOmniConfig) -> torch.Tensor:
    """[N, frames] codes -> [frames, 1 + N] rows, main token = code 0 + semantic start."""
    codes = codes.to(device="cpu", dtype=torch.long)
    main = codes[:1] + config.semantic_start_token_id
    return torch.cat((main, codes)).T.contiguous()


def build_training_sequence(
    text: str,
    target_codes: torch.Tensor,
    tokenizer,
    config: FishOmniConfig,
    reference_texts: Sequence[str] = (),
    reference_codes: Sequence[torch.Tensor] = (),
    history: Sequence[HistoryTurn] = (),
    supervise_history: bool = True,
) -> TrainingSequence:
    """One supervised conversation ending in the assistant voice turn for text.

    history turns are earlier (text, codes) user / assistant pairs, as the chunked
    inference path writes them; their audio is supervised when supervise_history.
    """
    prompt = build_drama3_prompt(
        text,
        tokenizer,
        config,
        list(reference_texts),
        list(reference_codes),
        history=list(history),
    )
    num_codebooks = config.audio_decoder.num_codebooks
    eos_row = torch.zeros((1, 1 + num_codebooks), dtype=torch.long)
    eos_row[0, 0] = config.eos_token_id
    rows = torch.cat((prompt, audio_rows(target_codes, config), eos_row))

    tokens = rows[:, 0]
    semantic = (tokens >= config.semantic_start_token_id) & (
        tokens <= config.semantic_end_token_id
    )
    # The first <|im_end|> closes the system turn; every audio row after it belongs
    # to an assistant turn (references live only in the system turn).
    system_end = int((tokens == config.eos_token_id).nonzero()[0, 0])
    after_system = torch.arange(len(rows)) > system_end
    assistant_audio = semantic & after_system
    # <|im_end|> directly after assistant audio closes that turn.
    closes_audio = torch.zeros_like(semantic)
    closes_audio[1:] = (tokens[1:] == config.eos_token_id) & assistant_audio[:-1]
    supervised = assistant_audio | closes_audio

    if not supervise_history:
        current_start = len(prompt)
        supervised[:current_start] = False
    return TrainingSequence(rows=rows, supervised=supervised)
