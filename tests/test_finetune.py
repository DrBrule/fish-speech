"""Tiny-model checks for the Drama 3 fine-tuning stack (CPU, no checkpoints)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fish_speech.finetune.data import (
    Drama3Collator,
    Drama3FinetuneDataset,
    MixtureConfig,
    read_manifest,
)
from fish_speech.finetune.model import Drama3ForTraining, LossWeights
from fish_speech.finetune.peft_utils import (
    apply_finetune,
    lora_target_regex,
    save_finetune,
    set_full_trainable,
)
from fish_speech.finetune.sequence import build_training_sequence
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import (
    AudioDecoderConfig,
    FishOmniConfig,
    FishOmniTextConfig,
)
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder

SEMANTIC_START, SEMANTIC_END, EOS = 32, 39, 2


def tiny_config() -> FishOmniConfig:
    text = FishOmniTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    tower = AudioDecoderConfig(
        dim=16,
        intermediate_size=32,
        n_head=2,
        n_local_heads=1,
        head_dim=8,
        norm_eps=1e-6,
        rope_base=10000.0,
        rotary_percent=0.5,
        vocab_size=8,
        num_codebooks=3,
        shared_enrichment_layers=2,
        vq_backbone_codebooks=1,
    )
    return FishOmniConfig(
        text=text,
        audio_decoder=tower,
        semantic_start_token_id=SEMANTIC_START,
        semantic_end_token_id=SEMANTIC_END,
        eos_token_id=EOS,
    )


class HashTokenizer:
    """Each literal becomes 1-3 deterministic ids outside the semantic band."""

    def token_to_id(self, token: str) -> int | None:
        return {"<|semantic:0|>": 32, "<|semantic:7|>": 39, "<|im_end|>": 2}.get(token)

    def encode(self, text: str) -> SimpleNamespace:
        if text.startswith("<|im_end|>"):
            return SimpleNamespace(ids=[EOS, 5])
        seed = sum(map(ord, text))
        ids = [3 + (seed + i * 7) % 28 for i in range(1 + seed % 3)]
        return SimpleNamespace(ids=[i if i != EOS else 4 for i in ids])


def randomized(module: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.2)
    return module


def tiny_model(seed: int = 0) -> FishOmniModel:
    torch.manual_seed(seed)
    config = tiny_config()
    backbone = randomized(FishOmniBackbone(config.text, dtype=torch.float32))
    tower = randomized(FishOmniAudioDecoder(config, dtype=torch.float32))
    return FishOmniModel(config, backbone, tower, HashTokenizer())


def codes(frames: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 8, (3, frames), generator=generator)


class SequenceTest(unittest.TestCase):
    def test_prompt_prefix_and_supervision(self) -> None:
        config, tokenizer = tiny_config(), HashTokenizer()
        target, reference, earlier = codes(5, 1), codes(4, 2), codes(3, 3)
        sequence = build_training_sequence(
            "hello there",
            target,
            tokenizer,
            config,
            ["ref text"],
            [reference],
            history=[("before", earlier)],
        )
        prompt = build_drama3_prompt(
            "hello there", tokenizer, config, ["ref text"], [reference],
            history=[("before", earlier)],
        )
        torch.testing.assert_close(sequence.rows[: len(prompt)], prompt)
        rows, supervised = sequence.rows, sequence.supervised
        # Target audio and closing eos are supervised.
        self.assertTrue(supervised[len(prompt) :].all())
        torch.testing.assert_close(rows[len(prompt) : -1, 1:].T, target)
        self.assertEqual(int(rows[-1, 0]), EOS)
        # Reference audio is never supervised; history audio and its eos are.
        semantic = (rows[:, 0] >= SEMANTIC_START) & (rows[:, 0] <= SEMANTIC_END)
        self.assertEqual(int((semantic & ~supervised).sum()), 4)
        self.assertEqual(int((semantic & supervised).sum()), 5 + 3)
        self.assertEqual(int((~semantic & supervised).sum()), 2)

        no_history = build_training_sequence(
            "hi", target, tokenizer, config, history=[("before", earlier)],
            supervise_history=False,
        )
        self.assertEqual(int(no_history.supervised.sum()), 5 + 1)


class TeacherForcingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_model()
        self.wrapper = Drama3ForTraining(self.model, gradient_checkpointing=False).eval()
        self.sequence = build_training_sequence(
            "hello", codes(6, 4), HashTokenizer(), self.model.config, ["r"], [codes(4, 5)]
        )

    def test_backbone_matches_inference_and_padding(self) -> None:
        rows = self.sequence.rows
        backbone, tower = self.model.backbone, self.model.audio_decoder
        with torch.no_grad():
            expected = backbone(
                inputs_embeds=tower.embed_backbone(rows, backbone.embed(rows[:, 0]))[None]
            )
            actual = self.wrapper.run_backbone(rows[None])
            padded = torch.cat((rows, torch.zeros((5, rows.shape[1]), dtype=torch.long)))
            batch = self.wrapper.run_backbone(torch.stack((padded, padded)))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(batch[:, : len(rows)], expected.expand(2, -1, -1))

    def test_tower_matches_cached_decode(self) -> None:
        """Training head logits equal a cached, step-by-step decode of the same frame."""
        rows = self.sequence.rows
        tower = self.model.audio_decoder
        with torch.no_grad():
            hidden = self.wrapper.run_backbone(rows[None])[0, :-1]
            frames, next_frames = rows[:-1], rows[1:]
            keep = torch.ones(len(frames), dtype=torch.bool)
            states = self.wrapper.tower_hidden_states(hidden, frames, next_frames, keep)
            p = len(frames) - 2  # predicts the last audio frame
            tower.setup_caches(len(frames))
            tower.prefill(hidden[:p], torch.arange(p), frames[:p], next_frames[:p])
            position = torch.tensor([p])
            nxt = next_frames[p : p + 1]
            semantic = tower.is_semantic(nxt[:, 0])
            h = tower.run_shared(hidden[p : p + 1], nxt[:, 1], semantic, position, frames[p : p + 1])
            for k, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
                h = layer(h, nxt[:, k + 1], semantic, position)
                torch.testing.assert_close(head(h), head(states[k][p : p + 1]))
        for layer in (*tower.shared_layers, *tower.residual_layers):
            layer.transformer_layer.attention.k_cache = None
            layer.transformer_layer.attention.v_cache = None


class TrainingTest(unittest.TestCase):
    def batch(self, model: FishOmniModel) -> dict:
        tokenizer, config = HashTokenizer(), model.config
        items = []
        for i, refs in enumerate(([], ["ref"])):
            seq = build_training_sequence(
                f"line {i}", codes(5 + i, i), tokenizer, config, refs,
                [codes(4, 9)] if refs else [],
            )
            items.append(
                {"rows": seq.rows, "supervised": seq.supervised, "regularize": bool(refs)}
            )
        return Drama3Collator(pad_token_id=0)(items)

    def test_lora_step_save_and_merge(self) -> None:
        from peft import LoraConfig, get_peft_model

        model = tiny_model(1)
        wrapper = Drama3ForTraining(
            model, LossWeights(kl=0.1), gradient_checkpointing=True
        ).train()
        for p in wrapper.parameters():
            p.requires_grad_(False)
        peft_model = get_peft_model(
            wrapper,
            LoraConfig(r=4, lora_alpha=8, target_modules=lora_target_regex("backbone", "lora")),
        )
        wrapper._disable_adapter = peft_model.disable_adapter
        trained = set_full_trainable(peft_model, ["tower"])  # LoRA + full tower is legal
        self.assertGreater(trained, 0)
        lora_names = [n for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" in n]
        self.assertTrue(any("linear_attn.in_proj_qkv" in n for n in lora_names))
        self.assertTrue(any("shared_expert.down_proj" in n for n in lora_names))

        batch = self.batch(model)
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=1e-2
        )
        out = peft_model(**batch, return_parts=True)
        for key in ("text", "semantic", "codebook", "kl"):
            self.assertIn(key, out["parts"])
            self.assertTrue(torch.isfinite(out["parts"][key]))
        first = float(out["loss"])
        for _ in range(15):
            optimizer.zero_grad()
            loss = peft_model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), first)
        for name, p in peft_model.named_parameters():
            if ".mlp.experts." in name or "embed_tokens" in name:
                self.assertIsNone(p.grad, name)

        peft_model.eval()
        with torch.no_grad():
            reference = peft_model(**batch, return_parts=True)["parts"]
        with tempfile.TemporaryDirectory() as tmp:
            save_finetune(peft_model, tmp, {"test": True})
            fresh = apply_finetune(tiny_model(1), tmp)
        merged = Drama3ForTraining(fresh, gradient_checkpointing=False).eval()
        with torch.no_grad():
            parts = merged(**batch, return_parts=True)["parts"]
        for key in ("text", "semantic", "codebook"):
            torch.testing.assert_close(parts[key], reference[key], atol=1e-4, rtol=1e-4)


class DatasetTest(unittest.TestCase):
    def test_mixture_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lines = []
            for i in range(6):
                role = "target" if i < 4 else "replay"
                speaker = "me" if role == "target" else "other"
                np.save(tmp / f"{i}.npy", codes(4 + i, i).numpy())
                lines.append(
                    {
                        "id": str(i), "text": f"line {i}", "codes": f"{i}.npy",
                        "frames": 4 + i, "speaker": speaker, "role": role,
                        "group": "g" if role == "target" else None, "order": i,
                        "language": "en", "style": "warm" if i == 1 else None,
                    }
                )
            (tmp / "manifest.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
            records = read_manifest(tmp / "manifest.jsonl")
            dataset = Drama3FinetuneDataset(
                records, HashTokenizer(), tiny_config(), MixtureConfig(seed=0)
            )
            tasks = set()
            for _ in range(10):
                for i in range(len(dataset)):
                    item = dataset[i]
                    tasks.add(item["task"])
                    self.assertTrue(item["supervised"].any())
                    if item["regularize"]:
                        self.assertEqual(item["task"], "clone")
            self.assertEqual(tasks, {"default", "history", "clone"})


if __name__ == "__main__":
    unittest.main()
