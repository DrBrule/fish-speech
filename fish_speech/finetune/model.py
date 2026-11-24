"""Teacher-forced training wrapper around the Drama 3 backbone and codebook-MTP tower.

Loss (technical report, Eq. 8-9), each term a mean over its own bucket:

    L = w_text * L_text + w_sem * L_sem + w_cb * L_cb      (defaults 1, 0.5, 0.5)

L_text: supervised text targets (the closing <|im_end|> of each assistant turn).
L_sem:  the semantic code 0 of supervised audio frames, from the backbone lm_head.
L_cb:   codes 1..9 of the same frames from the tower, averaged over the 9 codebooks;
        it back-propagates into the backbone.

The modules are the ones inference runs (fish_omni.backbone / tower), so training and
inference share the numerics. Caches are never set up here, so every forward is a
whole sequence from position 0.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.layers import LayerContext
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts


@dataclass
class LossWeights:
    text: float = 1.0
    semantic: float = 0.5
    codebook: float = 0.5
    # KL to the frozen base model on replay rows (LoRA only; see Drama3ForTraining).
    kl: float = 0.0
    # Switch-style router load-balancing loss; only useful when routers train.
    router_aux: float = 0.0


def enable_trainable_fp8_experts() -> None:
    """Route FP8 experts through the dequantize-per-expert path when grads are needed.

    The fused FP8 Triton kernel has no backward. The per-expert path dequantizes
    each hit expert to the activation dtype and is differentiable w.r.t. the input,
    so frozen FP8 experts still pass gradients to the layers below (QLoRA-style).
    """
    if getattr(MoEExperts, "_finetune_patched", False):
        return
    original = MoEExperts.forward

    def forward(self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor) -> Tensor:
        if self.fp8 and torch.is_grad_enabled():
            return self.forward_grouped(x, topk_ids, topk_weights)
        return original(self, x, topk_ids, topk_weights)

    MoEExperts.forward = forward
    MoEExperts._finetune_patched = True


class Drama3ForTraining(nn.Module):
    """Wraps a FishOmniModel; exposes backbone / audio_decoder for PEFT targeting."""

    # The loss is already a mean; Trainer must still divide it by accumulation steps.
    accepts_loss_kwargs = False

    def __init__(
        self,
        model: FishOmniModel,
        loss_weights: LossWeights | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.audio_decoder = model.audio_decoder
        # Not registered as a submodule: the two halves above are the parameters.
        self.__dict__["omni"] = model
        self.omni_config: FishOmniConfig = model.config
        self.loss_weights = loss_weights or LossWeights()
        self.gradient_checkpointing = gradient_checkpointing
        devices = {p.device for p in self.backbone.parameters()}
        devices.add(self.audio_decoder.device)
        if len(devices) > 1:
            # transformers.Trainer reads this to leave a multi-GPU model where it is.
            self.hf_device_map = {
                f"backbone.model.layers.{i}": str(layer.input_layernorm.weight.device)
                for i, layer in enumerate(self.backbone.model.layers)
            } | {"audio_decoder": str(self.audio_decoder.device)}
        self._router_logits: list[Tensor] = []
        self._router_hooks = []
        enable_trainable_fp8_experts()
        if self.loss_weights.router_aux > 0:
            self._install_router_hooks()

    # HF Trainer / PEFT conveniences ----------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def enable_input_require_grads(self) -> None:
        """Checkpointed blocks need an input that requires grad when the embedding is
        frozen (non-reentrant checkpointing does not, but keep PEFT's hook working)."""

        def hook(module, inputs, output):
            output.requires_grad_(True)

        self.backbone.model.embed_tokens.register_forward_hook(hook)

    @property
    def device(self) -> torch.device:
        return self.backbone.model.embed_tokens.weight.device

    # Forward ---------------------------------------------------------------------------

    def _maybe_checkpoint(self, fn, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def run_backbone(self, rows: Tensor) -> Tensor:
        """rows [B, T, 1 + N] -> final-norm hidden states [B, T, H] (no caches)."""
        backbone, tower = self.backbone, self.audio_decoder
        batch, length, width = rows.shape
        flat = rows.reshape(-1, width)
        embeddings = tower.embed_backbone(flat, backbone.embed(flat[:, 0]))
        hidden = embeddings.reshape(batch, length, -1).to(backbone.residual_dtype)

        config = backbone.config
        rotary_dim = config.rotary_dim
        contexts: dict[torch.device, LayerContext] = {}
        for layer in backbone.model.layers:
            device = layer.input_layernorm.weight.device
            if device not in contexts:
                exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                frequencies = 1.0 / (config.rope_theta ** (exponents / rotary_dim))
                positions = torch.arange(length, device=device)
                angles = positions.float()[:, None] * frequencies[None, :]
                angles = torch.cat((angles, angles), dim=-1)
                contexts[device] = LayerContext(
                    input_pos=positions,
                    cos=angles.cos(),
                    sin=angles.sin(),
                    kv_len=length,
                    continue_state=False,
                    use_cache=False,
                )
            context = contexts[device]
            hidden = self._maybe_checkpoint(
                lambda h, layer=layer, context=context: layer(h, context),
                hidden.to(device),
            )
        norm = backbone.model.norm
        return norm(hidden.to(norm.weight.device)).to(backbone.dtype)

    def tower_hidden_states(
        self, hidden: Tensor, frames: Tensor, next_frames: Tensor, keep: Tensor
    ) -> list[Tensor]:
        """Teacher-forced tower over one sequence; row p is labelled with frame p + 1.

        Returns, per residual layer k (predicting code k), its hidden states at the
        rows in keep (the supervised audio targets). Mirrors FishOmniAudioDecoder.prefill.
        """
        tower = self.audio_decoder
        hidden = hidden.to(tower.device)
        frames, next_frames, keep = (t.to(tower.device) for t in (frames, next_frames, keep))
        positions = torch.arange(len(frames), device=tower.device)
        semantic = tower.is_semantic(next_frames[:, 0])
        unseen = tower.embedding_sum(
            frames, 1 + tower.config.vq_backbone_codebooks, tower.config.num_codebooks
        )
        unseen = torch.where(tower.is_semantic(frames[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(tower.shared_layers):
            shared_input = unseen if index == 0 else None
            hidden = self._maybe_checkpoint(
                lambda h, layer=layer, s=shared_input: layer(
                    h, next_frames[:, 1], semantic, positions, s
                ),
                hidden,
            )
        kept = []
        for index, layer in enumerate(tower.residual_layers):
            hidden = self._maybe_checkpoint(
                lambda h, layer=layer, index=index: layer(
                    h, next_frames[:, index + 1], semantic, positions
                ),
                hidden,
            )
            kept.append(hidden[keep])
        return kept

    def forward(
        self,
        input_rows: Tensor,
        supervised: Tensor,
        lengths: Tensor,
        regularize: Tensor | None = None,
        return_parts: bool = False,
        **_: object,
    ) -> dict:
        config = self.omni_config
        weights = self.loss_weights
        self._router_logits.clear()
        rows = input_rows.to(self.device)
        batch, length, _ = rows.shape
        lengths = lengths.to(rows.device)
        supervised = supervised.to(rows.device)

        hidden = self.run_backbone(rows)  # [B, T, H]
        hidden = hidden[:, :-1]
        targets = rows[:, 1:]  # frame p + 1 labels position p
        valid = torch.arange(length - 1, device=rows.device)[None] < (lengths[:, None] - 1)
        target_supervised = supervised[:, 1:] & valid
        tokens = targets[..., 0]
        is_semantic = (tokens >= config.semantic_start_token_id) & (
            tokens <= config.semantic_end_token_id
        )
        sem_mask = target_supervised & is_semantic
        text_mask = target_supervised & ~is_semantic

        lm_head = self.backbone.lm_head
        losses = {}
        zero = hidden.new_zeros((), dtype=torch.float32)

        def head_ce(mask: Tensor) -> Tensor:
            if not mask.any():
                return zero
            logits = self.backbone.compute_logits(hidden[mask.to(hidden.device)])
            labels = tokens[mask].to(logits.device)
            return F.cross_entropy(logits.float(), labels)

        losses["text"] = head_ce(text_mask)
        losses["semantic"] = head_ce(sem_mask)

        # Codebook loss, summed over sequences then averaged over all kept frames.
        num_codebooks = config.audio_decoder.num_codebooks
        cb_sum = torch.zeros(num_codebooks - 1, device=self.audio_decoder.device)
        cb_count = 0
        for b in range(batch):
            n = int(lengths[b]) - 1
            keep = sem_mask[b, :n]
            if not keep.any():
                continue
            states = self.tower_hidden_states(
                hidden[b, :n], rows[b, :n], targets[b, :n], keep
            )
            labels = targets[b, :n][keep.to(targets.device)].to(self.audio_decoder.device)
            for k, (state, head) in enumerate(zip(states, self.audio_decoder.codebook_heads)):
                logits = head(state).float()
                cb_sum[k] = cb_sum[k] + F.cross_entropy(
                    logits, labels[:, k + 2], reduction="sum"
                )
            cb_count += int(keep.sum())
        losses["codebook"] = (
            (cb_sum / cb_count).mean() if cb_count else zero.to(cb_sum.device)
        )

        device = lm_head.weight.device
        loss = (
            weights.text * losses["text"].to(device)
            + weights.semantic * losses["semantic"].to(device)
            + weights.codebook * losses["codebook"].to(device)
        )

        if weights.kl > 0 and regularize is not None and regularize.any():
            losses["kl"] = self.kl_to_base(rows, hidden, sem_mask, regularize)
            loss = loss + weights.kl * losses["kl"].to(device)
        if weights.router_aux > 0 and self._router_logits:
            losses["router_aux"] = self.router_aux_loss()
            loss = loss + weights.router_aux * losses["router_aux"].to(device)

        out = {"loss": loss}
        if return_parts:
            out["parts"] = {k: v.detach().float().cpu() for k, v in losses.items()}
            out["frames"] = int(sem_mask.sum())
        return out

    # Regularizers -----------------------------------------------------------------------

    def kl_to_base(
        self, rows: Tensor, hidden: Tensor, sem_mask: Tensor, regularize: Tensor
    ) -> Tensor:
        """KL(base || tuned) of the semantic-token distribution on replay rows.

        The base model is this model with its LoRA adapters disabled; set by
        attach_base_switch (peft_utils). Restricted to eos + the semantic band, the
        tokens the sampler draws from.
        """
        switch = getattr(self, "_disable_adapter", None)
        if switch is None:
            raise RuntimeError("KL regularization needs a LoRA model (attach_base_switch)")
        mask = sem_mask & regularize.to(sem_mask.device)[:, None]
        if not mask.any():
            return hidden.new_zeros((), dtype=torch.float32)
        config = self.omni_config
        band = torch.cat(
            (
                torch.tensor([config.eos_token_id]),
                torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1),
            )
        ).to(self.backbone.lm_head.weight.device)
        tuned = self.backbone.compute_logits(hidden[mask])[:, band].float()
        with torch.no_grad(), switch():
            base_hidden = self.run_backbone(rows)[:, :-1]
            base = self.backbone.compute_logits(base_hidden[mask])[:, band].float()
        return F.kl_div(
            tuned.log_softmax(-1), base.log_softmax(-1), log_target=True, reduction="batchmean"
        )

    def _install_router_hooks(self) -> None:
        def hook(module, inputs, output):
            if module.training or torch.is_grad_enabled():
                self._router_logits.append(output)

        for layer in self.backbone.model.layers:
            self._router_hooks.append(layer.mlp.gate.register_forward_hook(hook))

    def router_aux_loss(self) -> Tensor:
        """Switch Transformer load balancing: E * sum_i f_i * P_i, averaged over layers."""
        top_k = self.backbone.config.num_experts_per_tok
        total = None
        for logits in self._router_logits:
            probabilities = logits.float().softmax(-1)
            experts = probabilities.shape[-1]
            chosen = probabilities.topk(top_k, dim=-1).indices
            fraction = F.one_hot(chosen, experts).float().sum(1).mean(0) / top_k
            term = experts * (fraction.to(probabilities.device) * probabilities.mean(0)).sum()
            total = term if total is None else total + term.to(total.device)
        self._router_logits.clear()
        return total / max(1, len(self.backbone.model.layers))


def frames_to_seconds(frames: int) -> float:
    return frames * 2048 / 44100


def count_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    buffers = sum(b.numel() for b in module.buffers())
    return trainable, total + buffers


__all__ = [
    "Drama3ForTraining",
    "LossWeights",
    "count_parameters",
    "enable_trainable_fp8_experts",
    "frames_to_seconds",
]
