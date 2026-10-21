"""Lifecycle-free, permutation-equivariant sparse note ownership."""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .adaptive_sparse_ownership import (
    AdaptiveSparseOwnership,
    AdaptiveSparseOwnershipConfig,
    sparsemax_notes,
)


class AdaptiveSetOwnership(AdaptiveSparseOwnership):
    """Let each note compare itself with a mean/max summary of its peers."""

    def __init__(self, config: AdaptiveSparseOwnershipConfig | None = None) -> None:
        super().__init__(config)
        channels = self.config.hidden_channels
        self.set_fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1),
            nn.SiLU(),
        )
        self.correction = nn.Conv2d(channels, 1, 1)
        self.confidence = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)

    def forward(
        self,
        mixture: torch.Tensor,
        anchors: torch.Tensor,
        raw: torch.Tensor,
        pitches: torch.Tensor,
        onsets: torch.Tensor | None = None,
        midi_offsets: torch.Tensor | None = None,
        acoustic_offsets: torch.Tensor | None = None,
        harmonic_logits: torch.Tensor | None = None,
        percussive_logits: torch.Tensor | None = None,
        *,
        confidence_mode: Literal["adaptive", "zero", "one"] = "adaptive",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del anchors, onsets, midi_offsets, acoustic_offsets, harmonic_logits, percussive_logits
        if mixture.ndim != 3 or raw.ndim != 4 or raw.shape[0] != mixture.shape[0]:
            raise ValueError("invalid mixture/note-bank geometry")
        if raw.shape[1] < 2 or raw.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("allocator requires at least two aligned notes")
        features, base = self._features(mixture, raw, pitches)
        batch, notes, _, frequency, frames = features.shape
        local = F.silu(self.input(features.flatten(0, 1)))
        local = self.blocks(local).view(batch, notes, -1, frequency, frames)
        mean = local.mean(1, keepdim=True).expand_as(local)
        maximum = local.amax(1, keepdim=True).expand_as(local)
        memory = self.set_fuse(torch.cat((local, mean, maximum), 2).flatten(0, 1))
        delta = self.config.maximum_simplex_correction * torch.tanh(
            self.correction(memory)[:, 0]
        ).view(batch, notes, frequency, frames)
        sparse = sparsemax_notes(base + delta)
        proposal_total = raw.abs().sum(1, keepdim=True)
        sparse = torch.where(proposal_total > 1e-20, sparse, torch.zeros_like(sparse))
        gate_note = self.confidence(memory)[:, 0].view(batch, notes, frequency, frames)
        gate = torch.sigmoid((base * gate_note).sum(1, keepdim=True))
        if confidence_mode == "zero":
            gate = torch.zeros_like(gate)
        elif confidence_mode == "one":
            gate = torch.ones_like(gate)
        elif confidence_mode != "adaptive":
            raise ValueError(f"invalid confidence mode: {confidence_mode}")
        ownership = (1 - gate) * base + gate * sparse
        output = ownership * mixture[:, None]
        return output, {
            "ownership": ownership,
            "base_ownership": base,
            "sparse_ownership": sparse,
            "confidence": gate,
            "mean_abs_correction": delta.abs().mean(),
            "mean_confidence": gate.mean(),
            "sum_error_max_abs": (output.sum(1) - mixture).abs().amax(),
        }

    def contract(self) -> dict[str, Any]:
        result = super().contract()
        result.update({
            "kind": "adaptive_set_ownership_v1",
            "cross_note_context": "per-TF permutation-invariant mean and maximum",
            "negotiation": "shared Deep-Sets correction; no attention or iteration",
        })
        return result


__all__ = ["AdaptiveSetOwnership"]
