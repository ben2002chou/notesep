"""Acoustic-boundary Temporal ASO with a relative ownership update."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)


class AcousticMultiplicativeTemporalOwnership(TemporalHybridSetOwnership):
    """Predict bounded log multipliers relative to raw ownership.

    A zero network output preserves the raw magnitude-share prior.  Unlike an
    absolute sparsemax displacement, a weak proposal can change only
    multiplicatively.  The model consumes the measured acoustic endpoint used
    by the frozen occurrence separator, not the nominal MIDI note-off.
    """

    hop_length = 512
    boundary_scale_samples = 2048

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6",
            phase_mode="mixture",
            temporal_blocks=2,
            magnitude_reference="mixture",
            exact_mixture_additivity=True,
        )
        if config.feature_set != "essential6" or config.phase_mode != "mixture":
            raise ValueError(
                "acoustic multiplicative ownership requires essential6 and mixture phase"
            )
        super().__init__(config)
        original = self.input
        replacement = nn.Conv2d(
            original.in_channels + 2,
            original.out_channels,
            original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:, : original.in_channels].copy_(original.weight)
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        self.input = replacement
        self._acoustic_coordinates: torch.Tensor | None = None

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, base = super()._features(mixture, raw, pitches)
        if self._acoustic_coordinates is None:
            raise RuntimeError("acoustic onset/offset coordinates were not supplied")
        coordinates = self._acoustic_coordinates[:, :, :, None, :].expand(
            -1, -1, -1, features.shape[-2], -1
        ).to(features.dtype)
        return torch.cat((features, coordinates), dim=2), base

    def _allocate_ownership(
        self, base: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        # Exponentiated-gradient/mirror-descent update on the simplex.  Exact
        # zero proposals remain impossible owners; all-empty bins remain
        # rejected rather than being assigned arbitrarily.
        work = base.float()
        weights = work * torch.exp(delta.float())
        total = weights.sum(1, keepdim=True)
        ownership = torch.where(
            total > 1e-20,
            weights / total.clamp_min(1e-20),
            torch.zeros_like(weights),
        )
        return ownership.to(base.dtype)

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
        hard_phase: bool = True,
    ):
        if onsets is None or acoustic_offsets is None:
            raise ValueError("onsets and measured acoustic offsets are required")
        del midi_offsets
        centers = (
            torch.arange(
                mixture.shape[-1], device=mixture.device, dtype=mixture.real.dtype
            )
            * self.hop_length
        )
        onset_distance = torch.tanh(
            (centers[None, None] - onsets[..., None].to(centers.dtype))
            / self.boundary_scale_samples
        )
        offset_distance = torch.tanh(
            (acoustic_offsets[..., None].to(centers.dtype) - centers[None, None])
            / self.boundary_scale_samples
        )
        self._acoustic_coordinates = torch.stack(
            (onset_distance, offset_distance), dim=2
        )
        try:
            return super().forward(
                mixture,
                anchors,
                raw,
                pitches,
                onsets,
                None,
                acoustic_offsets,
                harmonic_logits,
                percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._acoustic_coordinates = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "acoustic_multiplicative_temporal_ownership_v1",
                "features": (
                    "essential6 plus signed onset distance and signed measured "
                    "acoustic-offset distance"
                ),
                "onset_coordinate": "tanh((frame_center-onset)/2048)",
                "offset_coordinate": "tanh((acoustic_offset-frame_center)/2048)",
                "boundary_authority": "measured acoustic endpoint",
                "ownership_update": (
                    "normalize(base_ownership * exp(bounded_learned_log_gain))"
                ),
                "magnitude_allocation": (
                    "relative-multiplicative ownership over the configured shared "
                    "magnitude carrier"
                ),
                "zero_proposal_policy": "cannot acquire ownership",
                "all_empty_policy": "reject bin",
                "excluded_inputs": (
                    "symbolic MIDI offset, lifecycle state, instrument identity, background lane"
                ),
            }
        )
        return value


__all__ = ["AcousticMultiplicativeTemporalOwnership"]
