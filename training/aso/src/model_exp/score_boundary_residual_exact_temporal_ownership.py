"""Exact-additive Temporal ASO with a boundary-local score residual."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .temporal_hybrid_set_ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)


@dataclass(frozen=True)
class ScoreBoundaryResidualConfig:
    boundary_scale_samples: int = 2048
    maximum_boundary_correction: float = 0.15


class ScoreBoundaryResidualExactTemporalOwnership(TemporalHybridSetOwnership):
    """Freeze an exact Temporal ASO and learn only an outside-boundary residual.

    The residual is structurally zero inside the symbolic score interval.  A
    boundary that lies beyond the crop therefore has no effect on that side;
    this prevents saturated coordinates for incomplete crop-edge notes from
    globally changing their ownership.
    """

    hop_length = 512

    def __init__(
        self,
        config: TemporalHybridOwnershipConfig | None = None,
        boundary_config: ScoreBoundaryResidualConfig | None = None,
    ) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6",
            phase_mode="mixture",
            temporal_blocks=2,
            exact_mixture_additivity=True,
        )
        if (
            config.feature_set != "essential6"
            or config.phase_mode != "mixture"
            or config.magnitude_reference != "mixture"
            or not config.exact_mixture_additivity
        ):
            raise ValueError(
                "boundary residual requires essential6, mixture magnitude/phase, "
                "and exact additivity"
            )
        super().__init__(config)
        self.boundary_config = boundary_config or ScoreBoundaryResidualConfig()
        if (
            self.boundary_config.boundary_scale_samples < 1
            or not 0 < self.boundary_config.maximum_boundary_correction <= 0.5
        ):
            raise ValueError("invalid boundary residual configuration")
        self.boundary_delta = nn.Conv2d(8, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.boundary_delta.weight)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.boundary_delta.weight.requires_grad_(True)
        self._boundary_coordinates: torch.Tensor | None = None
        self._outside_support: torch.Tensor | None = None

    def _refine_magnitude_delta(
        self, features: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        if self._boundary_coordinates is None or self._outside_support is None:
            raise RuntimeError("symbolic boundary coordinates were not supplied")
        batch, notes, _, frequency, frames = features.shape
        coordinates = self._boundary_coordinates[:, :, :, None, :].expand(
            -1, -1, -1, frequency, -1
        ).to(features.dtype)
        inputs = torch.cat((features, coordinates), dim=2)
        correction = self.boundary_config.maximum_boundary_correction * torch.tanh(
            self.boundary_delta(inputs.flatten(0, 1))[:, 0]
        ).view(batch, notes, frequency, frames)
        return delta + self._outside_support[:, :, None, :] * correction

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
        if onsets is None or midi_offsets is None:
            raise ValueError("symbolic onsets and MIDI offsets are required")
        centers = (
            torch.arange(
                mixture.shape[-1], device=mixture.device, dtype=mixture.real.dtype
            )
            * self.hop_length
        )
        scale = self.boundary_config.boundary_scale_samples
        after_onset = torch.tanh(
            (centers[None, None] - onsets[..., None].to(centers.dtype)) / scale
        )
        before_offset = torch.tanh(
            (midi_offsets[..., None].to(centers.dtype) - centers[None, None]) / scale
        )
        # Only the score-violating sides can change: before an onset or after a
        # note-off.  Inside [onset, note-off], this support is exactly zero.
        pre_onset = (-after_onset).clamp_min(0.0)
        post_offset = (-before_offset).clamp_min(0.0)
        self._boundary_coordinates = torch.stack((after_onset, before_offset), dim=2)
        self._outside_support = torch.maximum(pre_onset, post_offset)
        try:
            return super().forward(
                mixture,
                anchors,
                raw,
                pitches,
                onsets,
                midi_offsets,
                acoustic_offsets,
                harmonic_logits,
                percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._boundary_coordinates = None
            self._outside_support = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "score_boundary_residual_exact_temporal_ownership_v1",
                "parent": "frozen exact-additive essential6 Temporal ASO",
                "trainable_adapter": "zero-initialized 8-to-1 3x3 convolution (72 weights)",
                "score_coordinates": (
                    "signed tanh distance to symbolic onset and MIDI note-off"
                ),
                "boundary_support": (
                    "smoothly nonzero only before symbolic onset or after MIDI note-off"
                ),
                "maximum_boundary_correction": (
                    self.boundary_config.maximum_boundary_correction
                ),
                "body_invariance": "boundary residual is exactly zero inside the score interval",
                "excluded_inputs": (
                    "acoustic endpoint, lifecycle state, instrument identity, background lane"
                ),
                "mixture_additivity": "exact complex STFT sum under shared mixture phase",
            }
        )
        return value


__all__ = [
    "ScoreBoundaryResidualConfig",
    "ScoreBoundaryResidualExactTemporalOwnership",
]
