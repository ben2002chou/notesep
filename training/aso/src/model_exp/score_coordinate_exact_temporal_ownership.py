"""Exactly additive Temporal ASO with deployable score-boundary coordinates."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .temporal_hybrid_set_ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)


class ScoreCoordinateExactTemporalOwnership(TemporalHybridSetOwnership):
    """Add signed onset and MIDI-offset distances to an exact partition.

    The coordinates are symbolic score inputs available at inference.  The
    measured acoustic endpoint is deliberately not exposed to the network.
    """

    hop_length = 512
    boundary_scale_samples = 2048

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
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
                "score-coordinate model requires essential6, mixture phase, "
                "mixture magnitude, and exact additivity"
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
        self._score_coordinates: torch.Tensor | None = None

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, base = super()._features(mixture, raw, pitches)
        if self._score_coordinates is None:
            raise RuntimeError("symbolic onset/offset coordinates were not supplied")
        coordinates = self._score_coordinates[:, :, :, None, :].expand(
            -1, -1, -1, features.shape[-2], -1
        ).to(features.dtype)
        return torch.cat((features, coordinates), dim=2), base

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
        frames = mixture.shape[-1]
        centers = (
            torch.arange(frames, device=mixture.device, dtype=mixture.real.dtype)
            * self.hop_length
        )
        onset_distance = torch.tanh(
            (centers[None, None] - onsets[..., None].to(centers.dtype))
            / self.boundary_scale_samples
        )
        offset_distance = torch.tanh(
            (midi_offsets[..., None].to(centers.dtype) - centers[None, None])
            / self.boundary_scale_samples
        )
        self._score_coordinates = torch.stack(
            (onset_distance, offset_distance), dim=2
        )
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
            self._score_coordinates = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "score_coordinate_exact_temporal_ownership_v1",
                "features": (
                    "essential6 plus signed distance from frame centre to "
                    "symbolic onset and signed distance from frame centre to MIDI offset"
                ),
                "onset_coordinate": "tanh((frame_center-onset)/2048)",
                "offset_coordinate": "tanh((midi_offset-frame_center)/2048)",
                "boundary_authority": "symbolic score only; acoustic endpoint is supervision/evaluation only",
                "excluded_inputs": "acoustic endpoint, lifecycle state, instrument identity, background lane",
                "mixture_additivity": "exact complex STFT sum under shared mixture phase",
            }
        )
        return value


__all__ = ["ScoreCoordinateExactTemporalOwnership"]
