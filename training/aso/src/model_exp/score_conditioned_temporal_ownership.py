"""Temporal sparse ownership with one literal score-on/off input channel."""
from __future__ import annotations

from typing import Any
import math

import torch
from torch import nn
import torch.nn.functional as F

from .temporal_hybrid_set_ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)


class ScoreOnOffTemporalOwnership(TemporalHybridSetOwnership):
    """Essential-six refiner plus a frame-centre score activity bit."""

    hop_length = 512

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6", phase_mode="mixture", temporal_blocks=2
        )
        if config.feature_set != "essential6" or config.phase_mode != "mixture":
            raise ValueError("score-on/off model requires essential6 and mixture phase")
        super().__init__(config)
        original = self.input
        replacement = nn.Conv2d(
            original.in_channels + 1, original.out_channels,
            original.kernel_size, stride=original.stride,
            padding=original.padding, bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight[:, :-1].copy_(original.weight)
            replacement.weight[:, -1].zero_()
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        self.input = replacement
        self._activity: torch.Tensor | None = None

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, base = super()._features(mixture, raw, pitches)
        if self._activity is None:
            raise RuntimeError("score activity was not supplied")
        activity = self._activity[:, :, None, None, :].expand(
            -1, -1, 1, features.shape[-2], -1
        ).to(features.dtype)
        return torch.cat((features, activity), dim=2), base

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
        offsets = acoustic_offsets if acoustic_offsets is not None else midi_offsets
        if onsets is None or offsets is None:
            raise ValueError("onsets and offsets are required")
        frames = mixture.shape[-1]
        centers = torch.arange(frames, device=mixture.device) * self.hop_length
        self._activity = (
            (centers[None, None] >= onsets[..., None])
            & (centers[None, None] < offsets[..., None])
        )
        try:
            return super().forward(
                mixture, anchors, raw, pitches, onsets, midi_offsets,
                acoustic_offsets, harmonic_logits, percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._activity = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update({
            "kind": "score_onoff_temporal_ownership_v1",
            "features": "essential6 plus one binary frame-centre score-on/off channel",
            "excluded_inputs": "release, distance-to-boundary, lifecycle, instrument identity, background lane",
            "score_activity": "one iff onset <= frame_center < offset",
        })
        return value


class _OffsetConditionedInput(nn.Module):
    """Frozen incumbent input plus a zero-initialized offset-only branch."""

    def __init__(self, incumbent: nn.Conv2d) -> None:
        super().__init__()
        self.base = incumbent
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.offset = nn.Conv2d(
            1, incumbent.out_channels, incumbent.kernel_size,
            stride=incumbent.stride, padding=incumbent.padding, bias=False,
        )
        nn.init.zeros_(self.offset.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value[:, :-1]) + self.offset(value[:, -1:])


class OffsetDistanceTemporalOwnership(TemporalHybridSetOwnership):
    """Selected Temporal ASO plus one smooth distance-to-acoustic-offset input."""

    hop_length = 512
    half_window = 1024

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6", phase_mode="mixture", temporal_blocks=2
        )
        if config.feature_set != "essential6" or config.phase_mode != "mixture":
            raise ValueError("offset-distance model requires essential6 and mixture phase")
        super().__init__(config)
        self.input = _OffsetConditionedInput(self.input)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.input.offset.weight.requires_grad_(True)
        self._offset_distance: torch.Tensor | None = None

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, base = super()._features(mixture, raw, pitches)
        if self._offset_distance is None:
            raise RuntimeError("acoustic offset distance was not supplied")
        distance = self._offset_distance[:, :, None, None, :].expand(
            -1, -1, 1, features.shape[-2], -1
        ).to(features.dtype)
        return torch.cat((features, distance), dim=2), base

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
        offsets = acoustic_offsets if acoustic_offsets is not None else midi_offsets
        if offsets is None:
            raise ValueError("acoustic offsets are required")
        centers = torch.arange(mixture.shape[-1], device=mixture.device) * self.hop_length
        self._offset_distance = torch.tanh(
            (offsets[..., None] - centers[None, None]).to(mixture.real.dtype)
            / self.half_window
        )
        try:
            return super().forward(
                mixture, anchors, raw, pitches, onsets, midi_offsets,
                acoustic_offsets, harmonic_logits, percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._offset_distance = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update({
            "kind": "offset_distance_temporal_ownership_v1",
            "features": "essential6 plus one tanh signed distance to acoustic offset",
            "offset_distance": "tanh((acoustic_offset-frame_center)/(n_fft/2))",
            "trainable_adapter": "zero-initialized 1x3x3 offset-input convolution only",
            "excluded_inputs": "onset, lifecycle state, instrument identity, background lane",
        })
        return value


class PostOffsetDistanceTemporalOwnership(OffsetDistanceTemporalOwnership):
    """Selected Temporal ASO with a feature supported only after the offset."""

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
        offsets = acoustic_offsets if acoustic_offsets is not None else midi_offsets
        if offsets is None:
            raise ValueError("acoustic offsets are required")
        centers = torch.arange(mixture.shape[-1], device=mixture.device) * self.hop_length
        self._offset_distance = torch.tanh(
            (centers[None, None] - offsets[..., None]).to(mixture.real.dtype)
            / self.half_window
        ).clamp_min(0.0)
        try:
            return TemporalHybridSetOwnership.forward(
                self, mixture, anchors, raw, pitches, onsets, midi_offsets,
                acoustic_offsets, harmonic_logits, percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._offset_distance = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update({
            "kind": "post_offset_distance_temporal_ownership_v2",
            "features": "essential6 plus one post-offset-only smooth distance channel",
            "offset_distance": "max(0,tanh((frame_center-acoustic_offset)/(n_fft/2)))",
            "body_invariance": "offset adapter input is exactly zero before acoustic offset",
        })
        return value


class PostOffsetSparsemaxResidualOwnership(TemporalHybridSetOwnership):
    """Frozen Temporal ASO plus a post-offset-only sparsemax-logit residual."""

    hop_length = 512
    half_window = 1024

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6", phase_mode="mixture", temporal_blocks=2
        )
        if config.feature_set != "essential6" or config.phase_mode != "mixture":
            raise ValueError("post-offset sparsemax residual requires essential6 and mixture phase")
        super().__init__(config)
        self.post_offset_delta = nn.Conv2d(6, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.post_offset_delta.weight)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.post_offset_delta.weight.requires_grad_(True)
        self._post_offset_weight: torch.Tensor | None = None

    def _refine_magnitude_delta(
        self, features: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        if self._post_offset_weight is None:
            raise RuntimeError("post-offset weight was not supplied")
        batch, notes, channels, frequency, frames = features.shape
        correction = self.config.maximum_simplex_correction * torch.tanh(
            self.post_offset_delta(features.flatten(0, 1))[:, 0]
        ).view(batch, notes, frequency, frames)
        return delta + self._post_offset_weight[:, :, None, :] * correction

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
        offsets = acoustic_offsets if acoustic_offsets is not None else midi_offsets
        if offsets is None:
            raise ValueError("acoustic offsets are required")
        centers = torch.arange(mixture.shape[-1], device=mixture.device) * self.hop_length
        self._post_offset_weight = torch.tanh(
            (centers[None, None] - offsets[..., None]).to(mixture.real.dtype)
            / self.half_window
        ).clamp_min(0.0)
        try:
            return super().forward(
                mixture, anchors, raw, pitches, onsets, midi_offsets,
                acoustic_offsets, harmonic_logits, percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._post_offset_weight = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update({
            "kind": "post_offset_sparsemax_residual_ownership_v3",
            "post_offset_weight": "max(0,tanh((frame_center-acoustic_offset)/(n_fft/2)))",
            "trainable_adapter": "zero-initialized essential6-to-one 3x3 sparsemax-logit residual",
            "trainable_parameters": 54,
            "body_invariance": "adapter correction is exactly zero before acoustic offset",
            "magnitude_allocation": "frozen Temporal ASO logits plus bounded post-offset residual, then source-axis sparsemax",
        })
        return value


class PostOffsetAttenuationOwnership(TemporalHybridSetOwnership):
    """Frozen Temporal ASO plus a learned, non-redistributing tail attenuator."""

    hop_length = 512
    half_window = 1024

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6", phase_mode="mixture", temporal_blocks=2
        )
        if config.feature_set != "essential6" or config.phase_mode != "mixture":
            raise ValueError("post-offset attenuation requires essential6 and mixture phase")
        super().__init__(config)
        self.post_offset_attenuation = nn.Conv2d(6, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.post_offset_attenuation.weight)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.post_offset_attenuation.weight.requires_grad_(True)
        self._post_offset_weight: torch.Tensor | None = None

    def _refine_output(
        self, features: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self._post_offset_weight is None:
            raise RuntimeError("post-offset weight was not supplied")
        batch, notes, channels, frequency, frames = features.shape
        logits = self.post_offset_attenuation(features.flatten(0, 1))[:, 0].view(
            batch, notes, frequency, frames
        )
        # This shifted softplus is exactly zero at the zero initialization but
        # has derivative 0.5, unlike a squared or ReLU-zero parameterization.
        strength = (F.softplus(logits) - math.log(2.0)).clamp(0.0, 1.0)
        scale = 1.0 - self._post_offset_weight[:, :, None, :] * strength
        return output * scale

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
        offsets = acoustic_offsets if acoustic_offsets is not None else midi_offsets
        if offsets is None:
            raise ValueError("acoustic offsets are required")
        centers = torch.arange(mixture.shape[-1], device=mixture.device) * self.hop_length
        self._post_offset_weight = torch.tanh(
            (centers[None, None] - offsets[..., None]).to(mixture.real.dtype)
            / self.half_window
        ).clamp_min(0.0)
        try:
            return super().forward(
                mixture, anchors, raw, pitches, onsets, midi_offsets,
                acoustic_offsets, harmonic_logits, percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._post_offset_weight = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update({
            "kind": "post_offset_nonredistributing_attenuation_v4",
            "post_offset_weight": "max(0,tanh((frame_center-acoustic_offset)/(n_fft/2)))",
            "trainable_adapter": "zero-initialized essential6-to-one 3x3 one-sided attenuation head",
            "trainable_parameters": 54,
            "initialization": "shifted-softplus strength is exactly zero with nonzero gradient",
            "body_invariance": "output is exactly unchanged before acoustic offset",
            "magnitude_allocation": "frozen Temporal ASO sparsemax followed by post-offset attenuation only",
            "redistribute_removed_energy": False,
            "magnitude_partition_consistency": False,
        })
        return value


__all__ = [
    "ScoreOnOffTemporalOwnership",
    "OffsetDistanceTemporalOwnership",
    "PostOffsetDistanceTemporalOwnership",
    "PostOffsetSparsemaxResidualOwnership",
    "PostOffsetAttenuationOwnership",
]
