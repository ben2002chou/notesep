"""Exact-additive complex residual on a frozen mixture-phase Temporal ASO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .temporal_hybrid_set_ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
    _unit,
)


@dataclass(frozen=True)
class ZeroSumComplexConfig:
    hidden_channels: int = 16
    maximum_relative_residual: float = 0.15


class ZeroSumComplexExactTemporalOwnership(TemporalHybridSetOwnership):
    """Learn bounded complex changes whose source-axis sum is exactly zero.

    The parent produces ``a_i X`` with nonnegative sparse ownership ``a`` and
    mixture phase.  A small temporal head proposes radial/quadrature updates in
    mixture-relative coordinates.  An ownership-weighted source-axis
    projection removes their common sum before the updates are added.
    """

    def __init__(
        self,
        config: TemporalHybridOwnershipConfig | None = None,
        complex_config: ZeroSumComplexConfig | None = None,
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
                "zero-sum complex residual requires an exact essential6 "
                "mixture-magnitude/phase parent"
            )
        super().__init__(config)
        self.complex_config = complex_config or ZeroSumComplexConfig()
        if (
            self.complex_config.hidden_channels < 4
            or not 0 < self.complex_config.maximum_relative_residual <= 0.5
        ):
            raise ValueError("invalid zero-sum complex configuration")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        channels = self.complex_config.hidden_channels
        self.complex_head = nn.Sequential(
            nn.Conv2d(7, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, (1, 5), padding=(0, 2), groups=channels
            ),
            nn.SiLU(),
            nn.Conv2d(channels, 2, 1),
        )
        nn.init.zeros_(self.complex_head[-1].weight)
        nn.init.zeros_(self.complex_head[-1].bias)
        self._relative_phase_sine: torch.Tensor | None = None
        self._last_complex_residual: torch.Tensor | None = None

    def _refine_output(
        self, features: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self._relative_phase_sine is None:
            raise RuntimeError("relative phase sine was not supplied")
        batch, notes, _, frequency, frames = features.shape
        phase_sine = self._relative_phase_sine[:, :, None].to(features.dtype)
        inputs = torch.cat((features, phase_sine), dim=2)
        components = self.complex_head(inputs.flatten(0, 1)).view(
            batch, notes, 2, frequency, frames
        )

        mixture = output.sum(1)
        mixture_magnitude = mixture.abs()
        mixture_unit = _unit(mixture, self.config.epsilon)
        ownership = torch.where(
            mixture_magnitude[:, None] > self.config.epsilon,
            output.abs() / mixture_magnitude[:, None].clamp_min(self.config.epsilon),
            torch.full_like(output.real, 1.0 / notes),
        )
        relative = torch.complex(
            torch.tanh(components[:, :, 0]),
            torch.tanh(components[:, :, 1]),
        )
        proposal = (
            self.complex_config.maximum_relative_residual
            * mixture_magnitude[:, None]
            * mixture_unit[:, None]
            * ownership.clamp_min(0).sqrt()
            * relative
        )
        # In exact arithmetic sum_i correction_i = 0 because sum_i a_i = 1.
        correction = proposal - ownership * proposal.sum(1, keepdim=True)
        self._last_complex_residual = correction
        return output + correction

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
        raw_unit = _unit(raw, self.config.epsilon)
        mixture_unit = _unit(mixture, self.config.epsilon)[:, None]
        self._relative_phase_sine = (raw_unit.conj() * mixture_unit).imag
        self._last_complex_residual = None
        try:
            output, diagnostics = super().forward(
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
            assert self._last_complex_residual is not None
            diagnostics["mean_complex_delta"] = (
                self._last_complex_residual.abs()
                / mixture.abs()[:, None].clamp_min(self.config.epsilon)
            ).mean()
            diagnostics["maximum_complex_delta"] = (
                self._last_complex_residual.abs()
                / mixture.abs()[:, None].clamp_min(self.config.epsilon)
            ).amax()
            return output, diagnostics
        finally:
            self._relative_phase_sine = None
            self._last_complex_residual = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "zero_sum_complex_exact_temporal_ownership_v1",
                "parent": "frozen exact-additive mixture-phase Temporal ASO",
                "complex_input": (
                    "essential6 plus signed raw-versus-mixture phase sine"
                ),
                "complex_head": (
                    "3x3 local convolution, 1x5 temporal depthwise convolution, "
                    "then radial/quadrature 1x1 output"
                ),
                "complex_coordinates": "radial and quadrature relative to mixture phase",
                "maximum_relative_residual": (
                    self.complex_config.maximum_relative_residual
                ),
                "complex_projection": (
                    "proposal_i - ownership_i * sum_j proposal_j"
                ),
                "complex_mixture_consistency": True,
                "magnitude_partition_consistency": False,
                "excluded_inputs": (
                    "target waveform, acoustic endpoint, lifecycle state, instrument identity, background lane"
                ),
            }
        )
        return value


__all__ = ["ZeroSumComplexConfig", "ZeroSumComplexExactTemporalOwnership"]
