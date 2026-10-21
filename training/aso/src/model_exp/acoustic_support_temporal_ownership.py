"""Temporal ASO trained through oracle measured-acoustic note support."""
from __future__ import annotations

from typing import Any

import torch

from .temporal_hybrid_set_ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)


class AcousticSupportTemporalOwnership(TemporalHybridSetOwnership):
    """Apply fractional STFT-frame support before the training loss.

    The measured endpoint comes from isolated-note target energy and is an
    oracle evaluation/training signal.  This class is therefore a ceiling, not
    a deployable score-only method.
    """

    hop_length = 512
    n_fft = 2048

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6",
            phase_mode="mixture",
            temporal_blocks=2,
            exact_mixture_additivity=False,
        )
        if (
            config.feature_set != "essential6"
            or config.phase_mode != "mixture"
            or config.magnitude_reference != "mixture"
            or config.exact_mixture_additivity
        ):
            raise ValueError("acoustic-support ASO requires nonexact mixture-magnitude allocation")
        super().__init__(config)
        self._fractional_support: torch.Tensor | None = None

    def _refine_output(
        self, features: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        del features
        if self._fractional_support is None:
            raise RuntimeError("fractional acoustic support was not supplied")
        support = self._fractional_support[:, :, None, :].to(output.real.dtype)
        weighted = output * support
        target_magnitude = output.abs().sum(1, keepdim=True)
        weighted_magnitude = weighted.abs().sum(1, keepdim=True)
        return torch.where(
            weighted_magnitude > 1e-20,
            weighted * target_magnitude / weighted_magnitude.clamp_min(1e-20),
            torch.zeros_like(weighted),
        )

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
        del midi_offsets
        if onsets is None or acoustic_offsets is None:
            raise ValueError("measured acoustic onset/offset support is required")
        centers = (
            torch.arange(mixture.shape[-1], device=mixture.device, dtype=torch.long)
            * self.hop_length
        )
        left = centers[None, None] - self.n_fft // 2
        right = centers[None, None] + self.n_fft // 2
        overlap = (
            torch.minimum(right, acoustic_offsets[..., None])
            - torch.maximum(left, onsets[..., None])
        ).clamp_min(0)
        self._fractional_support = overlap.to(mixture.real.dtype) / self.n_fft
        try:
            output, diagnostics = super().forward(
                mixture,
                anchors,
                raw,
                pitches,
                onsets,
                acoustic_offsets,
                acoustic_offsets,
                harmonic_logits,
                percussive_logits,
                hard_phase=hard_phase,
            )
            diagnostics["supported_note_frame_fraction"] = (
                self._fractional_support > 0
            ).to(mixture.real.dtype).mean()
            return output, diagnostics
        finally:
            self._fractional_support = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "oracle_acoustic_support_temporal_ownership_v1",
                "support": "fractional overlap of STFT frame with measured acoustic note interval",
                "support_application": "project, renormalize among supported notes, then train through projection",
                "endpoint_authority": "isolated-target measured acoustic endpoint; oracle and nondeployable",
                "mixture_additivity": "only on TF bins with at least one acoustically supported proposal",
            }
        )
        return value


__all__ = ["AcousticSupportTemporalOwnership"]
