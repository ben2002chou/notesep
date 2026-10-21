"""Two-branch occurrence separator for harmonic and off-harmonic audio."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .occurrence_joint_tfc_tdf_separator import OccurrenceJointTFCTDFSeparator


_CONTRACT_KIND = "occurrence_dual_harmonic_residual_separator_v1"


class OccurrenceDualHarmonicResidualSeparator(nn.Module):
    """Run independent ownership separators on complementary mixture views."""

    def __init__(self, preset: str = "30m") -> None:
        super().__init__()
        self.preset = str(preset)
        self.harmonic_branch = OccurrenceJointTFCTDFSeparator.from_preset(
            self.preset, ownership_parameterization=True
        )
        self.residual_branch = OccurrenceJointTFCTDFSeparator.from_preset(
            self.preset, ownership_parameterization=True
        )
        self.config = self.harmonic_branch.config

    @classmethod
    def from_preset(cls, preset: str) -> "OccurrenceDualHarmonicResidualSeparator":
        return cls(preset)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def contract(self) -> dict[str, Any]:
        return {
            "kind": _CONTRACT_KIND,
            "contract_version": 1,
            "preset": self.preset,
            "architecture": {
                "layout": "independent_harmonic_and_residual_ownership_branches",
                "merge": "waveform_sum",
                "harmonic_branch": self.harmonic_branch.contract(self.preset),
                "residual_branch": self.residual_branch.contract(self.preset),
            },
            "inputs": {
                "harmonic_branch": "pitch-relative harmonic projection of mixture",
                "residual_branch": "mixture minus pitch-relative harmonic projection",
                "query": ["pitch", "onset_sample", "offset_sample"],
            },
            "outputs": {
                "harmonic": "harmonic target estimate",
                "residual": "off-harmonic target estimate",
                "primary": "harmonic plus residual target estimate",
            },
            "dna_conditioning": False,
            "parameter_count": self.parameter_count(),
        }

    def forward(
        self,
        harmonic_mixture: torch.Tensor,
        residual_mixture: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if harmonic_mixture.shape != residual_mixture.shape:
            raise ValueError("Harmonic and residual mixtures must have equal shape")
        zeros = torch.zeros_like(harmonic_mixture)
        harmonic, harmonic_diagnostics = self.harmonic_branch(
            harmonic_mixture,
            zeros,
            zeros,
            zeros,
            pitch,
            valid_samples,
            onset,
            offset,
        )
        residual, residual_diagnostics = self.residual_branch(
            residual_mixture,
            zeros,
            zeros,
            zeros,
            pitch,
            valid_samples,
            onset,
            offset,
        )
        return harmonic + residual, {
            "harmonic": harmonic,
            "residual": residual,
            "harmonic_branch": harmonic_diagnostics,
            "residual_branch": residual_diagnostics,
        }


def occurrence_dual_harmonic_residual_separator_from_contract(
    raw: object,
) -> OccurrenceDualHarmonicResidualSeparator:
    if not isinstance(raw, dict) or raw.get("kind") != _CONTRACT_KIND:
        raise ValueError("Checkpoint lacks a dual harmonic/residual contract")
    preset = raw.get("preset")
    if not isinstance(preset, str):
        raise ValueError("Dual harmonic/residual contract lacks a preset")
    model = OccurrenceDualHarmonicResidualSeparator.from_preset(preset)
    if model.contract() != raw:
        raise ValueError("Dual harmonic/residual checkpoint contract changed")
    return model


__all__ = [
    "OccurrenceDualHarmonicResidualSeparator",
    "occurrence_dual_harmonic_residual_separator_from_contract",
]
