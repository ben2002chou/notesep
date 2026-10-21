"""Symmetric HPSS separator with one mandatory direct-complex H/P readout."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .occurrence_dual_hpss_symmetric_alignment import (
    OccurrenceDualHPSSSymmetricAlignment,
)


class OccurrenceDualHPSSJointComplexOutput(OccurrenceDualHPSSSymmetricAlignment):
    """Fuse mature branch spectra through the sole full-band output path.

    The 1x1 complex mixer starts exactly as H+P, but all final source samples
    must pass through its trainable weights.  It therefore preserves the
    checkpoint function without adding an ignorable residual side adapter.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.joint_complex_readout = nn.Conv2d(
            4, 2, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.reset_joint_readout_to_sum()

    def reset_joint_readout_to_sum(self) -> None:
        with torch.no_grad():
            self.joint_complex_readout.weight.zero_()
            # Input channels: H.real, H.imag, P.real, P.imag.
            self.joint_complex_readout.weight[0, 0, 0, 0] = 1.0
            self.joint_complex_readout.weight[0, 2, 0, 0] = 1.0
            self.joint_complex_readout.weight[1, 1, 0, 0] = 1.0
            self.joint_complex_readout.weight[1, 3, 0, 0] = 1.0

    def contract(self) -> dict[str, Any]:
        contract = super().contract()
        contract["contract_version"] = max(10, int(contract["contract_version"]))
        contract["architecture"]["merge"] = "mandatory_joint_complex_1x1_readout"
        contract["architecture"]["joint_output"] = {
            "type": "direct_complex_linear_hp_fusion",
            "input": "post_branch_physical_scale_projected_target_spectra",
            "channels": ["H.real", "H.imag", "P.real", "P.imag"],
            "output": ["source.real", "source.imag"],
            "kernel": [1, 1],
            "bias": False,
            "initialization": "exact_complex_H_plus_P",
            "external_sum_bypass": False,
            "free_residual_head": False,
            "mixture_projection": False,
        }
        contract["outputs"]["primary"] = (
            "one direct complex source formed by the mandatory learned H/P readout"
        )
        contract["parameter_count"] = self.parameter_count()
        return contract

    def _merge_branch_targets(
        self,
        harmonic_target: torch.Tensor,
        percussive_target: torch.Tensor,
        harmonic_diagnostics: dict[str, torch.Tensor],
        percussive_diagnostics: dict[str, torch.Tensor],
        reference_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        harmonic_spectrum = harmonic_diagnostics["projected_target_spectrum"]
        percussive_spectrum = percussive_diagnostics["projected_target_spectrum"]
        if harmonic_spectrum.shape != percussive_spectrum.shape:
            raise ValueError("H/P target spectra must have equal shape")
        readout_input = torch.stack(
            (
                harmonic_spectrum.real,
                harmonic_spectrum.imag,
                percussive_spectrum.real,
                percussive_spectrum.imag,
            ),
            dim=1,
        )
        # Preserve exact H+P initialization and form a valid complex tensor by
        # keeping this tiny readout in FP32 even under the model's BF16 autocast.
        with torch.autocast(device_type=reference_audio.device.type, enabled=False):
            fused_channels = self.joint_complex_readout(readout_input.float())
        fused_spectrum = torch.complex(fused_channels[:, 0], fused_channels[:, 1])
        samples = int(reference_audio.shape[-1])
        estimate = torch.istft(
            fused_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self.harmonic_branch._window(reference_audio.float()),
            center=True,
            length=samples,
        )
        sample_support = harmonic_diagnostics["sample_support"]
        estimate = estimate.to(reference_audio.dtype) * sample_support
        identity_spectrum = harmonic_spectrum + percussive_spectrum
        return estimate, {
            "fused_target_spectrum": fused_spectrum,
            "identity_sum_spectrum": identity_spectrum,
            "fusion_delta_spectrum": fused_spectrum - identity_spectrum,
            "readout_weight": self.joint_complex_readout.weight,
        }


__all__ = ["OccurrenceDualHPSSJointComplexOutput"]
