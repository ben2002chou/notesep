"""Frequency-aware mandatory direct-complex H/P fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .occurrence_dual_hpss_symmetric_alignment import (
    OccurrenceDualHPSSSymmetricAlignment,
)


class OccurrenceDualHPSSFrequencywiseJointOutput(
    OccurrenceDualHPSSSymmetricAlignment
):
    """Fuse H/P spectra with frequencywise ownership and local TF context.

    This is one direct linear renderer, not a correction or residual path. Its
    two factors are initialized so their composition is exactly complex H+P:
    a distinct 2x4 complex-channel mixer per STFT frequency followed by a
    center-identity 3x3 TF convolution.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        frequency_bins = int(self.config.n_fft // 2 + 1)
        self.frequencywise_joint_weight = nn.Parameter(
            torch.empty(frequency_bins, 2, 4)
        )
        self.local_tf_joint_filter = nn.Conv2d(
            2, 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.reset_joint_readout_to_sum()

    def reset_joint_readout_to_sum(self) -> None:
        with torch.no_grad():
            self.frequencywise_joint_weight.zero_()
            self.frequencywise_joint_weight[:, 0, 0] = 1.0
            self.frequencywise_joint_weight[:, 0, 2] = 1.0
            self.frequencywise_joint_weight[:, 1, 1] = 1.0
            self.frequencywise_joint_weight[:, 1, 3] = 1.0
            self.local_tf_joint_filter.weight.zero_()
            self.local_tf_joint_filter.weight[0, 0, 1, 1] = 1.0
            self.local_tf_joint_filter.weight[1, 1, 1, 1] = 1.0

    def contract(self) -> dict[str, Any]:
        contract = super().contract()
        contract["contract_version"] = max(11, int(contract["contract_version"]))
        contract["architecture"]["merge"] = (
            "mandatory_frequencywise_local_tf_joint_complex_readout"
        )
        contract["architecture"]["joint_output"] = {
            "type": "direct_complex_frequencywise_local_tf_hp_fusion",
            "input": "post_branch_physical_scale_projected_target_spectra",
            "channels": ["H.real", "H.imag", "P.real", "P.imag"],
            "output": ["source.real", "source.imag"],
            "frequencywise_matrix": [2, 4],
            "local_tf_kernel": [3, 3],
            "bias": False,
            "initialization": "exact_complex_H_plus_P",
            "external_sum_bypass": False,
            "free_residual_head": False,
            "mixture_projection": False,
        }
        contract["outputs"]["primary"] = (
            "one direct complex source formed by mandatory frequencywise/local-TF H/P fusion"
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
        del harmonic_target, percussive_target
        harmonic_spectrum = harmonic_diagnostics["projected_target_spectrum"]
        percussive_spectrum = percussive_diagnostics["projected_target_spectrum"]
        readout_input = torch.stack(
            (
                harmonic_spectrum.real,
                harmonic_spectrum.imag,
                percussive_spectrum.real,
                percussive_spectrum.imag,
            ),
            dim=1,
        )
        if readout_input.shape[-2] != self.frequencywise_joint_weight.shape[0]:
            raise ValueError("joint readout frequency dimension mismatch")
        with torch.autocast(device_type=reference_audio.device.type, enabled=False):
            frequencywise = torch.einsum(
                "bcft,foc->boft",
                readout_input.float(),
                self.frequencywise_joint_weight.float(),
            )
            fused_channels = self.local_tf_joint_filter(frequencywise)
        fused_spectrum = torch.complex(fused_channels[:, 0], fused_channels[:, 1])
        estimate = torch.istft(
            fused_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self.harmonic_branch._window(reference_audio.float()),
            center=True,
            length=int(reference_audio.shape[-1]),
        )
        sample_support = harmonic_diagnostics["sample_support"]
        estimate = estimate.to(reference_audio.dtype) * sample_support
        identity_spectrum = harmonic_spectrum + percussive_spectrum
        return estimate, {
            "fused_target_spectrum": fused_spectrum,
            "identity_sum_spectrum": identity_spectrum,
            "fusion_delta_spectrum": fused_spectrum - identity_spectrum,
            "frequencywise_joint_weight": self.frequencywise_joint_weight,
            "local_tf_joint_filter_weight": self.local_tf_joint_filter.weight,
        }


__all__ = ["OccurrenceDualHPSSFrequencywiseJointOutput"]
