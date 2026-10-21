"""CQT harmonic-comb spatial FiLM for the mature symmetric HPSS separator."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .dna_residual_tfc_framewise_conditioned import _SpatialFiLM
from .occurrence_dual_hpss_symmetric_alignment import (
    OccurrenceDualHPSSSymmetricAlignment,
)
from .occurrence_dual_hpss_separator import _first_contract_difference


CQTConditionMode = Literal["correct", "disabled", "shuffled_pitch"]


class OccurrenceDualHPSSCQTCombFiLM(OccurrenceDualHPSSSymmetricAlignment):
    """Condition only the harmonic stream with pitch-combed CQT evidence.

    The existing complex-STFT HPSS branches and outputs are unchanged.  Every
    added FiLM affine is zero initialized, so migrating a symmetric checkpoint
    preserves its exact step-zero waveform.
    """

    CONTRACT_KIND = "occurrence_dual_hpss_symmetric_cqt_comb_film_v1"

    def __init__(
        self,
        *args: Any,
        cqt_bins_per_octave: int = 60,
        cqt_fmin: float = 27.5,
        comb_sigma_cents: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if cqt_bins_per_octave < 24 or cqt_fmin <= 0 or comb_sigma_cents <= 0:
            raise ValueError("invalid CQT-comb geometry")
        from nnAudio.features import CQT2010v2

        self.cqt_bins_per_octave = int(cqt_bins_per_octave)
        self.cqt_fmin = float(cqt_fmin)
        self.comb_sigma_cents = float(comb_sigma_cents)
        self.cqt_hop_length = max(int(self.config.hop_length), 512)
        self.cqt_bins = int(math.floor(
            math.log2((self.config.sample_rate / 2) / self.cqt_fmin)
            * self.cqt_bins_per_octave
        ))
        self.cqt_transform = CQT2010v2(
            sr=self.config.sample_rate,
            hop_length=self.cqt_hop_length,
            fmin=self.cqt_fmin,
            n_bins=self.cqt_bins,
            bins_per_octave=self.cqt_bins_per_octave,
            output_format="Complex",
            earlydownsample=False,
            trainable=False,
            verbose=False,
        )
        # TFC-TDF folds each contiguous frequency quarter into channels before
        # the first convolution.  Fold the side condition identically rather
        # than interpolating 512 linear bins onto the 128-cell CWS axis.
        condition_channels = 3 * int(self.config.num_subbands)
        self.cqt_input_film = _SpatialFiLM(
            condition_channels, int(self.config.initial_channels)
        )
        self.cqt_condition_mode: CQTConditionMode = "correct"

    def set_cqt_condition_mode(self, mode: CQTConditionMode) -> None:
        if mode not in {"correct", "disabled", "shuffled_pitch"}:
            raise ValueError(f"invalid CQT condition mode: {mode}")
        self.cqt_condition_mode = mode

    def contract(self) -> dict[str, Any]:
        contract = super().contract()
        contract["kind"] = self.CONTRACT_KIND
        contract["contract_version"] = max(9, int(contract["contract_version"]))
        contract["architecture"]["harmonic_cqt_comb_film"] = {
            "input": "pitch_gated_hpss_harmonic_mixture",
            "representation": "fixed_complex_cqt_log_magnitude",
            "bins_per_octave": self.cqt_bins_per_octave,
            "cents_per_bin": 1200.0 / self.cqt_bins_per_octave,
            "fmin_hz": self.cqt_fmin,
            "bins": self.cqt_bins,
            "hop_length": self.cqt_hop_length,
            "comb_sigma_cents": self.comb_sigma_cents,
            "channels": ["log_magnitude", "soft_comb", "combed_log_magnitude"],
            "frequency_mapping": "fixed_cqt_frequency_to_linear_stft_grid_sample",
            "injection": "zero_initialized_spatial_film_before_encoder_scale_0",
            "percussive_branch": "unchanged",
            "output": "unchanged_complex_stft_harmonic_plus_percussive",
        }
        contract["parameter_count"] = self.parameter_count()
        return contract

    def _soft_comb(
        self, pitch: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        rows = torch.arange(self.cqt_bins, device=device, dtype=dtype)
        frequencies = self.cqt_fmin * torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            rows / float(self.cqt_bins_per_octave),
        )
        f0 = 440.0 * torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            (pitch.to(dtype) - 69.0) / 12.0,
        )
        ratio = frequencies[None] / f0[:, None]
        order = ratio.round().clamp_min(1.0)
        cents = 1200.0 * torch.log2(ratio.clamp_min(1e-8) / order)
        comb = torch.exp(-0.5 * (cents / self.comb_sigma_cents).square())
        valid = (ratio >= 0.5) & (order * f0[:, None] < self.config.sample_rate / 2)
        return comb * valid.to(comb.dtype)

    def _map_cqt_to_stft(
        self, features: torch.Tensor, frames: int, frequency_bins: int
    ) -> torch.Tensor:
        if features.shape[-1] != frames:
            features = F.interpolate(
                features,
                size=(features.shape[-2], frames),
                mode="bilinear",
                align_corners=False,
            )
        dtype, device = features.dtype, features.device
        frequencies = (
            torch.arange(frequency_bins, device=device, dtype=dtype)
            * float(self.config.sample_rate) / float(self.config.n_fft)
        )
        cqt_position = self.cqt_bins_per_octave * torch.log2(
            frequencies.clamp_min(self.cqt_fmin) / self.cqt_fmin
        )
        y = 2.0 * cqt_position / max(1, self.cqt_bins - 1) - 1.0
        y = torch.where(
            frequencies >= self.cqt_fmin,
            y,
            torch.full_like(y, -2.0),
        )
        x = torch.linspace(-1.0, 1.0, frames, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(
            features.shape[0], -1, -1, -1
        )
        return F.grid_sample(
            features, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )

    def _prepare_harmonic_side_condition(
        self,
        harmonic_state: dict[str, Any],
        harmonic_audio: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
    ) -> torch.Tensor:
        samples = harmonic_audio.shape[-1]
        support = (
            torch.arange(samples, device=harmonic_audio.device)[None]
            < valid_samples.long()[:, None]
        )
        normalized = harmonic_audio.float() * support / harmonic_state["level"]
        raw = self.cqt_transform(normalized, output_format="Complex")
        cqt = torch.complex(raw[..., 0].float(), raw[..., 1].float())
        log_magnitude = torch.log1p(cqt.abs())
        conditioning_pitch = pitch
        if self.cqt_condition_mode == "shuffled_pitch":
            conditioning_pitch = torch.roll(pitch, shifts=1, dims=0)
        comb = self._soft_comb(
            conditioning_pitch, log_magnitude.device, log_magnitude.dtype
        )
        features = torch.stack(
            (
                log_magnitude,
                comb[..., None].expand_as(log_magnitude),
                log_magnitude * comb[..., None],
            ),
            dim=1,
        )
        condition = self._map_cqt_to_stft(
            features,
            int(harmonic_state["frames"]),
            int(self.config.processed_frequency_bins),
        )
        if self.cqt_condition_mode == "disabled":
            condition = torch.zeros_like(condition)
        condition = condition * harmonic_state["frame_support"][:, None, None]
        # Match the carrier's complex-as-channel, waveform-subband frequency
        # folding exactly: [B,3,512,T] -> [B,12,128,T] -> [B,12,T,128].
        condition = self.harmonic_branch._cac_to_cws(condition)
        condition = condition.transpose(-1, -2)
        if not torch.isfinite(condition).all():
            raise RuntimeError("non-finite CQT-comb condition")
        return condition

    def _apply_harmonic_side_condition(
        self,
        rung: str,
        index: int | None,
        values: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if condition is None:
            raise RuntimeError("CQT-comb FiLM condition was not prepared")
        if rung == "input" and index is None:
            return self.cqt_input_film(values, condition)
        raise ValueError(f"invalid CQT-comb FiLM rung: {rung}:{index}")


def occurrence_dual_hpss_cqt_comb_film_from_contract(
    raw: object,
) -> OccurrenceDualHPSSCQTCombFiLM:
    """Strict reconstruction for evaluator and checkpoint-resume loaders."""

    if not isinstance(raw, dict) or raw.get("kind") != OccurrenceDualHPSSCQTCombFiLM.CONTRACT_KIND:
        raise ValueError("Checkpoint lacks a symmetric CQT-comb FiLM contract")
    preset = raw.get("preset")
    if not isinstance(preset, str):
        raise ValueError("CQT-comb canary currently requires a named preset")
    architecture = raw.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("CQT-comb contract lacks architecture metadata")
    communication = architecture.get("communication")
    cqt = architecture.get("harmonic_cqt_comb_film")
    if not isinstance(communication, dict) or not isinstance(cqt, dict):
        raise ValueError("CQT-comb contract lacks communication or CQT metadata")
    conditioning = raw.get("conditioning") or {}
    conditioning_type = conditioning.get("type")
    conditioning_mode = (
        "rt2_target_roll"
        if conditioning_type == "rt2_25hz_target_only_roll"
        else "rt2_onset_only"
        if conditioning_type == "rt2_25hz_onset_only_pitch"
        else "harmonic_lifecycle"
    )
    hold = conditioning.get("training_positive_hold_ms", [0.0, 0.0])
    if not isinstance(hold, list) or len(hold) != 2:
        raise ValueError("CQT-comb contract has invalid roll hold bounds")
    model = OccurrenceDualHPSSCQTCombFiLM(
        preset,
        str((raw.get("harmonic_gate") or {}).get("mode", "none")),
        str(architecture.get("percussive_parameterization", "ownership")),
        conditioning_mode,
        *map(float, hold),
        attention_heads=int(communication["attention_heads"]),
        harmonic_parameterization=str(
            architecture.get("harmonic_parameterization", "ownership")
        ),
        output_carrier_mode=str(
            architecture.get("output_carrier_mode", "encoder_input")
        ),
        complex_mask_mode=str(architecture.get("complex_mask_mode", "bounded")),
        cqt_bins_per_octave=int(cqt["bins_per_octave"]),
        cqt_fmin=float(cqt["fmin_hz"]),
        comb_sigma_cents=float(cqt["comb_sigma_cents"]),
    )
    initialization = raw.get("initialization_checkpoint")
    if not isinstance(initialization, dict):
        raise ValueError("CQT-comb contract lacks initialization provenance")
    model.communication_initialization = initialization
    reconstructed = model.contract()
    if reconstructed != raw:
        difference = _first_contract_difference(raw, reconstructed)
        raise ValueError(f"CQT-comb checkpoint contract changed: {difference}")
    return model


__all__ = [
    "CQTConditionMode",
    "OccurrenceDualHPSSCQTCombFiLM",
    "occurrence_dual_hpss_cqt_comb_film_from_contract",
]
