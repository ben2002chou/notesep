"""Differentiable harmonic-partial supervision aligned with DNA geometry.

The frequency geometry mirrors the canonical DNA evaluation metric: a periodic
Hann STFT, inharmonic partial centers, Gaussian frequency ownership, and a
20 Hz compact support.  Unlike the evaluator's equal-harmonic ratios, this loss
uses target-activity weights and a strongest-partial-relative denominator floor.
Its bounded relative errors therefore retain weak upper-partial supervision
without allowing nearly silent targets to dominate an update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import torch
from torch import nn


TensorLike: TypeAlias = torch.Tensor | int | float


@dataclass(frozen=True)
class DNAHarmonicPartialLossConfig:
    """Configuration with explicit canonical geometry and stable loss scales."""

    sample_rate: int = 16_000
    n_fft: int = 2_048
    hop_length: int = 256
    maximum_harmonics: int = 59
    inharmonicity: float = 0.0004
    support_hz: float = 20.0
    gaussian_sigma_hz: float = 7.5
    weak_partial_floor_ratio: float = 1e-6
    absolute_energy_floor: float = 1e-12
    partial_energy_weight: float = 1.0
    off_partial_weight: float = 0.0
    temporal_jaggedness_weight: float = 0.0
    hard_example_gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if self.n_fft != 2_048 or self.hop_length != 256:
            raise ValueError("DNA harmonic loss requires n_fft=2048 and hop_length=256")
        if self.maximum_harmonics != 59:
            raise ValueError(
                "DNA harmonic loss requires exactly 59 candidate harmonics"
            )
        if self.inharmonicity != 0.0004:
            raise ValueError("DNA harmonic loss requires inharmonicity=0.0004")
        if self.support_hz != 20.0 or self.gaussian_sigma_hz != 7.5:
            raise ValueError(
                "DNA harmonic loss requires 20 Hz support and sigma=7.5 Hz"
            )
        if not 0.0 < self.weak_partial_floor_ratio <= 1.0:
            raise ValueError("weak_partial_floor_ratio must be in (0, 1]")
        if self.absolute_energy_floor <= 0.0:
            raise ValueError("absolute_energy_floor must be positive")
        if (
            min(
                self.partial_energy_weight,
                self.off_partial_weight,
                self.temporal_jaggedness_weight,
            )
            < 0.0
        ):
            raise ValueError("loss weights must be nonnegative")
        if not math.isfinite(self.hard_example_gamma) or self.hard_example_gamma < 0.0:
            raise ValueError("hard_example_gamma must be finite and nonnegative")


@dataclass(frozen=True)
class DNAHarmonicPartialLossResult:
    """Scalar loss components plus tensors useful for scientific auditing."""

    total: torch.Tensor
    partial_energy: torch.Tensor
    off_partial_hallucination: torch.Tensor
    temporal_jaggedness: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _as_batch_vector(
    value: TensorLike,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or [batch], got {tuple(tensor.shape)}")
    return tensor


def _bounded_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    ratio = numerator / denominator
    return ratio / (1.0 + ratio)


def _weighted_batch_values(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, values.ndim))
    denominator = weights.sum(dim=dims)
    per_example = (values * weights).sum(dim=dims) / denominator.clamp_min(floor)
    active = denominator > floor
    return per_example, active


def _hard_example_mean(
    per_example: torch.Tensor,
    active: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    weights = active.to(per_example.dtype)
    if gamma > 0.0:
        # Detaching the focal weights keeps the update a stable reweighting of
        # example gradients instead of introducing a second-order incentive to
        # manipulate the difficulty estimate itself.
        weights = weights * (per_example.detach() + 0.05).pow(gamma)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-12)


class DNAHarmonicPartialLoss(nn.Module):
    """Target-activity-weighted correction loss over DNA harmonic partials."""

    def __init__(self, config: DNAHarmonicPartialLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or DNAHarmonicPartialLossConfig()

    def _geometry(
        self,
        pitch: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        dtype = torch.float32
        harmonics = torch.arange(
            1,
            config.maximum_harmonics + 1,
            device=device,
            dtype=dtype,
        )
        f0 = 440.0 * torch.pow(2.0, (pitch.to(dtype) - 69.0) / 12.0)
        centers = (
            f0[:, None]
            * harmonics[None, :]
            * torch.sqrt(1.0 + config.inharmonicity * harmonics.square())
        )
        valid_harmonics = centers <= config.sample_rate / 2.0
        frequencies = torch.fft.rfftfreq(
            config.n_fft,
            d=1.0 / config.sample_rate,
            device=device,
            dtype=dtype,
        )
        distance = (centers[:, :, None] - frequencies[None, None, :]).abs()
        raw = torch.exp(-0.5 * (distance / config.gaussian_sigma_hz).square())
        raw = raw * (distance < config.support_hz) * valid_harmonics[:, :, None]
        ownership = raw / raw.sum(dim=1, keepdim=True).clamp_min(1.0)
        off_partial = 1.0 - raw.sum(dim=1).clamp(max=1.0)
        return centers, valid_harmonics, ownership, off_partial

    def forward(
        self,
        predicted_target: torch.Tensor,
        exact_target: torch.Tensor,
        pitch: TensorLike,
        valid_samples: TensorLike,
        onset_samples: TensorLike,
        offset_samples: TensorLike,
    ) -> DNAHarmonicPartialLossResult:
        """Compute a loss for padded ``[batch, samples]`` target waveforms."""

        if predicted_target.ndim != 2 or predicted_target.shape != exact_target.shape:
            raise ValueError(
                "predicted_target and exact_target must have equal [batch, samples] shape"
            )
        if (
            not predicted_target.is_floating_point()
            or not exact_target.is_floating_point()
        ):
            raise ValueError("waveforms must use floating-point tensors")
        if predicted_target.device != exact_target.device:
            raise ValueError("predicted_target and exact_target must share a device")

        batch_size, original_samples = predicted_target.shape
        device = predicted_target.device
        pitch_tensor = _as_batch_vector(
            pitch,
            name="pitch",
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        valid = _as_batch_vector(
            valid_samples,
            name="valid_samples",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        onset = _as_batch_vector(
            onset_samples,
            name="onset_samples",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        offset = _as_batch_vector(
            offset_samples,
            name="offset_samples",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        if not bool(torch.all((pitch_tensor >= 0) & (pitch_tensor <= 127))):
            raise ValueError("pitch values must be MIDI pitches in [0, 127]")
        bounds_ok = (valid > 0) & (valid <= original_samples)
        bounds_ok &= (onset >= 0) & (onset < offset) & (offset <= valid)
        if not bool(torch.all(bounds_ok)):
            raise ValueError("require 0 <= onset < offset <= valid_samples <= samples")

        config = self.config
        autocast_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            predicted = predicted_target.float()
            target = exact_target.float()
            sample_index = torch.arange(original_samples, device=device)
            sample_support = sample_index[None, :] < valid[:, None]
            predicted = predicted * sample_support
            target = target * sample_support
            if original_samples < config.n_fft:
                padding = config.n_fft - original_samples
                predicted = torch.nn.functional.pad(predicted, (0, padding))
                target = torch.nn.functional.pad(target, (0, padding))

            window = torch.hann_window(
                config.n_fft,
                periodic=True,
                device=device,
                dtype=torch.float32,
            )
            stft_kwargs = {
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "win_length": config.n_fft,
                "window": window,
                "center": False,
                "return_complex": True,
            }
            predicted_power = torch.stft(predicted, **stft_kwargs).abs().square()
            target_power = torch.stft(target, **stft_kwargs).abs().square()

            centers, valid_harmonics, ownership, off_partial = self._geometry(
                pitch_tensor, device=device
            )
            predicted_partials = torch.einsum(
                "bhf,bft->bht", ownership, predicted_power
            )
            target_partials = torch.einsum("bhf,bft->bht", ownership, target_power)

            frame_count = target_power.shape[-1]
            frame_starts = torch.arange(frame_count, device=device) * config.hop_length
            frame_ends = frame_starts + config.n_fft
            valid_frame = frame_starts[None, :] < valid[:, None]
            note_frame = (frame_starts[None, :] < offset[:, None]) & (
                frame_ends[None, :] > onset[:, None]
            )
            frame_support = valid_frame & note_frame

            peak = target_partials.amax(dim=1, keepdim=True)
            energy_floor = (
                peak * config.weak_partial_floor_ratio + config.absolute_energy_floor
            )
            relative_error = _bounded_ratio(
                (predicted_partials - target_partials).abs(),
                target_partials + energy_floor,
            )
            activity_gate = target_partials / (target_partials + energy_floor)
            partial_weights = torch.sqrt(target_partials + energy_floor) * activity_gate
            partial_weights = partial_weights * valid_harmonics[:, :, None]
            partial_weights = partial_weights * frame_support[:, None, :]
            partial_energy_per_example, partial_energy_active = _weighted_batch_values(
                relative_error,
                partial_weights,
                floor=config.absolute_energy_floor,
            )
            partial_energy_loss = _hard_example_mean(
                partial_energy_per_example,
                partial_energy_active,
                gamma=config.hard_example_gamma,
            )

            off_predicted = (predicted_power * off_partial[:, :, None]).sum(dim=1)
            off_target = (target_power * off_partial[:, :, None]).sum(dim=1)
            total_target = target_power.sum(dim=1)
            spectral_floor = (
                total_target.amax(dim=1, keepdim=True) * config.weak_partial_floor_ratio
                + config.absolute_energy_floor
            )
            off_error = _bounded_ratio(
                torch.relu(off_predicted - off_target),
                total_target + spectral_floor,
            )
            off_weights = torch.sqrt(total_target + spectral_floor) * frame_support
            off_partial_per_example, off_partial_active = _weighted_batch_values(
                off_error,
                off_weights,
                floor=config.absolute_energy_floor,
            )
            off_partial_loss = _hard_example_mean(
                off_partial_per_example,
                off_partial_active,
                gamma=config.hard_example_gamma,
            )

            if frame_count >= 2:
                target_level = torch.log1p(target_partials / energy_floor)
                predicted_level = torch.log1p(predicted_partials / energy_floor)
                target_delta = target_level[..., 1:] - target_level[..., :-1]
                predicted_delta = predicted_level[..., 1:] - predicted_level[..., :-1]
                jagged_error = _bounded_ratio(
                    (predicted_delta - target_delta).abs(),
                    torch.ones_like(target_delta),
                )
                pair_support = frame_support[:, 1:] & frame_support[:, :-1]
                jagged_weights = 0.5 * (
                    partial_weights[..., 1:] + partial_weights[..., :-1]
                )
                jagged_weights = jagged_weights * pair_support[:, None, :]
                temporal_jaggedness_per_example, temporal_jaggedness_active = (
                    _weighted_batch_values(
                        jagged_error,
                        jagged_weights,
                        floor=config.absolute_energy_floor,
                    )
                )
                temporal_jaggedness_loss = _hard_example_mean(
                    temporal_jaggedness_per_example,
                    temporal_jaggedness_active,
                    gamma=config.hard_example_gamma,
                )
            else:
                temporal_jaggedness_loss = predicted_power.sum() * 0.0
                temporal_jaggedness_per_example = torch.zeros_like(
                    partial_energy_per_example
                )

            total = (
                config.partial_energy_weight * partial_energy_loss
                + config.off_partial_weight * off_partial_loss
                + config.temporal_jaggedness_weight * temporal_jaggedness_loss
            )
            signed_difference = predicted_partials - target_partials
            normalized_signed = signed_difference / (target_partials + energy_floor)
            normalized_signed = normalized_signed / (1.0 + normalized_signed.abs())
            diagnostic_weights = partial_weights / partial_weights.sum(
                dim=(1, 2), keepdim=True
            ).clamp_min(config.absolute_energy_floor)
            signed_error = (
                (normalized_signed * diagnostic_weights).sum(dim=(1, 2)).mean()
            )

        diagnostics = {
            "harmonic_centers_hz": centers,
            "valid_harmonics": valid_harmonics,
            "frequency_ownership": ownership,
            "off_partial_mass": off_partial,
            "predicted_partial_energy": predicted_partials,
            "target_partial_energy": target_partials,
            "partial_activity_weights": partial_weights,
            "frame_support": frame_support,
            "partial_energy_per_example": partial_energy_per_example,
            "off_partial_per_example": off_partial_per_example,
            "temporal_jaggedness_per_example": temporal_jaggedness_per_example,
            "bounded_signed_partial_error": signed_error,
        }
        return DNAHarmonicPartialLossResult(
            total=total,
            partial_energy=partial_energy_loss,
            off_partial_hallucination=off_partial_loss,
            temporal_jaggedness=temporal_jaggedness_loss,
            diagnostics=diagnostics,
        )


def dna_harmonic_partial_loss(
    predicted_target: torch.Tensor,
    exact_target: torch.Tensor,
    pitch: TensorLike,
    valid_samples: TensorLike,
    onset_samples: TensorLike,
    offset_samples: TensorLike,
    *,
    config: DNAHarmonicPartialLossConfig | None = None,
) -> DNAHarmonicPartialLossResult:
    """Functional entry point for :class:`DNAHarmonicPartialLoss`."""

    return DNAHarmonicPartialLoss(config)(
        predicted_target,
        exact_target,
        pitch,
        valid_samples,
        onset_samples,
        offset_samples,
    )
