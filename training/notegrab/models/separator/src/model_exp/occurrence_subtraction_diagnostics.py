"""Diagnostics for subtractive occurrence-separation anchors.

These helpers deliberately operate on complete waveforms and ordinary complex
STFTs.  HPSS is nonlinear, so independently decomposed HPSS components are not
expected to close under source subtraction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def closure_metrics(
    mixture: np.ndarray,
    target: np.ndarray,
    background: np.ndarray,
    *,
    epsilon: float = 1e-20,
) -> dict[str, float]:
    """Return additive closure diagnostics for ``mixture = target + background``."""

    mixture64 = np.asarray(mixture, dtype=np.float64).reshape(-1)
    target64 = np.asarray(target, dtype=np.float64).reshape(-1)
    background64 = np.asarray(background, dtype=np.float64).reshape(-1)
    if not mixture64.size or not (
        mixture64.shape == target64.shape == background64.shape
    ):
        raise ValueError("mixture, target, and background need equal nonzero shapes")
    error = (mixture64 - target64) - background64
    error_energy = float(np.dot(error, error))
    background_energy = float(np.dot(background64, background64))
    denominator = max(background_energy, float(epsilon))
    return {
        "closure_nmse": error_energy / denominator,
        "closure_max_abs": float(np.max(np.abs(error))),
        "closure_snr_db": float(
            10.0 * math.log10(denominator / max(error_energy, float(epsilon)))
        ),
        "background_energy": background_energy,
        "error_energy": error_energy,
    }


def ordinary_stft_subtraction_metrics(
    mixture: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    epsilon: float = 1e-20,
) -> dict[str, float]:
    """Verify ordinary STFT subtraction and inverse reconstruction.

    Settings intentionally match the occurrence separator: periodic Hann,
    ``win_length=n_fft``, and centered analysis/synthesis.
    """

    if mixture.ndim != 1 or target.shape != mixture.shape or not mixture.numel():
        raise ValueError("mixture and target need equal nonempty 1-D shapes")
    if n_fft < 2 or hop_length < 1:
        raise ValueError("invalid STFT geometry")
    mixture = mixture.float()
    target = target.float()
    background = mixture - target
    window = torch.hann_window(
        int(n_fft), periodic=True, dtype=mixture.dtype, device=mixture.device
    )

    def stft(values: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            values,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(n_fft),
            window=window,
            center=True,
            return_complex=True,
        )

    mixture_stft = stft(mixture)
    target_stft = stft(target)
    background_stft = stft(background)
    difference_stft = mixture_stft - target_stft
    spectral_error = difference_stft - background_stft
    spectral_reference = float(background_stft.abs().square().sum().item())
    spectral_error_energy = float(spectral_error.abs().square().sum().item())
    reconstructed = torch.istft(
        difference_stft,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        center=True,
        length=int(mixture.numel()),
    )
    waveform_error = reconstructed - background
    waveform_reference = float(background.square().sum().item())
    waveform_error_energy = float(waveform_error.square().sum().item())
    return {
        "stft_subtraction_nmse": spectral_error_energy
        / max(spectral_reference, float(epsilon)),
        "stft_subtraction_max_abs": float(spectral_error.abs().max().item()),
        "istft_residual_nmse": waveform_error_energy
        / max(waveform_reference, float(epsilon)),
        "istft_residual_max_abs": float(waveform_error.abs().max().item()),
    }


def projection_underprediction_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid_samples: torch.Tensor,
) -> torch.Tensor:
    """Penalize aligned target-amplitude gain below one, but not above one."""

    if estimate.ndim != 2 or target.shape != estimate.shape:
        raise ValueError("estimate and target need equal [batch, samples] shapes")
    if valid_samples.shape != estimate.shape[:1]:
        raise ValueError("valid_samples must have shape [batch]")
    positions = torch.arange(estimate.shape[-1], device=estimate.device)[None]
    support = positions < valid_samples.long()[:, None]
    estimate64 = estimate.double() * support
    target64 = target.double() * support
    target_energy = target64.square().sum(dim=-1).clamp_min(1e-12)
    gain = (estimate64 * target64).sum(dim=-1) / target_energy
    return torch.relu(1.0 - gain).square().mean().float()


def batch_anchor_quality_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    valid_samples: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-batch means for recovery, leakage, and cancellation quality."""

    if estimate.ndim != 2 or not (target.shape == mixture.shape == estimate.shape):
        raise ValueError("estimate, target, and mixture need equal [batch, samples] shapes")
    if valid_samples.shape != estimate.shape[:1]:
        raise ValueError("valid_samples must have shape [batch]")
    positions = torch.arange(estimate.shape[-1], device=estimate.device)[None]
    support = positions < valid_samples.long()[:, None]
    estimate64 = estimate.double() * support
    target64 = target.double() * support
    mixture64 = mixture.double() * support
    target_energy = target64.square().sum(dim=-1).clamp_min(1e-12)
    mixture_energy = mixture64.square().sum(dim=-1).clamp_min(1e-12)
    estimate_energy = estimate64.square().sum(dim=-1)
    gain = (estimate64 * target64).sum(dim=-1) / target_energy
    off_target = estimate64 - gain[:, None] * target64
    estimate_error = estimate64 - target64
    return {
        "anchor_estimate_to_target_energy_ratio": (
            estimate_energy / target_energy
        ).mean().float(),
        "anchor_target_projection_gain": gain.mean().float(),
        "anchor_target_projected_energy_ratio": gain.square().mean().float(),
        "anchor_off_target_energy_ratio": (
            off_target.square().sum(dim=-1) / target_energy
        ).mean().float(),
        "anchor_residual_oracle_nmse": (
            estimate_error.square().sum(dim=-1) / target_energy
        ).mean().float(),
        "target_to_mixture_energy_ratio": (
            target_energy / mixture_energy
        ).mean().float(),
    }


def summarize_metric_rows(
    rows: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Summarize equal-schema scalar metric rows."""

    if not rows:
        return {}
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("metric rows must have the same ordered fields")
    result: dict[str, dict[str, float]] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(np.max(values)),
        }
    return result


__all__ = [
    "closure_metrics",
    "batch_anchor_quality_metrics",
    "ordinary_stft_subtraction_metrics",
    "projection_underprediction_loss",
    "summarize_metric_rows",
]
