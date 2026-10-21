"""Pitch-relative spectral guards for HPSS harmonic occurrence branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


GATE_MODES = (
    "none",
    "below_f0",
    "below_f0_input_only",
    "subharmonic_notches_2_4",
    "suboctave_ratio_cap_0p5",
    "suboctave_ratio_cap_0p2",
    "suboctave_neighbor_smooth",
    "soft_harmonic_comb_input_only",
)

SOFT_HARMONIC_COMB_SIGMA_CENTS = 50.0
SOFT_HARMONIC_COMB_FLOOR = 0.15


def gate_applies_to_target_and_prediction(mode: str) -> bool:
    return mode in {"below_f0", "subharmonic_notches_2_4"}


def _gate_mask(
    frequencies: torch.Tensor,
    pitches: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode not in GATE_MODES:
        raise ValueError(f"Unknown harmonic gate mode: {mode}")
    f0 = 440.0 * torch.pow(2.0, (pitches.float() - 69.0) / 12.0)
    if mode == "none" or mode.startswith("suboctave_ratio_cap_"):
        return torch.ones(
            pitches.shape[0], frequencies.shape[0],
            device=frequencies.device, dtype=frequencies.dtype,
        )
    if mode == "soft_harmonic_comb_input_only":
        # Compare each positive-frequency bin with its nearest integer partial.
        # The deliberately broad 50-cent Gaussian tolerates vibrato, tuning
        # error, inharmonicity, and coarse low-frequency STFT resolution.  A
        # nonzero floor keeps non-harmonic timbre available to the H encoder;
        # this is an evidence prior, not an output-domain deletion rule.
        safe_frequency = frequencies[None].clamp_min(1e-6)
        harmonic_order = torch.round(
            safe_frequency / f0[:, None].clamp_min(1e-6)
        ).clamp_min(1.0)
        partial_frequency = harmonic_order * f0[:, None]
        cents = 1200.0 * torch.abs(
            torch.log2(safe_frequency / partial_frequency.clamp_min(1e-6))
        )
        affinity = torch.exp(
            -0.5 * (cents / SOFT_HARMONIC_COMB_SIGMA_CENTS).square()
        )
        affinity = torch.where(
            frequencies[None] > 0,
            affinity,
            torch.zeros_like(affinity),
        )
        return SOFT_HARMONIC_COMB_FLOOR + (
            1.0 - SOFT_HARMONIC_COMB_FLOOR
        ) * affinity
    if mode in {"below_f0", "below_f0_input_only"}:
        cutoff = torch.clamp(f0 - 60.0, min=30.0)
        return (frequencies[None] >= cutoff[:, None]).to(frequencies.dtype)

    # Preserve arbitrary low-frequency color while removing only plausible
    # fundamentals at one octave, a twelfth, and two octaves below the query.
    # Adaptive widths avoid erasing the query fundamental in the low register.
    keep = torch.ones(
        pitches.shape[0], frequencies.shape[0],
        device=frequencies.device, dtype=frequencies.dtype,
    )
    for divisor in (2.0, 3.0, 4.0):
        center = f0 / divisor
        inner = torch.minimum(torch.full_like(center, 15.0), 0.08 * center)
        outer = torch.minimum(torch.full_like(center, 60.0), 0.20 * center)
        distance = (frequencies[None] - center[:, None]).abs()
        transition = ((distance - inner[:, None]) / (
            outer - inner
        ).clamp_min(1e-6)[:, None]).clamp(0.0, 1.0)
        notch = 0.5 - 0.5 * torch.cos(torch.pi * transition)
        keep = torch.minimum(keep, notch)
    return keep


def _raised_cosine_band(
    frequencies: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    inner = torch.minimum(torch.full_like(center, 15.0), 0.08 * center)
    outer = torch.minimum(torch.full_like(center, 60.0), 0.20 * center)
    distance = (frequencies[None] - center[:, None]).abs()
    transition = ((distance - inner[:, None]) / (
        outer - inner
    ).clamp_min(1e-6)[:, None]).clamp(0.0, 1.0)
    return 0.5 + 0.5 * torch.cos(torch.pi * transition)


def _suboctave_ratio_mask(
    spectrum: torch.Tensor,
    frequencies: torch.Tensor,
    pitches: torch.Tensor,
    *,
    cap: float,
) -> torch.Tensor:
    """Smoothly cap local f0/2 RMS relative to f0 RMS in each frame."""

    f0 = 440.0 * torch.pow(2.0, (pitches.float() - 69.0) / 12.0)
    sub_band = _raised_cosine_band(frequencies, f0 / 2.0)
    f0_band = _raised_cosine_band(frequencies, f0)
    power = spectrum.abs().square()
    sub_rms = (
        (power * sub_band[:, :, None]).sum(dim=-2)
        / sub_band.sum(dim=-1).clamp_min(1e-6)[:, None]
    ).sqrt()
    f0_rms = (
        (power * f0_band[:, :, None]).sum(dim=-2)
        / f0_band.sum(dim=-1).clamp_min(1e-6)[:, None]
    ).sqrt()
    frame_rms = power.mean(dim=-2).sqrt()
    requested = (float(cap) * f0_rms / sub_rms.clamp_min(1e-8)).clamp(max=1.0)
    # Ignore numerically tiny suboctave texture and smooth frame gains to avoid
    # time-varying spectral chatter around attacks.
    requested = torch.where(sub_rms >= 0.01 * frame_rms, requested, torch.ones_like(requested))
    gain = F.avg_pool1d(requested[:, None], kernel_size=5, stride=1, padding=2)[:, 0]
    return 1.0 - sub_band[:, :, None] * (1.0 - gain[:, None, :])


def _suboctave_neighbor_smoothing_mask(
    spectrum: torch.Tensor,
    frequencies: torch.Tensor,
    pitches: torch.Tensor,
) -> torch.Tensor:
    """Flatten a narrow f0/2 peak toward, but never below, its neighbors."""

    f0 = 440.0 * torch.pow(2.0, (pitches.float() - 69.0) / 12.0)
    center = f0 / 2.0
    core_band = _raised_cosine_band(frequencies, center)
    distance = (frequencies[None] - center[:, None]).abs()
    # Estimate the local floor outside the 60 Hz suppression skirt.  A cosine
    # window avoids making the estimate depend sharply on a single STFT bin.
    neighbor_position = ((distance - 60.0) / 60.0).clamp(0.0, 1.0)
    neighbor_band = torch.sin(torch.pi * neighbor_position).square()
    neighbor_band = neighbor_band * (
        (distance >= 60.0) & (distance <= 120.0)
    ).to(neighbor_band.dtype)

    power = spectrum.abs().square()
    core_power = (
        (power * core_band[:, :, None]).sum(dim=-2)
        / core_band.sum(dim=-1).clamp_min(1e-6)[:, None]
    )
    neighbor_power = (
        (power * neighbor_band[:, :, None]).sum(dim=-2)
        / neighbor_band.sum(dim=-1).clamp_min(1e-6)[:, None]
    )
    # Only treat a locally prominent peak as octave evidence.  The square-root
    # power ratio is the complex-STFT gain that brings its average power to the
    # neighboring floor.  It cannot create a below-neighbor dip.
    instantaneous_floor = (
        neighbor_power / core_power.clamp_min(1e-12)
    ).clamp(0.0, 1.0).sqrt()
    requested = torch.where(
        core_power > 1.25 * neighbor_power,
        instantaneous_floor,
        torch.ones_like(instantaneous_floor),
    )
    smoothed = F.avg_pool1d(
        requested[:, None], kernel_size=5, stride=1, padding=2
    )[:, 0]
    # Temporal smoothing may carry attenuation into a frame whose local peak
    # has already disappeared.  This lower bound restores the instantaneous
    # no-dip guarantee in those frames.
    gain = torch.maximum(smoothed, instantaneous_floor)
    gain = torch.where(
        core_power > 1.25 * neighbor_power,
        gain,
        torch.ones_like(gain),
    )
    return 1.0 - core_band[:, :, None] * (1.0 - gain[:, None, :])


def pitch_relative_harmonic_gate(
    audio: torch.Tensor,
    pitches: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    mode: str,
) -> torch.Tensor:
    """Apply a differentiable pitch-relative STFT gate and preserve padding."""

    if audio.ndim != 2:
        raise ValueError("audio must have shape [batch, samples]")
    if pitches.shape != audio.shape[:1] or valid_samples.shape != audio.shape[:1]:
        raise ValueError("pitches and valid_samples must have shape [batch]")
    if mode == "none":
        return audio
    samples = int(audio.shape[-1])
    support = (
        torch.arange(samples, device=audio.device)[None]
        < valid_samples.long()[:, None]
    )
    values = audio.float() * support
    window = torch.hann_window(n_fft, device=audio.device, dtype=values.dtype)
    spectrum = torch.stft(
        values,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    frequencies = (
        torch.arange(spectrum.shape[-2], device=audio.device, dtype=values.dtype)
        * float(sample_rate) / float(n_fft)
    )
    if mode.startswith("suboctave_ratio_cap_"):
        cap = 0.5 if mode.endswith("0p5") else 0.2
        mask = _suboctave_ratio_mask(
            spectrum, frequencies, pitches.to(audio.device), cap=cap
        )
    elif mode == "suboctave_neighbor_smooth":
        mask = _suboctave_neighbor_smoothing_mask(
            spectrum, frequencies, pitches.to(audio.device)
        )
    else:
        mask = _gate_mask(frequencies, pitches.to(audio.device), mode=mode)[:, :, None]
    gated = torch.istft(
        spectrum * mask,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=samples,
    )
    return gated.to(audio.dtype) * support


__all__ = [
    "GATE_MODES",
    "gate_applies_to_target_and_prediction",
    "pitch_relative_harmonic_gate",
    "SOFT_HARMONIC_COMB_FLOOR",
    "SOFT_HARMONIC_COMB_SIGMA_CENTS",
]
