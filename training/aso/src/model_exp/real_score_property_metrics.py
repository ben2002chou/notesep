"""Label-free property metrics for full-score real-mixture note separation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def _sample_mask(valid_samples: torch.Tensor, samples: int) -> torch.Tensor:
    return (
        torch.arange(samples, device=valid_samples.device)[None]
        < valid_samples[:, None]
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value.mul(mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def full_score_allocation_metrics(
    note_estimates: torch.Tensor,
    note_to_mixture: torch.Tensor,
    mixture: torch.Tensor,
    valid_samples: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure closure and anti-cancellation without an isolated-note target.

    These are pre-projection metrics.  Closure alone is not sufficient: the
    cancellation and absolute-allocation ratios expose solutions that explain
    the mixture only after large, oppositely signed note outputs cancel.
    """

    if note_estimates.ndim != 2 or mixture.ndim != 2:
        raise ValueError("estimates and mixture must be rank-two waveforms")
    if note_to_mixture.shape != (note_estimates.shape[0],):
        raise ValueError("note_to_mixture must have one index per note")
    if valid_samples.shape != (mixture.shape[0],):
        raise ValueError("valid_samples must have one value per mixture")
    if note_estimates.shape[1] != mixture.shape[1]:
        raise ValueError("waveform sample counts differ")
    if torch.any((note_to_mixture < 0) | (note_to_mixture >= mixture.shape[0])):
        raise ValueError("note_to_mixture contains an invalid index")

    summed = torch.zeros_like(mixture)
    summed.index_add_(0, note_to_mixture, note_estimates)
    absolute_sum = torch.zeros_like(mixture)
    absolute_sum.index_add_(0, note_to_mixture, note_estimates.abs())
    mask = _sample_mask(valid_samples, mixture.shape[-1])
    mixture_l1 = _masked_mean(mixture.abs(), mask).clamp_min(1e-7)
    mixture_rms = _masked_mean(mixture.square(), mask).sqrt().clamp_min(1e-7)
    residual = mixture - summed
    summed_absolute = summed.abs()
    cancellation = (absolute_sum - summed_absolute).clamp_min(0.0)
    return {
        "summed_estimate": summed,
        "residual": residual,
        "closure_relative_l1": _masked_mean(residual.abs(), mask) / mixture_l1,
        "closure_relative_rmse": (
            _masked_mean(residual.square(), mask).sqrt() / mixture_rms
        ),
        "absolute_allocation_to_mixture_l1": (
            _masked_mean(absolute_sum, mask) / mixture_l1
        ),
        "cancellation_to_allocation_l1": (
            _masked_mean(cancellation, mask)
            / _masked_mean(absolute_sum, mask).clamp_min(1e-7)
        ),
    }


def temporal_residual_metrics(
    residual: torch.Tensor,
    mixture: torch.Tensor,
    valid_samples: torch.Tensor,
    score_events: Sequence[Sequence[tuple[int, int, int]]],
    *,
    onset_samples: int,
    tail_pre_samples: int,
    tail_post_samples: int,
) -> dict[str, torch.Tensor]:
    """Measure unexplained waveform energy near scored attacks and releases."""

    if residual.shape != mixture.shape or residual.ndim != 2:
        raise ValueError("residual and mixture must have aligned [B, T] shapes")
    if len(score_events) != mixture.shape[0]:
        raise ValueError("score_events must have one row per mixture")
    if min(onset_samples, tail_pre_samples, tail_post_samples) < 0:
        raise ValueError("temporal widths must be nonnegative")
    onset_ratios: list[torch.Tensor] = []
    tail_ratios: list[torch.Tensor] = []
    for row, events in enumerate(score_events):
        valid = int(valid_samples[row])
        onset_mask = torch.zeros(valid, device=mixture.device, dtype=torch.bool)
        tail_mask = torch.zeros_like(onset_mask)
        for _, onset, offset in events:
            onset = max(0, min(valid, int(onset)))
            offset = max(onset, min(valid, int(offset)))
            onset_mask[onset : min(offset, onset + onset_samples)] = True
            tail_mask[
                max(onset, offset - tail_pre_samples) : min(
                    valid, offset + tail_post_samples
                )
            ] = True
        for mask, destination in (
            (onset_mask, onset_ratios),
            (tail_mask, tail_ratios),
        ):
            if bool(mask.any()):
                numerator = residual[row, :valid][mask].square().mean().sqrt()
                denominator = (
                    mixture[row, :valid][mask].square().mean().sqrt().clamp_min(1e-7)
                )
                destination.append(numerator / denominator)
            else:
                destination.append(torch.full((), float("nan"), device=mixture.device))
    return {
        "onset_residual_rms_ratio": torch.stack(onset_ratios),
        "tail_residual_rms_ratio": torch.stack(tail_ratios),
    }


def _harmonic_bins(
    pitch: int,
    *,
    sample_rate: int,
    n_fft: int,
    harmonics: int,
    radius: int,
) -> set[int]:
    fundamental = 440.0 * 2.0 ** ((int(pitch) - 69) / 12.0)
    maximum = n_fft // 2
    bins: set[int] = set()
    for harmonic in range(1, harmonics + 1):
        frequency = fundamental * harmonic
        if frequency >= sample_rate / 2:
            break
        center = int(round(frequency * n_fft / sample_rate))
        bins.update(range(max(0, center - radius), min(maximum, center + radius) + 1))
    return bins


def scored_harmonic_residual_ratio(
    residual: torch.Tensor,
    mixture: torch.Tensor,
    valid_samples: torch.Tensor,
    score_events: Sequence[Sequence[tuple[int, int, int]]],
    *,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    harmonics: int = 8,
    bin_radius: int = 1,
) -> torch.Tensor:
    """Return residual/mix energy on score-active harmonic time-frequency bins."""

    if residual.shape != mixture.shape or residual.ndim != 2:
        raise ValueError("residual and mixture must have aligned [B, T] shapes")
    if len(score_events) != mixture.shape[0]:
        raise ValueError("score_events must have one row per mixture")
    window = torch.hann_window(n_fft, device=mixture.device, dtype=mixture.dtype)
    residual_stft = torch.stft(
        residual,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    mixture_stft = torch.stft(
        mixture,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    ratios: list[torch.Tensor] = []
    for row, events in enumerate(score_events):
        mask = torch.zeros_like(residual_stft[row].real, dtype=torch.bool)
        valid_frames = min(
            mask.shape[-1], int(valid_samples[row].item()) // hop_length + 1
        )
        for pitch, onset, offset in events:
            start = max(0, int(onset) // hop_length)
            stop = min(valid_frames, (int(offset) + hop_length - 1) // hop_length + 1)
            bins = sorted(
                _harmonic_bins(
                    pitch,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    harmonics=harmonics,
                    radius=bin_radius,
                )
            )
            if bins and start < stop:
                mask[bins, start:stop] = True
        if bool(mask.any()):
            numerator = residual_stft[row].abs().square()[mask].mean()
            denominator = mixture_stft[row].abs().square()[mask].mean().clamp_min(1e-12)
            ratios.append((numerator / denominator).sqrt())
        else:
            ratios.append(torch.full((), float("nan"), device=mixture.device))
    return torch.stack(ratios)


def competing_harmonic_leakage_ratio(
    note_estimates: torch.Tensor,
    query_events: Sequence[tuple[int, int, int]],
    competing_events: Sequence[Iterable[tuple[int, int, int]]],
    *,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    harmonics: int = 8,
    bin_radius: int = 1,
) -> torch.Tensor:
    """Measure energy at competing-note-only harmonics during temporal overlap."""

    if note_estimates.ndim != 2 or len(query_events) != note_estimates.shape[0]:
        raise ValueError("query events must align with note estimates")
    if len(competing_events) != note_estimates.shape[0]:
        raise ValueError("competing events must align with note estimates")
    window = torch.hann_window(n_fft, device=note_estimates.device, dtype=note_estimates.dtype)
    spectra = torch.stft(
        note_estimates,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs().square()
    ratios: list[torch.Tensor] = []
    for row, (pitch, onset, offset) in enumerate(query_events):
        own_bins = _harmonic_bins(
            pitch,
            sample_rate=sample_rate,
            n_fft=n_fft,
            harmonics=harmonics,
            radius=bin_radius,
        )
        competitor_mask = torch.zeros_like(spectra[row], dtype=torch.bool)
        overlap_frames = torch.zeros(
            spectra.shape[-1], device=spectra.device, dtype=torch.bool
        )
        for other_pitch, other_onset, other_offset in competing_events[row]:
            overlap_start = max(int(onset), int(other_onset))
            overlap_stop = min(int(offset), int(other_offset))
            other_bins = _harmonic_bins(
                other_pitch,
                sample_rate=sample_rate,
                n_fft=n_fft,
                harmonics=harmonics,
                radius=bin_radius,
            ) - own_bins
            start = max(0, overlap_start // hop_length)
            stop = min(
                spectra.shape[-1],
                (overlap_stop + hop_length - 1) // hop_length + 1,
            )
            if other_bins and start < stop:
                competitor_mask[sorted(other_bins), start:stop] = True
                overlap_frames[start:stop] = True
        if bool(competitor_mask.any()):
            # Use the same overlapping frames in numerator and denominator.
            # This is an energy fraction in [0, 1], not a comparison between
            # a sparse-bin mean and a full-spectrogram mean.
            competing_energy = spectra[row][competitor_mask].sum()
            total_energy = spectra[row][:, overlap_frames].sum().clamp_min(1e-12)
            ratios.append(competing_energy / total_energy)
        else:
            ratios.append(torch.full((), float("nan"), device=note_estimates.device))
    return torch.stack(ratios)
