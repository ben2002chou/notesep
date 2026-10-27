"""Spectral and waveform losses used to train NoteGrab."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _per_example_complex_error(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    fft_size: int,
    hop_length: int,
    valid_samples: torch.Tensor | None = None,
    normalizer: torch.Tensor | None = None,
) -> torch.Tensor:
    if normalizer is not None and normalizer.shape != reference.shape:
        raise ValueError("complex-error normalizer must match reference shape")
    if valid_samples is not None:
        sample_mask = (
            torch.arange(estimate.shape[-1], device=estimate.device)[None]
            < valid_samples.long()[:, None]
        )
        estimate = estimate * sample_mask
        reference = reference * sample_mask
        if normalizer is not None:
            normalizer = normalizer * sample_mask
    window = torch.hann_window(fft_size, device=estimate.device, dtype=torch.float32)
    estimate_spectrum = torch.stft(
        estimate.float(),
        fft_size,
        hop_length,
        fft_size,
        window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    reference_spectrum = torch.stft(
        reference.float(),
        fft_size,
        hop_length,
        fft_size,
        window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    difference = (estimate_spectrum - reference_spectrum).abs()
    if normalizer is None:
        magnitude = reference_spectrum.abs()
    else:
        magnitude = torch.stft(
            normalizer.float(),
            fft_size,
            hop_length,
            fft_size,
            window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()
    if valid_samples is None:
        numerator = difference.mean(dim=(-2, -1))
        denominator = magnitude.mean(dim=(-2, -1)).clamp_min(1e-5)
    else:
        counts = (
            torch.div(
                valid_samples.long() + hop_length - 1,
                hop_length,
                rounding_mode="floor",
            )
            .add(1)
            .clamp(max=difference.shape[-1])
        )
        mask = (
            torch.arange(difference.shape[-1], device=estimate.device)[None]
            < counts[:, None]
        )
        denominator_count = (
            (counts * difference.shape[-2]).clamp_min(1).to(difference.dtype)
        )
        numerator = (difference * mask[:, None]).sum(dim=(-2, -1)) / denominator_count
        denominator = (magnitude * mask[:, None]).sum(dim=(-2, -1)) / denominator_count
        denominator = denominator.clamp_min(1e-5)
    return numerator / denominator


def _per_example_log_magnitude_error(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    fft_size: int,
    hop_length: int,
    valid_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_samples is not None:
        sample_mask = (
            torch.arange(estimate.shape[-1], device=estimate.device)[None]
            < valid_samples.long()[:, None]
        )
        estimate = estimate * sample_mask
        reference = reference * sample_mask
    window = torch.hann_window(fft_size, device=estimate.device, dtype=torch.float32)

    def magnitude(values: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            values.float(),
            fft_size,
            hop_length,
            fft_size,
            window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()

    error = (
        torch.log1p(100.0 * magnitude(estimate))
        - torch.log1p(100.0 * magnitude(reference))
    ).abs()
    if valid_samples is None:
        return error.mean(dim=(-2, -1))
    counts = (
        torch.div(
            valid_samples.long() + hop_length - 1,
            hop_length,
            rounding_mode="floor",
        )
        .add(1)
        .clamp(max=error.shape[-1])
    )
    mask = torch.arange(error.shape[-1], device=estimate.device)[None] < counts[:, None]
    denominator = (counts * error.shape[-2]).clamp_min(1).to(error.dtype)
    return (error * mask[:, None]).sum(dim=(-2, -1)) / denominator


def dna_residual_losses(
    estimate: torch.Tensor,
    target: torch.Tensor,
    dna_note: torch.Tensor,
    correction: torch.Tensor,
    *,
    valid_samples: torch.Tensor | None = None,
    complex_normalizer: torch.Tensor | None = None,
    fft_sizes: Sequence[int] = (256, 512, 1024),
    complex_weight: float = 1.0,
    complex_error_cap: float = 0.0,
    log_magnitude_weight: float = 0.5,
    waveform_weight: float = 0.2,
    envelope_weight: float = 0.2,
    phase_coherence_weight: float = 0.0,
    do_no_harm_weight: float = 0.0,
    correction_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the multi-resolution losses in the paper training recipe."""
    if phase_coherence_weight or do_no_harm_weight or correction_weight:
        raise ValueError("The NoteGrab recipe does not use these extra losses")
    if complex_error_cap < 0:
        raise ValueError("complex_error_cap must be non-negative")
    complex_terms, log_terms = [], []
    for fft_size in fft_sizes:
        hop = max(1, int(fft_size) // 4)
        complex_terms.append(
            _per_example_complex_error(
                estimate,
                target,
                fft_size=int(fft_size),
                hop_length=hop,
                valid_samples=valid_samples,
                normalizer=complex_normalizer,
            )
        )
        log_terms.append(
            _per_example_log_magnitude_error(
                estimate,
                target,
                fft_size=int(fft_size),
                hop_length=hop,
                valid_samples=valid_samples,
            )
        )
    complex_per_example = torch.stack(complex_terms).mean(dim=0)
    log_per_example = torch.stack(log_terms).mean(dim=0)
    if valid_samples is None:
        sample_mask = None
        masked_estimate = estimate
        masked_target = target
        waveform = (estimate.float() - target.float()).abs().mean()
    else:
        sample_mask = (
            torch.arange(estimate.shape[-1], device=estimate.device)[None]
            < valid_samples.long()[:, None]
        )
        waveform = (
            ((estimate.float() - target.float()).abs() * sample_mask).sum(dim=-1)
            / valid_samples.clamp_min(1).to(estimate.dtype)
        ).mean()
        masked_estimate = estimate * sample_mask
        masked_target = target * sample_mask

    envelope_terms = []
    for window_size in (64, 256, 1024):
        size = min(window_size, estimate.shape[-1])
        if size:
            estimate_envelope = F.avg_pool1d(
                masked_estimate.abs()[:, None], size, max(1, size // 4)
            )
            target_envelope = F.avg_pool1d(
                masked_target.abs()[:, None], size, max(1, size // 4)
            )
            error = (
                torch.log(estimate_envelope.clamp_min(1e-5))
                - torch.log(target_envelope.clamp_min(1e-5))
            ).abs()
            if valid_samples is None:
                envelope_terms.append(error.mean())
            else:
                stride = max(1, size // 4)
                counts = (
                    torch.div(
                        (valid_samples.long() - size).clamp_min(0),
                        stride,
                        rounding_mode="floor",
                    )
                    + 1
                ).clamp(max=error.shape[-1])
                mask = (
                    torch.arange(error.shape[-1], device=estimate.device)[None]
                    < counts[:, None]
                )
                envelope_terms.append(
                    ((error[:, 0] * mask).sum(dim=-1) / counts.clamp_min(1)).mean()
                )
    envelope = torch.stack(envelope_terms).mean()
    complex_loss = complex_per_example.mean()
    complex_optimized = (
        complex_per_example.clamp_max(float(complex_error_cap)).mean()
        if float(complex_error_cap) > 0.0
        else complex_loss
    )
    log_magnitude = log_per_example.mean()
    loss = (
        float(complex_weight) * complex_optimized
        + float(log_magnitude_weight) * log_magnitude
        + float(waveform_weight) * waveform
        + float(envelope_weight) * envelope
    )
    return {
        "loss": loss,
        "complex": complex_loss,
        "complex_optimized": complex_optimized,
        "log_magnitude": log_magnitude,
        "waveform_l1": waveform,
        "envelope": envelope,
    }
