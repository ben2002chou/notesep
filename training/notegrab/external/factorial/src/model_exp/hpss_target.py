"""Batched differentiable-device HPSS decomposition for training targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _median_envelopes(
    magnitude: torch.Tensor, *, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if magnitude.ndim != 3:
        raise ValueError("magnitude must have shape [batch, frequency, time]")
    if kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd and at least three")
    radius = kernel_size // 2
    if magnitude.shape[-1] <= radius or magnitude.shape[-2] <= radius:
        raise ValueError("STFT is too small for the requested HPSS kernel")
    harmonic = (
        F.pad(magnitude, (radius, radius), mode="reflect")
        .unfold(-1, kernel_size, 1)
        .median(dim=-1)
        .values
    )
    percussive = (
        F.pad(magnitude, (0, 0, radius, radius), mode="reflect")
        .unfold(-2, kernel_size, 1)
        .median(dim=-1)
        .values
    )
    return harmonic, percussive


def project_hpss_batch(
    audio: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordinary soft-mask HPSS harmonic and percussive waveforms.

    At margin one, the two masks are complementary.  Computing the second
    waveform as the exact time-domain complement avoids introducing a third
    residual target through numerical STFT reconstruction error.
    """

    if audio.ndim != 2:
        raise ValueError("audio must have shape [batch, samples]")
    if valid_samples.shape != audio.shape[:1]:
        raise ValueError("valid_samples must have shape [batch]")
    if power <= 0.0 or margin < 1.0:
        raise ValueError("power must be positive and margin must be at least one")
    samples = int(audio.shape[-1])
    support = torch.arange(samples, device=audio.device)[None] < valid_samples.long()[:, None]
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
    harmonic_envelope, percussive_envelope = _median_envelopes(
        spectrum.abs(), kernel_size=kernel_size
    )
    harmonic_weight = harmonic_envelope.pow(power)
    percussive_weight = (float(margin) * percussive_envelope).pow(power)
    harmonic_mask = harmonic_weight / (
        harmonic_weight + percussive_weight
    ).clamp_min(1e-12)
    harmonic = torch.istft(
        spectrum * harmonic_mask,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=samples,
    ) * support
    percussive = (values - harmonic) * support
    return harmonic, percussive


def project_complex_hpss_spectrum(
    spectrum: torch.Tensor,
    *,
    kernel_size: int = 31,
    power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a complex STFT into exactly complementary HPSS carriers."""

    if spectrum.ndim != 3 or not torch.is_complex(spectrum):
        raise ValueError("spectrum must have shape [batch,frequency,time]")
    if power <= 0.0:
        raise ValueError("HPSS power must be positive")
    harmonic_envelope, percussive_envelope = _median_envelopes(
        spectrum.abs(), kernel_size=kernel_size
    )
    harmonic_weight = harmonic_envelope.pow(power)
    percussive_weight = percussive_envelope.pow(power)
    harmonic_mask = harmonic_weight / (
        harmonic_weight + percussive_weight
    ).clamp_min(1e-12)
    harmonic = harmonic_mask * spectrum
    percussive = spectrum - harmonic
    return harmonic, percussive


def project_with_reference_hpss_mask_batch(
    reference_audio: torch.Tensor,
    audio: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``audio`` with the HPSS mask derived from ``reference_audio``.

    This is intended for matched diagnostics.  Deriving one mask from the
    rendered target and applying that same linear operator to the mixture and
    every estimate prevents method-dependent HPSS boundaries from confounding
    a harmonic-versus-percussive comparison.  The percussive waveform is the
    exact complement, so the two returned components add back to ``audio`` on
    valid support.
    """

    if reference_audio.ndim != 2 or audio.shape != reference_audio.shape:
        raise ValueError(
            "reference_audio and audio must have equal [batch, samples] shapes"
        )
    if valid_samples.shape != audio.shape[:1]:
        raise ValueError("valid_samples must have shape [batch]")
    if power <= 0.0 or margin < 1.0:
        raise ValueError("power must be positive and margin must be at least one")
    samples = int(audio.shape[-1])
    support = (
        torch.arange(samples, device=audio.device)[None]
        < valid_samples.long()[:, None]
    )
    reference = reference_audio.float() * support
    values = audio.float() * support
    window = torch.hann_window(n_fft, device=audio.device, dtype=values.dtype)

    def spectrum(waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

    reference_spectrum = spectrum(reference)
    audio_spectrum = spectrum(values)
    harmonic_envelope, percussive_envelope = _median_envelopes(
        reference_spectrum.abs(), kernel_size=kernel_size
    )
    harmonic_weight = harmonic_envelope.pow(power)
    percussive_weight = (float(margin) * percussive_envelope).pow(power)
    harmonic_mask = harmonic_weight / (
        harmonic_weight + percussive_weight
    ).clamp_min(1e-12)
    harmonic = torch.istft(
        audio_spectrum * harmonic_mask,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=samples,
    ) * support
    percussive = (values - harmonic) * support
    return harmonic, percussive


__all__ = [
    "project_complex_hpss_spectrum",
    "project_hpss_batch",
    "project_with_reference_hpss_mask_batch",
]
