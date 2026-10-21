"""Identity-preserving complex residual correction for DNA note estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DNAResidualCorrectorConfig:
    n_fft: int = 512
    hop_length: int = 128
    channels: int = 48
    blocks: int = 6
    pitch_embed_dim: int = 16
    use_rt2_prior: bool = False
    use_region_context: bool = False
    maximum_correction_scale: float = 2.0
    output_mode: str = "bounded_residual"
    soft_replace_gate_bias: float = -12.0
    use_harmonic_features: bool = False
    use_frequency_dilated_blocks: bool = False
    use_timing_features: bool = False
    sample_rate: int = 16_000


class _ResidualBlock(nn.Module):
    def __init__(
        self, channels: int, dilation: int, frequency_dilation: int = 1
    ) -> None:
        super().__init__()
        groups = max(1, channels // 8)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=(frequency_dilation, dilation),
            dilation=(frequency_dilation, dilation),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm1(values))
        residual = self.conv1(residual)
        residual = self.conv2(F.silu(self.norm2(residual)))
        return values + residual


def _complex_channels(spectrum: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            spectrum.real,
            spectrum.imag,
            torch.log1p(10.0 * spectrum.abs()),
        ),
        dim=1,
    )


def _pitch_relative_harmonic_features(
    pitch: torch.Tensor,
    *,
    frequency_bins: int,
    time_frames: int,
    sample_rate: int,
    n_fft: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return pitch-relative frequency, harmonic proximity, and index maps."""

    if pitch.ndim != 1:
        raise ValueError("pitch must have shape [batch]")
    if frequency_bins != n_fft // 2 + 1 or time_frames < 1:
        raise ValueError("feature geometry does not match the configured STFT")
    frequencies = torch.arange(
        frequency_bins, device=pitch.device, dtype=torch.float32
    ) * (float(sample_rate) / float(n_fft))
    fundamental = 440.0 * torch.pow(
        torch.tensor(2.0, device=pitch.device),
        (pitch.float() - 69.0) / 12.0,
    )
    ratio = frequencies[None, :] / fundamental[:, None]
    positive_ratio = ratio.clamp_min(1e-7)

    relative_octaves = torch.log2(positive_ratio).clamp(-4.0, 4.0) / 4.0
    relative_octaves = torch.where(
        frequencies[None, :] > 0.0,
        relative_octaves,
        torch.full_like(relative_octaves, -1.0),
    )

    nearest_harmonic = ratio.round().clamp_min(1.0)
    cents_to_harmonic = 1200.0 * torch.log2(
        positive_ratio / nearest_harmonic
    )
    harmonic_proximity = torch.exp(-0.5 * (cents_to_harmonic / 50.0).square())
    harmonic_proximity = torch.where(
        ratio >= 0.5, harmonic_proximity, torch.zeros_like(harmonic_proximity)
    )

    maximum_harmonic = max(1.0, float(frequency_bins - 1))
    harmonic_index = torch.log1p(nearest_harmonic.clamp_max(maximum_harmonic))
    harmonic_index = harmonic_index / math.log1p(maximum_harmonic)
    harmonic_index = torch.where(
        ratio >= 0.5, harmonic_index, torch.zeros_like(harmonic_index)
    )

    features = torch.stack(
        (relative_octaves, harmonic_proximity, harmonic_index), dim=1
    )
    return features.to(dtype=dtype).unsqueeze(-1).expand(-1, -1, -1, time_frames)


class DNAComplexResidualCorrector(nn.Module):
    """Predict a bounded complex correction while preserving DNA at step zero.

    The selected DNA note is always an explicit input and skip connection. The
    optional RT2 note is only a prior; it never replaces the DNA identity path.
    """

    def __init__(self, config: DNAResidualCorrectorConfig) -> None:
        super().__init__()
        if config.n_fft < 16 or not 0 < config.hop_length <= config.n_fft:
            raise ValueError("Invalid STFT geometry")
        if config.sample_rate < 1:
            raise ValueError("Sample rate must be positive")
        if config.channels < 8 or config.blocks < 1:
            raise ValueError("Corrector capacity must be positive")
        if config.output_mode not in {
            "bounded_residual",
            "complex_mask_delta",
            "soft_replace",
            "unassigned_residual_filter",
            "ungated_residual",
        }:
            raise ValueError(f"Unsupported residual output mode: {config.output_mode}")
        self.config = config
        source_count = 2 + int(config.use_rt2_prior) + 2 * int(config.use_region_context)
        feature_channels = (
            3 * source_count
            + 3 * int(config.use_harmonic_features)
            + 4 * int(config.use_timing_features)
        )
        self.input = nn.Conv2d(feature_channels, config.channels, kernel_size=3, padding=1)
        self.pitch_embedding = nn.Embedding(128, config.pitch_embed_dim)
        self.pitch_projection = nn.Linear(config.pitch_embed_dim, config.channels)
        self.blocks = nn.ModuleList(
            _ResidualBlock(
                config.channels,
                dilation=2 ** (index % 4),
                frequency_dilation=(
                    2 ** (index % 4) if config.use_frequency_dilated_blocks else 1
                ),
            )
            for index in range(config.blocks)
        )
        self.output = nn.Conv2d(config.channels, 3, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if config.output_mode == "soft_replace":
            with torch.no_grad():
                self.output.bias[2] = float(config.soft_replace_gate_bias)

    def _window(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(
            self.config.n_fft,
            device=waveform.device,
            dtype=waveform.dtype,
        )

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform.float(),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(waveform.float()),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )

    def forward(
        self,
        mixture: torch.Tensor,
        dna_note: torch.Tensor,
        pitch: torch.Tensor,
        rt2_note: torch.Tensor | None = None,
        valid_samples: torch.Tensor | None = None,
        other_dna: torch.Tensor | None = None,
        unassigned_residual: torch.Tensor | None = None,
        note_onset_samples: torch.Tensor | None = None,
        note_offset_samples: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mixture.ndim != 2 or dna_note.shape != mixture.shape:
            raise ValueError("mixture and dna_note must have shape [batch, samples]")
        if pitch.shape != (mixture.shape[0],):
            raise ValueError("pitch must have shape [batch]")
        if self.config.use_rt2_prior != (rt2_note is not None):
            raise ValueError("RT2 prior presence does not match model configuration")
        if rt2_note is not None and rt2_note.shape != mixture.shape:
            raise ValueError("rt2_note must match mixture shape")
        has_region_context = other_dna is not None and unassigned_residual is not None
        if self.config.use_region_context != has_region_context:
            raise ValueError("region context presence does not match model configuration")
        if has_region_context and (
            other_dna.shape != mixture.shape or unassigned_residual.shape != mixture.shape
        ):
            raise ValueError("region context waveforms must match mixture shape")
        if valid_samples is not None and valid_samples.shape != (mixture.shape[0],):
            raise ValueError("valid_samples must have shape [batch]")
        has_timing = note_onset_samples is not None and note_offset_samples is not None
        if self.config.use_timing_features != has_timing:
            raise ValueError("timing feature presence does not match model configuration")
        if has_timing:
            if valid_samples is None:
                raise ValueError("timing features require valid_samples")
            if (
                note_onset_samples.shape != (mixture.shape[0],)
                or note_offset_samples.shape != (mixture.shape[0],)
            ):
                raise ValueError("note timing must have shape [batch]")
            if torch.any(note_onset_samples < 0) or torch.any(
                note_offset_samples < note_onset_samples
            ) or torch.any(note_offset_samples > valid_samples):
                raise ValueError("note timing is outside valid_samples")

        # Per-example level normalization keeps quiet and loud notes numerically
        # comparable without changing the identity skip connection.
        if valid_samples is None:
            level = mixture.float().square().mean(dim=-1, keepdim=True).sqrt()
        else:
            positions = torch.arange(mixture.shape[-1], device=mixture.device)[None]
            mask = positions < valid_samples.long()[:, None]
            level = (
                (mixture.float().square() * mask).sum(dim=-1, keepdim=True)
                / valid_samples.clamp_min(1).to(mixture.dtype)[:, None]
            ).sqrt()
        level = level.clamp_min(1e-4)
        mixture_spectrum = self._stft(mixture / level)
        dna_spectrum = self._stft(dna_note / level)
        spectra = [mixture_spectrum, dna_spectrum]
        unassigned_spectrum = None
        if has_region_context:
            spectra.append(self._stft(other_dna / level))
            unassigned_spectrum = self._stft(unassigned_residual / level)
            spectra.append(unassigned_spectrum)
        if rt2_note is not None:
            spectra.append(self._stft(rt2_note / level))
        features = torch.cat([_complex_channels(value) for value in spectra], dim=1)
        if self.config.use_harmonic_features:
            features = torch.cat(
                (
                    features,
                    _pitch_relative_harmonic_features(
                        pitch,
                        frequency_bins=mixture_spectrum.shape[-2],
                        time_frames=mixture_spectrum.shape[-1],
                        sample_rate=self.config.sample_rate,
                        n_fft=self.config.n_fft,
                        dtype=features.dtype,
                    ),
                ),
                dim=1,
            )
        if has_timing:
            frame_samples = (
                torch.arange(
                    mixture_spectrum.shape[-1],
                    device=mixture.device,
                    dtype=torch.float32,
                )[None, :]
                * float(self.config.hop_length)
            )
            onset = note_onset_samples.float()[:, None]
            offset = note_offset_samples.float()[:, None]
            duration = (offset - onset).clamp_min(1.0)
            active = ((frame_samples >= onset) & (frame_samples < offset)).float()
            since_onset = ((frame_samples - onset) / duration).clamp(0.0, 1.0)
            until_offset = ((offset - frame_samples) / duration).clamp(0.0, 1.0)
            post_offset = (frame_samples >= offset).float()
            timing = torch.stack(
                (active, since_onset, until_offset, post_offset), dim=1
            ).to(dtype=features.dtype)
            timing = timing[:, :, None, :].expand(
                -1, -1, mixture_spectrum.shape[-2], -1
            )
            features = torch.cat((features, timing), dim=1)
        hidden = self.input(features)
        pitch_bias = self.pitch_projection(self.pitch_embedding(pitch.long()))
        hidden = hidden + pitch_bias[:, :, None, None]
        for block in self.blocks:
            hidden = block(hidden)

        raw = self.output(hidden)
        # torch.complex does not accept BF16 components. Keep convolutional
        # compute under autocast, then construct the phase-sensitive residual
        # in complex64 for stable STFT-domain arithmetic.
        delta_unit = torch.complex(
            torch.tanh(raw[:, 0]).float(),
            torch.tanh(raw[:, 1]).float(),
        )
        gate = torch.sigmoid(raw[:, 2])
        available = mixture_spectrum.abs() + dna_spectrum.abs()
        # RT2 is a learned feature, not a physical energy authority. Keeping it
        # out of the correction bound makes zero RT2-channel weights exactly
        # recover the DNA-only model and prevents hallucinated RT2 energy from
        # expanding the residual amplitude budget.
        if self.config.output_mode == "bounded_residual":
            correction_spectrum = (
                float(self.config.maximum_correction_scale)
                * gate
                * delta_unit
                * available.clamp_min(1e-6)
            )
            correction = torch.istft(
                correction_spectrum,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.n_fft,
                window=self._window(mixture.float()),
                center=True,
                length=mixture.shape[-1],
            ) * level
            estimate = dna_note.float() + correction
        elif self.config.output_mode == "ungated_residual":
            gate = torch.ones_like(gate)
            correction_spectrum = (
                float(self.config.maximum_correction_scale)
                * delta_unit
                * available.clamp_min(1e-6)
            )
            correction = torch.istft(
                correction_spectrum,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.n_fft,
                window=self._window(mixture.float()),
                center=True,
                length=mixture.shape[-1],
            ) * level
            estimate = dna_note.float() + correction
        elif self.config.output_mode == "complex_mask_delta":
            gate = torch.ones_like(gate)
            # DNA is a real mixture-phase mask. Predicting a complex mask delta
            # lets the corrector rotate and rescale mixture bins directly while
            # retaining an exact DNA identity at the zero initialization.
            correction_spectrum = (
                float(self.config.maximum_correction_scale)
                * delta_unit
                * mixture_spectrum
            )
            correction = torch.istft(
                correction_spectrum,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.n_fft,
                window=self._window(mixture.float()),
                center=True,
                length=mixture.shape[-1],
            ) * level
            estimate = dna_note.float() + correction
        elif self.config.output_mode == "unassigned_residual_filter":
            if unassigned_spectrum is None:
                raise ValueError("unassigned_residual_filter requires region context")
            gate = torch.ones_like(gate)
            # Filter only energy that DNA left unassigned after extracting every
            # note in the region. This keeps the update grounded in observed
            # mixture content instead of synthesizing a free correction.
            correction_spectrum = (
                float(self.config.maximum_correction_scale)
                * delta_unit
                * unassigned_spectrum
            )
            correction = torch.istft(
                correction_spectrum,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.n_fft,
                window=self._window(mixture.float()),
                center=True,
                length=mixture.shape[-1],
            ) * level
            estimate = dna_note.float() + correction
        else:
            generated_spectrum = (
                float(self.config.maximum_correction_scale)
                * delta_unit
                * available.clamp_min(1e-6)
            )
            # This is an initialization prior, not a permanent identity skip:
            # gate=1 removes DNA completely and uses the generated spectrum.
            estimate_spectrum = dna_spectrum + gate * (
                generated_spectrum - dna_spectrum
            )
            correction_spectrum = estimate_spectrum - dna_spectrum
            estimate_normalized = torch.istft(
                estimate_spectrum,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.n_fft,
                window=self._window(mixture.float()),
                center=True,
                length=mixture.shape[-1],
            )
            estimate = estimate_normalized * level
            correction = estimate - dna_note.float()
        return estimate, {
            "correction": correction,
            "gate": gate,
            "correction_spectrum": correction_spectrum,
            "dna_spectrum": dna_spectrum,
            "available_spectrum": available,
            "level": level,
        }


def oracle_gate_supervision_loss(
    predicted_gate: torch.Tensor,
    target_spectrum: torch.Tensor,
    dna_spectrum: torch.Tensor,
    available_spectrum: torch.Tensor,
    *,
    maximum_correction_scale: float,
    valid_samples: torch.Tensor | None = None,
    hop_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teach the learned gate where target evidence requires changing DNA."""

    if predicted_gate.shape != target_spectrum.shape:
        raise ValueError("Gate and target spectrum shapes must match")
    if dna_spectrum.shape != target_spectrum.shape:
        raise ValueError("DNA and target spectrum shapes must match")
    if available_spectrum.shape != target_spectrum.shape:
        raise ValueError("Available and target spectrum shapes must match")
    if maximum_correction_scale <= 0.0 or hop_length <= 0:
        raise ValueError("Invalid gate supervision geometry")
    needed = (target_spectrum - dna_spectrum).abs()
    capacity = float(maximum_correction_scale) * available_spectrum.clamp_min(1e-6)
    oracle_gate = (needed / capacity).clamp(0.0, 1.0).sqrt().detach()
    error = F.smooth_l1_loss(
        predicted_gate.float(), oracle_gate.float(), reduction="none", beta=0.1
    )
    if valid_samples is None:
        return error.mean(), oracle_gate.mean()
    frames = torch.div(
        valid_samples.long() + hop_length - 1, hop_length, rounding_mode="floor"
    ).add(1).clamp(max=error.shape[-1])
    mask = torch.arange(error.shape[-1], device=error.device)[None] < frames[:, None]
    denominator = (frames * error.shape[-2]).clamp_min(1).to(error.dtype)
    per_example = (error * mask[:, None]).sum(dim=(-2, -1)) / denominator
    oracle_mean = (
        oracle_gate * mask[:, None]
    ).sum(dim=(-2, -1)) / denominator
    return per_example.mean(), oracle_mean.mean()


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
        estimate.float(), fft_size, hop_length, fft_size, window,
        center=True, pad_mode="constant", return_complex=True,
    )
    reference_spectrum = torch.stft(
        reference.float(), fft_size, hop_length, fft_size, window,
        center=True, pad_mode="constant", return_complex=True,
    )
    difference = (estimate_spectrum - reference_spectrum).abs()
    if normalizer is None:
        magnitude = reference_spectrum.abs()
    else:
        magnitude = torch.stft(
            normalizer.float(), fft_size, hop_length, fft_size, window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
    if valid_samples is None:
        numerator = difference.mean(dim=(-2, -1))
        denominator = magnitude.mean(dim=(-2, -1)).clamp_min(1e-5)
    else:
        counts = torch.div(
            valid_samples.long() + hop_length - 1, hop_length, rounding_mode="floor"
        ).add(1).clamp(max=difference.shape[-1])
        mask = torch.arange(difference.shape[-1], device=estimate.device)[None] < counts[:, None]
        denominator_count = (counts * difference.shape[-2]).clamp_min(1).to(difference.dtype)
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
            values.float(), fft_size, hop_length, fft_size, window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
    error = (
        torch.log1p(100.0 * magnitude(estimate))
        - torch.log1p(100.0 * magnitude(reference))
    ).abs()
    if valid_samples is None:
        return error.mean(dim=(-2, -1))
    counts = torch.div(
        valid_samples.long() + hop_length - 1, hop_length, rounding_mode="floor"
    ).add(1).clamp(max=error.shape[-1])
    mask = torch.arange(error.shape[-1], device=estimate.device)[None] < counts[:, None]
    denominator = (counts * error.shape[-2]).clamp_min(1).to(error.dtype)
    return (error * mask[:, None]).sum(dim=(-2, -1)) / denominator


def _per_example_phase_coherence_error(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    fft_size: int,
    hop_length: int,
    valid_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference-energy-weighted phase disagreement in [0, 2]."""

    if valid_samples is not None:
        sample_mask = (
            torch.arange(estimate.shape[-1], device=estimate.device)[None]
            < valid_samples.long()[:, None]
        )
        estimate = estimate * sample_mask
        reference = reference * sample_mask
    window = torch.hann_window(fft_size, device=estimate.device, dtype=torch.float32)
    estimate_spectrum = torch.stft(
        estimate.float(), fft_size, hop_length, fft_size, window,
        center=True, pad_mode="constant", return_complex=True,
    )
    reference_spectrum = torch.stft(
        reference.float(), fft_size, hop_length, fft_size, window,
        center=True, pad_mode="constant", return_complex=True,
    )
    estimate_magnitude = estimate_spectrum.abs()
    reference_magnitude = reference_spectrum.abs()
    cosine = (
        (estimate_spectrum * reference_spectrum.conj()).real
        / (estimate_magnitude * reference_magnitude).clamp_min(1e-8)
    ).clamp(-1.0, 1.0)
    error = 1.0 - cosine
    weights = reference_magnitude.sqrt()
    if valid_samples is not None:
        counts = torch.div(
            valid_samples.long() + hop_length - 1, hop_length, rounding_mode="floor"
        ).add(1).clamp(max=error.shape[-1])
        frame_mask = (
            torch.arange(error.shape[-1], device=estimate.device)[None]
            < counts[:, None]
        )
        weights = weights * frame_mask[:, None]
    return (error * weights).sum(dim=(-2, -1)) / weights.sum(
        dim=(-2, -1)
    ).clamp_min(1e-8)


def _per_example_boundary_derivative_error(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    valid_samples: torch.Tensor | None = None,
    sample_rate: int = 16_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized onset/offset click errors for each example.

    Boundaries are derived only from the fixed reference waveform, so this is
    differentiable with respect to the estimate and does not require target
    audio at inference time. The local maximum mirrors the release-click
    diagnostic used by the held-out lifecycle evaluator.
    """

    if estimate.shape != reference.shape or estimate.ndim != 2:
        raise ValueError("estimate and reference must have shape [batch, samples]")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    batch_size, samples = estimate.shape
    if samples < 2:
        zeros = estimate.new_zeros(batch_size, dtype=torch.float32)
        return zeros, zeros
    if valid_samples is None:
        valid = torch.full(
            (batch_size,), samples, device=estimate.device, dtype=torch.long
        )
    else:
        valid = valid_samples.long().clamp(min=0, max=samples)

    positions = torch.arange(samples, device=estimate.device)[None]
    sample_mask = positions < valid[:, None]
    reference_abs = reference.float().abs() * sample_mask
    peak = reference_abs.amax(dim=-1, keepdim=True)
    threshold = torch.maximum(
        peak * 1e-3,
        torch.full_like(peak, 1e-6),
    )
    active = (reference_abs >= threshold) & sample_mask
    has_active = active.any(dim=-1)
    first = torch.where(active, positions, samples).amin(dim=-1)
    last = torch.where(active, positions, -1).amax(dim=-1)
    boundaries = torch.stack((first, last + 1), dim=1).clamp(min=1, max=samples - 1)

    derivative_positions = torch.arange(1, samples, device=estimate.device)[None, None]
    derivative_error = (
        torch.diff(estimate.float(), dim=-1)
        - torch.diff(reference.float(), dim=-1)
    ).abs()[:, None]
    click_radius = max(1, int(round(0.003 * sample_rate)))
    click_mask = (
        (derivative_positions - boundaries[:, :, None]).abs() <= click_radius
    ) & (derivative_positions < valid[:, None, None])
    worst_derivative_error = derivative_error.masked_fill(
        ~click_mask, 0.0
    ).amax(dim=-1)

    context_radius = max(1, int(round(0.020 * sample_rate)))
    context_mask = (
        (positions[:, None] - boundaries[:, :, None]).abs() <= context_radius
    ) & sample_mask[:, None]
    context_count = context_mask.sum(dim=-1).clamp_min(1)
    local_rms = (
        (reference.float().square()[:, None] * context_mask).sum(dim=-1)
        / context_count
        + 1e-12
    ).sqrt()
    normalized = worst_derivative_error / local_rms.clamp_min(1e-6)
    normalized = torch.where(has_active[:, None], normalized, torch.zeros_like(normalized))
    return normalized[:, 0], normalized[:, 1]


def _per_example_tail_energy_error(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    valid_samples: torch.Tensor | None = None,
    sample_rate: int = 16_000,
    tail_seconds: float = 0.150,
) -> torch.Tensor:
    """Return absolute post-offset energy error in dB for each example.

    This mirrors the held-out lifecycle metric: the reference offset is the
    first sample after its final sample above -60 dB relative to peak, and the
    tail spans up to 150 ms after that offset. Reference-derived masks are
    constants with respect to the estimate, so the result remains
    differentiable while matching the evaluation boundary semantics.
    """

    if estimate.shape != reference.shape or estimate.ndim != 2:
        raise ValueError("estimate and reference must have shape [batch, samples]")
    if sample_rate <= 0 or tail_seconds <= 0:
        raise ValueError("sample_rate and tail_seconds must be positive")
    batch_size, samples = estimate.shape
    if valid_samples is None:
        valid = torch.full(
            (batch_size,), samples, device=estimate.device, dtype=torch.long
        )
    else:
        valid = valid_samples.long().clamp(min=0, max=samples)

    positions = torch.arange(samples, device=estimate.device)[None]
    valid_mask = positions < valid[:, None]
    reference_abs = reference.float().abs() * valid_mask
    peak = reference_abs.amax(dim=-1, keepdim=True)
    threshold = torch.maximum(
        peak * 1e-3,
        torch.full_like(peak, 1e-6),
    )
    active = (reference_abs >= threshold) & valid_mask
    has_active = active.any(dim=-1)
    last = torch.where(active, positions, -1).amax(dim=-1)
    offset = torch.where(has_active, last + 1, torch.zeros_like(last))
    tail_samples = max(1, int(round(float(tail_seconds) * int(sample_rate))))
    stop = torch.minimum(offset + tail_samples, valid)
    tail_mask = (positions >= offset[:, None]) & (positions < stop[:, None])
    has_tail = tail_mask.any(dim=-1)

    estimate_energy = (estimate.float().square() * tail_mask).sum(dim=-1)
    reference_energy = (reference.float().square() * tail_mask).sum(dim=-1)
    error_db = (10.0 / math.log(10.0)) * (
        torch.log(estimate_energy + 1e-12) - torch.log(reference_energy + 1e-12)
    ).abs()
    return torch.where(has_tail, error_db, torch.zeros_like(error_db))


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
    do_no_harm_weight: float = 0.5,
    correction_weight: float = 0.01,
    quiet_region_weight: float = 0.0,
    quiet_do_no_harm_weight: float = 0.0,
    onset_derivative_weight: float = 0.0,
    offset_derivative_weight: float = 0.0,
    boundary_do_no_harm_weight: float = 0.0,
    tail_energy_weight: float = 0.0,
    tail_do_no_harm_weight: float = 0.0,
    tail_cvar_weight: float = 0.0,
    tail_cvar_fraction: float = 0.25,
    sample_rate: int = 16_000,
) -> dict[str, torch.Tensor]:
    if not 0.0 < float(tail_cvar_fraction) <= 1.0:
        raise ValueError("tail_cvar_fraction must be in (0, 1]")
    if float(complex_error_cap) < 0.0:
        raise ValueError("complex_error_cap must be non-negative")
    complex_terms = []
    dna_complex_terms = []
    log_terms = []
    dna_log_terms = []
    phase_terms = []
    dna_phase_terms = []
    for fft_size in fft_sizes:
        hop = max(1, int(fft_size) // 4)
        complex_terms.append(
            _per_example_complex_error(
                estimate, target, fft_size=int(fft_size), hop_length=hop,
                valid_samples=valid_samples,
                normalizer=complex_normalizer,
            )
        )
        dna_complex_terms.append(
            _per_example_complex_error(
                dna_note, target, fft_size=int(fft_size), hop_length=hop,
                valid_samples=valid_samples,
                normalizer=complex_normalizer,
            )
        )
        log_terms.append(
            _per_example_log_magnitude_error(
                estimate, target, fft_size=int(fft_size), hop_length=hop,
                valid_samples=valid_samples,
            )
        )
        dna_log_terms.append(
            _per_example_log_magnitude_error(
                dna_note, target, fft_size=int(fft_size), hop_length=hop,
                valid_samples=valid_samples,
            )
        )
        phase_terms.append(
            _per_example_phase_coherence_error(
                estimate, target, fft_size=int(fft_size), hop_length=hop,
                valid_samples=valid_samples,
            )
        )
        dna_phase_terms.append(
            _per_example_phase_coherence_error(
                dna_note, target, fft_size=int(fft_size), hop_length=hop,
                valid_samples=valid_samples,
            )
        )
    complex_per_example = torch.stack(complex_terms).mean(dim=0)
    dna_complex_per_example = torch.stack(dna_complex_terms).mean(dim=0)
    log_per_example = torch.stack(log_terms).mean(dim=0)
    dna_log_per_example = torch.stack(dna_log_terms).mean(dim=0)
    phase_per_example = torch.stack(phase_terms).mean(dim=0)
    dna_phase_per_example = torch.stack(dna_phase_terms).mean(dim=0)
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
    do_no_harm = (
        F.relu(complex_per_example - dna_complex_per_example)
        + F.relu(log_per_example - dna_log_per_example)
    ).mean()
    if sample_mask is None:
        target_power = target.float().square().mean(dim=-1)
        correction_power = correction.float().square().mean(dim=-1)
    else:
        denominator = valid_samples.clamp_min(1).to(target.dtype)
        target_power = (target.float().square() * sample_mask).sum(dim=-1) / denominator
        correction_power = (
            correction.float().square() * sample_mask
        ).sum(dim=-1) / denominator
    target_rms = target_power.sqrt().clamp_min(1e-5)
    correction_ratio = (
        correction_power.clamp_min(1e-10).sqrt() / target_rms
    ).mean()
    quiet_window = min(512, target.shape[-1])
    quiet_envelope = F.avg_pool1d(
        target.float().square()[:, None],
        quiet_window,
        stride=1,
        padding=quiet_window // 2,
    ).sqrt()[:, 0, : target.shape[-1]]
    quiet_threshold = quiet_envelope.amax(dim=-1, keepdim=True) * (10.0 ** (-45.0 / 20.0))
    quiet_mask = quiet_envelope <= quiet_threshold.clamp_min(1e-7)
    if sample_mask is not None:
        quiet_mask = quiet_mask & sample_mask
    quiet_count = quiet_mask.sum(dim=-1)
    quiet_error_per_example = (
        ((estimate.float() - target.float()).abs() * quiet_mask).sum(dim=-1)
        / quiet_count.clamp_min(1)
        / target_rms
    )
    quiet_error = torch.where(
        quiet_count > 0, quiet_error_per_example, torch.zeros_like(quiet_error_per_example)
    ).mean()
    quiet_denominator = quiet_count.clamp_min(1).to(estimate.dtype)
    estimate_quiet_rms = (
        ((estimate.float() - target.float()).square() * quiet_mask).sum(dim=-1)
        / quiet_denominator
        + 1e-14
    ).sqrt()
    dna_quiet_rms = (
        ((dna_note.float() - target.float()).square() * quiet_mask).sum(dim=-1)
        / quiet_denominator
        + 1e-14
    ).sqrt()
    quiet_do_no_harm_per_example = F.relu(
        torch.log(estimate_quiet_rms + 1e-7) - torch.log(dna_quiet_rms + 1e-7)
    )
    quiet_do_no_harm = torch.where(
        quiet_count > 0,
        quiet_do_no_harm_per_example,
        torch.zeros_like(quiet_do_no_harm_per_example),
    ).mean()
    use_boundary_loss = any(
        float(weight) > 0.0
        for weight in (
            onset_derivative_weight,
            offset_derivative_weight,
            boundary_do_no_harm_weight,
        )
    )
    if use_boundary_loss:
        onset_derivative_per_example, offset_derivative_per_example = (
            _per_example_boundary_derivative_error(
                estimate,
                target,
                valid_samples=valid_samples,
                sample_rate=sample_rate,
            )
        )
    else:
        onset_derivative_per_example = estimate.new_zeros(
            estimate.shape[0], dtype=torch.float32
        )
        offset_derivative_per_example = onset_derivative_per_example
    if float(boundary_do_no_harm_weight) > 0.0:
        dna_onset_derivative, dna_offset_derivative = (
            _per_example_boundary_derivative_error(
                dna_note,
                target,
                valid_samples=valid_samples,
                sample_rate=sample_rate,
            )
        )
    else:
        dna_onset_derivative = estimate.new_zeros(
            estimate.shape[0], dtype=torch.float32
        )
        dna_offset_derivative = dna_onset_derivative
    onset_derivative = onset_derivative_per_example.mean()
    offset_derivative = offset_derivative_per_example.mean()
    boundary_do_no_harm = (
        F.relu(onset_derivative_per_example - dna_onset_derivative)
        + F.relu(offset_derivative_per_example - dna_offset_derivative)
    ).mean()
    use_tail_loss = any(
        float(weight) > 0.0
        for weight in (tail_energy_weight, tail_do_no_harm_weight, tail_cvar_weight)
    )
    if use_tail_loss:
        tail_energy_per_example = _per_example_tail_energy_error(
            estimate,
            target,
            valid_samples=valid_samples,
            sample_rate=sample_rate,
        )
        dna_tail_energy_per_example = _per_example_tail_energy_error(
            dna_note,
            target,
            valid_samples=valid_samples,
            sample_rate=sample_rate,
        )
    else:
        tail_energy_per_example = estimate.new_zeros(
            estimate.shape[0], dtype=torch.float32
        )
        dna_tail_energy_per_example = tail_energy_per_example
    tail_energy = tail_energy_per_example.mean()
    dna_tail_energy = dna_tail_energy_per_example.mean()
    tail_do_no_harm = F.relu(
        tail_energy_per_example - dna_tail_energy_per_example
    ).mean()
    hard_count = max(
        1, int(math.ceil(estimate.shape[0] * float(tail_cvar_fraction)))
    )
    tail_cvar = torch.topk(
        tail_energy_per_example, k=min(hard_count, estimate.shape[0])
    ).values.mean()
    complex_loss = complex_per_example.mean()
    complex_optimized = (
        complex_per_example.clamp_max(float(complex_error_cap)).mean()
        if float(complex_error_cap) > 0.0
        else complex_loss
    )
    log_magnitude = log_per_example.mean()
    phase_coherence = phase_per_example.mean()
    total = (
        float(complex_weight) * complex_optimized
        + float(log_magnitude_weight) * log_magnitude
        + float(waveform_weight) * waveform
        + float(envelope_weight) * envelope
        + float(phase_coherence_weight) * phase_coherence
        + float(do_no_harm_weight) * do_no_harm
        + float(correction_weight) * correction_ratio
        + float(quiet_region_weight) * quiet_error
        + float(quiet_do_no_harm_weight) * quiet_do_no_harm
        + float(onset_derivative_weight) * onset_derivative
        + float(offset_derivative_weight) * offset_derivative
        + float(boundary_do_no_harm_weight) * boundary_do_no_harm
        + float(tail_energy_weight) * tail_energy
        + float(tail_do_no_harm_weight) * tail_do_no_harm
        + float(tail_cvar_weight) * tail_cvar
    )
    return {
        "loss": total,
        "complex": complex_loss,
        "complex_optimized": complex_optimized,
        "dna_complex": dna_complex_per_example.mean(),
        "log_magnitude": log_magnitude,
        "dna_log_magnitude": dna_log_per_example.mean(),
        "waveform_l1": waveform,
        "envelope": envelope,
        "phase_coherence": phase_coherence,
        "dna_phase_coherence": dna_phase_per_example.mean(),
        "do_no_harm": do_no_harm,
        "correction_ratio": correction_ratio,
        "quiet_region_error": quiet_error,
        "quiet_do_no_harm": quiet_do_no_harm,
        "onset_derivative_error": onset_derivative,
        "dna_onset_derivative_error": dna_onset_derivative.mean(),
        "offset_derivative_error": offset_derivative,
        "dna_offset_derivative_error": dna_offset_derivative.mean(),
        "boundary_do_no_harm": boundary_do_no_harm,
        "tail_energy_error_db": tail_energy,
        "dna_tail_energy_error_db": dna_tail_energy,
        "tail_do_no_harm": tail_do_no_harm,
        "tail_cvar_error_db": tail_cvar,
        "complex_improvement": (dna_complex_per_example - complex_per_example).mean(),
        "log_magnitude_improvement": (dna_log_per_example - log_per_example).mean(),
        "phase_coherence_improvement": (
            dna_phase_per_example - phase_per_example
        ).mean(),
    }
