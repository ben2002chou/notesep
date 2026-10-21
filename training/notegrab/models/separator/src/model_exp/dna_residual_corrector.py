"""Identity-preserving complex residual correction for DNA note estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


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
    cents_to_harmonic = 1200.0 * torch.log2(positive_ratio / nearest_harmonic)
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
    return (
        features.to(dtype=dtype).unsqueeze(-1).expand(-1, -1, -1, time_frames)
    )


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
            raise ValueError(
                f"Unsupported residual output mode: {config.output_mode}"
            )
        self.config = config
        source_count = (
            2 + int(config.use_rt2_prior) + 2 * int(config.use_region_context)
        )
        feature_channels = (
            3 * source_count
            + 3 * int(config.use_harmonic_features)
            + 4 * int(config.use_timing_features)
        )
        self.input = nn.Conv2d(
            feature_channels, config.channels, kernel_size=3, padding=1
        )
        self.pitch_embedding = nn.Embedding(128, config.pitch_embed_dim)
        self.pitch_projection = nn.Linear(
            config.pitch_embed_dim, config.channels
        )
        self.blocks = nn.ModuleList(
            _ResidualBlock(
                config.channels,
                dilation=2 ** (index % 4),
                frequency_dilation=(
                    2 ** (index % 4)
                    if config.use_frequency_dilated_blocks
                    else 1
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
            raise ValueError(
                "mixture and dna_note must have shape [batch, samples]"
            )
        if pitch.shape != (mixture.shape[0],):
            raise ValueError("pitch must have shape [batch]")
        if self.config.use_rt2_prior != (rt2_note is not None):
            raise ValueError(
                "RT2 prior presence does not match model configuration"
            )
        if rt2_note is not None and rt2_note.shape != mixture.shape:
            raise ValueError("rt2_note must match mixture shape")
        has_region_context = (
            other_dna is not None and unassigned_residual is not None
        )
        if self.config.use_region_context != has_region_context:
            raise ValueError(
                "region context presence does not match model configuration"
            )
        if has_region_context and (
            other_dna.shape != mixture.shape
            or unassigned_residual.shape != mixture.shape
        ):
            raise ValueError(
                "region context waveforms must match mixture shape"
            )
        if valid_samples is not None and valid_samples.shape != (
            mixture.shape[0],
        ):
            raise ValueError("valid_samples must have shape [batch]")
        has_timing = (
            note_onset_samples is not None and note_offset_samples is not None
        )
        if self.config.use_timing_features != has_timing:
            raise ValueError(
                "timing feature presence does not match model configuration"
            )
        if has_timing:
            if valid_samples is None:
                raise ValueError("timing features require valid_samples")
            if note_onset_samples.shape != (
                mixture.shape[0],
            ) or note_offset_samples.shape != (mixture.shape[0],):
                raise ValueError("note timing must have shape [batch]")
            if (
                torch.any(note_onset_samples < 0)
                or torch.any(note_offset_samples < note_onset_samples)
                or torch.any(note_offset_samples > valid_samples)
            ):
                raise ValueError("note timing is outside valid_samples")

        # Per-example level normalization keeps quiet and loud notes numerically
        # comparable without changing the identity skip connection.
        if valid_samples is None:
            level = mixture.float().square().mean(dim=-1, keepdim=True).sqrt()
        else:
            positions = torch.arange(mixture.shape[-1], device=mixture.device)[
                None
            ]
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
        features = torch.cat(
            [_complex_channels(value) for value in spectra], dim=1
        )
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
            frame_samples = torch.arange(
                mixture_spectrum.shape[-1],
                device=mixture.device,
                dtype=torch.float32,
            )[None, :] * float(self.config.hop_length)
            onset = note_onset_samples.float()[:, None]
            offset = note_offset_samples.float()[:, None]
            duration = (offset - onset).clamp_min(1.0)
            active = (
                (frame_samples >= onset) & (frame_samples < offset)
            ).float()
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
            correction = (
                torch.istft(
                    correction_spectrum,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.n_fft,
                    window=self._window(mixture.float()),
                    center=True,
                    length=mixture.shape[-1],
                )
                * level
            )
            estimate = dna_note.float() + correction
        elif self.config.output_mode == "ungated_residual":
            gate = torch.ones_like(gate)
            correction_spectrum = (
                float(self.config.maximum_correction_scale)
                * delta_unit
                * available.clamp_min(1e-6)
            )
            correction = (
                torch.istft(
                    correction_spectrum,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.n_fft,
                    window=self._window(mixture.float()),
                    center=True,
                    length=mixture.shape[-1],
                )
                * level
            )
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
            correction = (
                torch.istft(
                    correction_spectrum,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.n_fft,
                    window=self._window(mixture.float()),
                    center=True,
                    length=mixture.shape[-1],
                )
                * level
            )
            estimate = dna_note.float() + correction
        elif self.config.output_mode == "unassigned_residual_filter":
            if unassigned_spectrum is None:
                raise ValueError(
                    "unassigned_residual_filter requires region context"
                )
            gate = torch.ones_like(gate)
            # Filter only energy that DNA left unassigned after extracting every
            # note in the region. This keeps the update grounded in observed
            # mixture content instead of synthesizing a free correction.
            correction_spectrum = (
                float(self.config.maximum_correction_scale)
                * delta_unit
                * unassigned_spectrum
            )
            correction = (
                torch.istft(
                    correction_spectrum,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.n_fft,
                    window=self._window(mixture.float()),
                    center=True,
                    length=mixture.shape[-1],
                )
                * level
            )
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
