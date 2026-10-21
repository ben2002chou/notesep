"""TFC-TDF-UNet v3 donor for signed complex DNA residual correction.

Architecture flow is adapted from KUIELab's official SDX23 implementation:
https://github.com/kuielab/sdx23/blob/mdx_AB/my_submission/src/tfc_tdf_v3.py

The donor implementation is MIT licensed (Copyright 2023 IELab at Korea
University). This adaptation preserves its CWS subbands, TFC/TDF residual
blocks, encoder/decoder shape flow, multiplicative artifact gate, and final
mixture skip. Task-specific changes are documented in the public model class.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DNAResidualTFCTDFUNetConfig:
    """Exact checkpoint contract for the mono DNA residual donor."""

    n_fft: int = 1024
    hop_length: int = 128
    sample_rate: int = 16_000
    processed_frequency_bins: int = 512
    num_subbands: int = 4
    num_scales: int = 5
    num_blocks_per_scale: int = 2
    initial_channels: int = 64
    growth: int = 64
    tdf_bottleneck_factor: int = 4
    conditioning_channels: int = 32
    pitch_embed_dim: int = 16
    bands: int = 24
    minimum_log_frequency: float = 31.25
    band_radius: float = 1.0
    band_scale_floor_ratio: float = 1e-3
    maximum_correction_scale: float = 2.0


# The presets intentionally keep the official depth, growth pattern, CWS
# subbands, InstanceNorm, and GELU. Their exact parameter counts are exposed by
# ``parameter_count`` so integration code can bind the realized contract.
DNA_RESIDUAL_TFC_TDF_UNET_PRESETS: dict[str, DNAResidualTFCTDFUNetConfig] = {
    "30m": DNAResidualTFCTDFUNetConfig(initial_channels=64, growth=64),
    "46m": DNAResidualTFCTDFUNetConfig(initial_channels=96, growth=80),
}


def dna_residual_tfc_tdf_unet_config(
    preset: str,
) -> DNAResidualTFCTDFUNetConfig:
    """Return an immutable preset config and reject unknown names."""

    try:
        return replace(DNA_RESIDUAL_TFC_TDF_UNET_PRESETS[preset])
    except KeyError as error:
        choices = tuple(DNA_RESIDUAL_TFC_TDF_UNET_PRESETS)
        raise ValueError(f"Unknown TFC-TDF preset {preset!r}; expected {choices}") from error


def _instance_norm(channels: int) -> nn.Module:
    return nn.InstanceNorm2d(channels, affine=True)


class _Downscale(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _instance_norm(input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, output_channels, 2, 2, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _Upscale(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _instance_norm(input_channels),
            nn.GELU(),
            nn.ConvTranspose2d(input_channels, output_channels, 2, 2, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _TFCTDFBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        frequency_bins: int,
        bottleneck_factor: int,
    ) -> None:
        super().__init__()
        hidden_frequency = frequency_bins // bottleneck_factor
        if hidden_frequency < 1:
            raise ValueError("TDF bottleneck collapsed to zero frequency bins")
        self.tfc1 = nn.Sequential(
            _instance_norm(input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
        )
        self.tdf = nn.Sequential(
            _instance_norm(output_channels),
            nn.GELU(),
            nn.Linear(frequency_bins, hidden_frequency, bias=False),
            _instance_norm(output_channels),
            nn.GELU(),
            nn.Linear(hidden_frequency, frequency_bins, bias=False),
        )
        self.tfc2 = nn.Sequential(
            _instance_norm(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
        )
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(values)
        values = self.tfc1(values)
        values = values + self.tdf(values)
        values = self.tfc2(values)
        return values + shortcut


class _TFCTDFStack(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        blocks: int,
        frequency_bins: int,
        bottleneck_factor: int,
    ) -> None:
        super().__init__()
        layers = []
        for _ in range(blocks):
            layers.append(
                _TFCTDFBlock(
                    input_channels,
                    output_channels,
                    frequency_bins,
                    bottleneck_factor,
                )
            )
            input_channels = output_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


def _log_band_weights(
    frequency_bins: int,
    *,
    n_fft: int,
    sample_rate: int,
    bands: int,
    minimum_frequency: float,
    radius: float,
) -> torch.Tensor:
    if bands == 1:
        return torch.ones(1, frequency_bins)
    frequencies = torch.arange(frequency_bins, dtype=torch.float64)
    frequencies.mul_(float(sample_rate) / float(n_fft))
    positions = torch.log1p(frequencies / float(minimum_frequency))
    positions.mul_(float(bands - 1) / float(positions[-1].clamp_min(1e-12)))
    centers = torch.arange(bands, dtype=torch.float64)[:, None]
    weights = (1.0 - (positions[None] - centers).abs() / float(radius)).clamp_min(0)
    uncovered = weights.sum(dim=0) == 0
    if torch.any(uncovered):
        nearest = positions[uncovered].round().long().clamp(0, bands - 1)
        weights[:, uncovered] = F.one_hot(nearest, bands).T.to(weights)
    return (weights / weights.sum(dim=0, keepdim=True)).float()


class DNAResidualTFCTDFUNet(nn.Module):
    """Predict a signed complex correction and add it to the DNA waveform.

    Relative to the official SDX23 network, this version is mono, consumes four
    complex source views (mixture, DNA, other DNA, and unassigned residual),
    injects score/support conditioning, and predicts a zero-initialized complex
    correction instead of an absolute source. Variable frame lengths are padded
    only around the U-Net and cropped before reconstruction.
    """

    _SOURCE_COUNT = 4

    def __init__(self, config: DNAResidualTFCTDFUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or DNAResidualTFCTDFUNetConfig()
        config = self.config
        frequency_bins = config.n_fft // 2 + 1
        if config.n_fft < 16 or not 0 < config.hop_length <= config.n_fft // 2:
            raise ValueError("Invalid STFT geometry")
        if config.sample_rate < 1 or config.processed_frequency_bins < 1:
            raise ValueError("Sample rate and processed frequency bins must be positive")
        if config.processed_frequency_bins > frequency_bins:
            raise ValueError("processed_frequency_bins exceeds the STFT")
        if min(
            config.num_subbands,
            config.num_scales,
            config.num_blocks_per_scale,
            config.initial_channels,
            config.growth,
            config.tdf_bottleneck_factor,
            config.conditioning_channels,
            config.pitch_embed_dim,
            config.bands,
        ) < 1:
            raise ValueError("Model capacities must be positive")
        divisor = config.num_subbands * 2**config.num_scales
        if config.processed_frequency_bins % divisor:
            raise ValueError(
                "processed_frequency_bins must be divisible by "
                "num_subbands * 2**num_scales"
            )
        if config.minimum_log_frequency <= 0 or config.band_radius <= 0.5:
            raise ValueError("Invalid diagnostic band geometry")
        if not 0 < config.band_scale_floor_ratio < 1:
            raise ValueError("band_scale_floor_ratio must be in (0, 1)")
        if config.maximum_correction_scale <= 0:
            raise ValueError("maximum_correction_scale must be positive")

        source_channels = self._SOURCE_COUNT * 2 * config.num_subbands
        network_channels = source_channels + config.conditioning_channels
        channels = config.initial_channels
        frequency = config.processed_frequency_bins // config.num_subbands
        self.pitch_embedding = nn.Embedding(128, config.pitch_embed_dim)
        self.conditioning_projection = nn.Sequential(
            nn.Linear(config.pitch_embed_dim + 3, config.conditioning_channels),
            nn.GELU(),
            nn.Linear(config.conditioning_channels, config.conditioning_channels),
        )
        self.first_conv = nn.Conv2d(network_channels, channels, 1, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(config.num_scales):
            block = nn.ModuleDict(
                {
                    "tfc_tdf": _TFCTDFStack(
                        channels,
                        channels,
                        config.num_blocks_per_scale,
                        frequency,
                        config.tdf_bottleneck_factor,
                    ),
                    "downscale": _Downscale(channels, channels + config.growth),
                }
            )
            self.encoder_blocks.append(block)
            channels += config.growth
            frequency //= 2

        self.bottleneck_block = _TFCTDFStack(
            channels,
            channels,
            config.num_blocks_per_scale,
            frequency,
            config.tdf_bottleneck_factor,
        )
        self.decoder_blocks = nn.ModuleList()
        for _ in range(config.num_scales):
            next_channels = channels - config.growth
            frequency *= 2
            block = nn.ModuleDict(
                {
                    "upscale": _Upscale(channels, next_channels),
                    "tfc_tdf": _TFCTDFStack(
                        2 * next_channels,
                        next_channels,
                        config.num_blocks_per_scale,
                        frequency,
                        config.tdf_bottleneck_factor,
                    ),
                }
            )
            self.decoder_blocks.append(block)
            channels = next_channels

        self.final_projection = nn.Sequential(
            nn.Conv2d(channels + source_channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 2 * config.num_subbands, 1, bias=False),
        )
        nn.init.zeros_(self.final_projection[-1].weight)

        weights = _log_band_weights(
            frequency_bins,
            n_fft=config.n_fft,
            sample_rate=config.sample_rate,
            bands=config.bands,
            minimum_frequency=config.minimum_log_frequency,
            radius=config.band_radius,
        )
        self.register_buffer("band_weights", weights, persistent=True)

    @classmethod
    def from_preset(cls, preset: str) -> DNAResidualTFCTDFUNet:
        return cls(dna_residual_tfc_tdf_unet_config(preset))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _window(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(
            self.config.n_fft,
            periodic=True,
            dtype=reference.dtype,
            device=reference.device,
        )

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(waveform),
            center=True,
            return_complex=True,
        )

    def _cac_to_cws(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = values.shape
        subbands = self.config.num_subbands
        return values.reshape(
            batch, channels, subbands, frequency // subbands, frames
        ).reshape(batch, channels * subbands, frequency // subbands, frames)

    def _cws_to_cac(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = values.shape
        subbands = self.config.num_subbands
        return values.reshape(
            batch, channels // subbands, subbands, frequency, frames
        ).reshape(batch, channels // subbands, frequency * subbands, frames)

    def _validate_inputs(
        self,
        mixture: torch.Tensor,
        dna: torch.Tensor,
        other_dna: torch.Tensor,
        unassigned_residual: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
    ) -> None:
        if mixture.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        for name, value in (
            ("dna", dna),
            ("other_dna", other_dna),
            ("unassigned_residual", unassigned_residual),
        ):
            if value.shape != mixture.shape:
                raise ValueError(f"{name} must match mixture shape")
        batch, samples = mixture.shape
        for name, value in (
            ("pitch", pitch),
            ("valid_samples", valid_samples),
            ("onset", onset),
            ("offset", offset),
        ):
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if torch.any(valid_samples < 1) or torch.any(valid_samples > samples):
            raise ValueError("valid_samples falls outside waveform bounds")
        if torch.any(onset < 0) or torch.any(onset > offset):
            raise ValueError("onset must be nonnegative and no later than offset")
        if torch.any(offset > valid_samples):
            raise ValueError("offset must not exceed valid_samples")
        if torch.any(pitch < 0) or torch.any(pitch > 127):
            raise ValueError("pitch must be a MIDI value in [0, 127]")

    def _band_scales(
        self, spectra: list[torch.Tensor], frame_support: torch.Tensor
    ) -> torch.Tensor:
        condition_power = torch.stack(
            [spectrum.abs().square() for spectrum in spectra], dim=1
        ).sum(dim=1)
        support = frame_support[:, None].to(condition_power.dtype)
        mixture_power = spectra[0].abs().square() * support
        denominator = frame_support.sum(dim=-1).clamp_min(1) * mixture_power.shape[1]
        global_scale = (mixture_power.sum(dim=(-2, -1)) / denominator).sqrt()
        floor = (global_scale * self.config.band_scale_floor_ratio).clamp_min(
            torch.finfo(condition_power.dtype).eps
        )
        scales = []
        for weights in self.band_weights:
            weighted = condition_power * weights[None, :, None] * support
            count = weights.sum() * frame_support.sum(dim=-1).clamp_min(1)
            scales.append((weighted.sum(dim=(-2, -1)) / count).sqrt())
        return torch.maximum(torch.stack(scales, dim=1), floor[:, None]).detach()

    def forward(
        self,
        mixture: torch.Tensor,
        dna: torch.Tensor,
        other_dna: torch.Tensor,
        unassigned_residual: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(
            mixture,
            dna,
            other_dna,
            unassigned_residual,
            pitch,
            valid_samples,
            onset,
            offset,
        )
        batch, samples = mixture.shape
        sample_positions = torch.arange(samples, device=mixture.device)[None]
        sample_support = sample_positions < valid_samples.long()[:, None]
        sources = [mixture, dna, other_dna, unassigned_residual]
        masked_sources = [source.float() * sample_support for source in sources]
        denominator = valid_samples.float().clamp_min(1)[:, None]
        levels = [
            (source.square().sum(dim=-1, keepdim=True) / denominator).sqrt()
            for source in masked_sources
        ]
        level = torch.stack(levels).amax(dim=0).clamp_min(1e-4)
        spectra = [self._stft(source / level) for source in masked_sources]
        frequency_bins, frames = spectra[0].shape[-2:]

        frame_samples = (
            torch.arange(frames, device=mixture.device)[None]
            * self.config.hop_length
        )
        frame_support = frame_samples < valid_samples.long()[:, None]
        valid_scale = valid_samples.float().clamp_min(1)[:, None]
        frame_context = torch.stack(
            (
                frame_support.float(),
                ((frame_samples - onset[:, None]) / valid_scale).clamp(-1, 1),
                ((offset[:, None] - frame_samples) / valid_scale).clamp(-1, 1),
            ),
            dim=-1,
        )
        pitch_features = self.pitch_embedding(pitch.long())[:, None].expand(-1, frames, -1)
        conditioning = self.conditioning_projection(
            torch.cat((pitch_features, frame_context), dim=-1)
        ).transpose(1, 2)

        complex_channels = torch.cat(
            [torch.stack((spectrum.real, spectrum.imag), dim=1) for spectrum in spectra],
            dim=1,
        )[..., : self.config.processed_frequency_bins, :]
        source_cws = self._cac_to_cws(complex_channels)
        frequency = source_cws.shape[-2]
        conditioning = conditioning[:, :, None].expand(-1, -1, frequency, -1)
        network_input = torch.cat((source_cws, conditioning), dim=1)

        frame_divisor = 2**self.config.num_scales
        padded_frames = math.ceil(frames / frame_divisor) * frame_divisor
        frame_padding = padded_frames - frames
        if frame_padding:
            network_input = F.pad(network_input, (0, frame_padding))
            source_cws = F.pad(source_cws, (0, frame_padding))

        first_conv_out = hidden = self.first_conv(network_input)
        hidden = hidden.transpose(-1, -2)
        encoder_outputs = []
        for block in self.encoder_blocks:
            hidden = block["tfc_tdf"](hidden)
            encoder_outputs.append(hidden)
            hidden = block["downscale"](hidden)
        hidden = self.bottleneck_block(hidden)
        for block in self.decoder_blocks:
            hidden = block["upscale"](hidden)
            hidden = torch.cat((hidden, encoder_outputs.pop()), dim=1)
            hidden = block["tfc_tdf"](hidden)
        hidden = hidden.transpose(-1, -2)
        hidden = hidden * first_conv_out
        raw_cws = self.final_projection(torch.cat((source_cws, hidden), dim=1))
        raw = self._cws_to_cac(raw_cws)[..., :frames]

        available = (
            spectra[0].abs() + spectra[1].abs() + spectra[3].abs()
        )[..., : self.config.processed_frequency_bins, :]
        # torch.complex does not accept BF16 components under autocast. Keep
        # synthesis in FP32, matching torch.stft/istft's complex64 contract.
        unit = torch.complex(
            torch.tanh(raw[:, 0]).float(), torch.tanh(raw[:, 1]).float()
        )
        processed_correction = (
            self.config.maximum_correction_scale
            * unit
            * available.clamp_min(1e-6)
        )
        correction_spectrum = spectra[0].new_zeros(batch, frequency_bins, frames)
        correction_spectrum[..., : self.config.processed_frequency_bins, :] = (
            processed_correction
        )
        correction_spectrum = correction_spectrum * frame_support[:, None]
        correction = torch.istft(
            correction_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(mixture.float()),
            center=True,
            length=samples,
        ) * level
        correction = correction * sample_support
        estimate = torch.where(sample_support, dna.float() + correction, dna.float())

        band_scales = self._band_scales(spectra, frame_support)
        per_band = correction_spectrum[:, None] * self.band_weights[None, :, :, None]
        return estimate, {
            "estimate": estimate,
            "correction": correction,
            "correction_spectrum": correction_spectrum,
            "per_band_correction_spectra": per_band,
            "band_weights": self.band_weights,
            "band_scales": band_scales,
            "frame_support": frame_support,
            "level": level,
        }
