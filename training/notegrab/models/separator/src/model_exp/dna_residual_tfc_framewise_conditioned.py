"""Score-conditioned TFC-TDF canaries with no target-audio conditioning."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .dna_residual_tfc_tdf_unet import (
    DNAResidualTFCTDFUNet,
    DNAResidualTFCTDFUNetConfig,
    dna_residual_tfc_tdf_unet_config,
)


ConditioningArm = Literal["g1_input", "g2_film", "g3_competitor_film"]


class _SpatialFiLM(nn.Module):
    def __init__(self, condition_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.affine = nn.Conv2d(condition_channels, 2 * feature_channels, 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = F.interpolate(
            condition,
            size=values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        return values * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)


class DNAResidualTFCFramewiseConditioned(DNAResidualTFCTDFUNet):
    """TFC-TDF with explicit harmonic/lifecycle conditioning.

    G1 concatenates the score map at the input. G2 additionally applies
    zero-initialized spatial FiLM at every scale. G3 adds score-only competitor
    exclusive/shared support to the same path. The output remains a signed
    complex additive correction anchored to the original DNA waveform.
    """

    def __init__(
        self,
        config: DNAResidualTFCTDFUNetConfig,
        *,
        arm: ConditioningArm,
    ) -> None:
        if arm not in {"g1_input", "g2_film", "g3_competitor_film"}:
            raise ValueError(f"Unknown conditioning arm: {arm}")
        super().__init__(config)
        self.output_parameterization = "bounded_residual"
        self.arm = arm
        self.score_channels = 7 if arm == "g3_competitor_film" else 5
        score_cws_channels = self.score_channels * config.num_subbands
        old = self.first_conv
        replacement = nn.Conv2d(
            old.in_channels + score_cws_channels,
            old.out_channels,
            old.kernel_size,
            bias=False,
        )
        with torch.no_grad():
            replacement.weight[:, : old.in_channels].copy_(old.weight)
            replacement.weight[:, old.in_channels :].zero_()
        self.first_conv = replacement

        self.encoder_film = nn.ModuleList()
        self.decoder_film = nn.ModuleList()
        channels = config.initial_channels
        for _ in range(config.num_scales):
            self.encoder_film.append(_SpatialFiLM(self.score_channels, channels))
            channels += config.growth
        self.bottleneck_film = _SpatialFiLM(self.score_channels, channels)
        for _ in range(config.num_scales):
            channels -= config.growth
            self.decoder_film.append(_SpatialFiLM(self.score_channels, channels))
        self._competitor_events: tuple[torch.Tensor, ...] | None = None
        self.use_activation_checkpointing = True

    @classmethod
    def from_preset(
        cls, preset: str, *, arm: ConditioningArm
    ) -> DNAResidualTFCFramewiseConditioned:
        return cls(dna_residual_tfc_tdf_unet_config(preset), arm=arm)

    def contract(self, preset: str) -> dict[str, object]:
        return {
            "kind": "tfc_framewise_conditioned",
            "preset": preset,
            "conditioning_arm": self.arm,
            "score_channels": self.score_channels,
            "target_audio_conditioning": False,
            "activation_checkpointing": self.use_activation_checkpointing,
            **asdict(self.config),
        }

    def _run_checkpointed(
        self, module: nn.Module, values: torch.Tensor
    ) -> torch.Tensor:
        if self.training and self.use_activation_checkpointing:
            return checkpoint(module, values, use_reentrant=False)
        return module(values)

    def set_competitor_events(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if not (pitches.shape == onsets.shape == offsets.shape == valid.shape):
            raise ValueError("Competitor event tensors must have equal [batch, events] shape")
        self._competitor_events = (pitches, onsets, offsets, valid)

    def _harmonic_support(
        self, pitches: torch.Tensor, active: torch.Tensor, frequency_bins: int
    ) -> torch.Tensor:
        frequency = (
            torch.arange(frequency_bins, device=pitches.device, dtype=torch.float32)
            * float(self.config.sample_rate)
            / float(self.config.n_fft)
        )
        f0 = 440.0 * torch.pow(2.0, (pitches.float() - 69.0) / 12.0)
        harmonics = torch.arange(1, 60, device=pitches.device, dtype=torch.float32)
        centers = f0[..., None] * harmonics
        distance = (frequency[None, None, None, :] - centers[..., None]).abs()
        support = torch.exp(-0.5 * (distance / 7.5).square()).amax(dim=-2)
        return support[:, :, None] * active[..., None].to(support.dtype)

    def _score_map(
        self,
        *,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        frequency_bins: int,
        frames: int,
    ) -> torch.Tensor:
        frame_samples = (
            torch.arange(frames, device=pitch.device)[None] * self.config.hop_length
        )
        frame_valid = frame_samples < valid_samples[:, None]
        onset_region = (
            (frame_samples >= onset[:, None])
            & (frame_samples < torch.minimum(offset, onset + 1_600)[:, None])
        )
        sustain = (
            (frame_samples >= torch.minimum(offset, onset + 1_600)[:, None])
            & (frame_samples < offset[:, None])
        )
        release = (
            (frame_samples >= offset[:, None])
            & (frame_samples < torch.minimum(valid_samples, offset + 4_000)[:, None])
        )
        silence = frame_valid & ~(onset_region | sustain | release)
        target_active = frame_valid & (frame_samples >= onset[:, None]) & (
            frame_samples < torch.minimum(valid_samples, offset + 4_000)[:, None]
        )
        target = self._harmonic_support(
            pitch[:, None], target_active[:, None], frequency_bins
        )[:, 0].transpose(1, 2)
        lifecycle = torch.stack(
            (onset_region, sustain, release, silence), dim=1
        ).to(target.dtype)
        lifecycle = lifecycle[:, :, None].expand(-1, -1, frequency_bins, -1)
        channels = [target[:, None], lifecycle]

        if self.arm == "g3_competitor_film":
            if self._competitor_events is None:
                raise RuntimeError("G3 requires score-derived competitor events per batch")
            competitor_pitch, competitor_onset, competitor_offset, competitor_valid = (
                self._competitor_events
            )
            event_active = (
                competitor_valid.bool()[:, :, None]
                & (frame_samples[:, None] >= competitor_onset[:, :, None])
                & (frame_samples[:, None] < competitor_offset[:, :, None])
                & frame_valid[:, None]
            )
            competitor = self._harmonic_support(
                competitor_pitch, event_active, frequency_bins
            ).amax(dim=1).transpose(1, 2)
            shared = torch.minimum(target, competitor)
            exclusive = competitor * (target < 0.1).to(competitor.dtype)
            channels.extend((exclusive[:, None], shared[:, None]))
        result = torch.cat(channels, dim=1)
        return result * frame_valid[:, None, None].to(result.dtype)

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
            mixture, dna, other_dna, unassigned_residual, pitch, valid_samples, onset, offset
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
            torch.arange(frames, device=mixture.device)[None] * self.config.hop_length
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
        score_map = self._score_map(
            pitch=pitch,
            valid_samples=valid_samples,
            onset=onset,
            offset=offset,
            frequency_bins=self.config.processed_frequency_bins,
            frames=frames,
        )
        score_cws = self._cac_to_cws(score_map)
        network_input = torch.cat((source_cws, conditioning, score_cws), dim=1)
        frame_divisor = 2**self.config.num_scales
        padded_frames = math.ceil(frames / frame_divisor) * frame_divisor
        frame_padding = padded_frames - frames
        if frame_padding:
            network_input = F.pad(network_input, (0, frame_padding))
            source_cws = F.pad(source_cws, (0, frame_padding))
            score_map = F.pad(score_map, (0, frame_padding))

        first_conv_out = hidden = self.first_conv(network_input)
        hidden = hidden.transpose(-1, -2)
        spatial_condition = score_map.transpose(-1, -2)
        encoder_outputs = []
        use_film = self.arm != "g1_input"
        for index, block in enumerate(self.encoder_blocks):
            hidden = self._run_checkpointed(block["tfc_tdf"], hidden)
            if use_film:
                hidden = self.encoder_film[index](hidden, spatial_condition)
            encoder_outputs.append(hidden)
            hidden = block["downscale"](hidden)
        hidden = self._run_checkpointed(self.bottleneck_block, hidden)
        if use_film:
            hidden = self.bottleneck_film(hidden, spatial_condition)
        for index, block in enumerate(self.decoder_blocks):
            hidden = block["upscale"](hidden)
            hidden = torch.cat((hidden, encoder_outputs.pop()), dim=1)
            hidden = self._run_checkpointed(block["tfc_tdf"], hidden)
            if use_film:
                hidden = self.decoder_film[index](hidden, spatial_condition)
        hidden = hidden.transpose(-1, -2)
        hidden = hidden * first_conv_out
        raw_cws = self.final_projection(torch.cat((source_cws, hidden), dim=1))
        raw = self._cws_to_cac(raw_cws)[..., :frames]
        if self.output_parameterization == "direct_unbounded_complex":
            processed_correction = torch.complex(
                raw[:, 0].float(), raw[:, 1].float()
            )
        elif self.output_parameterization == "bounded_residual":
            available = (
                spectra[0].abs() + spectra[1].abs() + spectra[3].abs()
            )[..., : self.config.processed_frequency_bins, :]
            unit = torch.complex(
                torch.tanh(raw[:, 0]).float(), torch.tanh(raw[:, 1]).float()
            )
            processed_correction = (
                self.config.maximum_correction_scale
                * unit
                * available.clamp_min(1e-6)
            )
        else:
            raise RuntimeError(
                f"Unknown output parameterization: {self.output_parameterization}"
            )
        correction_spectrum = spectra[0].new_zeros(batch, frequency_bins, frames)
        correction_spectrum[..., : self.config.processed_frequency_bins, :] = processed_correction
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
        self._competitor_events = None
        return estimate, {
            "estimate": estimate,
            "correction": correction,
            "correction_spectrum": correction_spectrum,
            "per_band_correction_spectra": per_band,
            "band_weights": self.band_weights,
            "band_scales": band_scales,
            "frame_support": frame_support,
            "level": level,
            "score_condition": score_map[..., :frames],
        }
