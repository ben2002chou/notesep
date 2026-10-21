"""Compact flow-ready complex band expert for DNA residual correction.

The model keeps frequency local while parallelizing the expensive temporal
blocks: bands are folded into the batch dimension for a ConvNeXt-style TCN,
then exchange information only with adjacent bands at a pooled bottleneck.
Zero-initialized band heads make the waveform path an exact DNA identity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DNAResidualBandExpertConfig:
    n_fft: int = 1024
    hop_length: int = 128
    sample_rate: int = 16_000
    bands: int = 24
    channels: int = 16
    temporal_blocks: int = 8
    temporal_kernel: int = 7
    bottleneck_stride: int = 4
    pitch_embed_dim: int = 8
    minimum_log_frequency: float = 31.25
    band_radius: float = 1.0
    band_scale_floor_ratio: float = 1e-3
    maximum_correction_scale: float = 2.0


@dataclass(frozen=True)
class DNAResidualBandDualRegionExpertConfig(DNAResidualBandExpertConfig):
    """Flat checkpoint contract for active-note and post-offset experts."""

    crossfade_frames: int = 8


@dataclass(frozen=True)
class DNAResidualBandAttackContinuationExpertConfig(
    DNAResidualBandDualRegionExpertConfig
):
    """Flat checkpoint contract for attack and continuation experts."""

    attack_samples: int = 1_600


def _complex_features(spectrum: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (spectrum.real, spectrum.imag, torch.log1p(10.0 * spectrum.abs())), dim=1
    )


def _log_band_weights(
    frequency_bins: int,
    *,
    n_fft: int,
    sample_rate: int,
    bands: int,
    minimum_frequency: float,
    radius: float,
) -> torch.Tensor:
    """Return triangular log-frequency tapers normalized for exact overlap-add."""

    if bands == 1:
        return torch.ones(1, frequency_bins)
    frequencies = torch.arange(frequency_bins, dtype=torch.float64)
    frequencies.mul_(float(sample_rate) / float(n_fft))
    log_position = torch.log1p(frequencies / float(minimum_frequency))
    log_position.mul_(float(bands - 1) / float(log_position[-1].clamp_min(1e-12)))
    centers = torch.arange(bands, dtype=torch.float64)[:, None]
    raw = (1.0 - (log_position[None] - centers).abs() / float(radius)).clamp_min(0.0)
    # A radius below one can leave gaps between adjacent triangular supports.
    # Assign any such bin to its nearest center before normalization.
    uncovered = raw.sum(dim=0) == 0
    if torch.any(uncovered):
        nearest = log_position[uncovered].round().long().clamp(0, bands - 1)
        raw[:, uncovered] = F.one_hot(nearest, bands).T.to(raw)
    return (raw / raw.sum(dim=0, keepdim=True)).to(torch.float32)


class _TemporalConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.flow_film = nn.Linear(channels, 2 * channels)
        self.expand = nn.Conv1d(channels, 2 * channels, 1)
        self.contract = nn.Conv1d(2 * channels, channels, 1)

    def forward(
        self,
        values: torch.Tensor,
        support: torch.Tensor,
        flow_condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.depthwise(values)
        hidden = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        scale, shift = self.flow_film(flow_condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale[:, :, None]) + shift[:, :, None]
        hidden = self.contract(F.gelu(self.expand(hidden)))
        return (values + hidden) * support


class _SparseCrossBandMixer(nn.Module):
    """Mix each band with only its immediate frequency neighbors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.neighbor = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels
        )
        self.channel = nn.Conv2d(channels, channels, 1)

    def forward(self, values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # values: [batch, bands, channels, frames]
        hidden = values.permute(0, 2, 1, 3)
        hidden = self.channel(F.gelu(self.neighbor(hidden)))
        hidden = hidden.permute(0, 2, 1, 3)
        return (values + hidden) * support[:, None]


class DNAResidualBandExpert(nn.Module):
    """Predict a signed complex correction using parallel frequency experts.

    ``mode="deterministic"`` returns ``dna + correction``. In
    ``mode="velocity"`` the same first return value is the residual velocity;
    ``diagnostics["estimate"]`` still exposes ``dna + velocity`` for callers
    that need a waveform candidate during future flow training.
    """

    _SOURCE_COUNT = 5  # mixture, DNA, other DNA, unassigned, interpolant
    # Global context plus two band-local scale/activity channels.
    _CONTEXT_CHANNELS = 7
    _TEMPORAL_DILATION_CYCLE = 8

    def __init__(self, config: DNAResidualBandExpertConfig | None = None) -> None:
        super().__init__()
        self.config = config or DNAResidualBandExpertConfig()
        config = self.config
        if config.n_fft < 16 or not 0 < config.hop_length <= config.n_fft // 2:
            raise ValueError("Invalid STFT geometry")
        if config.sample_rate < 1 or config.minimum_log_frequency <= 0:
            raise ValueError("Frequency configuration must be positive")
        if config.bands < 1 or config.channels < 4 or config.temporal_blocks < 1:
            raise ValueError("Band expert capacity must be positive")
        if config.temporal_kernel < 3 or config.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd and at least three")
        if config.bottleneck_stride < 1 or config.band_radius <= 0.5:
            raise ValueError("Invalid overlap or bottleneck geometry")
        if config.maximum_correction_scale <= 0:
            raise ValueError("maximum_correction_scale must be positive")
        if not 0 < config.band_scale_floor_ratio < 1:
            raise ValueError("band_scale_floor_ratio must be in (0, 1)")

        frequency_bins = config.n_fft // 2 + 1
        weights = _log_band_weights(
            frequency_bins,
            n_fft=config.n_fft,
            sample_rate=config.sample_rate,
            bands=config.bands,
            minimum_frequency=config.minimum_log_frequency,
            radius=config.band_radius,
        )
        self.register_buffer("band_weights", weights, persistent=True)
        supports: list[tuple[int, int]] = []
        for weight in weights:
            nonzero = torch.nonzero(weight > 0, as_tuple=False).flatten()
            if len(nonzero) == 0:
                raise ValueError("Too many bands for the configured STFT geometry")
            supports.append((int(nonzero[0]), int(nonzero[-1]) + 1))
        self.band_supports = tuple(supports)

        input_channels = 3 * self._SOURCE_COUNT + self._CONTEXT_CHANNELS
        self.input_projections = nn.ModuleList(
            nn.Conv2d(input_channels, config.channels, kernel_size=(end - start, 1))
            for start, end in self.band_supports
        )
        self.output_projections = nn.ModuleList(
            nn.Conv1d(config.channels, 2 * (end - start), 1)
            for start, end in self.band_supports
        )
        self.band_embeddings = nn.Parameter(torch.zeros(config.bands, config.channels))
        self.pitch_embedding = nn.Embedding(128, config.pitch_embed_dim)
        self.pitch_projection = nn.Linear(config.pitch_embed_dim, config.channels)
        self.flow_condition = nn.Sequential(
            nn.Linear(2, config.channels),
            nn.SiLU(),
            nn.Linear(config.channels, config.channels),
        )
        self.temporal_dilations = tuple(
            2 ** (index % self._TEMPORAL_DILATION_CYCLE)
            for index in range(config.temporal_blocks)
        )
        self.temporal = nn.ModuleList(
            _TemporalConvNeXtBlock(
                config.channels,
                config.temporal_kernel,
                dilation=dilation,
            )
            for dilation in self.temporal_dilations
        )
        self.cross_band_mixer = _SparseCrossBandMixer(config.channels)
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def overlap_reconstruction_weights(self) -> torch.Tensor:
        """Expose fixed [band, frequency] tapers used by frequency overlap-add."""

        return self.band_weights

    def temporal_receptive_field_frames(self) -> int:
        """Return the exact temporal TCN receptive field in STFT frames."""

        return 1 + (self.config.temporal_kernel - 1) * sum(
            self.temporal_dilations
        )

    def temporal_receptive_field_seconds(self) -> float:
        """Return the temporal TCN receptive field duration at this hop size."""

        return (
            self.temporal_receptive_field_frames()
            * self.config.hop_length
            / self.config.sample_rate
        )

    def _band_condition_statistics(
        self,
        spectra: list[torch.Tensor],
        frame_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute detached local RMS scales and activity for every band."""

        support = frame_support[:, None].to(spectra[0].real.dtype)
        condition_power = torch.stack(
            [spectrum.abs().square() for spectrum in spectra], dim=1
        ).sum(dim=1)
        mixture_power = spectra[0].abs().square() * support
        mixture_denominator = (
            frame_support.sum(dim=-1).clamp_min(1) * mixture_power.shape[1]
        )
        global_mixture_scale = (
            mixture_power.sum(dim=(-2, -1)) / mixture_denominator
        ).sqrt()
        absolute_floor = torch.finfo(spectra[0].real.dtype).eps
        floor = (
            global_mixture_scale * self.config.band_scale_floor_ratio
        ).clamp_min(absolute_floor)

        energies = []
        for band, (start, end) in enumerate(self.band_supports):
            weights = self.band_weights[band, start:end]
            weighted_power = (
                condition_power[:, start:end]
                * weights[None, :, None]
                * support
            )
            denominator = weights.sum() * frame_support.sum(dim=-1).clamp_min(1)
            energies.append((weighted_power.sum(dim=(-2, -1)) / denominator).sqrt())
        energy = torch.stack(energies, dim=1).detach()
        scales = torch.maximum(energy, floor[:, None]).detach()
        activity = (energy / scales).clamp(0.0, 1.0).detach()
        normalized_energy = (energy / scales).detach()
        return scales, activity, normalized_energy

    def _window(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(
            self.config.n_fft, device=waveform.device, dtype=torch.float32
        )

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform.float(),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(waveform),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )

    @staticmethod
    def _check_waveform(name: str, value: torch.Tensor, shape: torch.Size) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape [batch, samples]")

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
        mode: str,
        interpolant_residual: torch.Tensor | None,
        flow_time: torch.Tensor | None,
    ) -> None:
        if mixture.ndim != 2:
            raise ValueError("mixture must have shape [batch, samples]")
        for name, value in (
            ("dna", dna),
            ("other_dna", other_dna),
            ("unassigned_residual", unassigned_residual),
        ):
            self._check_waveform(name, value, mixture.shape)
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
        if mode not in {"deterministic", "velocity"}:
            raise ValueError("mode must be 'deterministic' or 'velocity'")
        if mode == "velocity" and (interpolant_residual is None or flow_time is None):
            raise ValueError("velocity mode requires interpolant_residual and flow_time")
        if interpolant_residual is not None:
            self._check_waveform("interpolant_residual", interpolant_residual, mixture.shape)
        if flow_time is not None and flow_time.shape != (batch,):
            raise ValueError("flow_time must have shape [batch]")

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
        *,
        mode: str = "deterministic",
        interpolant_residual: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
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
            mode,
            interpolant_residual,
            flow_time,
        )
        batch, samples = mixture.shape
        positions = torch.arange(samples, device=mixture.device)[None]
        sample_support = positions < valid_samples.long()[:, None]
        sources = [mixture, dna, other_dna, unassigned_residual]
        if interpolant_residual is None:
            interpolant_residual = torch.zeros_like(mixture)
        sources.append(interpolant_residual)
        masked_sources = [source.float() * sample_support for source in sources]

        denominator = valid_samples.float().clamp_min(1.0)[:, None]
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
        scale = valid_samples.float().clamp_min(1.0)[:, None]
        onset_distance = (frame_samples - onset[:, None]).float() / scale
        offset_distance = (offset[:, None] - frame_samples).float() / scale
        time = (
            torch.zeros(batch, device=mixture.device)
            if flow_time is None
            else flow_time.float()
        )
        global_context = torch.stack(
            (
                frame_support.float(),
                onset_distance.clamp(-1.0, 1.0),
                offset_distance.clamp(-1.0, 1.0),
                time[:, None].expand(-1, frames),
                torch.full(
                    (batch, frames),
                    float(mode == "velocity"),
                    device=mixture.device,
                ),
            ),
            dim=1,
        )
        band_scales, band_activity, normalized_band_energy = (
            self._band_condition_statistics(spectra, frame_support)
        )
        log_band_scales = torch.log10(
            band_scales.clamp_min(torch.finfo(band_scales.dtype).eps)
        ).clamp(-4.0, 4.0) / 4.0

        pitch_bias = self.pitch_projection(self.pitch_embedding(pitch.long()))
        band_hidden = []
        for band, ((start, end), projection) in enumerate(
            zip(self.band_supports, self.input_projections)
        ):
            width = end - start
            band_scale = band_scales[:, band, None, None]
            local_features = [
                _complex_features(spectrum[:, start:end] / band_scale)
                for spectrum in spectra
            ]
            local_context = global_context[:, :, None].expand(-1, -1, width, -1)
            local_statistics = torch.stack(
                (log_band_scales[:, band], band_activity[:, band]), dim=1
            )
            local_statistics = local_statistics[:, :, None, None].expand(
                -1, -1, width, frames
            )
            features = torch.cat(
                local_features + [local_context, local_statistics], dim=1
            )
            hidden = projection(features).squeeze(2)
            hidden = hidden + pitch_bias[:, :, None]
            hidden = hidden + self.band_embeddings[band][None, :, None]
            band_hidden.append(hidden)
        hidden = torch.stack(band_hidden, dim=1)
        support = frame_support[:, None, None].to(hidden.dtype)
        hidden = hidden * support

        folded = hidden.flatten(0, 1)
        folded_support = support.expand(-1, self.config.bands, -1, -1).flatten(0, 1)
        velocity_flag = torch.full_like(time, float(mode == "velocity"))
        flow_condition = self.flow_condition(torch.stack((time, velocity_flag), dim=1))
        folded_flow_condition = flow_condition[:, None].expand(
            -1, self.config.bands, -1
        ).flatten(0, 1)
        for block in self.temporal:
            folded = block(folded, folded_support, folded_flow_condition)
        hidden = folded.unflatten(0, (batch, self.config.bands))

        stride = self.config.bottleneck_stride
        pooled = F.avg_pool1d(
            hidden.flatten(0, 1), stride, stride, ceil_mode=True
        ).unflatten(0, (batch, self.config.bands))
        pooled_support = F.max_pool1d(
            frame_support.float()[:, None], stride, stride, ceil_mode=True
        )
        mixed = self.cross_band_mixer(pooled, pooled_support)
        mixed = mixed.repeat_interleave(stride, dim=-1)[..., :frames]
        hidden = (hidden + mixed) * support

        available = spectra[0].abs() + spectra[1].abs() + spectra[3].abs()
        per_band = spectra[0].new_zeros(
            (batch, self.config.bands, frequency_bins, frames)
        )
        for band, ((start, end), projection) in enumerate(
            zip(self.band_supports, self.output_projections)
        ):
            width = end - start
            raw = projection(hidden[:, band]).unflatten(1, (2, width))
            unit = torch.complex(torch.tanh(raw[:, 0]).float(), torch.tanh(raw[:, 1]).float())
            local = (
                float(self.config.maximum_correction_scale)
                * unit
                * available[:, start:end].clamp_min(1e-6)
            )
            local = local * self.band_weights[band, start:end][None, :, None]
            per_band[:, band, start:end] = local
        correction_spectrum = per_band.sum(dim=1)
        correction = torch.istft(
            correction_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(mixture),
            center=True,
            length=samples,
        ) * level
        correction = correction * sample_support
        estimate = torch.where(sample_support, dna.float() + correction, dna.float())
        output = estimate if mode == "deterministic" else correction
        return output, {
            "estimate": estimate,
            "velocity": correction,
            "correction": correction,
            "correction_spectrum": correction_spectrum,
            "per_band_correction_spectra": per_band,
            "band_weights": self.band_weights,
            "band_scales": band_scales,
            "band_activity": band_activity,
            "normalized_band_condition_energy": normalized_band_energy,
            "frame_support": frame_support,
            "level": level,
        }

    def per_band_corrections(self, *args: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Run the model and return [batch, band, frequency, frame] corrections."""

        _, diagnostics = self.forward(*args, **kwargs)
        return diagnostics["per_band_correction_spectra"]


class DNAResidualBandDualRegionExpert(nn.Module):
    """Blend independent active and tail experts around the target-note offset.

    Each branch is a complete :class:`DNAResidualBandExpert` with independent
    parameters and a zero-initialized output path. A fixed raised-cosine gate
    blends their per-band complex spectra before a single inverse STFT. Thus
    the wrapper retains exact DNA identity at initialization and gives each
    branch exclusive capacity away from the configurable transition region.
    """

    def __init__(
        self, config: DNAResidualBandDualRegionExpertConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or DNAResidualBandDualRegionExpertConfig()
        if self.config.crossfade_frames < 1:
            raise ValueError("crossfade_frames must be positive")
        branch_config = DNAResidualBandExpertConfig(
            **{
                item.name: getattr(self.config, item.name)
                for item in fields(DNAResidualBandExpertConfig)
            }
        )
        self.active = DNAResidualBandExpert(branch_config)
        self.tail = DNAResidualBandExpert(branch_config)

    @property
    def band_weights(self) -> torch.Tensor:
        return self.active.band_weights

    def overlap_reconstruction_weights(self) -> torch.Tensor:
        return self.active.overlap_reconstruction_weights()

    def temporal_receptive_field_frames(self) -> int:
        return self.active.temporal_receptive_field_frames()

    def temporal_receptive_field_seconds(self) -> float:
        return self.active.temporal_receptive_field_seconds()

    def _window(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.active._window(waveform)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.active._stft(waveform)

    def _region_gates(
        self,
        frame_support: torch.Tensor,
        offset_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return complementary [batch, frames] raised-cosine gates."""

        if frame_support.ndim != 2:
            raise ValueError("frame_support must have shape [batch, frames]")
        if offset_samples.shape != (frame_support.shape[0],):
            raise ValueError("offset_samples must have shape [batch]")
        frame_positions = torch.arange(
            frame_support.shape[-1],
            device=frame_support.device,
            dtype=torch.float32,
        )[None]
        center = offset_samples.float()[:, None] / float(
            self.config.hop_length
        )
        width = float(self.config.crossfade_frames)
        progress = ((frame_positions - center) / width + 0.5).clamp(0.0, 1.0)
        tail = 0.5 - 0.5 * torch.cos(torch.pi * progress)
        support = frame_support.to(tail.dtype)
        tail = tail * support
        active = (1.0 - tail) * support
        return active, tail

    def _gate_boundary_samples(
        self,
        onset_samples: torch.Tensor,
        offset_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Return the sample boundary around which branch spectra crossfade."""

        del onset_samples
        return offset_samples

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
        *,
        mode: str = "deterministic",
        interpolant_residual: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_kwargs = {
            "mode": mode,
            "interpolant_residual": interpolant_residual,
            "flow_time": flow_time,
        }
        _, active_info = self.active(
            mixture,
            dna,
            other_dna,
            unassigned_residual,
            pitch,
            valid_samples,
            onset,
            offset,
            **branch_kwargs,
        )
        _, tail_info = self.tail(
            mixture,
            dna,
            other_dna,
            unassigned_residual,
            pitch,
            valid_samples,
            onset,
            offset,
            **branch_kwargs,
        )
        if not torch.equal(active_info["frame_support"], tail_info["frame_support"]):
            raise RuntimeError("Dual-region branches produced different frame support")
        if not torch.equal(active_info["level"], tail_info["level"]):
            raise RuntimeError("Dual-region branches produced different input levels")

        gate_boundary = self._gate_boundary_samples(onset, offset)
        active_gate, tail_gate = self._region_gates(
            active_info["frame_support"], gate_boundary
        )
        active_raw = active_info["per_band_correction_spectra"]
        tail_raw = tail_info["per_band_correction_spectra"]
        active_gated = active_raw * active_gate[:, None, None, :]
        tail_gated = tail_raw * tail_gate[:, None, None, :]
        per_band = active_gated + tail_gated
        correction_spectrum = per_band.sum(dim=1)
        correction = torch.istft(
            correction_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(mixture),
            center=True,
            length=mixture.shape[-1],
        ) * active_info["level"]
        sample_support = (
            torch.arange(mixture.shape[-1], device=mixture.device)[None]
            < valid_samples.long()[:, None]
        )
        correction = correction * sample_support
        estimate = torch.where(sample_support, dna.float() + correction, dna.float())
        output = estimate if mode == "deterministic" else correction
        return output, {
            "estimate": estimate,
            "velocity": correction,
            "correction": correction,
            "correction_spectrum": correction_spectrum,
            "per_band_correction_spectra": per_band,
            "band_weights": active_info["band_weights"],
            "band_scales": active_info["band_scales"],
            "band_activity": active_info["band_activity"],
            "normalized_band_condition_energy": active_info[
                "normalized_band_condition_energy"
            ],
            "frame_support": active_info["frame_support"],
            "level": active_info["level"],
            "active_frame_gate": active_gate,
            "tail_frame_gate": tail_gate,
            "active_per_band_correction_spectra": active_raw,
            "tail_per_band_correction_spectra": tail_raw,
            "active_gated_per_band_correction_spectra": active_gated,
            "tail_gated_per_band_correction_spectra": tail_gated,
            "active_band_scales": active_info["band_scales"],
            "tail_band_scales": tail_info["band_scales"],
            "region_gate_boundary_samples": gate_boundary,
        }

    def per_band_corrections(
        self, *args: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Return the smoothly gated sum of both branch spectra."""

        _, diagnostics = self.forward(*args, **kwargs)
        return diagnostics["per_band_correction_spectra"]


class DNAResidualBandAttackContinuationExpert(DNAResidualBandDualRegionExpert):
    """Use independent experts before and after a bounded attack region.

    The crossfade is centered at the earlier of the target-note offset and
    ``onset + attack_samples``. Short notes therefore never route their
    post-offset frames back through the attack expert, while sustained notes
    hand off after a fixed attack duration.
    """

    def __init__(
        self,
        config: DNAResidualBandAttackContinuationExpertConfig | None = None,
    ) -> None:
        selected = config or DNAResidualBandAttackContinuationExpertConfig()
        if selected.attack_samples < 1:
            raise ValueError("attack_samples must be positive")
        super().__init__(selected)
        self.config = selected

    @property
    def attack(self) -> DNAResidualBandExpert:
        return self.active

    @property
    def continuation(self) -> DNAResidualBandExpert:
        return self.tail

    def _gate_boundary_samples(
        self,
        onset_samples: torch.Tensor,
        offset_samples: torch.Tensor,
    ) -> torch.Tensor:
        return torch.minimum(
            offset_samples,
            onset_samples + int(self.config.attack_samples),
        )

    def forward(
        self, *args: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output, diagnostics = super().forward(*args, **kwargs)
        diagnostics.update(
            {
                "attack_frame_gate": diagnostics["active_frame_gate"],
                "continuation_frame_gate": diagnostics["tail_frame_gate"],
                "attack_per_band_correction_spectra": diagnostics[
                    "active_per_band_correction_spectra"
                ],
                "continuation_per_band_correction_spectra": diagnostics[
                    "tail_per_band_correction_spectra"
                ],
                "attack_gated_per_band_correction_spectra": diagnostics[
                    "active_gated_per_band_correction_spectra"
                ],
                "continuation_gated_per_band_correction_spectra": diagnostics[
                    "tail_gated_per_band_correction_spectra"
                ],
            }
        )
        return output, diagnostics
