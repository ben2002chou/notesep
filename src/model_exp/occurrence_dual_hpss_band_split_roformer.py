"""Occurrence-conditioned dual-HPSS Band-Split RoFormer.

This is a deterministic backbone ablation: each HPSS component is encoded as
disjoint complex-frequency bands, processed by alternating time- and
band-axis RoPE Transformer blocks, and decoded as a complex ratio mask.  The
model keeps the established pitch query, RT2-style target roll, HPSS targets,
and optional input-only below-f0 gate so the backbone is the primary changed
variable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .pitch_relative_harmonic_gate import (
    GATE_MODES,
    gate_applies_to_target_and_prediction,
    pitch_relative_harmonic_gate,
)


_CONTRACT_KIND = "occurrence_dual_hpss_band_split_roformer_v2"


def _paper_16khz_band_widths() -> tuple[int, ...]:
    """Adapt the paper's 2/4/12/24-bin schedule to 16 kHz / 1024 STFT."""

    widths = []
    for begin, end, width in (
        (0, 64, 2),
        (64, 128, 4),
        (128, 256, 12),
        (256, 512, 24),
    ):
        cursor = begin
        while cursor < end:
            current = min(width, end - cursor)
            widths.append(current)
            cursor += current
    if sum(widths) != 512:
        raise AssertionError("paper-adapted bands do not cover the spectrum")
    return tuple(widths)


@dataclass(frozen=True)
class OccurrenceBandSplitRoFormerConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    processed_frequency_bins: int = 512
    num_bands: int = 32
    model_dim: int = 256
    depth: int = 8
    attention_heads: int = 8
    mlp_expansion: int = 4
    dropout: float = 0.0
    mask_limit: float = 2.0
    pitch_embed_dim: int = 128
    band_scheme: str = "equal"

    def validate(self) -> None:
        positive = (
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.processed_frequency_bins,
            self.num_bands,
            self.model_dim,
            self.depth,
            self.attention_heads,
            self.mlp_expansion,
            self.pitch_embed_dim,
        )
        if min(map(int, positive)) < 1:
            raise ValueError("RoFormer dimensions must be positive")
        if self.n_fft < 16 or self.hop_length > self.n_fft // 2:
            raise ValueError("Invalid RoFormer STFT geometry")
        if self.processed_frequency_bins > self.n_fft // 2 + 1:
            raise ValueError("processed_frequency_bins exceeds the STFT")
        if self.band_scheme == "equal":
            if self.processed_frequency_bins % self.num_bands:
                raise ValueError("processed_frequency_bins must divide evenly into bands")
        elif self.band_scheme == "paper_2_4_12_24_bins_16khz":
            if (
                self.sample_rate != 16_000
                or self.n_fft != 1024
                or self.processed_frequency_bins != 512
                or self.num_bands != len(_paper_16khz_band_widths())
            ):
                raise ValueError("paper-adapted bands require fixed 16 kHz geometry")
        else:
            raise ValueError(f"Unknown RoFormer band scheme: {self.band_scheme}")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        head_dim = self.model_dim // self.attention_heads
        if head_dim % 2:
            raise ValueError("RoPE attention head dimensions must be even")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mask_limit <= 0.0:
            raise ValueError("mask_limit must be positive")


def occurrence_band_split_roformer_config(
    preset: str,
) -> OccurrenceBandSplitRoFormerConfig:
    presets = {
        "tiny": OccurrenceBandSplitRoFormerConfig(
            processed_frequency_bins=96,
            num_bands=8,
            model_dim=32,
            depth=1,
            attention_heads=4,
            pitch_embed_dim=16,
            n_fft=192,
            hop_length=48,
        ),
        "canary": OccurrenceBandSplitRoFormerConfig(
            num_bands=len(_paper_16khz_band_widths()),
            model_dim=64,
            depth=2,
            attention_heads=4,
            pitch_embed_dim=32,
            dropout=0.1,
            band_scheme="paper_2_4_12_24_bins_16khz",
        ),
        "medium": OccurrenceBandSplitRoFormerConfig(
            num_bands=len(_paper_16khz_band_widths()),
            model_dim=128,
            depth=4,
            attention_heads=8,
            pitch_embed_dim=64,
            dropout=0.1,
            band_scheme="paper_2_4_12_24_bins_16khz",
        ),
        "30m": OccurrenceBandSplitRoFormerConfig(
            num_bands=len(_paper_16khz_band_widths()),
            dropout=0.1,
            band_scheme="paper_2_4_12_24_bins_16khz",
        ),
    }
    try:
        config = presets[str(preset)]
    except KeyError as error:
        raise ValueError(f"Unknown Band-Split RoFormer preset: {preset}") from error
    config.validate()
    return config


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rope(values: torch.Tensor) -> torch.Tensor:
    # values: [batch, heads, sequence, head_dim]
    head_dim = values.shape[-1]
    positions = torch.arange(values.shape[-2], device=values.device, dtype=torch.float32)
    inverse = torch.pow(
        10_000.0,
        -torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32)
        / float(head_dim),
    )
    angles = torch.outer(positions, inverse)
    angles = torch.repeat_interleave(angles, 2, dim=-1)[None, None]
    cosine = angles.cos().to(values.dtype)
    sine = angles.sin().to(values.dtype)
    return values * cosine + _rotate_half(values) * sine


class _RMSNorm(nn.Module):
    """Version-independent RMSNorm matching the BS-RoFormer normalization."""

    def __init__(self, dimension: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(int(dimension)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        ).to(values.dtype)
        return values * scale * self.weight.to(values.dtype)


class _RoPESelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dim = self.dimension // self.heads
        self.qkv = nn.Linear(self.dimension, 3 * self.dimension, bias=False)
        self.output = nn.Linear(self.dimension, self.dimension, bias=False)
        self.dropout = float(dropout)

    def forward(
        self, values: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, sequence, _ = values.shape
        qkv = self.qkv(values).reshape(
            batch, sequence, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        query = _apply_rope(query)
        key = _apply_rope(key)
        attention_mask = None
        if valid is not None:
            if valid.shape != (batch, sequence):
                raise ValueError("attention validity mask has the wrong shape")
            attention_mask = valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = self.output(
            attended.transpose(1, 2).reshape(batch, sequence, self.dimension)
        )
        if valid is not None:
            result = result * valid[..., None].to(result.dtype)
        return result


class _RoFormerBlock(nn.Module):
    def __init__(self, config: OccurrenceBandSplitRoFormerConfig) -> None:
        super().__init__()
        self.attention_norm = _RMSNorm(config.model_dim)
        self.attention = _RoPESelfAttention(
            config.model_dim, config.attention_heads, config.dropout
        )
        self.mlp_norm = _RMSNorm(config.model_dim)
        hidden = config.model_dim * config.mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, 2 * hidden, bias=False),
            nn.GLU(dim=-1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.model_dim, bias=False),
        )

    def forward(
        self, values: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values), valid)
        values = values + self.mlp(self.mlp_norm(values))
        if valid is not None:
            values = values * valid[..., None].to(values.dtype)
        return values


class _DualAxisRoFormerLayer(nn.Module):
    def __init__(self, config: OccurrenceBandSplitRoFormerConfig) -> None:
        super().__init__()
        self.time = _RoFormerBlock(config)
        self.band = _RoFormerBlock(config)

    def forward(self, values: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        # [batch, frames, bands, dimension]
        batch, frames, bands, dimension = values.shape
        time_values = values.permute(0, 2, 1, 3).reshape(
            batch * bands, frames, dimension
        )
        time_valid = frame_valid[:, None].expand(-1, bands, -1).reshape(
            batch * bands, frames
        )
        time_values = self.time(time_values, time_valid)
        values = time_values.reshape(batch, bands, frames, dimension).permute(0, 2, 1, 3)
        band_values = values.reshape(batch * frames, bands, dimension)
        band_values = self.band(band_values)
        values = band_values.reshape(batch, frames, bands, dimension)
        return values * frame_valid[:, :, None, None].to(values.dtype)


class OccurrenceBandSplitRoFormerBranch(nn.Module):
    def __init__(self, config: OccurrenceBandSplitRoFormerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.band_widths = (
            _paper_16khz_band_widths()
            if config.band_scheme == "paper_2_4_12_24_bins_16khz"
            else (config.processed_frequency_bins // config.num_bands,) * config.num_bands
        )
        if len(self.band_widths) != config.num_bands:
            raise ValueError("band-width count does not match num_bands")
        boundaries = [0]
        for width in self.band_widths:
            boundaries.append(boundaries[-1] + width)
        if boundaries[-1] != config.processed_frequency_bins:
            raise ValueError("band widths do not cover processed frequencies")
        self.band_slices = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
        self.band_projections = nn.ModuleList(
            [
                nn.Sequential(
                    _RMSNorm(2 * width),
                    nn.Linear(2 * width, config.model_dim, bias=True),
                )
                for width in self.band_widths
            ]
        )
        self.pitch_embedding = nn.Embedding(128, config.pitch_embed_dim)
        self.pitch_projection = nn.Linear(
            config.pitch_embed_dim, config.model_dim, bias=False
        )
        self.roll_state_embedding = nn.Embedding(3, config.model_dim)
        self.frame_context_projection = nn.Sequential(
            nn.Linear(2, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.relative_frequency_projection = nn.Sequential(
            nn.Linear(3, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.layers = nn.ModuleList(
            [_DualAxisRoFormerLayer(config) for _ in range(config.depth)]
        )
        self.output_norm = _RMSNorm(config.model_dim)
        self.mask_heads = nn.ModuleList(
            [
                nn.Sequential(
                    _RMSNorm(config.model_dim),
                    nn.Linear(config.model_dim, 4 * config.model_dim),
                    nn.Tanh(),
                    nn.Linear(config.model_dim * 4, 4 * width),
                    nn.GLU(dim=-1),
                )
                for width in self.band_widths
            ]
        )
        self.ownership_heads = nn.ModuleList(
            [
                nn.Sequential(
                    _RMSNorm(config.model_dim),
                    nn.Linear(config.model_dim, width),
                )
                for width in self.band_widths
            ]
        )
        for head in self.mask_heads:
            final = head[3]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        for head in self.ownership_heads:
            final = head[1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, -2.0)

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

    def _roll_states(
        self,
        onset: torch.Tensor,
        offset: torch.Tensor,
        valid_samples: torch.Tensor,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_samples = (
            torch.arange(frames, device=onset.device)[None] * self.config.hop_length
        )
        frame_valid = frame_samples < valid_samples.long()[:, None]
        roll_frame = torch.div(
            frame_samples * 25, self.config.sample_rate, rounding_mode="floor"
        )
        onset_frame = torch.div(
            onset.long() * 25, self.config.sample_rate, rounding_mode="floor"
        )
        offset_frame = torch.div(
            offset.long() * 25 + self.config.sample_rate - 1,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        held = (
            frame_valid
            & (offset > onset)[:, None]
            & (roll_frame >= onset_frame[:, None])
            & (roll_frame < offset_frame[:, None])
        )
        state = held.long()
        state = torch.where(
            held & (roll_frame == onset_frame[:, None]),
            torch.full_like(state, 2),
            state,
        )
        return state, frame_valid

    def _relative_frequency_features(self, pitch: torch.Tensor) -> torch.Tensor:
        band_center_bin = torch.tensor(
            [(begin + end - 1) / 2.0 for begin, end in self.band_slices],
            device=pitch.device,
            dtype=torch.float32,
        )
        frequency = band_center_bin * self.config.sample_rate / self.config.n_fft
        fundamental = 440.0 * torch.pow(2.0, (pitch.float() - 69.0) / 12.0)
        ratio = frequency[None] / fundamental[:, None]
        positive = ratio.clamp_min(1e-5)
        relative_octave = torch.log2(positive).clamp(-4.0, 6.0) / 6.0
        fundamental_proximity = torch.exp(
            -0.5 * (1200.0 * torch.log2(positive) / 150.0).square()
        )
        harmonic = ratio.round().clamp_min(1.0)
        harmonic_proximity = torch.exp(
            -0.5 * (1200.0 * torch.log2(positive / harmonic) / 100.0).square()
        ) * (ratio >= 0.5)
        return torch.stack(
            (relative_octave, fundamental_proximity, harmonic_proximity), dim=-1
        )

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "occurrence_band_split_roformer_branch_v1",
            "config": asdict(self.config),
            "band_partition": {
                "type": self.config.band_scheme,
                "bands": self.config.num_bands,
                "band_width_bins": list(self.band_widths),
                "frequency_regions_hz": [0, 1000, 2000, 4000, 8000],
                "paper_region_width_bins": [2, 4, 12, 24],
            },
            "separator": {
                "layers": "alternating_time_and_band_rope_transformers",
                "normalization": "rms_norm",
                "attention": "scaled_dot_product_flash_eligible",
                "target_head": "per_band_rmsnorm_linear_tanh_linear_glu_complex_ratio_mask",
                "mask_bounding": "unbounded_glu_output",
                "ownership_head": "sigmoid_target_background_magnitude_ratio",
            },
            "conditioning": {
                "roll": "rt2_25hz_off_held_onset_target_only",
                "explicit_lifecycle_labels": False,
                "query": "midi_pitch_embedding",
                "relative_frequency": [
                    "log2_band_center_over_f0",
                    "fundamental_proximity",
                    "nearest_harmonic_proximity",
                ],
            },
            "parameter_count": self.parameter_count(),
        }

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
        del dna, other_dna, unassigned_residual
        if mixture.ndim != 2:
            raise ValueError("mixture must have shape [batch, samples]")
        batch, samples = mixture.shape
        for name, value in (
            ("pitch", pitch),
            ("valid_samples", valid_samples),
            ("onset", onset),
            ("offset", offset),
        ):
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        positions = torch.arange(samples, device=mixture.device)[None]
        sample_support = positions < valid_samples.long()[:, None]
        clean = mixture.float() * sample_support
        denominator = valid_samples.float().clamp_min(1)[:, None]
        level = (clean.square().sum(dim=-1, keepdim=True) / denominator).sqrt().clamp_min(1e-4)
        spectrum = self._stft(clean / level)
        processed = spectrum[..., : self.config.processed_frequency_bins, :]
        frames = processed.shape[-1]
        state, frame_valid = self._roll_states(onset, offset, valid_samples, frames)

        band_tokens = []
        for (begin, end), projection in zip(
            self.band_slices, self.band_projections, strict=True
        ):
            band = processed[:, begin:end].transpose(1, 2)
            features = torch.cat((band.real, band.imag), dim=-1)
            band_tokens.append(projection(features))
        hidden = torch.stack(band_tokens, dim=2)
        valid_scale = valid_samples.float().clamp_min(1)[:, None]
        frame_samples = (
            torch.arange(frames, device=mixture.device)[None] * self.config.hop_length
        )
        frame_context = torch.stack(
            (
                ((frame_samples - onset[:, None]) / valid_scale).clamp(-1, 1),
                ((offset[:, None] - frame_samples) / valid_scale).clamp(-1, 1),
            ),
            dim=-1,
        )
        temporal_condition = (
            self.pitch_projection(self.pitch_embedding(pitch.long()))[:, None]
            + self.roll_state_embedding(state)
            + self.frame_context_projection(frame_context)
        )
        frequency_condition = self.relative_frequency_projection(
            self._relative_frequency_features(pitch)
        )
        hidden = (
            hidden
            + temporal_condition[:, :, None]
            + frequency_condition[:, None]
        )
        hidden = hidden * frame_valid[:, :, None, None].to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, frame_valid)
        hidden = self.output_norm(hidden)

        masks = []
        ownership = []
        for index, (mask_head, ownership_head) in enumerate(
            zip(self.mask_heads, self.ownership_heads)
        ):
            values = hidden[:, :, index]
            raw = mask_head(values).transpose(1, 2)
            real, imaginary = raw.chunk(2, dim=1)
            masks.append(
                torch.complex(real.float(), imaginary.float())
            )
            ownership.append(ownership_head(values).transpose(1, 2).float())
        complex_mask = torch.cat(masks, dim=1)
        ownership_logits = torch.cat(ownership, dim=1)
        complex_mask = complex_mask * frame_valid[:, None]
        ownership_logits = ownership_logits * frame_valid[:, None]
        target_processed = processed * complex_mask
        target_spectrum = spectrum.new_zeros(spectrum.shape)
        target_spectrum[..., : self.config.processed_frequency_bins, :] = target_processed
        target = torch.istft(
            target_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window(mixture.float()),
            center=True,
            length=samples,
        ) * level
        target = target * sample_support
        background_processed = processed - target_processed
        return target, {
            "target": target,
            "estimate": target,
            "target_processed_spectrum": target_processed,
            "background_head_processed_spectrum": background_processed,
            "complex_mask": complex_mask,
            "ownership_logits": ownership_logits,
            "ownership_mask": torch.sigmoid(ownership_logits) * frame_valid[:, None],
            "frame_support": frame_valid,
            "level": level,
        }


class OccurrenceDualHPSSBandSplitRoFormer(nn.Module):
    def __init__(
        self,
        preset: str = "30m",
        harmonic_gate_mode: str = "below_f0_input_only",
        *,
        config: OccurrenceBandSplitRoFormerConfig | None = None,
    ) -> None:
        super().__init__()
        if harmonic_gate_mode not in GATE_MODES:
            raise ValueError(f"Unknown harmonic gate mode: {harmonic_gate_mode}")
        if gate_applies_to_target_and_prediction(harmonic_gate_mode):
            raise ValueError("The first RoFormer ablation supports input-only gating")
        self.preset = str(preset)
        self.harmonic_gate_mode = str(harmonic_gate_mode)
        self.config = config or occurrence_band_split_roformer_config(preset)
        self.harmonic_branch = OccurrenceBandSplitRoFormerBranch(self.config)
        self.percussive_branch = OccurrenceBandSplitRoFormerBranch(self.config)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _gate_harmonic_mixture(
        self,
        harmonic_mixture: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        input_gate_strength: torch.Tensor | None,
    ) -> torch.Tensor:
        gated = pitch_relative_harmonic_gate(
            harmonic_mixture,
            pitch,
            valid_samples,
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            mode=self.harmonic_gate_mode,
        )
        if input_gate_strength is None:
            return gated
        if input_gate_strength.shape != pitch.shape:
            raise ValueError("input_gate_strength must have shape [batch]")
        strength = input_gate_strength.to(harmonic_mixture.dtype)
        if not bool(torch.all((strength >= 0) & (strength <= 1))):
            raise ValueError("input_gate_strength must lie in [0, 1]")
        return torch.lerp(harmonic_mixture, gated, strength[:, None])

    def contract(self) -> dict[str, Any]:
        return {
            "kind": _CONTRACT_KIND,
            "contract_version": 2,
            "preset": self.preset,
            "architecture": {
                "layout": "independent_hpss_harmonic_and_percussive_band_split_roformer",
                "merge": "waveform_sum",
                "harmonic_branch": self.harmonic_branch.contract(),
                "percussive_branch": self.percussive_branch.contract(),
                "percussive_parameterization": "ownership",
            },
            "harmonic_gate": {
                "mode": self.harmonic_gate_mode,
                "application": ["mixture_input"],
            },
            "conditioning": {
                "type": "rt2_25hz_target_only_roll",
                "states": ["off", "held", "onset"],
                "explicit_lifecycle_labels": False,
            },
            "inputs": {
                "harmonic_branch": "HPSS harmonic mixture",
                "percussive_branch": "ordinary margin-1 HPSS percussive mixture",
                "query": ["pitch", "onset_sample", "offset_sample"],
            },
            "outputs": {
                "harmonic": "HPSS harmonic target complex-mask estimate",
                "percussive": "HPSS percussive target complex-mask estimate",
                "primary": "harmonic plus percussive estimate",
            },
            "dna_conditioning": False,
            "parameter_count": self.parameter_count(),
            "architecture_reference": {
                "paper": "arXiv:2309.02612v2",
                "preserved": [
                    "non_overlapping_uneven_band_split",
                    "per_band_rmsnorm_projection",
                    "alternating_time_and_band_transformers",
                    "rope_on_query_and_key",
                    "rmsnorm_transformer_blocks",
                    "per_band_tanh_glu_complex_mask_heads",
                ],
                "occurrence_adaptations": [
                    "16khz_frequency_geometry",
                    "target_pitch_and_rt2_roll_conditioning",
                    "dual_hpss_branches",
                    "input_only_below_f0_gate",
                    "matched_existing_occurrence_losses",
                ],
            },
        }

    def forward(
        self,
        harmonic_mixture: torch.Tensor,
        percussive_mixture: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        *,
        input_gate_strength: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if harmonic_mixture.shape != percussive_mixture.shape:
            raise ValueError("Harmonic and percussive mixtures must have equal shape")
        zeros = torch.zeros_like(harmonic_mixture)
        harmonic_input = self._gate_harmonic_mixture(
            harmonic_mixture, pitch, valid_samples, input_gate_strength
        )
        harmonic, harmonic_diagnostics = self.harmonic_branch(
            harmonic_input, zeros, zeros, zeros,
            pitch, valid_samples, onset, offset,
        )
        percussive, percussive_diagnostics = self.percussive_branch(
            percussive_mixture, zeros, zeros, zeros,
            pitch, valid_samples, onset, offset,
        )
        return harmonic + percussive, {
            "harmonic": harmonic,
            "percussive": percussive,
            "harmonic_input": harmonic_input,
            "harmonic_branch": harmonic_diagnostics,
            "percussive_branch": percussive_diagnostics,
            "conditioned_offset_samples": offset,
            "conditioning_hold_extension_samples": torch.zeros_like(offset),
            "input_gate_strength": (
                torch.ones_like(pitch, dtype=harmonic_mixture.dtype)
                if input_gate_strength is None
                else input_gate_strength.to(harmonic_mixture.dtype)
            ),
        }


def occurrence_dual_hpss_band_split_roformer_from_contract(
    raw: object,
) -> OccurrenceDualHPSSBandSplitRoFormer:
    if not isinstance(raw, dict) or raw.get("kind") != _CONTRACT_KIND:
        raise ValueError("Checkpoint lacks a dual HPSS Band-Split RoFormer contract")
    branch = raw.get("architecture", {}).get("harmonic_branch", {})
    config_raw = branch.get("config")
    if not isinstance(config_raw, dict):
        raise ValueError("RoFormer contract lacks its branch config")
    config = OccurrenceBandSplitRoFormerConfig(**config_raw)
    model = OccurrenceDualHPSSBandSplitRoFormer(
        str(raw.get("preset", "30m")),
        str(raw.get("harmonic_gate", {}).get("mode", "below_f0_input_only")),
        config=config,
    )
    if model.contract() != raw:
        raise ValueError("Dual HPSS Band-Split RoFormer contract changed")
    return model


__all__ = [
    "OccurrenceBandSplitRoFormerBranch",
    "OccurrenceBandSplitRoFormerConfig",
    "OccurrenceDualHPSSBandSplitRoFormer",
    "occurrence_band_split_roformer_config",
    "occurrence_dual_hpss_band_split_roformer_from_contract",
]
