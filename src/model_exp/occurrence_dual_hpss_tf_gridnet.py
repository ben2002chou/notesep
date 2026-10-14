"""Note-conditioned dual-HPSS separator with a frequency-pooled TF-GridNet trunk.

The recurrent spectral and temporal paths follow TF-GridNet's patch-BLSTM
contract. Frame attention pools frequency before attention because NoteSep's
513 bins at 16 kHz make the paper's full-frequency value vectors expensive.
This is a named adaptation, not a claim of exact paper reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pitch_relative_harmonic_gate import pitch_relative_harmonic_gate


_KIND = "occurrence_dual_hpss_tf_gridnet_v1"


@dataclass(frozen=True)
class OccurrenceTFGridNetConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    processed_frequency_bins: int = 513
    model_channels: int = 64
    lstm_hidden: int = 128
    blocks: int = 2
    patch_size: int = 4
    patch_stride: int = 2
    attention_heads: int = 4
    attention_key_channels: int = 4
    attention_frequency_bins: int = 32
    activation_checkpointing: bool = True

    def validate(self) -> None:
        values = (
            self.sample_rate, self.n_fft, self.hop_length,
            self.processed_frequency_bins, self.model_channels,
            self.lstm_hidden, self.blocks, self.patch_size,
            self.patch_stride, self.attention_heads,
            self.attention_key_channels, self.attention_frequency_bins,
        )
        if min(values) < 1:
            raise ValueError("all TF-GridNet dimensions must be positive")
        if self.n_fft < 32 or self.hop_length > self.n_fft // 2:
            raise ValueError("invalid STFT geometry")
        if self.processed_frequency_bins > self.n_fft // 2 + 1:
            raise ValueError("processed frequencies exceed the STFT")
        if self.patch_stride > self.patch_size:
            raise ValueError("patch stride cannot exceed patch size")
        if self.model_channels % self.attention_heads:
            raise ValueError("model channels must divide attention heads")
        if self.attention_frequency_bins > self.processed_frequency_bins:
            raise ValueError("attention pool exceeds processed frequencies")


def occurrence_tf_gridnet_config(preset: str) -> OccurrenceTFGridNetConfig:
    configs = {
        "tiny": OccurrenceTFGridNetConfig(
            n_fft=192, hop_length=48, processed_frequency_bins=97,
            model_channels=16, lstm_hidden=32, blocks=1, patch_size=3,
            patch_stride=2, attention_frequency_bins=16,
        ),
        "canary": OccurrenceTFGridNetConfig(),
        "scale93": OccurrenceTFGridNetConfig(
            model_channels=160, lstm_hidden=320, blocks=8,
            attention_heads=8, attention_frequency_bins=64,
        ),
    }
    try:
        config = configs[preset]
    except KeyError as error:
        raise ValueError(f"unknown TF-GridNet preset: {preset}") from error
    config.validate()
    return config


class _PatchBLSTM(nn.Module):
    def __init__(self, channels: int, hidden: int, patch: int, stride: int, *, axis: str):
        super().__init__()
        if axis not in {"frequency", "time"}:
            raise ValueError("patch BLSTM axis must be frequency or time")
        self.axis = axis
        self.patch = patch
        self.stride = stride
        self.channels = channels
        self.norm = nn.LayerNorm(channels * patch)
        self.lstm = nn.LSTM(
            channels * patch, hidden, batch_first=True, bidirectional=True
        )
        self.expand = nn.ConvTranspose1d(2 * hidden, channels, patch, stride=stride)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Move the sequence axis last, unfold with explicit end padding, then
        # restore by transpose-convolution. Every reshape follows a permute.
        if self.axis == "time":
            values = values.transpose(-1, -2)
        batch, channels, groups, length = values.shape
        steps = max(1, math.ceil(max(length - self.patch, 0) / self.stride) + 1)
        padded = (steps - 1) * self.stride + self.patch
        patches = F.pad(values, (0, padded - length)).unfold(-1, self.patch, self.stride)
        sequence = patches.permute(0, 2, 3, 1, 4).reshape(
            batch * groups, steps, channels * self.patch
        )
        sequence, _ = self.lstm(self.norm(sequence))
        restored = self.expand(sequence.transpose(1, 2))[..., :length]
        restored = restored.reshape(batch, groups, channels, length).permute(0, 2, 1, 3)
        result = values + restored
        return result.transpose(-1, -2) if self.axis == "time" else result


class _FrameAttention(nn.Module):
    def __init__(self, config: OccurrenceTFGridNetConfig) -> None:
        super().__init__()
        channels = config.model_channels
        heads = config.attention_heads
        self.heads = heads
        self.pool_bins = config.attention_frequency_bins
        self.key_channels = config.attention_key_channels
        self.value_channels = channels // heads
        self.norm = nn.GroupNorm(1, channels)
        self.query = nn.Conv2d(channels, heads * self.key_channels, 1)
        self.key = nn.Conv2d(channels, heads * self.key_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, bins = values.shape
        pooled = F.adaptive_avg_pool2d(self.norm(values), (frames, self.pool_bins))
        q = self.query(pooled).reshape(
            batch, self.heads, self.key_channels, frames, self.pool_bins
        ).permute(0, 1, 3, 2, 4).flatten(-2)
        k = self.key(pooled).reshape(
            batch, self.heads, self.key_channels, frames, self.pool_bins
        ).permute(0, 1, 3, 2, 4).flatten(-2)
        v = self.value(pooled).reshape(
            batch, self.heads, self.value_channels, frames, self.pool_bins
        ).permute(0, 1, 3, 2, 4).flatten(-2)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=frame_valid[:, None, None, :], dropout_p=0.0
        )
        attended = attended.reshape(
            batch, self.heads, frames, self.value_channels, self.pool_bins
        ).permute(0, 1, 3, 2, 4).reshape(batch, channels, frames, self.pool_bins)
        attended = F.interpolate(
            attended, size=(frames, bins), mode="bilinear", align_corners=False
        )
        return (values + self.output(attended)) * frame_valid[:, None, :, None]


class _TFGridNetBlock(nn.Module):
    def __init__(self, config: OccurrenceTFGridNetConfig) -> None:
        super().__init__()
        self.spectral = _PatchBLSTM(
            config.model_channels, config.lstm_hidden, config.patch_size,
            config.patch_stride, axis="frequency",
        )
        self.temporal = _PatchBLSTM(
            config.model_channels, config.lstm_hidden, config.patch_size,
            config.patch_stride, axis="time",
        )
        self.attention = _FrameAttention(config)

    def forward(self, values: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        values = self.spectral(values)
        values = self.temporal(values)
        return self.attention(values, frame_valid)


class OccurrenceTFGridNetBranch(nn.Module):
    def __init__(self, config: OccurrenceTFGridNetConfig, *, output_mode: str) -> None:
        super().__init__()
        config.validate()
        if output_mode not in {"complex_mapping", "real_mask"}:
            raise ValueError("invalid TF-GridNet output mode")
        self.config = config
        self.output_mode = output_mode
        self.ownership_parameterization = output_mode == "real_mask"
        self.complex_mask_parameterization = False
        channels = config.model_channels
        self.input_projection = nn.Conv2d(2, channels, 3, padding=1)
        self.input_norm = nn.GroupNorm(1, channels)
        self.pitch_embedding = nn.Embedding(128, channels)
        self.roll_embedding = nn.Embedding(3, channels)
        self.time_projection = nn.Linear(2, channels)
        self.frequency_projection = nn.Linear(3, channels)
        self.layers = nn.ModuleList(_TFGridNetBlock(config) for _ in range(config.blocks))
        self.output_norm = nn.GroupNorm(1, channels)
        self.ownership_head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.ownership_head.weight)
        nn.init.constant_(self.ownership_head.bias, -2.0)
        if output_mode == "complex_mapping":
            self.complex_head = nn.Conv2d(channels, 2, 3, padding=1)
            nn.init.normal_(self.complex_head.weight, std=0.01)
            nn.init.zeros_(self.complex_head.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _window(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(
            self.config.n_fft, device=waveform.device, dtype=waveform.dtype
        )

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform, self.config.n_fft, self.config.hop_length,
            window=self._window(waveform), center=True, return_complex=True,
        )

    def _conditioning(
        self, pitch: torch.Tensor, onset: torch.Tensor, offset: torch.Tensor,
        valid_samples: torch.Tensor, frames: int, bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = pitch.device
        frame_samples = torch.arange(frames, device=device)[None] * self.config.hop_length
        frame_valid = frame_samples < valid_samples[:, None]
        roll_frame = torch.div(
            frame_samples * 25, self.config.sample_rate, rounding_mode="floor"
        )
        onset_frame = torch.div(
            onset[:, None] * 25, self.config.sample_rate, rounding_mode="floor"
        )
        offset_frame = torch.div(
            offset[:, None] * 25 + self.config.sample_rate - 1,
            self.config.sample_rate, rounding_mode="floor",
        )
        held = (
            frame_valid & (offset > onset)[:, None]
            & (roll_frame >= onset_frame) & (roll_frame < offset_frame)
        )
        states = torch.where(held & (roll_frame == onset_frame), 2, held.long())
        time_features = torch.stack((
            ((frame_samples - onset[:, None]) / valid_samples[:, None].clamp_min(1)).clamp(-1, 1),
            ((offset[:, None] - frame_samples) / valid_samples[:, None].clamp_min(1)).clamp(-1, 1),
        ), dim=-1).float()
        temporal = (
            self.pitch_embedding(pitch)[:, None]
            + self.roll_embedding(states)
            + self.time_projection(time_features)
        )
        freq_hz = torch.arange(bins, device=device).float() * self.config.sample_rate / self.config.n_fft
        f0 = 440.0 * torch.pow(2.0, (pitch.float() - 69.0) / 12.0)
        ratio = (freq_hz[None] / f0[:, None]).clamp_min(1e-5)
        log_ratio = torch.log2(ratio)
        nearest = ratio.round().clamp_min(1.0)
        frequency_features = torch.stack((
            log_ratio.clamp(-4.0, 6.0) / 6.0,
            torch.exp(-0.5 * (1200.0 * log_ratio / 150.0).square()),
            torch.exp(-0.5 * (1200.0 * torch.log2(ratio / nearest) / 100.0).square())
            * (ratio >= 0.5),
        ), dim=-1)
        frequency = self.frequency_projection(frequency_features)
        condition = temporal.permute(0, 2, 1)[:, :, :, None] + frequency.permute(0, 2, 1)[:, :, None, :]
        return condition * frame_valid[:, None, :, None], frame_valid

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "occurrence_tf_gridnet_branch_v1",
            "config": asdict(self.config),
            "output_mode": self.output_mode,
            "output_carrier": "ungated_hpss_waveform",
            "architecture": "patch_spectral_blstm_patch_temporal_blstm_frequency_pooled_frame_attention",
            "conditioning": "rt2_25hz_off_held_onset_pitch_and_relative_frequency",
            "parameters": self.parameter_count(),
        }

    def forward(
        self, input_mixture: torch.Tensor, output_carrier: torch.Tensor,
        pitch: torch.Tensor, valid_samples: torch.Tensor,
        onset: torch.Tensor, offset: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if input_mixture.ndim != 2 or output_carrier.shape != input_mixture.shape:
            raise ValueError("input mixture and output carrier must match [batch,samples]")
        batch, samples = input_mixture.shape
        if any(value.shape != (batch,) for value in (pitch, valid_samples, onset, offset)):
            raise ValueError("query tensors must have shape [batch]")
        support = torch.arange(samples, device=input_mixture.device)[None] < valid_samples[:, None]
        clean_input = input_mixture.float() * support
        clean_carrier = output_carrier.float() * support
        level = (
            clean_carrier.square().sum(-1, keepdim=True) / valid_samples[:, None].clamp_min(1)
        ).sqrt().clamp_min(1e-4)
        spectrum = self._stft(clean_input / level)
        carrier_spectrum = self._stft(clean_carrier / level)
        processed = spectrum[:, :self.config.processed_frequency_bins]
        processed_carrier = carrier_spectrum[:, :self.config.processed_frequency_bins]
        frames, bins = processed.shape[-1], processed.shape[-2]
        condition, frame_valid = self._conditioning(
            pitch.long(), onset.long(), offset.long(), valid_samples.long(), frames, bins
        )
        features = torch.stack(
            (processed.real.transpose(-1, -2), processed.imag.transpose(-1, -2)),
            dim=1,
        )
        hidden = self.input_norm(self.input_projection(features)) + condition
        hidden = hidden * frame_valid[:, None, :, None]
        for layer in self.layers:
            if self.config.activation_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, frame_valid, use_reentrant=False)
            else:
                hidden = layer(hidden, frame_valid)
        hidden = self.output_norm(hidden)
        logits = self.ownership_head(hidden).squeeze(1).transpose(-1, -2).float()
        if self.output_mode == "real_mask":
            target_processed = processed_carrier * torch.sigmoid(logits)
        else:
            mapped = self.complex_head(hidden)
            target_processed = torch.complex(
                mapped[:, 0].transpose(-1, -2).float(),
                mapped[:, 1].transpose(-1, -2).float(),
            )
        target_processed = target_processed * frame_valid[:, None]
        target_spectrum = torch.zeros_like(spectrum)
        target_spectrum[:, :bins] = target_processed
        target = torch.istft(
            target_spectrum, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            window=self._window(clean_input), center=True, length=samples,
        ) * level
        target = target * support
        return target, {
            "target": target,
            "estimate": target,
            "target_processed_spectrum": target_processed,
            "background_head_processed_spectrum": processed_carrier - target_processed,
            "ownership_logits": logits,
            "ownership_mask": torch.sigmoid(logits) * frame_valid[:, None],
            "frame_support": frame_valid,
            "level": level,
        }


class OccurrenceDualHPSSTFGridNet(nn.Module):
    def __init__(
        self, preset: str = "canary", *, output_mode: str = "complex_mapping",
        harmonic_gate_mode: str = "below_f0_input_only",
        config: OccurrenceTFGridNetConfig | None = None,
    ) -> None:
        super().__init__()
        if harmonic_gate_mode != "below_f0_input_only":
            raise ValueError("GridNet experiment requires the established input-only gate")
        self.preset = preset
        self.output_mode = output_mode
        self.harmonic_gate_mode = harmonic_gate_mode
        self.config = config or occurrence_tf_gridnet_config(preset)
        self.harmonic_branch = OccurrenceTFGridNetBranch(self.config, output_mode=output_mode)
        self.percussive_branch = OccurrenceTFGridNetBranch(self.config, output_mode=output_mode)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def contract(self) -> dict[str, Any]:
        return {
            "kind": _KIND,
            "contract_version": 1,
            "preset": self.preset,
            "output_mode": self.output_mode,
            "architecture": {
                "layout": "independent_hpss_harmonic_and_percussive_tf_gridnet",
                "merge": "waveform_sum",
                "harmonic_branch": self.harmonic_branch.contract(),
                "percussive_branch": self.percussive_branch.contract(),
            },
            "harmonic_gate": {
                "mode": self.harmonic_gate_mode, "application": ["mixture_input"]
            },
            "conditioning": {
                "type": "rt2_25hz_target_only_roll", "states": ["off", "held", "onset"]
            },
            "inputs": {
                "harmonic_branch": "HPSS harmonic mixture",
                "percussive_branch": "ordinary margin-1 HPSS percussive mixture",
                "query": ["pitch", "onset_sample", "offset_sample"],
            },
            "outputs": {
                "harmonic": self.output_mode,
                "percussive": self.output_mode,
                "primary": "harmonic plus percussive target estimate",
                "output_carrier_mode": "ungated_hpss",
            },
            "dna_conditioning": False,
            "parameter_count": self.parameter_count(),
        }

    def forward(
        self, harmonic_mixture: torch.Tensor, percussive_mixture: torch.Tensor,
        pitch: torch.Tensor, valid_samples: torch.Tensor,
        onset: torch.Tensor, offset: torch.Tensor,
        *, input_gate_strength: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if harmonic_mixture.shape != percussive_mixture.shape:
            raise ValueError("HPSS branch inputs differ in shape")
        harmonic_gated = pitch_relative_harmonic_gate(
            harmonic_mixture, pitch, valid_samples,
            sample_rate=self.config.sample_rate, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, mode=self.harmonic_gate_mode,
        )
        if input_gate_strength is not None:
            if input_gate_strength.shape != pitch.shape:
                raise ValueError("input gate strength must have shape [batch]")
            harmonic_gated = torch.lerp(
                harmonic_mixture, harmonic_gated,
                input_gate_strength.to(harmonic_mixture.dtype)[:, None],
            )
        harmonic, hdiag = self.harmonic_branch(
            harmonic_gated, harmonic_mixture, pitch, valid_samples, onset, offset
        )
        percussive, pdiag = self.percussive_branch(
            percussive_mixture, percussive_mixture, pitch, valid_samples, onset, offset
        )
        estimate = harmonic + percussive
        background = harmonic_mixture + percussive_mixture - estimate
        return estimate, {
            "harmonic": harmonic,
            "percussive": percussive,
            "background": background,
            "waveform_reconstruction": estimate + background,
            "harmonic_input": harmonic_gated,
            "harmonic_carrier": harmonic_mixture,
            "percussive_carrier": percussive_mixture,
            "harmonic_branch": hdiag,
            "percussive_branch": pdiag,
            "conditioned_offset_samples": offset,
            "conditioning_hold_extension_samples": torch.zeros_like(offset),
            "input_gate_strength": (
                torch.ones_like(pitch, dtype=harmonic_mixture.dtype)
                if input_gate_strength is None else input_gate_strength.to(harmonic_mixture.dtype)
            ),
        }


def occurrence_dual_hpss_tf_gridnet_from_contract(raw: object) -> OccurrenceDualHPSSTFGridNet:
    if not isinstance(raw, dict) or raw.get("kind") != _KIND:
        raise ValueError("checkpoint lacks a dual-HPSS TF-GridNet contract")
    branch = raw.get("architecture", {}).get("harmonic_branch", {})
    config_raw = branch.get("config")
    if not isinstance(config_raw, dict):
        raise ValueError("TF-GridNet contract lacks branch config")
    model = OccurrenceDualHPSSTFGridNet(
        str(raw.get("preset", "canary")),
        output_mode=str(raw.get("output_mode", "complex_mapping")),
        harmonic_gate_mode=str(raw.get("harmonic_gate", {}).get("mode", "")),
        config=OccurrenceTFGridNetConfig(**config_raw),
    )
    if model.contract() != raw:
        raise ValueError("dual-HPSS TF-GridNet contract changed")
    return model
