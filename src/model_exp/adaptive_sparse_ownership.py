"""Small lifecycle-free adaptive sparse ownership for aligned note proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F


def sparsemax_notes(values: torch.Tensor) -> torch.Tensor:
    """Project the note axis of [B,K,F,T] values onto the simplex."""
    work = values.float()
    ordered = work.sort(dim=1, descending=True).values
    cumulative = ordered.cumsum(dim=1)
    rank = torch.arange(
        1, values.shape[1] + 1, device=values.device, dtype=work.dtype
    )[None, :, None, None]
    support = (1.0 + rank * ordered > cumulative).sum(1, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(1, support - 1) - 1.0) / support
    projected = (work - tau).clamp_min(0.0)
    return (projected / projected.sum(1, keepdim=True).clamp_min(1e-12)).to(values.dtype)


@dataclass(frozen=True)
class AdaptiveSparseOwnershipConfig:
    hidden_channels: int = 24
    local_blocks: int = 2
    maximum_simplex_correction: float = 0.5
    sample_rate: int = 16_000
    n_fft: int = 2_048
    epsilon: float = 1e-8


class _LocalBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(self.depthwise(F.silu(self.norm(value))))


class AdaptiveSparseOwnership(nn.Module):
    """Refine raw magnitude votes with local evidence and adaptive sparsemax.

    The score separator has already used pitch/onset/offset to create ``raw``.
    This allocator intentionally receives no timing state.  Pitch is used only
    to express frequency relative to the queried fundamental.
    """

    feature_channels = 11

    def __init__(self, config: AdaptiveSparseOwnershipConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdaptiveSparseOwnershipConfig()
        if (
            self.config.hidden_channels < 4
            or self.config.local_blocks < 1
            or self.config.maximum_simplex_correction <= 0
            or self.config.epsilon <= 0
        ):
            raise ValueError("invalid adaptive sparse ownership configuration")
        channels = self.config.hidden_channels
        self.input = nn.Conv2d(self.feature_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(_LocalBlock(channels) for _ in range(self.config.local_blocks)))
        self.correction = nn.Conv2d(channels, 1, 1)
        self.confidence = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, notes, frequency, frames = raw.shape
        magnitude = raw.abs()
        total = magnitude.sum(1, keepdim=True)
        # Match the frozen magnitude-partition authority exactly: normalize
        # every numerically nonzero proposal bin, and do not invent ownership
        # when all note proposals are truly empty.
        ownership_tiny = 1e-20
        base = torch.where(
            total > ownership_tiny,
            magnitude / total.clamp_min(ownership_tiny),
            torch.zeros_like(magnitude),
        )
        mix_mag = mixture.abs()[:, None]
        phase_cosine = (raw.conj() * mixture[:, None]).real / (
            magnitude * mix_mag
        ).clamp_min(self.config.epsilon)
        log_ratio = torch.log((magnitude + self.config.epsilon) /
                              (mix_mag + self.config.epsilon)).clamp(-6, 6) / 6
        raw_log = torch.log1p(20 * magnitude) / 4
        mix_log = torch.log1p(20 * mix_mag).expand_as(raw_log) / 4
        flux = F.pad(
            torch.log1p(20 * mixture.abs())[..., 1:]
            - torch.log1p(20 * mixture.abs())[..., :-1],
            (1, 0),
        )[:, None].expand_as(raw_log).clamp(-2, 2) / 2

        hz = (
            torch.arange(frequency, device=raw.device, dtype=raw.real.dtype)
            * self.config.sample_rate / self.config.n_fft
        ).clamp_min(self.config.sample_rate / self.config.n_fft)
        f0 = 440.0 * torch.pow(raw.real.new_tensor(2.0), (pitches.to(raw.real.dtype) - 69) / 12)
        semitone = 12 * torch.log2(hz[None, None] / f0[..., None].clamp_min(1e-5))
        ratio = hz[None, None] / f0[..., None].clamp_min(1e-5)
        harmonic = ratio.round().clamp_min(1)
        cents = 1200 * torch.log2(ratio.clamp_min(1e-6) / harmonic)
        pitch_features = (
            (semitone / 96).clamp(-2, 2),
            torch.sin(torch.pi * semitone / 12),
            torch.cos(torch.pi * semitone / 12),
            (cents / 100).clamp(-4, 4),
            (torch.log2(harmonic) / 7).clamp(0, 1),
        )
        pitch_features = tuple(x[..., None].expand(-1, -1, -1, frames) for x in pitch_features)
        features = torch.stack(
            (raw_log, mix_log, log_ratio, phase_cosine.clamp(-1, 1), base, flux, *pitch_features),
            dim=2,
        )
        if features.shape[2] != AdaptiveSparseOwnership.feature_channels:
            raise RuntimeError(f"feature contract changed: {tuple(features.shape)}")
        return features, base

    def forward(
        self,
        mixture: torch.Tensor,
        anchors: torch.Tensor,
        raw: torch.Tensor,
        pitches: torch.Tensor,
        onsets: torch.Tensor | None = None,
        midi_offsets: torch.Tensor | None = None,
        acoustic_offsets: torch.Tensor | None = None,
        harmonic_logits: torch.Tensor | None = None,
        percussive_logits: torch.Tensor | None = None,
        *,
        confidence_mode: Literal["adaptive", "zero", "one"] = "adaptive",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del anchors, onsets, midi_offsets, acoustic_offsets, harmonic_logits, percussive_logits
        if mixture.ndim != 3 or raw.ndim != 4 or raw.shape[0] != mixture.shape[0]:
            raise ValueError("invalid mixture/note-bank geometry")
        if raw.shape[1] < 2 or raw.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("allocator requires at least two aligned notes")
        features, base = self._features(mixture, raw, pitches)
        batch, notes, _, frequency, frames = features.shape
        memory = F.silu(self.input(features.flatten(0, 1)))
        memory = self.blocks(memory)
        delta = self.config.maximum_simplex_correction * torch.tanh(
            self.correction(memory)[:, 0]
        ).view(batch, notes, frequency, frames)
        sparse = sparsemax_notes(base + delta)
        total = raw.abs().sum(1, keepdim=True)
        sparse = torch.where(total > 1e-20, sparse, torch.zeros_like(sparse))
        gate_note = self.confidence(memory)[:, 0].view(batch, notes, frequency, frames)
        gate = torch.sigmoid((base * gate_note).sum(1, keepdim=True))
        if confidence_mode == "zero":
            gate = torch.zeros_like(gate)
        elif confidence_mode == "one":
            gate = torch.ones_like(gate)
        elif confidence_mode != "adaptive":
            raise ValueError(f"invalid confidence mode: {confidence_mode}")
        ownership = (1 - gate) * base + gate * sparse
        output = ownership * mixture[:, None]
        return output, {
            "ownership": ownership,
            "base_ownership": base,
            "sparse_ownership": sparse,
            "confidence": gate,
            "mean_abs_correction": delta.abs().mean(),
            "mean_confidence": gate.mean(),
            "sum_error_max_abs": (output.sum(1) - mixture).abs().amax(),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "adaptive_sparse_ownership_v1",
            **asdict(self.config),
            "initializer": "raw symmetric estimate magnitude partition",
            "features": "raw magnitude, mixture magnitude, magnitude ratio, phase agreement, raw ownership share, mixture flux, pitch-relative harmonic geometry",
            "excluded_inputs": "onset, offset, lifecycle state, instrument identity, background lane",
            "joint_operation": "bounded shared local scorer plus source-axis sparsemax",
            "adaptivity": "per-bin confidence interpolation between dense raw shares and sparse ownership",
            "carrier": "mixture complex STFT",
            "mixture_consistency": "exact across scored notes",
            "source_complexity": "linear in active notes and spectrogram bins",
        }


__all__ = ["AdaptiveSparseOwnership", "AdaptiveSparseOwnershipConfig", "sparsemax_notes"]
