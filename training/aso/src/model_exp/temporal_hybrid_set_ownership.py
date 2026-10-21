"""Temporal, lifecycle-free joint ownership with conservative phase reuse.

The score separator supplies aligned complex note proposals.  This module only
refines their joint time-frequency allocation; it does not perform separator
inference and receives no onset/offset/lifecycle state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from .simplex import sparsemax_notes


FeatureSet = Literal[
    "legacy11", "essential6", "essential7_relative_peak",
    "essential7_harmonic_prominence",
    "essential7_relative_harmonic_loudness", "essential6_phase"
]
PhaseMode = Literal["mixture", "hybrid_delta", "direct_complex"]
MagnitudeReference = Literal["mixture", "raw_total"]


@dataclass(frozen=True)
class TemporalHybridOwnershipConfig:
    feature_set: FeatureSet = "essential6_phase"
    phase_mode: PhaseMode = "hybrid_delta"
    magnitude_reference: MagnitudeReference = "mixture"
    hidden_channels: int = 24
    local_blocks: int = 2
    temporal_blocks: int = 2
    maximum_simplex_correction: float = 0.5
    # Conservative inference-time trust in the learned sparsemax allocation.
    # One preserves the trained endpoint; zero returns the magnitude partition.
    ownership_trust: float = 1.0
    phase_delta_threshold: float = 0.10
    phase_switch_temperature: float = 0.02
    maximum_complex_delta: float = 0.5
    phase_cutoff_hz: float = 4_800.0
    sample_rate: int = 16_000
    n_fft: int = 2_048
    epsilon: float = 1e-8
    relative_peak_radius_bins: int = 4
    gradient_checkpointing: bool = True
    exact_mixture_additivity: bool = False


class _LocalBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(self.depthwise(F.silu(self.norm(value))))


class _TemporalBlock(nn.Module):
    """Depthwise temporal convolution followed by channel mixing."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.temporal = nn.Conv2d(
            channels,
            channels,
            (1, 5),
            padding=(0, 2 * dilation),
            dilation=(1, dilation),
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(self.temporal(F.silu(self.norm(value))))


def _unit(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    magnitude = value.abs()
    return torch.where(
        magnitude > epsilon,
        value / magnitude.clamp_min(epsilon),
        torch.ones_like(value),
    )


def _straight_through_binary(probability: torch.Tensor) -> torch.Tensor:
    hard = (probability >= 0.5).to(probability.dtype)
    return hard.detach() - probability.detach() + probability


class TemporalHybridSetOwnership(nn.Module):
    """Shared temporal scorer, Deep-Sets context, sparsemax, and phase policy."""

    _FEATURE_CHANNELS = {
        "legacy11": 11,
        "essential6": 6,
        "essential7_relative_peak": 7,
        "essential7_harmonic_prominence": 7,
        "essential7_relative_harmonic_loudness": 7,
        "essential6_phase": 10,
    }

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalHybridOwnershipConfig()
        if self.config.feature_set not in self._FEATURE_CHANNELS:
            raise ValueError(f"invalid feature set: {self.config.feature_set}")
        if self.config.phase_mode not in {"mixture", "hybrid_delta", "direct_complex"}:
            raise ValueError(f"invalid phase mode: {self.config.phase_mode}")
        if self.config.magnitude_reference not in {"mixture", "raw_total"}:
            raise ValueError(f"invalid magnitude reference: {self.config.magnitude_reference}")
        if self.config.phase_mode != "mixture" and self.config.feature_set != "essential6_phase":
            raise ValueError("phase-aware output requires essential6_phase inputs")
        if (
            self.config.hidden_channels < 4
            or self.config.local_blocks < 1
            or self.config.temporal_blocks < 0
            or self.config.maximum_simplex_correction <= 0
            or not 0 <= self.config.ownership_trust <= 1
            or self.config.phase_switch_temperature <= 0
            or self.config.relative_peak_radius_bins < 0
            or not 0 < self.config.phase_cutoff_hz <= self.config.sample_rate / 2
        ):
            raise ValueError("invalid temporal hybrid ownership configuration")

        channels = self.config.hidden_channels
        self.input = nn.Conv2d(
            self._FEATURE_CHANNELS[self.config.feature_set], channels, 3, padding=1
        )
        self.local = nn.Sequential(
            *(_LocalBlock(channels) for _ in range(self.config.local_blocks))
        )
        self.temporal = nn.Sequential(
            *(
                _TemporalBlock(channels, 2**index)
                for index in range(self.config.temporal_blocks)
            )
        )
        self.set_fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1),
            nn.SiLU(),
        )
        self.magnitude_delta = nn.Conv2d(channels, 1, 1)
        self.complex_delta = (
            nn.Conv2d(channels, 2, 1)
            if self.config.phase_mode == "direct_complex"
            else None
        )
        self.phase_confidence = (
            nn.Conv2d(channels, 1, 1)
            if self.config.phase_mode == "direct_complex"
            else None
        )
        nn.init.zeros_(self.magnitude_delta.weight)
        nn.init.zeros_(self.magnitude_delta.bias)
        if self.complex_delta is not None:
            nn.init.zeros_(self.complex_delta.weight)
            nn.init.zeros_(self.complex_delta.bias)
            nn.init.zeros_(self.phase_confidence.weight)
            # Start infinitesimally on the corrected-phase side of the hard
            # switch.  The correction itself is exactly zero, so the initial
            # waveform still uses raw phase, while the complex head receives
            # a nonzero gradient on the first update.
            nn.init.constant_(self.phase_confidence.bias, -0.1)

    def _common_features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch, notes, frequency, frames = raw.shape
        del batch, notes
        magnitude = raw.abs()
        total = magnitude.sum(1, keepdim=True)
        base = torch.where(
            total > 1e-20,
            magnitude / total.clamp_min(1e-20),
            torch.zeros_like(magnitude),
        )
        mix_mag = mixture.abs()[:, None]
        raw_unit = _unit(raw, self.config.epsilon)
        mix_unit = _unit(mixture, self.config.epsilon)[:, None]
        phase_cosine = (raw_unit.conj() * mix_unit).real.clamp(-1, 1)
        log_ratio = torch.log(
            (magnitude + self.config.epsilon) / (mix_mag + self.config.epsilon)
        ).clamp(-6, 6) / 6
        raw_log = torch.log1p(20 * magnitude) / 4
        mix_log = torch.log1p(20 * mix_mag).expand_as(raw_log) / 4
        flux = F.pad(
            torch.log1p(20 * mixture.abs())[..., 1:]
            - torch.log1p(20 * mixture.abs())[..., :-1],
            (1, 0),
        )[:, None].expand_as(raw_log).clamp(-2, 2) / 2

        hz = (
            torch.arange(frequency, device=raw.device, dtype=raw.real.dtype)
            * self.config.sample_rate
            / self.config.n_fft
        ).clamp_min(self.config.sample_rate / self.config.n_fft)
        f0 = 440.0 * torch.pow(
            raw.real.new_tensor(2.0),
            (pitches.to(raw.real.dtype) - 69) / 12,
        )
        semitone = 12 * torch.log2(hz[None, None] / f0[..., None].clamp_min(1e-5))
        ratio = hz[None, None] / f0[..., None].clamp_min(1e-5)
        harmonic = ratio.round().clamp_min(1)
        cents = 1200 * torch.log2(ratio.clamp_min(1e-6) / harmonic)
        # For every candidate note/bin, compare the raw magnitude at that bin
        # with the strongest nearby raw bin from the same note proposal.  The
        # value is one on a local spectral ridge and lies in [0, 1] elsewhere.
        # Harmonic distance remains separate, allowing the network to combine
        # "is this a peak for this note?" with "is it where its partial belongs?".
        radius = self.config.relative_peak_radius_bins
        nearest_peak_magnitude = F.max_pool2d(
            magnitude,
            kernel_size=(2 * radius + 1, 1),
            stride=1,
            padding=(radius, 0),
        )
        relative_nearest_peak = torch.where(
            nearest_peak_magnitude > self.config.epsilon,
            magnitude / nearest_peak_magnitude.clamp_min(self.config.epsilon),
            torch.zeros_like(magnitude),
        ).clamp(0, 1)
        # Corrected prominence: compare the peak around the nearest expected
        # harmonic with the strongest expected-harmonic peak anywhere in this
        # note at the same frame.  Unlike ``relative_nearest_peak``, this is not
        # tautologically one on every local ridge.  The nearest-harmonic-cents
        # channel remains separate, so off-comb bins cannot masquerade as
        # harmonics merely because their nearest expected partial is strong.
        bin_hz = self.config.sample_rate / self.config.n_fft
        harmonic_center_bin = torch.round(
            harmonic * f0[..., None] / bin_hz
        ).long().clamp(0, frequency - 1)
        harmonic_peak = nearest_peak_magnitude.gather(
            2, harmonic_center_bin[..., None].expand(-1, -1, -1, frames)
        )
        harmonic_center_valid = (
            harmonic * f0[..., None]
            <= self.config.sample_rate / 2
        )
        harmonic_peak = torch.where(
            harmonic_center_valid[..., None], harmonic_peak,
            torch.zeros_like(harmonic_peak),
        )
        strongest_harmonic_peak = harmonic_peak.amax(2, keepdim=True)
        relative_harmonic_prominence = torch.where(
            strongest_harmonic_peak > self.config.epsilon,
            harmonic_peak / strongest_harmonic_peak.clamp_min(self.config.epsilon),
            torch.zeros_like(harmonic_peak),
        ).clamp(0, 1)
        # A peak-free alternative: the RMS energy in the nearest expected
        # harmonic band relative to the complete raw note's framewise RMS.
        # This is stable when every local maximum is merely low-level noise.
        band_power = F.avg_pool2d(
            magnitude.square(),
            kernel_size=(2 * radius + 1, 1),
            stride=1,
            padding=(radius, 0),
        ) * (2 * radius + 1)
        harmonic_band_power = band_power.gather(
            2, harmonic_center_bin[..., None].expand(-1, -1, -1, frames)
        )
        harmonic_band_power = torch.where(
            harmonic_center_valid[..., None], harmonic_band_power,
            torch.zeros_like(harmonic_band_power),
        )
        note_power = magnitude.square().sum(2, keepdim=True)
        relative_harmonic_loudness = torch.where(
            note_power > self.config.epsilon**2,
            torch.sqrt(
                harmonic_band_power
                / note_power.clamp_min(self.config.epsilon**2)
            ),
            torch.zeros_like(harmonic_band_power),
        ).clamp(0, 1)
        pitch = {
            "relative_semitone": (semitone / 96).clamp(-2, 2),
            "pitch_sine": torch.sin(torch.pi * semitone / 12),
            "pitch_cosine": torch.cos(torch.pi * semitone / 12),
            "nearest_harmonic_cents": (cents / 100).clamp(-4, 4),
            "harmonic_number": (torch.log2(harmonic) / 7).clamp(0, 1),
        }
        pitch = {
            key: value[..., None].expand(-1, -1, -1, frames)
            for key, value in pitch.items()
        }
        return {
            "raw_log": raw_log,
            "mixture_log": mix_log,
            "log_ratio": log_ratio,
            "phase_agreement": phase_cosine,
            "base_ownership": base,
            "relative_nearest_peak": relative_nearest_peak,
            "relative_harmonic_prominence": relative_harmonic_prominence,
            "relative_harmonic_loudness": relative_harmonic_loudness,
            "mixture_flux": flux,
            **pitch,
            "raw_phase_real": raw_unit.real,
            "raw_phase_imag": raw_unit.imag,
            "mixture_phase_real": mix_unit.real.expand_as(raw_log),
            "mixture_phase_imag": mix_unit.imag.expand_as(raw_log),
        }, base

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, base = self._common_features(mixture, raw, pitches)
        if self.config.feature_set == "legacy11":
            names = (
                "raw_log", "mixture_log", "log_ratio", "phase_agreement",
                "base_ownership", "mixture_flux", "relative_semitone",
                "pitch_sine", "pitch_cosine", "nearest_harmonic_cents",
                "harmonic_number",
            )
        elif self.config.feature_set in {
            "essential6", "essential7_relative_peak",
            "essential7_harmonic_prominence",
            "essential7_relative_harmonic_loudness",
        }:
            names = (
                "mixture_log", "log_ratio", "base_ownership",
                "phase_agreement", "nearest_harmonic_cents", "harmonic_number",
            )
            if self.config.feature_set == "essential7_relative_peak":
                names = (*names, "relative_nearest_peak")
            elif self.config.feature_set == "essential7_harmonic_prominence":
                names = (*names, "relative_harmonic_prominence")
            elif self.config.feature_set == "essential7_relative_harmonic_loudness":
                names = (*names, "relative_harmonic_loudness")
        else:
            names = (
                "mixture_log", "log_ratio", "base_ownership",
                "phase_agreement", "nearest_harmonic_cents", "harmonic_number",
                "raw_phase_real", "raw_phase_imag",
                "mixture_phase_real", "mixture_phase_imag",
            )
        return torch.stack(tuple(values[name] for name in names), dim=2), base

    def _refine_magnitude_delta(
        self, features: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Optional structured residual before the joint sparsemax projection."""
        del features
        return delta

    def _allocate_ownership(
        self, base: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Map a learned correction to source-simplex ownership.

        Kept as a hook so matched experiments can change the geometry of the
        ownership update without duplicating the temporal/set network.
        """
        return sparsemax_notes(base + delta)

    def _trust_magnitude_delta(
        self,
        features: torch.Tensor,
        memory: torch.Tensor,
        base: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally damp each proposed logit/log-gain before allocation.

        The default is an exact no-op.  Keeping this hook before
        ``_allocate_ownership`` lets relative-allocation variants express
        note-specific confidence without blending already-normalized owners.
        """
        del features, memory, base
        trust = delta.new_ones(delta.shape)
        return delta, trust

    def _blend_ownership(
        self,
        features: torch.Tensor,
        memory: torch.Tensor,
        base: torch.Tensor,
        learned: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend a learned allocation with its raw-magnitude starting point."""
        del features, memory
        trust = base.new_full(
            (base.shape[0], 1, base.shape[-2], base.shape[-1]),
            self.config.ownership_trust,
        )
        return base + trust * (learned - base), trust

    def _refine_output(
        self, features: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Optional non-redistributing refinement after sparse allocation."""
        del features
        return output

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
        hard_phase: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del anchors, onsets, midi_offsets, acoustic_offsets, harmonic_logits, percussive_logits
        if mixture.ndim != 3 or raw.ndim != 4 or raw.shape[0] != mixture.shape[0]:
            raise ValueError("invalid mixture/note-bank geometry")
        if raw.shape[1] < 2 or raw.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("allocator requires at least two aligned notes")

        features, base = self._features(mixture, raw, pitches)
        batch, notes, _, frequency, frames = features.shape
        local = F.silu(self.input(features.flatten(0, 1)))
        if self.training and self.config.gradient_checkpointing:
            local = checkpoint_sequential(
                self.local, len(self.local), local, use_reentrant=False
            )
            if len(self.temporal):
                local = checkpoint_sequential(
                    self.temporal, len(self.temporal), local, use_reentrant=False
                )
        else:
            local = self.temporal(self.local(local))
        local = local.view(
            batch, notes, -1, frequency, frames
        )
        mean = local.mean(1, keepdim=True).expand_as(local)
        maximum = local.amax(1, keepdim=True).expand_as(local)
        memory = self.set_fuse(
            torch.cat((local, mean, maximum), 2).flatten(0, 1)
        )
        delta = self.config.maximum_simplex_correction * torch.tanh(
            self.magnitude_delta(memory)[:, 0]
        ).view(batch, notes, frequency, frames)
        delta = self._refine_magnitude_delta(features, delta)
        delta, magnitude_delta_trust = self._trust_magnitude_delta(
            features,
            memory.view(batch, notes, -1, frequency, frames),
            base,
            delta,
        )
        learned_ownership = self._allocate_ownership(base, delta)
        ownership, ownership_trust = self._blend_ownership(
            features,
            memory.view(batch, notes, -1, frequency, frames),
            base,
            learned_ownership,
        )
        proposal_active = raw.abs().sum(1, keepdim=True) > 1e-20
        if not self.config.exact_mixture_additivity:
            ownership = torch.where(
                proposal_active, ownership, torch.zeros_like(ownership)
            )

        mix_unit = _unit(mixture, self.config.epsilon)[:, None]
        raw_unit = _unit(raw, self.config.epsilon)
        hz = (
            torch.arange(frequency, device=raw.device, dtype=raw.real.dtype)
            * self.config.sample_rate
            / self.config.n_fft
        )[None, None, :, None]
        below_cutoff = hz < self.config.phase_cutoff_hz

        if self.config.phase_mode == "mixture":
            carrier = mix_unit
            raw_phase_probability = torch.zeros_like(ownership)
            complex_change = torch.zeros_like(ownership)
        elif self.config.phase_mode == "hybrid_delta":
            probability = torch.sigmoid(
                (self.config.phase_delta_threshold - delta.abs())
                / self.config.phase_switch_temperature
            )
            probability = probability * below_cutoff
            selector = _straight_through_binary(probability) if hard_phase else probability
            carrier = _unit(
                selector * raw_unit + (1 - selector) * mix_unit,
                self.config.epsilon,
            )
            raw_phase_probability = probability
            complex_change = torch.zeros_like(ownership)
        else:
            assert self.complex_delta is not None and self.phase_confidence is not None
            components = self.complex_delta(memory).view(
                batch, notes, 2, frequency, frames
            )
            complex_change_value = self.config.maximum_complex_delta * torch.complex(
                torch.tanh(components[:, :, 0]), torch.tanh(components[:, :, 1])
            )
            corrected_unit = _unit(raw_unit + complex_change_value, self.config.epsilon)
            probability = torch.sigmoid(
                self.phase_confidence(memory)[:, 0]
            ).view(batch, notes, frequency, frames) * below_cutoff
            selector = _straight_through_binary(probability) if hard_phase else probability
            # High confidence preserves the raw note phase; low confidence uses
            # the learned bounded complex correction.  Above 4.8 kHz the
            # mixture phase is always used.
            low_band = _unit(
                selector * raw_unit + (1 - selector) * corrected_unit,
                self.config.epsilon,
            )
            carrier = torch.where(below_cutoff, low_band, mix_unit)
            raw_phase_probability = probability
            complex_change = complex_change_value.abs()

        magnitude_reference = (
            mixture.abs()[:, None]
            if self.config.magnitude_reference == "mixture"
            else raw.abs().sum(1, keepdim=True)
        )
        output = ownership * magnitude_reference * carrier
        output = self._refine_output(features, output)
        time_tv = delta[..., 1:].sub(delta[..., :-1]).abs().mean()
        freq_tv = delta[..., 1:, :].sub(delta[..., :-1, :]).abs().mean()
        closure_target = (
            magnitude_reference[:, 0]
            if self.config.exact_mixture_additivity
            else magnitude_reference[:, 0] * proposal_active[:, 0]
        )
        complex_closure_target = (
            mixture
            if self.config.exact_mixture_additivity
            and self.config.magnitude_reference == "mixture"
            and self.config.phase_mode == "mixture"
            else None
        )
        magnitude_closure_abs = (output.abs().sum(1) - closure_target).abs().amax()
        complex_closure_abs = (
            (output.sum(1) - complex_closure_target).abs().amax()
            if complex_closure_target is not None
            else output.real.new_tensor(float("nan"))
        )
        closure_scale = magnitude_reference[:, 0].abs().amax().clamp_min(1.0)
        return output, {
            "ownership": ownership,
            "base_ownership": base,
            "magnitude_delta": delta,
            "mean_abs_correction": delta.abs().mean(),
            "magnitude_delta_trust": magnitude_delta_trust,
            "mean_magnitude_delta_trust": magnitude_delta_trust.mean(),
            "ownership_trust": ownership_trust,
            "mean_ownership_trust": ownership_trust.mean(),
            "smoothness_penalty": time_tv + freq_tv,
            "time_smoothness_penalty": time_tv,
            "frequency_smoothness_penalty": freq_tv,
            "raw_phase_probability": raw_phase_probability,
            "mean_raw_phase_probability": raw_phase_probability.mean(),
            "mean_complex_delta": complex_change.mean(),
            "magnitude_closure_max_abs": magnitude_closure_abs,
            "magnitude_closure_max_relative": magnitude_closure_abs / closure_scale,
            "complex_closure_max_abs": complex_closure_abs,
            "complex_closure_max_relative": complex_closure_abs / closure_scale,
            "magnitude_reference": self.config.magnitude_reference,
        }

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "temporal_hybrid_set_ownership_v1",
            **asdict(self.config),
            "features": self.config.feature_set,
            "excluded_inputs": "onset, offset, lifecycle, instrument identity, background lane",
            "temporal_model": "depthwise 1x5 residual convolutions with dilations 1,2,...",
            "cross_note_context": "per-TF permutation-invariant mean and maximum",
            "magnitude_allocation": (
                "bounded learned delta, source-axis sparsemax, then convex trust "
                "blend with the input magnitude partition"
            ),
            "magnitude_reference": self.config.magnitude_reference,
            "magnitude_smoothness": "energy-weighted TF total-variation training penalty",
            "phase_policy": self.config.phase_mode,
            "high_frequency_phase": "mixture phase at and above phase_cutoff_hz",
            "complex_mixture_consistency": (
                self.config.exact_mixture_additivity
                and self.config.magnitude_reference == "mixture"
                and self.config.phase_mode == "mixture"
            ),
            "magnitude_partition_consistency": self.config.magnitude_reference == "mixture",
            "raw_empty_bin_policy": (
                "assign every mixture bin across notes"
                if self.config.exact_mixture_additivity
                else "leave bins unassigned when every raw proposal is zero"
            ),
            "raw_total_magnitude_conservation": self.config.magnitude_reference == "raw_total",
            "test_data_used": False,
        }


__all__ = ["TemporalHybridOwnershipConfig", "TemporalHybridSetOwnership"]
