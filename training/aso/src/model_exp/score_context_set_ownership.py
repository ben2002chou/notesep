"""ASO with minimal onset/offset support and score-harmonic collision cues."""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .adaptive_set_ownership import AdaptiveSetOwnership
from .adaptive_sparse_ownership import AdaptiveSparseOwnershipConfig, sparsemax_notes


class ScoreContextSetOwnership(AdaptiveSetOwnership):
    """Append one active flag and four collision-count channels to ASO."""

    base_feature_channels = 11
    score_feature_channels = 5
    feature_channels = base_feature_channels + score_feature_channels
    collision_radius_hz = 35.0

    def __init__(self, config: AdaptiveSparseOwnershipConfig | None = None) -> None:
        super().__init__(config)

    def score_features(
        self,
        raw: torch.Tensor,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        acoustic_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, notes, frequency, frames = raw.shape
        if onsets.shape != (batch, notes) or acoustic_offsets.shape != (batch, notes):
            raise ValueError("score timing geometry mismatch")
        centers = torch.arange(frames, device=raw.device) * (self.config.n_fft // 4)
        active = (centers[None, None] >= onsets[..., None]) & (
            centers[None, None] < acoustic_offsets[..., None]
        )
        hz = (
            torch.arange(frequency, device=raw.device, dtype=raw.real.dtype)
            * self.config.sample_rate / self.config.n_fft
        )
        f0 = 440.0 * torch.pow(
            raw.real.new_tensor(2.0), (pitches.to(raw.real.dtype) - 69) / 12
        )
        ratio = hz[None, None, :] / f0[..., None].clamp_min(1e-5)
        harmonic = ratio.round().clamp_min(1)
        distance = (hz[None, None, :] - harmonic * f0[..., None]).abs()
        plausible = distance <= self.collision_radius_hz
        active_harmonic = plausible[..., None] & active[:, :, None, :]
        competitor_count = active_harmonic.sum(1, keepdim=True) - active_harmonic.to(torch.long)
        competitor_count = competitor_count.clamp(0, 3)
        collision = F.one_hot(competitor_count, num_classes=4).permute(0, 1, 4, 2, 3)
        active_channel = active[:, :, None, None, :].expand(-1, -1, 1, frequency, -1)
        score = torch.cat((active_channel, collision), 2).to(raw.real.dtype)
        if score.shape != (batch, notes, self.score_feature_channels, frequency, frames):
            raise RuntimeError("score feature contract changed")
        return score, competitor_count

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
        score_mode: Literal["normal", "zero"] = "normal",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del anchors, midi_offsets, harmonic_logits, percussive_logits
        if onsets is None or acoustic_offsets is None:
            raise ValueError("score-context ASO requires onset and acoustic offset")
        if mixture.ndim != 3 or raw.ndim != 4 or raw.shape[0] != mixture.shape[0]:
            raise ValueError("invalid mixture/note-bank geometry")
        if raw.shape[1] < 2 or raw.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("allocator requires at least two aligned notes")
        base_features, base = super()._features(mixture, raw, pitches)
        score, competitor_count = self.score_features(raw, pitches, onsets, acoustic_offsets)
        if score_mode == "zero":
            score = torch.zeros_like(score)
        elif score_mode != "normal":
            raise ValueError(f"invalid score mode: {score_mode}")
        features = torch.cat((base_features, score), 2)
        batch, notes, _, frequency, frames = features.shape
        local = F.silu(self.input(features.flatten(0, 1)))
        local = self.blocks(local).view(batch, notes, -1, frequency, frames)
        mean = local.mean(1, keepdim=True).expand_as(local)
        maximum = local.amax(1, keepdim=True).expand_as(local)
        memory = self.set_fuse(torch.cat((local, mean, maximum), 2).flatten(0, 1))
        delta = self.config.maximum_simplex_correction * torch.tanh(
            self.correction(memory)[:, 0]
        ).view(batch, notes, frequency, frames)
        sparse = sparsemax_notes(base + delta)
        proposal_total = raw.abs().sum(1, keepdim=True)
        sparse = torch.where(proposal_total > 1e-20, sparse, torch.zeros_like(sparse))
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
            "score_active_fraction": score[:, :, 0].mean(),
            "score_collision_mean": competitor_count.float().mean(),
        }

    def contract(self) -> dict[str, Any]:
        result = super().contract()
        result.update({
            "kind": "score_context_set_ownership_v1",
            "score_features": "single onset-to-acoustic-offset active flag plus one-hot 0/1/2/3+ active harmonic competitor count",
            "collision_radius_hz": self.collision_radius_hz,
            "excluded_score_features": "attack, sustain, release, distance-to-boundary, instrument identity",
        })
        return result


class FrozenCoreScoreContextSetOwnership(AdaptiveSetOwnership):
    """Add a zero-initialized score adapter beside an untouched ASO input."""

    score_feature_channels = ScoreContextSetOwnership.score_feature_channels
    collision_radius_hz = ScoreContextSetOwnership.collision_radius_hz

    def __init__(self, config: AdaptiveSparseOwnershipConfig | None = None) -> None:
        super().__init__(config)
        self.score_input = nn.Conv2d(
            self.score_feature_channels,
            self.config.hidden_channels,
            3,
            padding=1,
            bias=False,
        )
        nn.init.zeros_(self.score_input.weight)

    def score_features(
        self,
        raw: torch.Tensor,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        acoustic_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ScoreContextSetOwnership.score_features(
            self, raw, pitches, onsets, acoustic_offsets
        )

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
        score_mode: Literal["normal", "zero"] = "normal",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del anchors, midi_offsets, harmonic_logits, percussive_logits
        if onsets is None or acoustic_offsets is None:
            raise ValueError("frozen-core score ASO requires onset and acoustic offset")
        if mixture.ndim != 3 or raw.ndim != 4 or raw.shape[0] != mixture.shape[0]:
            raise ValueError("invalid mixture/note-bank geometry")
        if raw.shape[1] < 2 or raw.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("allocator requires at least two aligned notes")
        base_features, base = super()._features(mixture, raw, pitches)
        score, competitor_count = self.score_features(raw, pitches, onsets, acoustic_offsets)
        if score_mode == "zero":
            score = torch.zeros_like(score)
        elif score_mode != "normal":
            raise ValueError(f"invalid score mode: {score_mode}")
        batch, notes, _, frequency, frames = base_features.shape
        base_local = self.input(base_features.flatten(0, 1))
        score_local = self.score_input(score.flatten(0, 1))
        local = F.silu(base_local + score_local)
        local = self.blocks(local).view(batch, notes, -1, frequency, frames)
        mean = local.mean(1, keepdim=True).expand_as(local)
        maximum = local.amax(1, keepdim=True).expand_as(local)
        memory = self.set_fuse(torch.cat((local, mean, maximum), 2).flatten(0, 1))
        delta = self.config.maximum_simplex_correction * torch.tanh(
            self.correction(memory)[:, 0]
        ).view(batch, notes, frequency, frames)
        sparse = sparsemax_notes(base + delta)
        proposal_total = raw.abs().sum(1, keepdim=True)
        sparse = torch.where(proposal_total > 1e-20, sparse, torch.zeros_like(sparse))
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
            "score_active_fraction": score[:, :, 0].mean(),
            "score_collision_mean": competitor_count.float().mean(),
        }

    def contract(self) -> dict[str, Any]:
        result = super().contract()
        result.update({
            "kind": "frozen_core_score_context_set_ownership_v1",
            "score_adapter": "parallel zero-initialized 3x3 input; base ASO frozen",
            "score_features": "single onset-to-acoustic-offset active flag plus one-hot 0/1/2/3+ active harmonic competitor count",
            "collision_radius_hz": self.collision_radius_hz,
            "excluded_score_features": "attack, sustain, release, distance-to-boundary, instrument identity",
        })
        return result


class ActiveMaskedScoreAdapterOwnership(FrozenCoreScoreContextSetOwnership):
    """Frozen-core adapter whose score evidence exists only while a note is active."""

    def score_features(
        self,
        raw: torch.Tensor,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        acoustic_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score, competitor_count = super().score_features(
            raw, pitches, onsets, acoustic_offsets
        )
        active = score[:, :, :1]
        # In the original encoding, the zero-competitor one-hot channel was
        # one at every inactive frame.  That was a global learned bias, not
        # score evidence.  Restrict every collision category to the same
        # measured acoustic support as the target note.
        score = torch.cat((active, score[:, :, 1:] * active), dim=2)
        return score, competitor_count

    def contract(self) -> dict[str, Any]:
        result = super().contract()
        result.update({
            "kind": "active_masked_frozen_core_score_adapter_v1",
            "inactive_score_behavior": "all five score channels exactly zero",
        })
        return result


def initialize_from_aso(
    target: ScoreContextSetOwnership, source_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    target_state = target.state_dict()
    with torch.no_grad():
        for key, value in source_state.items():
            if key == "input.weight":
                if target_state[key].shape[1] != value.shape[1] + target.score_feature_channels:
                    raise ValueError("input feature expansion mismatch")
                target_state[key][:, : value.shape[1]].copy_(value)
                target_state[key][:, value.shape[1] :].zero_()
            else:
                if key not in target_state or target_state[key].shape != value.shape:
                    raise ValueError(f"warm-start state mismatch: {key}")
                target_state[key].copy_(value)
    target.load_state_dict(target_state, strict=True)
    return {
        "old_parameter_slices_bit_exact": all(
            torch.equal(
                target.state_dict()[key][:, : value.shape[1]],
                value.to(target.state_dict()[key].device),
            ) if key == "input.weight" else torch.equal(
                target.state_dict()[key], value.to(target.state_dict()[key].device)
            )
            for key, value in source_state.items()
        ),
        "new_input_slices_exactly_zero": bool(
            torch.count_nonzero(target.state_dict()["input.weight"][:, 11:]) == 0
        ),
    }


def initialize_frozen_core_from_aso(
    target: FrozenCoreScoreContextSetOwnership,
    source_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    target_state = target.state_dict()
    with torch.no_grad():
        for key, value in source_state.items():
            if key not in target_state or target_state[key].shape != value.shape:
                raise ValueError(f"frozen-core state mismatch: {key}")
            target_state[key].copy_(value)
        target_state["score_input.weight"].zero_()
    target.load_state_dict(target_state, strict=True)
    return {
        "base_parameter_tensors_bit_exact": all(
            torch.equal(target.state_dict()[key], value.to(target.state_dict()[key].device))
            for key, value in source_state.items()
        ),
        "score_adapter_exactly_zero": bool(
            torch.count_nonzero(target.state_dict()["score_input.weight"]) == 0
        ),
    }


__all__ = [
    "ScoreContextSetOwnership",
    "FrozenCoreScoreContextSetOwnership",
    "ActiveMaskedScoreAdapterOwnership",
    "initialize_from_aso",
    "initialize_frozen_core_from_aso",
]
