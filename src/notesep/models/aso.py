"""Acoustic-boundary Temporal ASO with a relative ownership update."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .ownership import (
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)


class AcousticMultiplicativeTemporalOwnership(TemporalHybridSetOwnership):
    """Predict bounded log multipliers relative to raw ownership.

    A zero network output preserves the raw magnitude-share prior.  Unlike an
    absolute sparsemax displacement, a weak proposal can change only
    multiplicatively.  The model consumes the measured acoustic endpoint used
    by the frozen occurrence separator, not the nominal MIDI note-off.
    """

    hop_length = 512
    boundary_scale_samples = 2048

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        config = config or TemporalHybridOwnershipConfig(
            feature_set="essential6",
            phase_mode="mixture",
            temporal_blocks=2,
            magnitude_reference="mixture",
            exact_mixture_additivity=True,
        )
        if config.feature_set != "essential6" or config.phase_mode != "mixture":
            raise ValueError(
                "acoustic multiplicative ownership requires essential6 and mixture phase"
            )
        super().__init__(config)
        original = self.input
        replacement = nn.Conv2d(
            original.in_channels + 2,
            original.out_channels,
            original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:, : original.in_channels].copy_(original.weight)
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        self.input = replacement
        self._acoustic_coordinates: torch.Tensor | None = None

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, base = super()._features(mixture, raw, pitches)
        if self._acoustic_coordinates is None:
            raise RuntimeError("acoustic onset/offset coordinates were not supplied")
        coordinates = self._acoustic_coordinates[:, :, :, None, :].expand(
            -1, -1, -1, features.shape[-2], -1
        ).to(features.dtype)
        return torch.cat((features, coordinates), dim=2), base

    def _allocate_ownership(
        self, base: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        # Exponentiated-gradient/mirror-descent update on the simplex.  Exact
        # zero proposals remain impossible owners; all-empty bins remain
        # rejected rather than being assigned arbitrarily.
        work = base.float()
        weights = work * torch.exp(delta.float())
        total = weights.sum(1, keepdim=True)
        ownership = torch.where(
            total > 1e-20,
            weights / total.clamp_min(1e-20),
            torch.zeros_like(weights),
        )
        return ownership.to(base.dtype)

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
    ):
        if onsets is None or acoustic_offsets is None:
            raise ValueError("onsets and measured acoustic offsets are required")
        del midi_offsets
        centers = (
            torch.arange(
                mixture.shape[-1], device=mixture.device, dtype=mixture.real.dtype
            )
            * self.hop_length
        )
        onset_distance = torch.tanh(
            (centers[None, None] - onsets[..., None].to(centers.dtype))
            / self.boundary_scale_samples
        )
        offset_distance = torch.tanh(
            (acoustic_offsets[..., None].to(centers.dtype) - centers[None, None])
            / self.boundary_scale_samples
        )
        self._acoustic_coordinates = torch.stack(
            (onset_distance, offset_distance), dim=2
        )
        try:
            return super().forward(
                mixture,
                anchors,
                raw,
                pitches,
                onsets,
                None,
                acoustic_offsets,
                harmonic_logits,
                percussive_logits,
                hard_phase=hard_phase,
            )
        finally:
            self._acoustic_coordinates = None

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "acoustic_multiplicative_temporal_ownership_v1",
                "features": (
                    "essential6 plus signed onset distance and signed measured "
                    "acoustic-offset distance"
                ),
                "onset_coordinate": "tanh((frame_center-onset)/2048)",
                "offset_coordinate": "tanh((acoustic_offset-frame_center)/2048)",
                "boundary_authority": "measured acoustic endpoint",
                "ownership_update": (
                    "normalize(base_ownership * exp(bounded_learned_log_gain))"
                ),
                "magnitude_allocation": (
                    "relative-multiplicative ownership over the configured shared "
                    "magnitude carrier"
                ),
                "zero_proposal_policy": "cannot acquire ownership",
                "all_empty_policy": "reject bin",
                "excluded_inputs": (
                    "symbolic MIDI offset, lifecycle state, instrument identity, background lane"
                ),
            }
        )
        return value




@torch.no_grad()
def raw_energy_budget_threshold(
    magnitude: torch.Tensor, budget: float = 0.01
) -> torch.Tensor:
    """Return detached per-note thresholds with shape ``[B,K,1,1]``.

    Sort each note's magnitudes and find where their squared sum reaches
    the requested fraction of its total energy. Zero notes receive zero.
    """
    if magnitude.ndim != 4 or not 0 < budget < 1:
        raise ValueError("expected [batch,notes,frequency,time], 0 < budget < 1")
    batch, notes = magnitude.shape[:2]
    flat = magnitude.detach().reshape(batch * notes, -1)
    if flat.shape[-1] == 0:
        raise ValueError("empty time-frequency axes")
    sorted_magnitude = flat.sort(dim=1).values
    cumulative = sorted_magnitude.double().square().cumsum(1)
    total = cumulative[:, -1:]
    index = torch.searchsorted(
        cumulative.contiguous(), (budget * total).contiguous()
    ).clamp_max(flat.shape[1] - 1)
    return sorted_magnitude.gather(1, index).reshape(batch, notes, 1, 1)


def threshold_credibility(
    magnitude: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """Zero below half-threshold, smoothstep to one at the threshold.

    No straight-through estimator is used.  Gradients to the threshold flow
    through the transition band only.  A zero threshold leaves the proposal
    unchanged, including the exact zero magnitude of originally empty bins.
    """
    position = (
        (magnitude - 0.5 * threshold)
        / (0.5 * threshold).clamp_min(1e-30)
    ).clamp(0, 1)
    smooth = position.square() * (3 - 2 * position)
    return torch.where(threshold > 0, smooth, torch.ones_like(magnitude))


class ThresholdPrunedTemporalOwnership(AcousticMultiplicativeTemporalOwnership):
    """Learned-threshold ASO used for the instrument results."""

    def __init__(
        self,
        config: TemporalHybridOwnershipConfig | None = None,
        *,
        arm: str = "learned_gate_aso",
        energy_budget: float = 0.01,
    ) -> None:
        if arm != "learned_gate_aso":
            raise ValueError("expected learned_gate_aso")
        if not 0 < energy_budget < 1:
            raise ValueError("energy budget must be strictly between zero and one")
        super().__init__(config)
        if self.config.magnitude_reference != "mixture":
            raise ValueError("matched pruning experiment requires mixture magnitude")
        if self.config.ownership_trust != 1:
            raise ValueError("matched pruning experiment requires ownership_trust=1")
        original = self.input
        replacement = nn.Conv2d(
            original.in_channels + 1, original.out_channels,
            original.kernel_size, stride=original.stride, padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:, :-1].copy_(original.weight)
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        self.input = replacement
        self.arm = arm
        self.energy_budget = energy_budget
        self.log_threshold_multiplier = nn.Parameter(torch.zeros(()))
        self._pruning_context: dict[str, torch.Tensor] | None = None

    def _features(
        self, mixture: torch.Tensor, raw: torch.Tensor, pitches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, base = super()._features(mixture, raw, pitches)
        if self._pruning_context is None:
            raise RuntimeError("pruning context was not supplied")
        return torch.cat((features, self._pruning_context["gate"][:, :, None]), dim=2), base

    def _allocate_ownership(
        self, base: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        if self._pruning_context is None:
            raise RuntimeError("pruning context was not supplied")
        ctx = self._pruning_context
        # Weight the proposals by their gate and predicted correction.
        weights = ctx["raw_magnitude"].float() * ctx["gate"].float()
        weights = weights * torch.exp(delta.float())
        total = weights.sum(1, keepdim=True)
        active = total > 1e-20
        ownership = torch.where(
            active, weights / total.clamp_min(1e-20), torch.zeros_like(weights)
        ).to(base.dtype)
        ctx["active"] = active
        ctx["ownership"] = ownership
        return ownership

    def _refine_output(
        self, features: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self._pruning_context is None:
            raise RuntimeError("pruning context was not supplied")
        ctx = self._pruning_context
        # If all proposals are gated out, retain their magnitudes with mixture phase.
        mix = ctx["mixture"][:, None]
        phase = torch.polar(torch.ones_like(mix.real), torch.angle(mix))
        return torch.where(
            ctx["active"], ctx["ownership"] * mix,
            ctx["raw_magnitude"] * phase,
        )

    def _predict_magnitude_delta(self, memory: torch.Tensor) -> torch.Tensor:
        return self.config.maximum_simplex_correction * torch.tanh(
            self.magnitude_delta(memory)[:, 0]
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
        hard_phase: bool = True,
        raw_threshold: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        args = (
            mixture, anchors, raw, pitches, onsets, midi_offsets,
            acoustic_offsets, harmonic_logits, percussive_logits,
        )
        if mixture.ndim != 3 or raw.ndim != 4 or raw.shape[0] != mixture.shape[0]:
            raise ValueError("invalid mixture/note-bank geometry")
        if raw.shape[1] < 2 or raw.shape[-2:] != mixture.shape[-2:]:
            raise ValueError("allocator requires at least two aligned notes")
        if onsets is None or acoustic_offsets is None:
            raise ValueError("onsets and measured acoustic offsets are required")
        if self._pruning_context is not None:
            raise RuntimeError("reentrant pruning forward is unsupported")
        magnitude = raw.abs()
        if raw_threshold is None:
            raw_threshold = raw_energy_budget_threshold(magnitude, self.energy_budget)
        if raw_threshold.shape != (*raw.shape[:2], 1, 1):
            raise ValueError("raw_threshold must have shape [B,K,1,1]")
        threshold = raw_threshold.detach().to(magnitude) * self.log_threshold_multiplier.exp()
        gate = threshold_credibility(magnitude, threshold)
        self._pruning_context = {
            "raw_magnitude": magnitude, "mixture": mixture, "gate": gate,
            "active": (magnitude * gate).sum(1, keepdim=True) > 1e-20,
        }
        try:
            output, diagnostics = super().forward(*args, hard_phase=hard_phase)
            allocated = self._pruning_context["ownership"]
            active = self._pruning_context["active"]
            diagnostics["allocated_ownership"] = allocated
            diagnostics["ownership"] = torch.where(
                active, allocated, diagnostics["base_ownership"]
            )
            diagnostics["fallback_mask"] = ~active
            diagnostics.update({
                "pruning_gate": gate,
                "raw_threshold": raw_threshold.detach(),
                "effective_threshold": threshold,
                "threshold_multiplier": self.log_threshold_multiplier.exp(),
                "all_pruned": ~self._pruning_context["active"],
                "all_pruned_fraction": (~self._pruning_context["active"]).float().mean(),
            })
            return output, diagnostics
        finally:
            self._pruning_context = None



__all__ = ["AcousticMultiplicativeTemporalOwnership", "ThresholdPrunedTemporalOwnership", "raw_energy_budget_threshold", "threshold_credibility"]
