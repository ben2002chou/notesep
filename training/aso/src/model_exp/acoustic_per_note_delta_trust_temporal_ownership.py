"""Relative acoustic ASO with learned per-note log-gain trust."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .acoustic_multiplicative_temporal_ownership import (
    AcousticMultiplicativeTemporalOwnership,
)
from .temporal_hybrid_set_ownership import TemporalHybridOwnershipConfig


class AcousticPerNoteDeltaTrustTemporalOwnership(
    AcousticMultiplicativeTemporalOwnership
):
    """Damp each note's proposed log gain before relative normalization.

    The shared 1x1 head is evaluated independently on every note memory.  It
    therefore adds only ``hidden_channels + 1`` parameters while producing a
    distinct trust value for every note/time/frequency cell.  Relative
    multiplicative normalization still preserves structural zeros and the
    source simplex.
    """

    minimum_delta_trust = 0.25

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        super().__init__(config)
        if self.config.ownership_trust != 1.0:
            raise ValueError(
                "per-note delta trust requires ownership_trust=1.0; "
                "the learned head is the only trust mechanism"
            )
        self.delta_trust_head = nn.Conv2d(self.config.hidden_channels, 1, 1)
        nn.init.zeros_(self.delta_trust_head.weight)
        nn.init.zeros_(self.delta_trust_head.bias)

    def _trust_magnitude_delta(
        self,
        features: torch.Tensor,
        memory: torch.Tensor,
        base: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del features, base
        batch, notes, channels, frequency, frames = memory.shape
        trust = self.minimum_delta_trust + (
            1.0 - self.minimum_delta_trust
        ) * torch.sigmoid(
            self.delta_trust_head(memory.flatten(0, 1))
        ).view(batch, notes, frequency, frames)
        return trust * delta, trust

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "acoustic_per_note_delta_trust_temporal_ownership_v1",
                "learned_delta_trust": (
                    "one shared 1x1 head evaluated per note and TF cell; trust "
                    "multiplies bounded log gain before relative normalization"
                ),
                "delta_trust_range": [self.minimum_delta_trust, 1.0],
                "delta_trust_context": (
                    "each note's existing memory after local, temporal, and "
                    "permutation-invariant cross-note mean/max fusion"
                ),
                "delta_trust_parameters": self.config.hidden_channels + 1,
                "ownership_update": (
                    "normalize(base_ownership * "
                    "exp(per_note_TF_trust * bounded_learned_log_gain))"
                ),
                "ownership_sum_preserved_by_normalization": True,
                "zero_proposal_policy": "cannot acquire ownership",
            }
        )
        return value


__all__ = ["AcousticPerNoteDeltaTrustTemporalOwnership"]
