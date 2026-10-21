"""Relative acoustic ASO with one learned shared trust value per TF cell."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .acoustic_multiplicative_temporal_ownership import (
    AcousticMultiplicativeTemporalOwnership,
)
from .temporal_hybrid_set_ownership import TemporalHybridOwnershipConfig


class AcousticAdaptiveTrustTemporalOwnership(
    AcousticMultiplicativeTemporalOwnership
):
    """Learn when to trust a joint ownership transfer without per-note gating.

    A single trust scalar is shared by every note at a time-frequency cell, so
    interpolating between the base and learned ownership preserves the source
    simplex exactly. The trust head sees the permutation-invariant mean of the
    existing note memories and adds only 25 parameters at the default width.
    """

    minimum_trust = 0.25

    def __init__(self, config: TemporalHybridOwnershipConfig | None = None) -> None:
        super().__init__(config)
        if self.config.ownership_trust != 1.0:
            raise ValueError(
                "learned shared TF trust requires ownership_trust=1.0; "
                "the learned head supplies the only trust value"
            )
        self.trust_head = nn.Conv2d(self.config.hidden_channels, 1, 1)
        nn.init.zeros_(self.trust_head.weight)
        nn.init.zeros_(self.trust_head.bias)

    def _blend_ownership(
        self,
        features: torch.Tensor,
        memory: torch.Tensor,
        base: torch.Tensor,
        learned: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del features
        pooled = memory.mean(1)
        trust = self.minimum_trust + (1 - self.minimum_trust) * torch.sigmoid(
            self.trust_head(pooled)
        )
        return base + trust * (learned - base), trust

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "kind": "acoustic_adaptive_trust_temporal_ownership_v1",
                "learned_ownership_trust": (
                    "one shared learned TF scalar applied to all note ownership "
                    "displacements"
                ),
                "ownership_trust_range": [self.minimum_trust, 1.0],
                "ownership_trust_context": (
                    "permutation-invariant mean of existing cross-note TF memory"
                ),
                "ownership_trust_parameters": self.config.hidden_channels + 1,
                "ownership_sum_preserved_by_trust": True,
            }
        )
        return value


__all__ = ["AcousticAdaptiveTrustTemporalOwnership"]
