"""Sparse score-conditioned merger for phase-preserving acoustic atoms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class OccurrenceAtomicMergerConfig:
    input_features: int = 19
    hidden_channels: int = 32
    blocks: int = 3


class _LocalBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(self.depthwise(values))
        return values + self.activation(self.norm(update))


class OccurrenceAtomicMerger(nn.Module):
    """Classify a fixed atom grid into active note rows plus background.

    Expensive audio decomposition is independent of the number of logical MIDI
    lanes.  This head is evaluated only for the J active score occurrences and
    one background row.  Shared local processing is O(JFT), and source
    competition is the final softmax rather than J independent waveforms.
    """

    def __init__(self, config: OccurrenceAtomicMergerConfig | None = None) -> None:
        super().__init__()
        self.config = config or OccurrenceAtomicMergerConfig()
        if (
            self.config.input_features < 1
            or self.config.hidden_channels < 4
            or self.config.blocks < 1
        ):
            raise ValueError("invalid atomic merger configuration")
        self.input = nn.Sequential(
            nn.Conv2d(
                self.config.input_features,
                self.config.hidden_channels,
                kernel_size=1,
            ),
            nn.GroupNorm(1, self.config.hidden_channels),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            *(_LocalBlock(self.config.hidden_channels) for _ in range(self.config.blocks))
        )
        # Local evidence plus source-set mean and max.  This keeps the model
        # permutation equivariant while allowing every active note to affect
        # every other note's claim without quadratic source attention.
        self.claim = nn.Sequential(
            nn.Conv2d(3 * self.config.hidden_channels, self.config.hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(self.config.hidden_channels, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.config.input_features:
            raise ValueError("features must have shape [sources, features, F, T]")
        if features.shape[0] < 2:
            raise ValueError("atomic merging requires at least one note and background")
        local = self.local(self.input(features))
        mean = local.mean(dim=0, keepdim=True).expand_as(local)
        maximum = local.amax(dim=0, keepdim=True).expand_as(local)
        return self.claim(torch.cat((local, mean, maximum), dim=1))[:, 0]

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "occurrence_sparse_atomic_merger_v1",
            **asdict(self.config),
            "atomizer": "fixed_phase_preserving_complex_STFT_tracklets",
            "outputs": "one_claim_per_active_note_plus_background",
            "assignment": "strict_source_axis_argmax_at_inference",
            "logical_128_lanes": "scatter_active_occurrences_then_leave_all_other_lanes_silent",
            "source_complexity": "O(JFT)",
            "physical_128_waveforms": False,
            "codec": False,
        }


__all__ = ["OccurrenceAtomicMerger", "OccurrenceAtomicMergerConfig"]
