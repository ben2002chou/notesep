"""Score-only directional policy for below-F0 harmonic input gating."""

from __future__ import annotations

from typing import Any

import torch


def lower_member_exact_octave_gate(
    batch: dict[str, Any],
    *,
    minimum_overlap_samples: int = 1280,
    minimum_overlap_fraction: float = 0.25,
) -> torch.Tensor:
    """Return one iff the query is the lower member of an overlapping octave."""

    required = (
        "full_score_pitches", "full_score_onsets", "full_score_offsets",
        "full_score_valid", "full_score_target",
    )
    if any(key not in batch for key in required):
        raise ValueError("directional selective gate requires complete score tensors")
    if minimum_overlap_samples < 1 or not 0 < minimum_overlap_fraction <= 1:
        raise ValueError("invalid directional selective-gate overlap threshold")
    query_pitch = batch["pitch"].long()[:, None]
    query_onset = batch["note_onset_samples"].long()[:, None]
    query_offset = batch["note_offset_samples"].long()[:, None]
    onset = batch["full_score_onsets"].long()
    offset = batch["full_score_offsets"].long()
    overlap = (
        torch.minimum(query_offset, offset) - torch.maximum(query_onset, onset)
    ).clamp_min(0)
    shorter = torch.minimum(query_offset - query_onset, offset - onset).clamp_min(1)
    eligible = (
        batch["full_score_valid"].bool()
        & ~batch["full_score_target"].bool()
        & (batch["full_score_pitches"].long() == query_pitch + 12)
        & (overlap >= minimum_overlap_samples)
        & (overlap.float() >= minimum_overlap_fraction * shorter.float())
    )
    return eligible.any(dim=1).to(batch["mixture"].dtype)
