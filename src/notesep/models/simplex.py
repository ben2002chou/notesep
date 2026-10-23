"""Projection used by the temporal ownership model."""

from __future__ import annotations
import torch


def sparsemax_notes(values: torch.Tensor) -> torch.Tensor:
    """Project the note axis of [B,K,F,T] values onto the simplex."""
    work = values.float()
    ordered = work.sort(dim=1, descending=True).values
    cumulative = ordered.cumsum(dim=1)
    rank = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=work.dtype)[
        None, :, None, None
    ]
    support = (1.0 + rank * ordered > cumulative).sum(1, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(1, support - 1) - 1.0) / support
    projected = (work - tau).clamp_min(0.0)
    return (projected / projected.sum(1, keepdim=True).clamp_min(1e-12)).to(
        values.dtype
    )
