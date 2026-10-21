"""Learned, source-permutation-equivariant weights for complete note banks.

The separator still predicts one queried occurrence at a time.  This module is
used only after all score occurrences for a mixture have been queried: it
predicts a positive uncertainty for each occurrence from query-local features,
then uses inverse uncertainty to distribute the *raw* bank-sum error.  There
is no residual source.  The raw sum error remains the training signal; the
projected bank is an exact-consistency diagnostic and a synthetic source-loss
target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class BankUncertaintyConfig:
    hidden_channels: int = 16
    log_variance_min: float = -6.0
    log_variance_max: float = 4.0
    epsilon: float = 1e-8


class LearnedBankUncertainty(nn.Module):
    """Predict one calibrated log variance per independently queried note.

    Features are intentionally query-local and available at inference: note
    estimate RMS, mixture RMS, estimated/mixture ratio, pitch, normalized
    duration, and group cardinality.  A variance cannot simply grow to erase a
    source loss because :func:`heteroscedastic_source_nll` includes ``log var``.
    """

    def __init__(self, config: BankUncertaintyConfig | None = None) -> None:
        super().__init__()
        self.config = config or BankUncertaintyConfig()
        if self.config.hidden_channels < 1 or self.config.epsilon <= 0.0:
            raise ValueError("invalid bank uncertainty configuration")
        self.network = nn.Sequential(
            nn.Linear(6, self.config.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.config.hidden_channels, 1),
        )
        # Start at unit variance, so attaching this module is conservative.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        note_estimates: torch.Tensor,
        mixture: torch.Tensor,
        pitch: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        valid_samples: torch.Tensor,
        note_to_mixture: torch.Tensor,
    ) -> torch.Tensor:
        if note_estimates.ndim != 2 or mixture.ndim != 2:
            raise ValueError("notes and mixtures must be [queries,samples] and [mixtures,samples]")
        queries, samples = note_estimates.shape
        if mixture.shape[1] != samples or pitch.shape != (queries,):
            raise ValueError("bank uncertainty geometry mismatch")
        if onset.shape != (queries,) or offset.shape != (queries,):
            raise ValueError("query bounds must have shape [queries]")
        if valid_samples.shape != (mixture.shape[0],) or note_to_mixture.shape != (queries,):
            raise ValueError("mixture assignment geometry mismatch")
        if torch.any(note_to_mixture < 0) or torch.any(note_to_mixture >= mixture.shape[0]):
            raise ValueError("query has an invalid mixture assignment")
        support = torch.arange(samples, device=note_estimates.device)[None] < valid_samples.index_select(0, note_to_mixture)[:, None]
        note_rms = torch.sqrt((note_estimates.float().square() * support).sum(-1) / support.sum(-1).clamp_min(1) + self.config.epsilon)
        mixture_for_note = mixture.index_select(0, note_to_mixture)
        mixture_rms = torch.sqrt((mixture_for_note.float().square() * support).sum(-1) / support.sum(-1).clamp_min(1) + self.config.epsilon)
        counts = torch.bincount(note_to_mixture, minlength=mixture.shape[0]).index_select(0, note_to_mixture).float()
        duration = (offset - onset).float().clamp_min(1.0) / valid_samples.index_select(0, note_to_mixture).float().clamp_min(1.0)
        features = torch.stack((
            note_rms.log(), mixture_rms.log(), (note_rms / mixture_rms).log(),
            pitch.float().div(127.0), duration, counts.log(),
        ), dim=-1)
        return self.network(features).squeeze(-1).clamp(self.config.log_variance_min, self.config.log_variance_max)

    def contract(self) -> dict[str, Any]:
        return {"kind": "learned_note_bank_uncertainty_v1", **asdict(self.config), "features": ["note_rms", "mixture_rms", "relative_rms", "pitch", "duration", "bank_cardinality"]}


def uncertainty_weighted_bank_projection(
    mixture: torch.Tensor,
    note_estimates: torch.Tensor,
    log_variance: torch.Tensor,
    note_to_mixture: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return an exactly additive note bank, with learned inverse-variance shares."""
    if mixture.ndim != 2 or note_estimates.ndim != 2:
        raise ValueError("mixture and note estimates must be rank two")
    queries, samples = note_estimates.shape
    if mixture.shape[1] != samples or log_variance.shape != (queries,):
        raise ValueError("uncertainty projection geometry mismatch")
    summed = torch.zeros_like(mixture).index_add(0, note_to_mixture, note_estimates)
    residual = mixture - summed
    local_energy = note_estimates.float().square().add(epsilon)
    precision = torch.exp(-log_variance).to(local_energy.dtype)[:, None]
    unnormalized = local_energy * precision
    denominator = torch.zeros_like(mixture).index_add(0, note_to_mixture, unnormalized).index_select(0, note_to_mixture)
    weights = unnormalized / denominator.clamp_min(
        torch.finfo(unnormalized.dtype).tiny
    )
    correction = weights * residual.index_select(0, note_to_mixture)
    projected = note_estimates + correction
    projected_sum = torch.zeros_like(mixture).index_add(0, note_to_mixture, projected)
    support = torch.arange(samples, device=mixture.device)[None] < valid_samples[:, None]
    # Symmetric final correction removes float accumulation noise without a privileged note.
    rounding = (mixture - projected_sum) * support.to(mixture.dtype)
    projected = projected + weights * rounding.index_select(0, note_to_mixture)
    final = torch.zeros_like(mixture).index_add(0, note_to_mixture, projected)
    return projected, {
        "raw_bank_residual_rms": torch.sqrt((residual.float().square() * support).sum() / support.sum().clamp_min(1)),
        "projected_bank_error_max_abs": ((final - mixture) * support).abs().amax(),
        "mean_max_assignment_weight": weights.amax(dim=0).mean(),
        "mean_log_variance": log_variance.mean(),
    }


def heteroscedastic_source_nll(
    estimate: torch.Tensor, target: torch.Tensor, log_variance: torch.Tensor
) -> torch.Tensor:
    """Robustly calibrate uncertainty without a high-variance escape.

    The Student-t-like ``0.5*s + log1p(error*exp(-s))`` objective has bounded
    influence for nearly silent or badly mismatched stems, while diverging as
    ``s`` approaches either extreme.  This avoids the very large gradients of
    a Gaussian relative-MSE likelihood on quiet note tails.
    """
    if estimate.shape != target.shape or estimate.ndim != 2 or log_variance.shape != (estimate.shape[0],):
        raise ValueError("heteroscedastic source loss geometry mismatch")
    relative_mse = (estimate.float() - target.float()).square().mean(-1) / target.float().square().mean(-1).clamp_min(1e-8)
    return (
        0.5 * log_variance
        + torch.log1p(torch.exp(-log_variance) * relative_mse)
    ).mean()


__all__ = ["BankUncertaintyConfig", "LearnedBankUncertainty", "heteroscedastic_source_nll", "uncertainty_weighted_bank_projection"]
