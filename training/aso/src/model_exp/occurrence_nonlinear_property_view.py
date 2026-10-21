"""Mixture-consistent natural-channel views for real-audio property SSL.

The view never creates a learnable residual.  A compressor-like shared gain
envelope is exactly additive across a note and its complement.  For mild
saturation, the two-player Shapley value allocates the interaction term between
the queried note and its complement, so those two attributed signals sum to the
processed mixture exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_pair(mixture: torch.Tensor, target: torch.Tensor) -> None:
    if mixture.ndim != 2 or target.shape != mixture.shape:
        raise ValueError("mixture and target must have matching [batch, samples] shape")
    if not torch.isfinite(mixture).all() or not torch.isfinite(target).all():
        raise FloatingPointError("nonlinear property view requires finite waveforms")


def causal_shared_gain_envelope(
    mixture: torch.Tensor,
    *,
    strength: float,
    window_samples: int,
) -> torch.Tensor:
    """Return a causal, mixture-derived compressor envelope in ``(0, 1]``."""

    if strength < 0.0:
        raise ValueError("shared dynamics strength must be nonnegative")
    if window_samples < 1:
        raise ValueError("shared dynamics window must be positive")
    if strength == 0.0:
        return torch.ones_like(mixture)
    kernel = torch.ones(
        (1, 1, int(window_samples)), device=mixture.device, dtype=torch.float32
    ) / float(window_samples)
    level = F.conv1d(
        F.pad(mixture.float().abs().unsqueeze(1), (int(window_samples) - 1, 0)),
        kernel,
    ).squeeze(1)
    return (1.0 + float(strength) * level.square()).rsqrt().to(mixture.dtype)


def soft_saturate(values: torch.Tensor, drive: float) -> torch.Tensor:
    """A peak-preserving odd soft clip; zero drive is the identity."""

    if drive < 0.0:
        raise ValueError("saturation drive must be nonnegative")
    if drive == 0.0:
        return values
    scale = torch.tanh(torch.tensor(float(drive), device=values.device)).to(values.dtype)
    return torch.tanh(values * float(drive)) / scale


def apply_nonlinear_property_view(
    mixture: torch.Tensor,
    target: torch.Tensor,
    *,
    shared_dynamics_strength: float = 0.0,
    shared_dynamics_window_samples: int = 1,
    saturation_drive: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Process a mixture and attribute it between target and complement.

    The returned ``processed_target + processed_complement`` equals
    ``processed_mixture`` by construction.  The saturation contribution uses
    the exact two-player Shapley value, avoiding an unowned interaction source.
    """

    _check_pair(mixture, target)
    envelope = causal_shared_gain_envelope(
        mixture,
        strength=shared_dynamics_strength,
        window_samples=shared_dynamics_window_samples,
    )
    envelope_mixture = envelope * mixture
    envelope_target = envelope * target
    envelope_complement = envelope * (mixture - target)
    if saturation_drive == 0.0:
        processed_mixture = envelope_mixture
        processed_target = envelope_target
    else:
        processed_mixture = soft_saturate(envelope_mixture, saturation_drive)
        saturated_target = soft_saturate(envelope_target, saturation_drive)
        saturated_complement = soft_saturate(
            envelope_complement, saturation_drive
        )
        processed_target = 0.5 * (
            saturated_target + processed_mixture - saturated_complement
        )
    processed_complement = processed_mixture - processed_target
    diagnostics = {
        "shared_gain_mean": envelope.float().mean(),
        "shared_gain_min": envelope.float().amin(),
        "saturation_drive": mixture.new_tensor(float(saturation_drive)),
        "mixture_consistency_max_abs": (
            processed_mixture - (processed_target + processed_complement)
        ).abs().amax(),
    }
    return processed_mixture, processed_target, processed_complement, diagnostics
