"""Mature-separator initialized recurrent HPSS refinement.

The base separator always reads the original mixture.  A zero-initialized
sidecar additionally exposes the previous target estimate to each branch, so
fine-tuning can learn one correction step without discarding the mature
ownership/timbre solution at initialization.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .occurrence_dual_hpss_symmetric_alignment import (
    OccurrenceDualHPSSSymmetricAlignment,
)


LOOP_ADAPTER_PREFIXES = (
    "harmonic_branch.loop_condition_adapter.",
    "percussive_branch.loop_condition_adapter.",
)


class OccurrenceDualHPSSLearnedLoop(OccurrenceDualHPSSSymmetricAlignment):
    """Symmetric HPSS separator with a previous-estimate sidecar."""

    def __init__(self, *args: Any, loop_steps: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if int(loop_steps) < 1:
            raise ValueError("loop_steps must be positive")
        self.loop_steps = int(loop_steps)
        for branch in (self.harmonic_branch, self.percussive_branch):
            adapter = nn.Conv2d(
                2 * branch.config.num_subbands,
                branch.config.initial_channels,
                kernel_size=1,
                bias=False,
            )
            nn.init.zeros_(adapter.weight)
            branch.loop_condition_adapter = adapter

    @classmethod
    def from_preset(
        cls,
        preset: str,
        harmonic_gate_mode: str = "none",
        percussive_parameterization: str = "ownership",
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        attention_heads: int = 8,
        harmonic_parameterization: str = "ownership",
        output_carrier_mode: str = "encoder_input",
        complex_mask_mode: str = "bounded",
        loop_steps: int = 4,
    ) -> "OccurrenceDualHPSSLearnedLoop":
        return cls(
            preset,
            harmonic_gate_mode,
            percussive_parameterization,
            conditioning_mode,
            roll_hold_min_ms,
            roll_hold_max_ms,
            attention_heads=attention_heads,
            harmonic_parameterization=harmonic_parameterization,
            output_carrier_mode=output_carrier_mode,
            complex_mask_mode=complex_mask_mode,
            loop_steps=loop_steps,
        )

    def contract(self) -> dict[str, Any]:
        contract = super().contract()
        contract["contract_version"] = max(6, int(contract["contract_version"]))
        contract["architecture"]["learned_loop"] = {
            "type": "zero_initialized_previous_estimate_complex_stft_sidecar",
            "injection": "branch_first_conv_output_residual",
            "condition": "previous_hpss_component_estimate",
            "original_mixture_visible_every_step": True,
            "steps": self.loop_steps,
            "can_add_and_remove_energy": True,
            "can_refine_phase": True,
        }
        contract["parameter_count"] = self.parameter_count()
        return contract


def load_mature_separator_initialization(
    model: OccurrenceDualHPSSLearnedLoop,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Strictly copy a mature symmetric separator into the learned-loop model."""

    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("Initialization checkpoint lacks model state")
    missing, unexpected = model.load_state_dict(source_state, strict=False)
    if unexpected or sorted(missing) != sorted(
        key for key in model.state_dict() if key.startswith(LOOP_ADAPTER_PREFIXES)
    ):
        raise ValueError(
            f"Learned-loop initialization differs: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for key in missing:
        if torch.count_nonzero(model.state_dict()[key]).item() != 0:
            raise ValueError(f"Loop adapter is not zero initialized: {key}")
    return {
        "source_global_step": int(checkpoint.get("global_step", -1)),
        "model_only": True,
        "optimizer_reset": True,
        "accepted_missing_prefixes": list(LOOP_ADAPTER_PREFIXES),
        "accepted_missing_keys": sorted(missing),
        "unexpected_keys": [],
        "initial_output_parity": "exact_when_loop_condition_is_absent_or_adapter_zero",
    }


__all__ = [
    "LOOP_ADAPTER_PREFIXES",
    "OccurrenceDualHPSSLearnedLoop",
    "load_mature_separator_initialization",
]
