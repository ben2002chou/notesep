#!/usr/bin/env python3
"""Train the joint target/background occurrence TFC-TDF separator."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_losses import dna_residual_losses  # noqa: E402
from model_exp.occurrence_joint_tfc_tdf_separator import (  # noqa: E402
    OccurrenceJointTFCTDFSeparator,
)
import scripts.train_separator as base  # noqa: E402


PRESET = os.environ.get("OCCURRENCE_JOINT_TFC_TDF_PRESET", "30m")


def make_model() -> OccurrenceJointTFCTDFSeparator:
    return OccurrenceJointTFCTDFSeparator.from_preset(PRESET)


def _source_losses(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    complex_normalizer: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    zeros = torch.zeros_like(target)
    return dna_residual_losses(
        estimate,
        target,
        zeros,
        estimate,
        valid_samples=valid_samples,
        complex_normalizer=complex_normalizer,
        **base.ACTIVE_BASE_LOSS_WEIGHTS,
    )


def metrics_from_outputs(
    target_estimate: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    background_target = batch["mixture"] - batch["target"]
    background_estimate = diagnostics["background"]
    target = _source_losses(
        target_estimate, batch["target"], batch["valid_samples"].long()
    )
    background_complex_normalizer = (
        batch["mixture"]
        if bool(batch.get("background_complex_mixture_normalized", False))
        else None
    )
    background = _source_losses(
        background_estimate,
        background_target,
        batch["valid_samples"].long(),
        complex_normalizer=background_complex_normalizer,
    )
    metrics: dict[str, torch.Tensor] = {
        "loss": 0.5 * (target["loss"] + background["loss"]),
        "complex": 0.5 * (target["complex"] + background["complex"]),
        "log_magnitude": 0.5
        * (target["log_magnitude"] + background["log_magnitude"]),
        "waveform_l1": 0.5
        * (target["waveform_l1"] + background["waveform_l1"]),
        "envelope": 0.5 * (target["envelope"] + background["envelope"]),
        "target_complex": target["complex"],
        "target_log_magnitude": target["log_magnitude"],
        "target_waveform_l1": target["waveform_l1"],
        "target_envelope": target["envelope"],
        "background_complex": background["complex"],
        "background_log_magnitude": background["log_magnitude"],
        "background_waveform_l1": background["waveform_l1"],
        "background_envelope": background["envelope"],
    }
    with torch.no_grad():
        support = (
            torch.arange(target_estimate.shape[-1], device=target_estimate.device)[
                None
            ]
            < batch["valid_samples"].long()[:, None]
        )
        target_energy = (batch["target"].square() * support).sum().clamp_min(1e-10)
        background_energy = (background_target.square() * support).sum().clamp_min(
            1e-10
        )
        metrics["target_output_energy_ratio"] = (
            target_estimate.square() * support
        ).sum() / target_energy
        metrics["background_output_energy_ratio"] = (
            background_estimate.square() * support
        ).sum() / background_energy
        metrics["mixture_consistency_max_abs"] = (
            diagnostics["waveform_reconstruction"] - batch["mixture"]
        ).mul(support).abs().max()
        metrics["complex_mixture_consistency_max_abs"] = (
            diagnostics["projected_spectrum_reconstruction"]
            - diagnostics["mixture_spectrum"]
        ).abs().max()
    return metrics


def forward_metrics(
    model: OccurrenceJointTFCTDFSeparator,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    zeros = torch.zeros_like(batch["mixture"])
    target_estimate, diagnostics = model(
        batch["mixture"],
        zeros,
        zeros,
        zeros,
        batch["pitch"].long(),
        batch["valid_samples"].long(),
        batch["note_onset_samples"].long(),
        batch["note_offset_samples"].long(),
    )
    return metrics_from_outputs(target_estimate, diagnostics, batch)


def active_loss_contract() -> dict[str, Any]:
    return {
        "name": "joint_target_background_free_target_v1",
        "task": "direct_occurrence_joint_target_background",
        "source_weighting": {"target": 0.5, "background": 0.5},
        **base.ACTIVE_BASE_LOSS_WEIGHTS,
        "dna_conditioning": False,
        "residual_prediction": False,
        "mixture_consistency": "orthogonal_equal_correction_per_complex_bin",
    }


prototype = make_model()
base.ACTIVE_MODEL_CONTRACT = prototype.contract(PRESET)
base.MODEL_KIND = "tfc_tdf_unet"
base.selected_model = make_model
base.forward_metrics = forward_metrics
base.active_loss_contract = active_loss_contract


if __name__ == "__main__":
    base.main()
