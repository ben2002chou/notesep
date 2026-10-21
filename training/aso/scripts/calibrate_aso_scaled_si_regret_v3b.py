#!/usr/bin/env python3
"""Direct waveform SI-regret calibration of frozen scaled ASO."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import calibrate_aso_scaled_confidence_v1 as base


ENERGY_FLOOR = 1e-8
DISTORTION_FLOOR = 1e-4
REGRET_MARGIN_DB = 0.10
MEAN_REGRET_WEIGHT = 2.0
SQUARED_REGRET_WEIGHT = 0.25


def _stable_log_si_distortion(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_energy = target.square().sum(-1)
    if bool((target_energy <= 1e-12).any()):
        raise ValueError("silent target passed to SI-regret objective")
    estimate_energy = value.square().sum(-1)
    dot = (value * target).sum(-1)
    rho2 = dot.square() / (
        (estimate_energy + ENERGY_FLOOR * target_energy)
        * target_energy
    ).clamp_min(1e-20)
    rho2 = rho2.clamp(0.0, 1.0)
    distortion = (1.0 - rho2 + DISTORTION_FLOOR) / (
        rho2 + DISTORTION_FLOOR
    )
    return torch.log(distortion)


def supervised_nonsilent(record: dict) -> torch.Tensor:
    """Keep scored queries whose waveform reference makes SI-SDR defined."""
    target_energy = record["targets_wave"].float().square().sum(-1)
    active = record["query_valid"] & (target_energy > 1e-12)
    if not bool(active.any()):
        raise ValueError("training scene contains no nonsilent target")
    return active


def _objective(candidate_spec: torch.Tensor, record: dict):
    estimate = base.scaled.istft(candidate_spec[0], record["length"]).float()
    magnitude = base.scaled.istft(record["base"], record["length"]).float()
    valid = supervised_nonsilent(record)
    target = record["targets_wave"].float()[valid]
    candidate_log_d = _stable_log_si_distortion(estimate[valid], target)
    baseline_log_d = _stable_log_si_distortion(magnitude[valid], target).detach()
    margin = REGRET_MARGIN_DB * math.log(10.0) / 10.0
    regret = torch.relu(candidate_log_d - baseline_log_d - margin)
    loss = (
        candidate_log_d.mean()
        + MEAN_REGRET_WEIGHT * regret.mean()
        + SQUARED_REGRET_WEIGHT * regret.square().mean()
    )
    return loss, {
        "mean_log_si_distortion": candidate_log_d.mean(),
        "mean_regret": regret.mean(),
        "mean_squared_regret": regret.square().mean(),
        "maximum_regret": regret.max(),
        "silent_targets_excluded": (
            record["query_valid"].sum() - valid.sum()
        ).to(dtype=loss.dtype),
    }


def training_components(model, record):
    candidate_spec, diagnostics = base.scaled.call(model, record)
    loss, risk = _objective(candidate_spec, record)
    diagnostics = diagnostics | {f"si_regret_{key}": value for key, value in risk.items()}
    return loss, diagnostics


def evaluation_components(diagnostics, record, candidate_spec):
    del diagnostics
    loss, _ = _objective(candidate_spec, record)
    one = loss.new_ones(())
    return loss, one, loss


def representative_schedule(authorities, mode):
    schedule, singleton_ids = base.scaled.make_schedule(authorities, "small_full")
    if mode == "canary":
        return schedule[:320], singleton_ids, None, 16
    return schedule, singleton_ids, None, 0


def objective_progress(initial_loss: float, final_loss: float) -> float:
    # Natural-log distortion units: 0.10 is about 0.43 dB of risk-adjusted
    # objective movement and remains meaningful if the objective is negative.
    return initial_loss - final_loss


def risk_gates(mode, initial, final):
    calibrated = final["comparisons"]["calibrated_minus_magnitude"]
    owner = final["comparisons"]["owner_minus_magnitude"]
    gates = {
        "severe_regressions_vs_magnitude_not_increased": (
            calibrated["losses_gt_3db"] <= owner["losses_gt_3db"]
        ),
        "paired_mean_vs_owner_not_below_minus_0p05db": (
            final["comparisons"]["calibrated_minus_owner"]["mean_db"] >= -0.05
        ),
    }
    if mode == "canary":
        gates["fixed_canary_coverage_261"] = (
            initial["coverage"] == final["coverage"] == 261
        )
        gates["owner_severe_regressions_vs_magnitude_exactly_7"] = (
            owner["losses_gt_3db"] == 7
        )
        gates["severe_regressions_vs_magnitude_reduced_by_one"] = (
            calibrated["losses_gt_3db"] <= owner["losses_gt_3db"] - 1
        )
        gates["ordinary_regressions_vs_owner_lte_10pct"] = (
            final["comparisons"]["calibrated_minus_owner"]["losses_gt_0p1db"]
            <= math.floor(0.10 * final["coverage"])
        )
        for kind in ("fresh", "recorded"):
            deltas = [
                row["calibrated"]["si_sdr_db"] - row["owner"]["si_sdr_db"]
                for row in final["events"]
                if row["dataset_kind"] == kind
            ]
            gates[f"{kind}_median_vs_owner_not_below_minus_0p10db"] = (
                bool(deltas) and float(base.np.median(deltas)) >= -0.10
            )
    if mode == "full":
        gates["severe_regressions_vs_magnitude_reduced_25pct"] = (
            calibrated["losses_gt_3db"] <= 153
        )
    return gates


if __name__ == "__main__":
    base.SCHEMA = "aso_scaled_si_regret_confidence_calibration_v3b"
    base.OBJECTIVE_IMPLEMENTATION_PATH = Path(__file__).resolve()
    base.PREREG_SHA256 = "e6c4a0de8a2449a213525ebb4c2b6c85631d46ee1cc6def83212dcab4e02f23b"
    base.TARGET_DESCRIPTION = (
        "mean stable waveform log SI distortion plus linear and squared "
        "positive regret beyond 0.1 dB versus magnitude partition"
    )
    base.WEIGHT_DESCRIPTION = (
        "all valid per-scene notes; energy floor=1e-8; distortion floor=1e-4; "
        "regret weights 2.0 linear and 0.25 squared"
    )
    base.training_components = training_components
    base.evaluation_components = evaluation_components
    base.schedule_plan = representative_schedule
    base.calibration_progress = objective_progress
    base.CALIBRATION_PROGRESS_DESCRIPTION = (
        "absolute reduction in risk-adjusted natural-log SI-distortion objective"
    )
    base.CALIBRATION_PROGRESS_GATE = 0.10
    base.additional_gates = risk_gates
    raise SystemExit(base.main())
