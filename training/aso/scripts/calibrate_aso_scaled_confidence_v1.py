#!/usr/bin/env python3
"""Calibrate only the confidence gate of the frozen scaled small ASO model."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import train_aso_scaled_source_diverse_v1 as scaled  # noqa: E402
from model_exp.adaptive_set_ownership import AdaptiveSetOwnership  # noqa: E402
from model_exp.adaptive_sparse_ownership import AdaptiveSparseOwnershipConfig  # noqa: E402


SEED = 20260907
SCHEMA = "aso_scaled_confidence_calibration_v1"
OWNER_SHA256 = "8d41713af8174254c77adafa8d8e225d7de48b0cfda3ada074e97d30c3284ac5"
PREREG_SHA256 = "043ec3591b5f7bd00216c0412c7f2fc41ab76f75d40a58c3c8eecce70e7933e0"
TARGET_DESCRIPTION = "per-bin oracle convex interpolation from magnitude ownership to frozen sparse ownership"
WEIGHT_DESCRIPTION = "mixture squared magnitude; q==b bins excluded"
STOP = False
CALIBRATION_PROGRESS_DESCRIPTION = "relative calibration-loss reduction"
CALIBRATION_PROGRESS_GATE = 0.10
BASE_RUNNER_PATH = Path(__file__).resolve()
OBJECTIVE_IMPLEMENTATION_PATH = BASE_RUNNER_PATH
DERIVED_EXPERIMENT_PATH = BASE_RUNNER_PATH
EXPECTED_TRAINABLE_NAMES: tuple[str, ...] | None = (
    "confidence.weight", "confidence.bias"
)
EXPECTED_TRAINABLE_PARAMETERS = 17
REQUIRE_PARAMETER_UPDATE = False
OPTIMIZER_INITIALIZATION_DESCRIPTION = "fresh AdamW state"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def configure_trainable(model: AdaptiveSetOwnership) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.confidence.parameters():
        parameter.requires_grad_(True)


def protected_state_key(key: str) -> bool:
    return not key.startswith("confidence.")


def initialize_optimizer(
    _optimizer: torch.optim.Optimizer, _model: AdaptiveSetOwnership
) -> None:
    """Optional exact-state continuation hook for derived experiments."""


def build_full_gates(
    common: dict[str, bool],
    initial: dict[str, Any],
    final: dict[str, Any],
    coverage_authority: dict[str, Any],
) -> dict[str, bool]:
    comparisons = final["comparisons"]
    owner_gain = comparisons["owner_minus_magnitude"]["median_db"]
    calibrated_gain = comparisons["calibrated_minus_magnitude"]["median_db"]
    return common | {
        "audited_coverage_9592": final["coverage"] == int(coverage_authority["expected_scored_coverage"]) == 9592,
        "median_within_0p10db_of_owner": final["summary"]["calibrated"]["si_sdr_db"]["median"] >= final["summary"]["owner"]["si_sdr_db"]["median"] - 0.10,
        "retains_80pct_owner_gain_over_magnitude": calibrated_gain >= 0.8 * owner_gain,
        "severe_losses_vs_magnitude_lte153": comparisons["calibrated_minus_magnitude"]["losses_gt_3db"] <= 153,
        "median_off_e_no_worse_than_owner": final["summary"]["calibrated"]["off_e"]["median"] <= final["summary"]["owner"]["off_e"]["median"],
    }


def build_canary_gates(
    common: dict[str, bool],
    initial: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, bool]:
    """Default legacy canary gates; derived experiments may replace these."""
    return common | {
        "frozen_development_coverage_unchanged": (
            final["coverage"] == initial["coverage"]
        ),
        "median_within_0p05db_of_owner": (
            final["summary"]["calibrated"]["si_sdr_db"]["median"]
            >= final["summary"]["owner"]["si_sdr_db"]["median"] - 0.05
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def load_owner(path: Path, device: torch.device) -> tuple[AdaptiveSetOwnership, dict[str, Any]]:
    scaled.forbid_test_path(path)
    if sha256(path) != OWNER_SHA256:
        raise ValueError("selected small ASO checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location=device, weights_only=False)
    contract = payload.get("contract", {})
    if contract.get("arm") != "small_full" or contract.get("test_data_used") is not False:
        raise ValueError("owner checkpoint contract mismatch")
    allowed = {item.name for item in fields(AdaptiveSparseOwnershipConfig)}
    config = AdaptiveSparseOwnershipConfig(
        **{key: value for key, value in contract["config"].items() if key in allowed}
    )
    model = AdaptiveSetOwnership(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, contract


def calibration_components(
    diagnostics: dict[str, torch.Tensor], targets: torch.Tensor, mixture: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = diagnostics["base_ownership"][0]
    direction = diagnostics["sparse_ownership"][0] - base
    denominator = direction.square().sum(0)
    target_magnitude = targets.abs()
    target_share = target_magnitude / target_magnitude.sum(0, keepdim=True).clamp_min(1e-8)
    oracle = ((direction * (target_share - base)).sum(0) / denominator.clamp_min(1e-12)).clamp(0, 1)
    eligible = denominator > 1e-8
    weight = mixture.abs().square() * eligible
    prediction = diagnostics["confidence"][0, 0]
    numerator = ((prediction - oracle).square() * weight).sum()
    weight_sum = weight.sum()
    return numerator, weight_sum, numerator / weight_sum.clamp_min(1e-12)


def training_components(
    model: AdaptiveSetOwnership, record: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the default calibration loss and model diagnostics.

    Kept as a small override point so task-aligned calibration experiments can
    reuse the frozen loading, scheduling, checkpointing, and validation
    machinery without changing the historical per-bin calibration objective.
    """
    _, diagnostics = scaled.call(model, record)
    _, _, loss = calibration_components(
        diagnostics, record["targets"], record["mixture"]
    )
    return loss, diagnostics


def evaluation_components(
    diagnostics: dict[str, torch.Tensor],
    record: dict[str, Any],
    _candidate_spec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return default held-development calibration sufficient statistics."""
    return calibration_components(
        diagnostics, record["targets"], record["mixture"]
    )


def schedule_plan(authorities: scaled.Authorities, mode: str):
    schedule, singleton_ids = scaled.make_schedule(authorities, "small_full")
    canary_records = None
    maximum = 0
    if mode == "canary":
        schedule = [("fresh", 0), ("recorded", 0)] * 100
        canary_records = {
            kind: authorities.record(kind, "train", 0, torch.device("cuda"))
            for kind in ("fresh", "recorded")
        }
        maximum = 2
    return schedule, singleton_ids, canary_records, maximum


def additional_gates(
    _mode: str, _initial: dict[str, Any], _final: dict[str, Any]
) -> dict[str, bool]:
    """Optional task-specific scientific gates for derived calibrators."""
    return {}


def calibration_progress(initial_loss: float, final_loss: float) -> float:
    """Return positive progress for the default positive calibration loss."""
    return 1.0 - final_loss / max(initial_loss, 1e-12)


def summarize(events: list[dict[str, Any]], method: str) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "median": float(np.median([row[method][metric] for row in events])),
            "mean": float(np.mean([row[method][metric] for row in events])),
        }
        for metric in ("si_sdr_db", "off_e", "gain", "scale_invariant_distortion")
    }


def compare(events: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    delta = np.asarray([row[left]["si_sdr_db"] - row[right]["si_sdr_db"] for row in events])
    return {
        "median_db": float(np.median(delta)),
        "mean_db": float(np.mean(delta)),
        "losses_gt_0p1db": int((delta < -0.1).sum()),
        "losses_gt_1db": int((delta < -1.0).sum()),
        "losses_gt_3db": int((delta < -3.0).sum()),
        "wins_gt_0p1db": int((delta > 0.1).sum()),
        "wins_gt_1db": int((delta > 1.0).sum()),
        "wins_gt_3db": int((delta > 3.0).sum()),
    }


@torch.no_grad()
def evaluate(
    calibrated: AdaptiveSetOwnership,
    owner: AdaptiveSetOwnership,
    authorities: scaled.Authorities,
    maximum: int,
) -> dict[str, Any]:
    calibrated.eval(); owner.eval()
    events: list[dict[str, Any]] = []
    closure = 0.0
    calibration_numerator = 0.0
    calibration_weight = 0.0
    silent_targets_excluded = 0
    for kind in ("fresh", "recorded"):
        count = len(authorities.ids(kind, "validation"))
        if maximum:
            count = min(count, maximum)
        for index in range(count):
            record = authorities.record(kind, "validation", index, torch.device("cuda"))
            if record["base"].shape[0] == 1:
                # A singleton has no ownership decision: the simplex has one
                # feasible member. Preserve it as an exact identity case in
                # reported coverage, but do not ask a K>=2 allocator to score
                # or learn from it.
                candidate_spec = record["base"][None]
                owner_spec = record["base"][None]
                diagnostics = {"sum_error_max_abs": 0.0}
                owner_diagnostics = {"sum_error_max_abs": 0.0}
            else:
                candidate_spec, diagnostics = scaled.call(calibrated, record)
                owner_spec, owner_diagnostics = scaled.call(owner, record)
                numerator, weight, _ = evaluation_components(
                    diagnostics, record, candidate_spec
                )
                calibration_numerator += float(numerator)
                calibration_weight += float(weight)
            candidate_wave = scaled.istft(candidate_spec[0], record["length"]).float()
            owner_wave = scaled.istft(owner_spec[0], record["length"]).float()
            magnitude_wave = scaled.istft(record["base"], record["length"]).float()
            target_energy = record["targets_wave"].double().square().sum(-1)
            silent_targets_excluded += int(
                (record["query_valid"] & (target_energy <= 1e-12)).sum()
            )
            scored_notes = torch.where(
                record["query_valid"] & (target_energy > 1e-12)
            )[0]
            if not len(scored_notes):
                continue
            values = {
                "calibrated": scaled.metrics(
                    record["targets_wave"][scored_notes],
                    candidate_wave[scored_notes],
                ),
                "owner": scaled.metrics(
                    record["targets_wave"][scored_notes], owner_wave[scored_notes]
                ),
                "magnitude": scaled.metrics(
                    record["targets_wave"][scored_notes],
                    magnitude_wave[scored_notes],
                ),
            }
            for local_note, note in enumerate(scored_notes.tolist()):
                events.append({
                    "dataset_kind": kind,
                    "mixture_id": record["mixture_id"],
                    "note": int(note),
                    **{
                        method: {
                            key: float(metric[key][local_note]) for key in metric
                        }
                        for method, metric in values.items()
                    },
                })
            closure = max(
                closure,
                float(diagnostics["sum_error_max_abs"]),
                float(owner_diagnostics["sum_error_max_abs"]),
            )
    if not events or calibration_weight <= 0:
        raise ValueError("empty calibration development evaluation")
    return {
        "coverage": len(events),
        "events": events,
        "summary": {method: summarize(events, method) for method in ("calibrated", "owner", "magnitude")},
        "comparisons": {
            "calibrated_minus_owner": compare(events, "calibrated", "owner"),
            "calibrated_minus_magnitude": compare(events, "calibrated", "magnitude"),
            "owner_minus_magnitude": compare(events, "owner", "magnitude"),
        },
        "calibration_loss": calibration_numerator / calibration_weight,
        "maximum_spectral_closure_abs": closure,
        "silent_targets_excluded": silent_targets_excluded,
    }


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--fresh-index", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--coverage-authority", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA compute required")
    for path in (
        args.owner_checkpoint, args.fresh_index, args.recorded_root, args.proposal_root,
        args.coverage_authority, args.preregister, args.output,
    ):
        scaled.forbid_test_path(path)
    if sha256(args.preregister) != PREREG_SHA256:
        raise ValueError("preregistration SHA-256 mismatch")
    if args.output.exists() and not (args.output / "RESUME.pt").exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGUSR1, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cache = scaled.ProposalCache(args.proposal_root)
    authorities = scaled.Authorities(args.fresh_index, args.recorded_root, cache)
    calibrated, owner_contract = load_owner(args.owner_checkpoint, device)
    owner, _ = load_owner(args.owner_checkpoint, device)
    owner.eval()
    initial_state_bit_exact = all(
        torch.equal(value, owner.state_dict()[key])
        for key, value in calibrated.state_dict().items()
    )
    if not initial_state_bit_exact:
        raise ValueError("candidate and frozen owner differ before calibration")
    configure_trainable(calibrated)
    frozen_before = {
        key: value.detach().cpu().clone()
        for key, value in calibrated.state_dict().items()
        if protected_state_key(key)
    }
    trainable_names = [name for name, value in calibrated.named_parameters() if value.requires_grad]
    trainable_parameters = sum(value.numel() for value in calibrated.parameters() if value.requires_grad)
    if (
        (EXPECTED_TRAINABLE_NAMES is not None and tuple(trainable_names) != EXPECTED_TRAINABLE_NAMES)
        or trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise ValueError(f"unexpected confidence parameter contract: {trainable_names}, {trainable_parameters}")
    optimized_parameters = [
        value for value in calibrated.parameters() if value.requires_grad
    ]
    schedule, singleton_ids, canary_records, maximum = schedule_plan(authorities, args.mode)
    coverage_authority = json.loads(args.coverage_authority.read_text(encoding="utf-8"))
    if (
        coverage_authority.get("status") != "passed"
        or coverage_authority.get("test_data_used") is not False
        or not all(coverage_authority.get("gates", {}).values())
    ):
        raise ValueError("development coverage authority failed")
    contract = {
        "schema": SCHEMA,
        "mode": args.mode,
        "seed": SEED,
        "steps": len(schedule),
        "owner_checkpoint": str(args.owner_checkpoint.resolve()),
        "owner_checkpoint_sha256": OWNER_SHA256,
        "owner_contract": owner_contract,
        "trainable_names": trainable_names,
        "trainable_parameters": trainable_parameters,
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "require_parameter_update": REQUIRE_PARAMETER_UPDATE,
        "initial_candidate_owner_state_bit_exact": initial_state_bit_exact,
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": 1.0,
        },
        "optimizer_initialization": OPTIMIZER_INITIALIZATION_DESCRIPTION,
        "checkpoint_policy": (
            "resume plus named 25/50/75/100 percent states; final selection "
            "uses held development only"
        ),
        "target": TARGET_DESCRIPTION,
        "weighting": WEIGHT_DESCRIPTION,
        "singleton_identity_rule": {"count": len(singleton_ids), "mixture_ids": singleton_ids},
        "fresh_index_sha256": scaled.FRESH_SHA,
        "recorded_frozen_sha256": scaled.RECORDED_SHA,
        "proposal_validation_sha256": cache.validation_sha256,
        "coverage_authority_sha256": sha256(args.coverage_authority),
        "preregister_sha256": PREREG_SHA256,
        "trainer_sha256": sha256(OBJECTIVE_IMPLEMENTATION_PATH),
        "objective_implementation_path": str(OBJECTIVE_IMPLEMENTATION_PATH),
        "objective_implementation_sha256": sha256(OBJECTIVE_IMPLEMENTATION_PATH),
        "base_runner_path": str(BASE_RUNNER_PATH),
        "base_runner_sha256": sha256(BASE_RUNNER_PATH),
        "derived_experiment_path": str(DERIVED_EXPERIMENT_PATH),
        "derived_experiment_sha256": sha256(DERIVED_EXPERIMENT_PATH),
        "model_sha256": sha256(ROOT / "src/model_exp/adaptive_set_ownership.py"),
        "base_model_sha256": sha256(
            ROOT / "src/model_exp/adaptive_sparse_ownership.py"
        ),
        "test_authority_forbidden_sha256": scaled.TEST_SHA,
        "test_data_used": False,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    contract_path = args.output / "CONTRACT.json"
    if contract_path.exists() and contract_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("resume contract mismatch")
    if not contract_path.exists():
        contract_path.write_text(encoded, encoding="utf-8")

    initial_path = args.output / "initial_development.json"
    if initial_path.exists():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
    else:
        initial = evaluate(calibrated, owner, authorities, maximum)
        initial_path.write_text(json.dumps(initial, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    optimizer = torch.optim.AdamW(
        optimized_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initialize_optimizer(optimizer, calibrated)
    start_step = 0
    history: list[dict[str, Any]] = []
    resume = args.output / "RESUME.pt"
    if resume.exists():
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("contract") != contract:
            raise ValueError("resume payload contract mismatch")
        calibrated.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        history = list(payload["history"])
    started = time.monotonic()
    try:
        for position in range(start_step, len(schedule)):
            kind, index = schedule[position]
            record = canary_records[kind] if canary_records is not None else authorities.record(kind, "train", index, device)
            calibrated.train(); optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = training_components(calibrated, record)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite calibration loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
            if not bool(torch.isfinite(gradient)):
                raise ValueError("nonfinite confidence gradient")
            optimizer.step()
            done = position + 1
            silent_excluded = int(
                diagnostics.get("si_regret_silent_targets_excluded", loss.new_zeros(())).detach()
            )
            if done == 1 or done % 100 == 0 or done == len(schedule) or silent_excluded:
                row = {
                    "step": done, "kind": kind, "loss": float(loss.detach()),
                    "gradient_norm": float(gradient.detach()),
                    "mean_confidence": float(diagnostics["mean_confidence"].detach()),
                    "silent_targets_excluded": silent_excluded,
                }
                history.append(row); print(json.dumps(row), flush=True)
            if done in {len(schedule) // 4, len(schedule) // 2, 3 * len(schedule) // 4, len(schedule)} or STOP:
                checkpoint_payload = {
                    "schema": SCHEMA, "contract": contract, "step": done,
                    "model": calibrated.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                }
                atomic_torch(resume, checkpoint_payload)
                if not STOP:
                    atomic_torch(
                        args.output / f"checkpoint-step-{done:06d}.pt",
                        checkpoint_payload,
                    )
            if STOP:
                return 75

        final = evaluate(calibrated, owner, authorities, maximum)
        frozen_exact = all(
            torch.equal(value, calibrated.state_dict()[key].detach().cpu())
            for key, value in frozen_before.items()
        )
        any_trainable_changed = any(
            not torch.equal(
                owner.state_dict()[key].detach().cpu(),
                calibrated.state_dict()[key].detach().cpu(),
            )
            for key in trainable_names
        )
        reduction = calibration_progress(
            float(initial["calibration_loss"]), float(final["calibration_loss"])
        )
        checkpoint_path = args.output / f"checkpoint-step-{len(schedule):06d}.pt"
        atomic_torch(checkpoint_path, {
            "schema": SCHEMA, "contract": contract, "step": len(schedule),
            "model": calibrated.state_dict(), "history": history,
        })
        common = {
            "trainable_parameter_count_expected": (
                trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS
            ),
            "initial_candidate_owner_state_bit_exact": initial_state_bit_exact,
            "initial_candidate_owner_metrics_exact": all(
                row["calibrated"] == row["owner"] for row in initial["events"]
            ),
            "nonconfidence_parameters_bit_exact": frozen_exact,
            "calibration_progress_gate": reduction >= CALIBRATION_PROGRESS_GATE,
            "spectral_closure_1e_minus4": final["maximum_spectral_closure_abs"] <= 1e-4,
            "all_metrics_finite": all(
                np.isfinite(row[method][metric])
                for row in final["events"]
                for method in ("calibrated", "owner", "magnitude")
                for metric in ("si_sdr_db", "off_e", "gain", "scale_invariant_distortion")
            ),
            "all_parameters_finite": all(
                bool(torch.isfinite(value).all())
                for value in calibrated.state_dict().values()
            ),
            "required_parameter_update_observed": (
                any_trainable_changed or not REQUIRE_PARAMETER_UPDATE
            ),
        }
        if args.mode == "canary":
            gates = build_canary_gates(common, initial, final)
        else:
            gates = build_full_gates(common, initial, final, coverage_authority)
        gates.update(additional_gates(args.mode, initial, final))
        passed = all(gates.values())
        result = {
            "schema": SCHEMA,
            "status": ("passed_canary" if args.mode == "canary" else "passed_development") if passed else ("failed_canary" if args.mode == "canary" else "failed_development"),
            "role": "SCNS-Train training/development only; SCNS-Eval-v2 forbidden",
            "mode": args.mode,
            "job_id": os.environ["SLURM_JOB_ID"],
            "device": torch.cuda.get_device_name(),
            "steps": len(schedule),
            "parameters_total": sum(value.numel() for value in calibrated.parameters()),
            "parameters_calibrated": trainable_parameters,
            "trainable_names": trainable_names,
            "nonconfidence_parameters_bit_exact": frozen_exact,
            "protected_parameters_bit_exact": frozen_exact,
            "any_trainable_parameter_changed": any_trainable_changed,
            "calibration_loss_relative_reduction": reduction,
            "calibration_progress": reduction,
            "calibration_progress_description": CALIBRATION_PROGRESS_DESCRIPTION,
            "calibration_progress_gate": CALIBRATION_PROGRESS_GATE,
            "initial": initial,
            "development": final,
            "gates": gates,
            "contract": contract,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "training_checkpoints": [
                {
                    "step": step,
                    "path": str(
                        (args.output / f"checkpoint-step-{step:06d}.pt").resolve()
                    ),
                    "sha256": sha256(
                        args.output / f"checkpoint-step-{step:06d}.pt"
                    ),
                }
                for step in sorted(
                    {
                        len(schedule) // 4,
                        len(schedule) // 2,
                        3 * len(schedule) // 4,
                        len(schedule),
                    }
                )
            ],
            "runtime_seconds": time.monotonic() - started,
            "test_data_used": False,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": result["status"], "calibration_loss_relative_reduction": reduction,
            "summary": final["summary"], "comparisons": final["comparisons"], "gates": gates,
            "runtime_seconds": result["runtime_seconds"],
        }, indent=2), flush=True)
        return 0 if passed else 2
    finally:
        authorities.close()


if __name__ == "__main__":
    raise SystemExit(main())
