#!/usr/bin/env python3
"""Train the selected acoustic-multiplicative ASO used by the paper pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training.aso import dataset  # noqa: E402
from training.aso import engine  # noqa: E402
from training.aso.datasets.synthetic import (  # noqa: E402
    FreshSourceCompleteDataset,
)
from training.aso.datasets.recorded import (  # noqa: E402
    RecordedNaturalScoreSceneDataset,
)

from notesep.models.aso import (  # noqa: E402
    AcousticMultiplicativeTemporalOwnership,
)
from notesep.models.ownership import (  # noqa: E402
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)
from notesep.training_paths import configure  # noqa: E402
from notesep.training_resume import validate_resume_contract  # noqa: E402

SEED = 20260909
FRESH_SHA256 = "202391feb2dc99ef94d28181783f325022de546733004011a9a7a68556e0c958"
FRESH_AUTHORITY_SHA256 = (
    "711659391e8f93849e0f2595061003fed584fef47d54526684e92a595ed203eb"
)
RECORDED_SHA256 = "24c30535b171b19a6ce06b29d886aece2f1cd65d7828f4f719c928df5ea534c7"
PROPOSAL_VALIDATION_SHA256 = (
    "7070a58fb977c14eb8fd1139cf300f848d39373bb460c74db49afe9e0c64cb82"
)
STOP = False
FINAL_ARM = "acoustic_multiplicative"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


class AcousticAuthorities(dataset.Authorities):
    """Source-complete train/dev authority with measured endpoints retained."""

    def __init__(
        self,
        fresh_index: Path,
        recorded_root: Path,
        cache: dataset.ProposalCache,
    ) -> None:
        for path in (fresh_index, recorded_root, cache.root):
            dataset.forbid_test_path(path)
        if sha256(fresh_index) != FRESH_SHA256:
            raise ValueError("corrected fresh index hash mismatch")
        if sha256(fresh_index.parent / "AUTHORITY.json") != FRESH_AUTHORITY_SHA256:
            raise ValueError("corrected acoustic authority hash mismatch")
        if sha256(recorded_root / "FROZEN.json") != RECORDED_SHA256:
            raise ValueError("recorded authority hash mismatch")
        self.cache = cache
        self.fresh = {
            split: FreshSourceCompleteDataset(
                fresh_index,
                split=split,
                crop_samples=49152,
                maximum_sources=64,
                preserve_signed_note_times=True,
                require_acoustic_offsets=True,
            )
            for split in ("train", "validation")
        }
        self.recorded = {
            split: RecordedNaturalScoreSceneDataset([recorded_root], split=split)
            for split in ("train", "validation")
        }


def call(model: TemporalHybridSetOwnership, record: dict[str, Any]):
    return model(
        record["mixture"][None],
        record["base"][None],
        record["raw"][None],
        record["pitches"][None],
        record["onsets"][None],
        record["symbolic_offsets"][None],
        record["offsets"][None],
        hard_phase=True,
    )


def shuffled_call(model: TemporalHybridSetOwnership, record: dict[str, Any]):
    shifted = dict(record)
    notes = len(record["raw"])
    order = torch.roll(torch.arange(notes, device=record["raw"].device), 1)
    shifted["onsets"] = record["onsets"][order]
    shifted["symbolic_offsets"] = record["symbolic_offsets"][order]
    if isinstance(model, AcousticMultiplicativeTemporalOwnership):
        shifted["offsets"] = record["offsets"][order]
    return call(model, shifted)


def raw_protected_objective(candidate_spec: torch.Tensor, record: dict):
    """The existing SI-regret loss, protecting the better input proposal."""
    estimate = dataset.istft(candidate_spec[0], record["length"]).float()
    magnitude = dataset.istft(record["base"], record["length"]).float()
    raw = record["raw_wave"].float()
    valid = engine.supervised_nonsilent(record)
    target = record["targets_wave"].float()[valid]
    candidate_log_d = engine.stable_log_si_distortion(estimate[valid], target)
    magnitude_log_d = engine.stable_log_si_distortion(magnitude[valid], target)
    raw_log_d = engine.stable_log_si_distortion(raw[valid], target)
    baseline_log_d = torch.minimum(magnitude_log_d, raw_log_d).detach()
    margin = engine.REGRET_MARGIN_DB * math.log(10.0) / 10.0
    regret = torch.relu(candidate_log_d - baseline_log_d - margin)
    loss = (
        candidate_log_d.mean()
        + engine.MEAN_REGRET_WEIGHT * regret.mean()
        + engine.SQUARED_REGRET_WEIGHT * regret.square().mean()
    )
    return loss, {
        "mean_log_si_distortion": candidate_log_d.mean(),
        "mean_regret": regret.mean(),
        "mean_squared_regret": regret.square().mean(),
        "maximum_regret": regret.max(),
        "raw_reference_selected_fraction": (raw_log_d < magnitude_log_d)
        .to(loss.dtype)
        .mean(),
        "silent_targets_excluded": (record["query_valid"].sum() - valid.sum()).to(
            dtype=loss.dtype
        ),
    }


def initialize(
    config: TemporalHybridOwnershipConfig,
    device: torch.device,
) -> tuple[TemporalHybridSetOwnership, TemporalHybridSetOwnership, float]:
    parent = TemporalHybridSetOwnership(config).to(device)
    parent_state = parent.state_dict()
    model = AcousticMultiplicativeTemporalOwnership(config).to(device)
    target = model.state_dict()
    for name, value in parent_state.items():
        if name == "input.weight":
            target[name].zero_()
            target[name][:, : value.shape[1]].copy_(value)
        else:
            target[name].copy_(value)
    model.load_state_dict(target, strict=True)
    return model, parent, 0.0


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(FINAL_ARM,), required=True)
    parser.add_argument("--fresh-index", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", type=Path)
    args = parser.parse_args()
    configure(args.path_map)
    if not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA compute required")
    for path in (
        args.fresh_index,
        args.recorded_root,
        args.proposal_root,
        args.output,
    ):
        dataset.forbid_test_path(path)
    if args.output.exists() and not (args.output / "RESUME.pt").exists():
        raise FileExistsError(args.output)
    proposal_validation_sha256 = PROPOSAL_VALIDATION_SHA256
    if sha256(args.proposal_root / "VALIDATION.json") != proposal_validation_sha256:
        raise ValueError("proposal cache validation hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    dataset.FRESH_SHA = FRESH_SHA256
    cache = dataset.ProposalCache(args.proposal_root)
    authorities = AcousticAuthorities(args.fresh_index, args.recorded_root, cache)
    config = TemporalHybridOwnershipConfig(
        feature_set="essential6",
        phase_mode="mixture",
        temporal_blocks=2,
        exact_mixture_additivity=False,
        magnitude_reference="mixture",
    )
    model, parent, _ = initialize(config, device)
    schedule, singleton_ids = dataset.make_schedule(authorities)
    maximum_mixtures = 0

    parity_kind, parity_index = next(
        (kind, index)
        for kind, index in schedule
        if len(authorities.record(kind, "train", index, device)["raw"]) >= 2
    )
    parity_record = authorities.record(parity_kind, "train", parity_index, device)
    model.eval()
    parent.eval()
    with torch.no_grad():
        candidate_initial, _ = call(model, parity_record)
        parent_initial, _ = call(parent, parity_record)
    initialization_parity = float((candidate_initial - parent_initial).abs().amax())
    if initialization_parity > 2e-6:
        raise ValueError(
            f"matched initialization parity failed: {initialization_parity}"
        )
    del candidate_initial, parent_initial, parity_record

    contract = {
        "schema": "temporal_aso_exact_score_training_v1",
        "arm": args.arm,
        "mode": "full",
        "seed": args.seed,
        "initialization": "scratch",
        "steps": len(schedule),
        "model_contract": model.contract(),
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "fresh_index": str(args.fresh_index.resolve()),
        "fresh_index_sha256": FRESH_SHA256,
        "fresh_authority_sha256": FRESH_AUTHORITY_SHA256,
        "recorded_authority": str((args.recorded_root / "FROZEN.json").resolve()),
        "recorded_authority_sha256": RECORDED_SHA256,
        "proposal_root": str(args.proposal_root.resolve()),
        "proposal_validation_sha256": proposal_validation_sha256,
        "objective": (
            "waveform SI-regret against the better of raw proposal and "
            "magnitude partition, plus target ownership and TF smoothness"
        ),
        "regret_reference": "best_raw_or_magnitude",
        "boundary_inputs": (
            "signed onset and measured acoustic-offset coordinates only"
        ),
        "acoustic_endpoint_role": (
            "model input, target support, and evaluation diagnostic"
        ),
        "singleton_identity_count": len(singleton_ids),
        "test_data_used": False,
    }
    contract_path = args.output / "CONTRACT.json"
    if contract_path.exists():
        recorded_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract = validate_resume_contract(
            contract,
            recorded_contract,
        )
    else:
        encoded = json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
        contract_path.write_text(encoded, encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start = 0
    history: list[dict[str, Any]] = []
    resume = args.output / "RESUME.pt"
    if resume.exists():
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("contract") != contract:
            raise ValueError("resume payload contract mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"])
        history = list(payload["history"])

    started = time.monotonic()
    try:
        for position in range(start, len(schedule)):
            kind, index = schedule[position]
            record = authorities.record(kind, "train", index, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = engine.train_loss(
                model, record, forward=call, objective=raw_protected_objective
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient)):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            done = position + 1
            if done == 1 or done % 100 == 0 or done == len(schedule):
                row = {
                    "step": done,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient.detach()),
                    "mean_abs_correction": float(
                        diagnostics["mean_abs_correction"].detach()
                    ),
                    "mean_ownership_trust": float(
                        diagnostics["mean_ownership_trust"].detach()
                    ),
                    "mean_magnitude_delta_trust": float(
                        diagnostics["mean_magnitude_delta_trust"].detach()
                    ),
                    "complex_closure_max_abs": float(
                        diagnostics["complex_closure_max_abs"].detach()
                    ),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
            checkpoints = {len(schedule) // 2, len(schedule)}
            if done in checkpoints or done % 500 == 0 or STOP:
                payload = {
                    "schema": "temporal_aso_exact_score_checkpoint_v1",
                    "contract": contract,
                    "step": done,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                }
                engine.atomic_torch(resume, payload)
                if done in checkpoints:
                    engine.atomic_torch(
                        args.output / f"checkpoint-step-{done:06d}.pt", payload
                    )
            if STOP:
                return 75

        development = engine.evaluate(
            model, authorities, maximum_mixtures, parent=parent, forward=call
        )
        shuffled = engine.evaluate(
            model, authorities, maximum_mixtures, parent=None, forward=shuffled_call
        )
        correct_by_id = {
            (row["dataset_kind"], row["mixture_id"], row["note"]): row
            for row in development["events"]
        }
        shuffled_delta = np.asarray(
            [
                correct_by_id[(row["dataset_kind"], row["mixture_id"], row["note"])][
                    "candidate"
                ]["si_sdr_db"]
                - row["candidate"]["si_sdr_db"]
                for row in shuffled["events"]
            ]
        )
        shuffled["correct_minus_shuffled"] = {
            "median_db": float(np.median(shuffled_delta)),
            "mean_db": float(np.mean(shuffled_delta)),
            "correct_wins_gt_0p1db": int((shuffled_delta > 0.1).sum()),
            "shuffled_wins_gt_0p1db": int((shuffled_delta < -0.1).sum()),
        }

        final_checkpoint = args.output / f"checkpoint-step-{len(schedule):06d}.pt"
        gates = {
            "initialization_numerically_matched": initialization_parity <= 2e-6,
            "training_completed": bool(history)
            and history[-1]["step"] == len(schedule),
            "all_parameters_finite": all(
                bool(torch.isfinite(value).all())
                for value in model.state_dict().values()
            ),
            "coverage_nonempty": development["coverage"] > 0,
        }
        result = {
            "schema": "temporal_aso_exact_score_result_v1",
            "status": "passed" if all(gates.values()) else "failed",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name(),
            "runtime_seconds": time.monotonic() - started,
            "contract": contract,
            "checkpoint": str(final_checkpoint.resolve()),
            "checkpoint_sha256": sha256(final_checkpoint),
            "initialization_parity_max_abs": initialization_parity,
            "final_mean_ownership_trust": (
                history[-1]["mean_ownership_trust"] if history else None
            ),
            "final_mean_magnitude_delta_trust": (
                history[-1]["mean_magnitude_delta_trust"] if history else None
            ),
            "development": development,
            "shuffled_timing_diagnostic": shuffled,
            "gates": gates,
            "test_data_used": False,
        }
        atomic_json(args.output / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "arm": args.arm,
                    "coverage": development["coverage"],
                    "median_si_sdr": development["summary"]["candidate"]["si_sdr_db"][
                        "median"
                    ],
                    "paired_median_vs_parent": development["comparisons"][
                        "candidate_minus_parent"
                    ]["median_db"],
                    "maximum_complex_closure_abs": development[
                        "maximum_complex_closure_abs"
                    ],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if result["status"] == "passed" else 2
    finally:
        authorities.close()


if __name__ == "__main__":
    raise SystemExit(main())
