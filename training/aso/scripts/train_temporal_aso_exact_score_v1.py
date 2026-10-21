#!/usr/bin/env python3
"""Matched exact-additive Temporal ASO control and onset/offset experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
from notesep.training_paths import configure
from notesep.training_resume import validate_resume_contract
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import train_aso_scaled_source_diverse_v1 as scaled  # noqa: E402
from model_exp.fresh_source_complete_data import FreshSourceCompleteDataset  # noqa: E402
from model_exp.recorded_natural_score_scene_data import (  # noqa: E402
    RecordedNaturalScoreSceneDataset,
)
from model_exp.score_coordinate_exact_temporal_ownership import (  # noqa: E402
    ScoreCoordinateExactTemporalOwnership,
)
from model_exp.score_boundary_residual_exact_temporal_ownership import (  # noqa: E402
    ScoreBoundaryResidualExactTemporalOwnership,
)
from model_exp.zero_sum_complex_exact_temporal_ownership import (  # noqa: E402
    ZeroSumComplexExactTemporalOwnership,
)
from model_exp.temporal_hybrid_set_ownership import (  # noqa: E402
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)
from model_exp.acoustic_support_temporal_ownership import (  # noqa: E402
    AcousticSupportTemporalOwnership,
)
from model_exp.acoustic_multiplicative_temporal_ownership import (  # noqa: E402
    AcousticMultiplicativeTemporalOwnership,
)
from model_exp.acoustic_adaptive_trust_temporal_ownership import (  # noqa: E402
    AcousticAdaptiveTrustTemporalOwnership,
)
from model_exp.acoustic_per_note_delta_trust_temporal_ownership import (  # noqa: E402
    AcousticPerNoteDeltaTrustTemporalOwnership,
)


LEGACY_SPEC = importlib.util.spec_from_file_location(
    "temporal_aso_legacy_runner", ROOT / "scripts/train_temporal_hybrid_aso_v1.py"
)
assert LEGACY_SPEC is not None and LEGACY_SPEC.loader is not None
legacy = importlib.util.module_from_spec(LEGACY_SPEC)
LEGACY_SPEC.loader.exec_module(legacy)

SEED = 20260909
FRESH_SHA256 = "202391feb2dc99ef94d28181783f325022de546733004011a9a7a68556e0c958"
FRESH_AUTHORITY_SHA256 = "711659391e8f93849e0f2595061003fed584fef47d54526684e92a595ed203eb"
RECORDED_SHA256 = "24c30535b171b19a6ce06b29d886aece2f1cd65d7828f4f719c928df5ea534c7"
PROPOSAL_VALIDATION_SHA256 = "e5d12608fc6a8d143a61b36d3f51ec1afa122abeaff7d29aad4e50bac4c74124"
WARM_START_SHA256 = "a7657fabe82693e19e79defb43f404b6820af5775c78a0e5623c06b356cf251b"
BOUNDARY_PARENT_SHA256 = "586c8b8271496260d1423a12436f8dd1522d799deea23cd474490c61e307969b"
PREREG = ROOT / "docs/preregister_temporal_aso_exact_score_v1_20260909.md"
BOUNDARY_PREREG = ROOT / "docs/preregister_temporal_aso_boundary_residual_v1_20260909.md"
BOUNDARY_PREREG_SHA256 = "8766e890e1f8dada25306671c9f28957e6fb639fa5064f9fabe86a32f3c11286"
COMPLEX_PREREG = ROOT / "docs/preregister_zero_sum_complex_temporal_aso_v1_20260909.md"
COMPLEX_PREREG_SHA256 = "6c091f563f367533e39bc4ca927e27d8193bcf8c968570c4f3807a25e54cd739"
AMENDMENT = ROOT / "docs/amendment_temporal_aso_exact_score_v1b_20260909.md"
AMENDMENT_SHA256 = "f9b74a5da9cb9db0670bb3006c2c69393b24a46933cfb79f72b7485fc8b06579"
RESUME_AMENDMENT = ROOT / "docs/amendment_temporal_aso_scaled_variants_v2b_resume_20260909.md"
RESUME_AMENDMENT_SHA256 = "19b64939c42c7696a387e29827ad5243c17000e1cd111f30c34adb284fca6060"
PRE_RESUME_RUNNER_SHA256S = {
    # Original exact-score runner admitted by the v2b resume amendment.
    "3692b57f68881dc54d8361be047d61a27989b78e53ff4c645619e99d03d4dfa6",
    # Acoustic-multiplicative runner that wrote the current full-run contract
    # and step-26,500 resume payload.  This is a resume-only provenance
    # admission; model, data, schedule, objective, and evaluator are unchanged.
    "3e0f70765cd6f8a35f630d0e2538b700fb5d12a70d8fa58f892c4eff755a0ade",
}
STOP = False
ACOUSTIC_MULTIPLICATIVE_ARMS = {
    "acoustic_multiplicative",
    "acoustic_multiplicative_raw_total",
    "acoustic_multiplicative_adaptive_trust",
    "acoustic_multiplicative_per_note_delta_trust",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


class AcousticAuthorities(scaled.Authorities):
    """Source-complete train/dev authority with measured endpoints retained."""

    def __init__(self, fresh_index: Path, recorded_root: Path, cache: scaled.ProposalCache) -> None:
        for path in (fresh_index, recorded_root, cache.root):
            scaled.forbid_test_path(path)
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


def select_regret_reference(
    magnitude_log_distortion: torch.Tensor,
    raw_log_distortion: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Choose the frozen per-note comparator for SI-regret training."""
    if mode == "magnitude":
        return magnitude_log_distortion
    if mode == "best_raw_or_magnitude":
        return torch.minimum(magnitude_log_distortion, raw_log_distortion)
    raise ValueError(f"unknown regret reference: {mode}")


def raw_protected_objective(candidate_spec: torch.Tensor, record: dict):
    """The existing SI-regret loss, protecting the better input proposal."""
    estimate = scaled.istft(candidate_spec[0], record["length"]).float()
    magnitude = scaled.istft(record["base"], record["length"]).float()
    raw = record["raw_wave"].float()
    valid = legacy.risk.supervised_nonsilent(record)
    target = record["targets_wave"].float()[valid]
    candidate_log_d = legacy.risk._stable_log_si_distortion(estimate[valid], target)
    magnitude_log_d = legacy.risk._stable_log_si_distortion(magnitude[valid], target)
    raw_log_d = legacy.risk._stable_log_si_distortion(raw[valid], target)
    baseline_log_d = select_regret_reference(
        magnitude_log_d, raw_log_d, "best_raw_or_magnitude"
    ).detach()
    margin = legacy.risk.REGRET_MARGIN_DB * math.log(10.0) / 10.0
    regret = torch.relu(candidate_log_d - baseline_log_d - margin)
    loss = (
        candidate_log_d.mean()
        + legacy.risk.MEAN_REGRET_WEIGHT * regret.mean()
        + legacy.risk.SQUARED_REGRET_WEIGHT * regret.square().mean()
    )
    return loss, {
        "mean_log_si_distortion": candidate_log_d.mean(),
        "mean_regret": regret.mean(),
        "mean_squared_regret": regret.square().mean(),
        "maximum_regret": regret.max(),
        "raw_reference_selected_fraction": (raw_log_d < magnitude_log_d)
        .to(loss.dtype)
        .mean(),
        "silent_targets_excluded": (
            record["query_valid"].sum() - valid.sum()
        ).to(dtype=loss.dtype),
    }


def initialize(
    arm: str,
    config: TemporalHybridOwnershipConfig,
    source: dict[str, Any] | None,
    device: torch.device,
) -> tuple[TemporalHybridSetOwnership, TemporalHybridSetOwnership, float]:
    model_type = (
        AcousticSupportTemporalOwnership
        if arm == "acoustic_support"
        else TemporalHybridSetOwnership
    )
    parent = model_type(config).to(device)
    if source is not None:
        parent.load_state_dict(source["model"], strict=True)
    parent_state = parent.state_dict()
    if arm in {"exact_no_timing", "acoustic_support"}:
        model: TemporalHybridSetOwnership = model_type(config).to(device)
        model.load_state_dict(parent_state, strict=True)
    elif arm in ACOUSTIC_MULTIPLICATIVE_ARMS:
        model_class = {
            "acoustic_multiplicative_adaptive_trust": (
                AcousticAdaptiveTrustTemporalOwnership
            ),
            "acoustic_multiplicative_per_note_delta_trust": (
                AcousticPerNoteDeltaTrustTemporalOwnership
            ),
        }.get(arm, AcousticMultiplicativeTemporalOwnership)
        model = model_class(config).to(device)
        target = model.state_dict()
        for name, value in parent_state.items():
            if name == "input.weight":
                target[name].zero_()
                target[name][:, : value.shape[1]].copy_(value)
            else:
                target[name].copy_(value)
        model.load_state_dict(target, strict=True)
    elif arm == "exact_boundary_residual":
        model = ScoreBoundaryResidualExactTemporalOwnership(config).to(device)
        target = model.state_dict()
        for name, value in source["model"].items():
            target[name].copy_(value)
        target["boundary_delta.weight"].zero_()
        model.load_state_dict(target, strict=True)
    elif arm == "exact_zero_sum_complex":
        model = ZeroSumComplexExactTemporalOwnership(config).to(device)
        target = model.state_dict()
        for name, value in source["model"].items():
            target[name].copy_(value)
        model.load_state_dict(target, strict=True)
    else:
        model = ScoreCoordinateExactTemporalOwnership(config).to(device)
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
    parser.add_argument(
        "--arm",
        choices=(
            "exact_no_timing", "exact_onset_offset", "exact_boundary_residual",
            "exact_zero_sum_complex",
            "acoustic_support",
            "acoustic_multiplicative",
            "acoustic_multiplicative_raw_total",
            "acoustic_multiplicative_adaptive_trust",
            "acoustic_multiplicative_per_note_delta_trust",
        ),
        required=True,
    )
    parser.add_argument("--mode", choices=("canary", "screen", "full"), required=True)
    parser.add_argument("--fresh-index", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument(
        "--expected-proposal-validation-sha256",
        default=PROPOSAL_VALIDATION_SHA256,
        help=(
            "Expected SHA-256 of proposal-root/VALIDATION.json. The default "
            "preserves the historical cache; successor separator caches must "
            "supply their independently validated hash explicitly."
        ),
    )
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument(
        "--initialization", choices=("warm_start", "scratch"), default="warm_start"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--regret-reference",
        choices=("magnitude", "best_raw_or_magnitude"),
        default="magnitude",
    )
    parser.add_argument("--scale-preregistration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", type=Path)
    args = parser.parse_args()
    configure(args.path_map)
    if not torch.cuda.is_available() or ("gilbreth" in socket.gethostname().lower() and (not os.environ.get("SLURM_JOB_ID") or "-fe" in socket.gethostname().lower())):
        raise RuntimeError("allocated CUDA compute required")
    for path in (
        args.fresh_index,
        args.recorded_root,
        args.proposal_root,
        args.output,
    ):
        scaled.forbid_test_path(path)
    if args.warm_start is not None:
        scaled.forbid_test_path(args.warm_start)
    if args.output.exists() and not (args.output / "RESUME.pt").exists():
        raise FileExistsError(args.output)
    proposal_validation_sha256 = args.expected_proposal_validation_sha256.lower()
    if len(proposal_validation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in proposal_validation_sha256
    ):
        raise ValueError("invalid expected proposal validation SHA-256")
    if sha256(args.proposal_root / "VALIDATION.json") != proposal_validation_sha256:
        raise ValueError("proposal cache validation hash mismatch")
    if args.initialization == "scratch":
        if args.warm_start is not None:
            raise ValueError("scratch initialization forbids a warm start")
        if args.arm not in {
            "exact_no_timing", "exact_onset_offset", "acoustic_support",
            "acoustic_multiplicative",
            "acoustic_multiplicative_raw_total",
            "acoustic_multiplicative_adaptive_trust",
            "acoustic_multiplicative_per_note_delta_trust",
        }:
            raise ValueError("scratch scaling is limited to the matched exact arms")
        if args.scale_preregistration is None:
            raise ValueError("scratch scaling requires a preregistration")
        active_prereg = args.scale_preregistration
        expected_prereg_sha256 = os.environ.get("PREREG_SHA256")
        expected_warm_start = None
    elif args.warm_start is None:
        raise ValueError("warm_start initialization requires --warm-start")
    else:
        expected_warm_start = (
            BOUNDARY_PARENT_SHA256
            if args.arm in {"exact_boundary_residual", "exact_zero_sum_complex"}
            else WARM_START_SHA256
        )
        if sha256(args.warm_start) != expected_warm_start:
            raise ValueError("warm-start checkpoint hash mismatch")
    if args.initialization != "scratch" and args.arm == "exact_boundary_residual":
        active_prereg = BOUNDARY_PREREG
        expected_prereg_sha256 = BOUNDARY_PREREG_SHA256
    elif args.initialization != "scratch" and args.arm == "exact_zero_sum_complex":
        active_prereg = COMPLEX_PREREG
        expected_prereg_sha256 = COMPLEX_PREREG_SHA256
    elif args.initialization != "scratch":
        active_prereg = PREREG
        expected_prereg_sha256 = os.environ.get("PREREG_SHA256")
    if sha256(active_prereg) != expected_prereg_sha256:
        raise ValueError("preregistration hash mismatch")
    if sha256(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError("numerical-closure amendment hash mismatch")
    if sha256(RESUME_AMENDMENT) != RESUME_AMENDMENT_SHA256:
        raise ValueError("resume amendment hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    scaled.FRESH_SHA = FRESH_SHA256
    cache = scaled.ProposalCache(args.proposal_root)
    authorities = AcousticAuthorities(args.fresh_index, args.recorded_root, cache)
    config = TemporalHybridOwnershipConfig(
        feature_set="essential6",
        phase_mode="mixture",
        temporal_blocks=2,
        exact_mixture_additivity=args.arm not in {
            "acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS,
        },
        magnitude_reference=(
            "raw_total"
            if args.arm == "acoustic_multiplicative_raw_total"
            else "mixture"
        ),
    )
    source = (
        torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if args.warm_start is not None else None
    )
    if source is not None and args.arm in {"exact_boundary_residual", "exact_zero_sum_complex"}:
        if (
            source.get("step") != 2000
            or source.get("contract", {}).get("arm") != "exact_no_timing"
            or not source.get("contract", {}).get("model_contract", {}).get(
                "exact_mixture_additivity"
            )
        ):
            raise ValueError("boundary parent checkpoint contract mismatch")
    elif source is not None and (
        source.get("step") != 29058
        or source.get("contract", {}).get("arm") != "essential6_temporal_mixture"
    ):
        raise ValueError("warm-start model contract mismatch")
    model, parent, _ = initialize(args.arm, config, source, device)
    del source
    if args.regret_reference == "best_raw_or_magnitude":
        legacy.risk._objective = raw_protected_objective
    schedule, singleton_ids = scaled.make_schedule(authorities, "small_full")
    if args.mode == "canary":
        schedule = schedule[:320]
        maximum_mixtures = 16
    elif args.mode == "screen":
        schedule = schedule[:2000]
        maximum_mixtures = 64
    else:
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
        candidate_initial, candidate_diag = call(model, parity_record)
        parent_initial, parent_diag = call(parent, parity_record)
    initialization_parity = float((candidate_initial - parent_initial).abs().amax())
    initialization_complex_closure = float(candidate_diag["complex_closure_max_abs"])
    parent_complex_closure = float(parent_diag["complex_closure_max_abs"])
    if (
        args.arm in ACOUSTIC_MULTIPLICATIVE_ARMS
        and initialization_parity > 2e-6
    ) or (
        args.arm not in ACOUSTIC_MULTIPLICATIVE_ARMS
        and initialization_parity != 0.0
    ):
        raise ValueError(f"matched initialization parity failed: {initialization_parity}")
    del candidate_initial, parent_initial, parity_record

    contract = {
        "schema": "temporal_aso_exact_score_training_v1",
        "arm": args.arm,
        "mode": args.mode,
        "seed": args.seed,
        "initialization": args.initialization,
        "steps": len(schedule),
        "model_contract": model.contract(),
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "warm_start": str(args.warm_start.resolve()) if args.warm_start else None,
        "warm_start_sha256": expected_warm_start,
        "fresh_index": str(args.fresh_index.resolve()),
        "fresh_index_sha256": FRESH_SHA256,
        "fresh_authority_sha256": FRESH_AUTHORITY_SHA256,
        "recorded_authority": str((args.recorded_root / "FROZEN.json").resolve()),
        "recorded_authority_sha256": RECORDED_SHA256,
        "proposal_root": str(args.proposal_root.resolve()),
        "proposal_validation_sha256": proposal_validation_sha256,
        "preregistration": str(active_prereg.resolve()),
        "preregistration_sha256": sha256(active_prereg),
        "closure_amendment": str(AMENDMENT.resolve()),
        "closure_amendment_sha256": AMENDMENT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "historical_runner_sha256": "4b08e16aa209d1309e611abf7bd23a83be5432bd8eeaa6573ecb9c174990daea",
        "model_source_sha256": sha256(ROOT / "src/model_exp/temporal_hybrid_set_ownership.py"),
        "score_model_source_sha256": sha256(ROOT / "src/model_exp/score_coordinate_exact_temporal_ownership.py"),
        "acoustic_multiplicative_model_source_sha256": (
            sha256(ROOT / "src/model_exp/acoustic_multiplicative_temporal_ownership.py")
            if args.arm in ACOUSTIC_MULTIPLICATIVE_ARMS else None
        ),
        "acoustic_adaptive_trust_model_source_sha256": (
            sha256(ROOT / "src/model_exp/acoustic_adaptive_trust_temporal_ownership.py")
            if args.arm == "acoustic_multiplicative_adaptive_trust" else None
        ),
        "acoustic_per_note_delta_trust_model_source_sha256": (
            sha256(
                ROOT
                / "src/model_exp/acoustic_per_note_delta_trust_temporal_ownership.py"
            )
            if args.arm == "acoustic_multiplicative_per_note_delta_trust"
            else None
        ),
        "acoustic_support_model_source_sha256": (
            sha256(ROOT / "src/model_exp/acoustic_support_temporal_ownership.py")
            if args.arm == "acoustic_support" else None
        ),
        "boundary_residual_source_sha256": (
            sha256(ROOT / "src/model_exp/score_boundary_residual_exact_temporal_ownership.py")
            if args.arm == "exact_boundary_residual" else None
        ),
        "zero_sum_complex_source_sha256": (
            sha256(ROOT / "src/model_exp/zero_sum_complex_exact_temporal_ownership.py")
            if args.arm == "exact_zero_sum_complex" else None
        ),
        "objective": (
            "waveform SI-regret against the better of raw proposal and magnitude partition, "
            "plus target ownership and TF smoothness"
            if args.regret_reference == "best_raw_or_magnitude"
            else "unchanged waveform SI-regret plus target ownership and TF smoothness"
        ),
        "regret_reference": args.regret_reference,
        "boundary_inputs": (
            "signed onset and measured acoustic-offset coordinates only"
            if args.arm in ACOUSTIC_MULTIPLICATIVE_ARMS
            else "signed symbolic onset and MIDI-offset coordinates only"
        ),
        "acoustic_endpoint_role": (
            "model input, target support, and evaluation diagnostic"
            if args.arm in ACOUSTIC_MULTIPLICATIVE_ARMS
            else "target support and evaluation diagnostic only"
        ),
        "singleton_identity_count": len(singleton_ids),
        "test_data_used": False,
    }
    contract_path = args.output / "CONTRACT.json"
    if contract_path.exists():
        recorded_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract = validate_resume_contract(
            contract, recorded_contract, active_runner_sha256=sha256(Path(__file__)),
            allowed_parent_runners=PRE_RESUME_RUNNER_SHA256S | {"4b08e16aa209d1309e611abf7bd23a83be5432bd8eeaa6573ecb9c174990daea"},
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

    legacy.call = call
    started = time.monotonic()
    try:
        for position in range(start, len(schedule)):
            kind, index = schedule[position]
            record = authorities.record(kind, "train", index, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = legacy.train_loss(model, record)
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
                    "mean_abs_correction": float(diagnostics["mean_abs_correction"].detach()),
                    "mean_ownership_trust": float(
                        diagnostics["mean_ownership_trust"].detach()
                    ),
                    "mean_magnitude_delta_trust": float(
                        diagnostics["mean_magnitude_delta_trust"].detach()
                    ),
                    "complex_closure_max_abs": float(diagnostics["complex_closure_max_abs"].detach()),
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
                legacy.atomic_torch(resume, payload)
                if done in checkpoints:
                    legacy.atomic_torch(args.output / f"checkpoint-step-{done:06d}.pt", payload)
            if STOP:
                return 75

        legacy.call = call
        development = legacy.evaluate(model, authorities, maximum_mixtures, parent=parent)
        shuffled = None
        if args.arm in {
            "exact_onset_offset", "exact_boundary_residual",
            "acoustic_multiplicative",
            "acoustic_multiplicative_raw_total",
            "acoustic_multiplicative_adaptive_trust",
            "acoustic_multiplicative_per_note_delta_trust",
        }:
            legacy.call = shuffled_call
            shuffled = legacy.evaluate(model, authorities, maximum_mixtures, parent=None)
            legacy.call = call
            correct_by_id = {
                (row["dataset_kind"], row["mixture_id"], row["note"]): row
                for row in development["events"]
            }
            shuffled_delta = np.asarray([
                correct_by_id[(row["dataset_kind"], row["mixture_id"], row["note"])]["candidate"]["si_sdr_db"]
                - row["candidate"]["si_sdr_db"]
                for row in shuffled["events"]
            ])
            shuffled["correct_minus_shuffled"] = {
                "median_db": float(np.median(shuffled_delta)),
                "mean_db": float(np.mean(shuffled_delta)),
                "correct_wins_gt_0p1db": int((shuffled_delta > 0.1).sum()),
                "shuffled_wins_gt_0p1db": int((shuffled_delta < -0.1).sum()),
            }

        final_checkpoint = args.output / f"checkpoint-step-{len(schedule):06d}.pt"
        gates = {
            "initialization_numerically_matched": initialization_parity <= 2e-6,
            (
                "initial_complex_closure_not_expected"
                if args.arm in {"acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS}
                else "initial_complex_closure_lte_5e_minus5"
            ): (
                True
                if args.arm in {"acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS}
                else max(initialization_complex_closure, parent_complex_closure) <= 5e-5
            ),
            "training_completed": bool(history) and history[-1]["step"] == len(schedule),
            "all_parameters_finite": all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
            "coverage_nonempty": development["coverage"] > 0,
            (
                "magnitude_closure_recorded_not_scale_gate"
                if args.arm in {"exact_boundary_residual", "exact_zero_sum_complex"}
                or args.initialization == "scratch"
                else "magnitude_closure_lte_5e_minus5"
            ): (
                True
                if args.arm in {"exact_boundary_residual", "exact_zero_sum_complex"}
                or args.initialization == "scratch"
                else development["maximum_magnitude_closure_abs"] <= 5e-5
            ),
            (
                "complex_closure_not_expected"
                if args.arm in {"acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS}
                else "complex_closure_lte_5e_minus5"
            ): (
                True
                if args.arm in {"acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS}
                else development["maximum_complex_closure_abs"] <= 5e-5
            ),
            (
                "magnitude_closure_relative_recorded_not_expected"
                if args.arm in {
                    "exact_zero_sum_complex", "acoustic_support",
                    "acoustic_multiplicative",
                    "acoustic_multiplicative_raw_total",
                    "acoustic_multiplicative_adaptive_trust",
                    "acoustic_multiplicative_per_note_delta_trust",
                }
                else "magnitude_closure_relative_lte_2e_minus6"
            ): (
                True
                if args.arm in {
                    "exact_zero_sum_complex", "acoustic_support",
                    "acoustic_multiplicative",
                    "acoustic_multiplicative_raw_total",
                    "acoustic_multiplicative_adaptive_trust",
                    "acoustic_multiplicative_per_note_delta_trust",
                }
                else development["maximum_magnitude_closure_relative"] <= 2e-6
            ),
            (
                "complex_closure_relative_not_expected"
                if args.arm in {"acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS}
                else "complex_closure_relative_lte_2e_minus6"
            ): (
                True
                if args.arm in {"acoustic_support", *ACOUSTIC_MULTIPLICATIVE_ARMS}
                else development["maximum_complex_closure_relative"] <= 2e-6
            ),
            "no_catastrophic_canary_instability": (
                True
                if args.initialization == "scratch"
                else development["comparisons"]["candidate_minus_parent"]["losses_gt_3db"] <= 2
                if args.mode == "canary"
                else True
            ),
            "test_data_unused": True,
        }
        result = {
            "schema": "temporal_aso_exact_score_result_v1",
            "status": "passed" if all(gates.values()) else "failed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "device": torch.cuda.get_device_name(),
            "runtime_seconds": time.monotonic() - started,
            "contract": contract,
            "resume_runner_sha256": sha256(Path(__file__)),
            "resume_amendment": str(RESUME_AMENDMENT.resolve()),
            "resume_amendment_sha256": sha256(RESUME_AMENDMENT),
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
        print(json.dumps({
            "status": result["status"],
            "arm": args.arm,
            "coverage": development["coverage"],
            "median_si_sdr": development["summary"]["candidate"]["si_sdr_db"]["median"],
            "paired_median_vs_parent": development["comparisons"]["candidate_minus_parent"]["median_db"],
            "maximum_complex_closure_abs": development["maximum_complex_closure_abs"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        }, indent=2), flush=True)
        return 0 if result["status"] == "passed" else 2
    finally:
        legacy.call = call
        authorities.close()


if __name__ == "__main__":
    raise SystemExit(main())
