#!/usr/bin/env python3
"""Matched train/development runner for temporal hybrid ASO v1."""
from __future__ import annotations

import argparse
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

import calibrate_aso_scaled_si_regret_v3b as risk  # noqa: E402
import train_aso_scaled_source_diverse_v1 as scaled  # noqa: E402
from model_exp.temporal_hybrid_set_ownership import (  # noqa: E402
    TemporalHybridOwnershipConfig,
    TemporalHybridSetOwnership,
)
from model_exp.score_conditioned_temporal_ownership import (  # noqa: E402
    OffsetDistanceTemporalOwnership,
    PostOffsetAttenuationOwnership,
    PostOffsetDistanceTemporalOwnership,
    PostOffsetSparsemaxResidualOwnership,
    ScoreOnOffTemporalOwnership,
)


SEED = 20260908
FRESH_SHA256 = "215bb4f714e3bc45f867c341a5589cd286049f54a4d66ad6853ab71e68fd334b"
RECORDED_SHA256 = "24c30535b171b19a6ce06b29d886aece2f1cd65d7828f4f719c928df5ea534c7"
CACHE_VALIDATION_SHA256 = "e5d12608fc6a8d143a61b36d3f51ec1afa122abeaff7d29aad4e50bac4c74124"
PREREG = ROOT / "docs/preregister_temporal_hybrid_aso_v1_20260908.md"
AMENDMENT = ROOT / "docs/amendment_temporal_hybrid_aso_v1b_20260908.md"
AMENDMENT_V1C = ROOT / "docs/amendment_temporal_hybrid_aso_v1c_20260908.md"
AMENDMENT_V1D = ROOT / "docs/amendment_temporal_hybrid_aso_v1d_20260908.md"
ONOFF_PREREG = ROOT / "docs/preregister_temporal_aso_onoff_channel_v1_20260908.md"
RELATIVE_PEAK_PREREG = ROOT / "docs/preregister_temporal_aso_relative_peak_canary_v1_20260908.md"
RELATIVE_PEAK_SCREEN = ROOT / "docs/amendment_temporal_aso_relative_peak_screen_v1_20260908.md"
HARMONIC_PROMINENCE_PREREG = ROOT / "docs/preregister_temporal_aso_harmonic_prominence_v1_20260908.md"
RELATIVE_HARMONIC_LOUDNESS_PREREG = ROOT / "docs/preregister_temporal_aso_relative_harmonic_loudness_v1_20260908.md"
RAW_TOTAL_PREREG = ROOT / "docs/preregister_temporal_aso_raw_total_transport_v1_20260908.md"
OFFSET_DISTANCE_PREREG = ROOT / "docs/preregister_temporal_aso_offset_distance_adapter_v1_20260908.md"
OFFSET_DISTANCE_V1B = ROOT / "docs/amendment_temporal_aso_offset_distance_adapter_v1b_20260908.md"
OFFSET_DISTANCE_V1C = ROOT / "docs/amendment_temporal_aso_offset_distance_adapter_v1c_20260908.md"
POST_OFFSET_DISTANCE_PREREG = ROOT / "docs/preregister_temporal_aso_post_offset_distance_v2_20260908.md"
POST_OFFSET_SPARSEMAX_PREREG = ROOT / "docs/preregister_temporal_aso_post_offset_sparsemax_residual_v3_20260908.md"
POST_OFFSET_SPARSEMAX_V3B = ROOT / "docs/amendment_temporal_aso_post_offset_sparsemax_residual_v3b_20260908.md"
POST_OFFSET_ATTENUATION_PREREG = ROOT / "docs/preregister_temporal_aso_post_offset_attenuation_v4_20260908.md"
SELECTED_TEMPORAL_SHA256 = "a7657fabe82693e19e79defb43f404b6820af5775c78a0e5623c06b356cf251b"
STOP = False
INPUT_OFFSET_ARMS = {
    "essential6_temporal_offset_distance",
    "essential6_temporal_post_offset_distance",
}
OFFSET_ARMS = INPUT_OFFSET_ARMS | {
    "essential6_temporal_post_offset_sparsemax",
    "essential6_temporal_post_offset_attenuation",
}

ARMS: dict[str, dict[str, Any]] = {
    "legacy11_local_mixture": dict(
        feature_set="legacy11", phase_mode="mixture", temporal_blocks=0
    ),
    "legacy11_temporal_mixture": dict(
        feature_set="legacy11", phase_mode="mixture", temporal_blocks=2
    ),
    "essential6_temporal_mixture": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2
    ),
    "essential6_temporal_raw_total": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2,
        magnitude_reference="raw_total",
    ),
    "essential7_relative_peak_temporal_mixture": dict(
        feature_set="essential7_relative_peak", phase_mode="mixture", temporal_blocks=2
    ),
    "essential7_harmonic_prominence_temporal_mixture": dict(
        feature_set="essential7_harmonic_prominence", phase_mode="mixture", temporal_blocks=2
    ),
    "essential7_relative_harmonic_loudness_temporal_mixture": dict(
        feature_set="essential7_relative_harmonic_loudness", phase_mode="mixture", temporal_blocks=2
    ),
    "phase10_temporal_mixture": dict(
        feature_set="essential6_phase", phase_mode="mixture", temporal_blocks=2
    ),
    "phase10_temporal_hybrid": dict(
        feature_set="essential6_phase", phase_mode="hybrid_delta", temporal_blocks=2
    ),
    "phase10_temporal_direct_complex": dict(
        feature_set="essential6_phase", phase_mode="direct_complex", temporal_blocks=2
    ),
    "essential6_temporal_score_onoff": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2
    ),
    "essential6_temporal_offset_distance": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2
    ),
    "essential6_temporal_post_offset_distance": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2
    ),
    "essential6_temporal_post_offset_sparsemax": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2
    ),
    "essential6_temporal_post_offset_attenuation": dict(
        feature_set="essential6", phase_mode="mixture", temporal_blocks=2
    ),
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


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def call(model: TemporalHybridSetOwnership, record: dict[str, Any]):
    return model(
        record["mixture"][None], record["base"][None], record["raw"][None],
        record["pitches"][None], record["onsets"][None], record["offsets"][None],
        record["offsets"][None], hard_phase=True,
    )


def train_loss(model: TemporalHybridSetOwnership, record: dict[str, Any]):
    candidate_spec, diagnostics = call(model, record)
    waveform_risk, risk_diagnostics = risk._objective(candidate_spec, record)
    target_magnitude = record["targets"].abs()
    target_share = target_magnitude / target_magnitude.sum(0, keepdim=True).clamp_min(1e-8)
    energy = record["mixture"].abs().square()
    energy = energy / energy.sum().clamp_min(1e-12)
    ownership = diagnostics["ownership"][0].clamp_min(1e-8)
    ownership_loss = -(target_share * ownership.log()).sum(0).mul(energy).sum()
    smoothness = diagnostics["smoothness_penalty"]
    loss = waveform_risk + ownership_loss + 0.01 * smoothness
    return loss, diagnostics | {
        "waveform_risk": waveform_risk,
        "ownership_loss": ownership_loss,
        "smoothness_loss": smoothness,
        **{f"risk_{key}": value for key, value in risk_diagnostics.items()},
    }


def summarize(events: list[dict[str, Any]], method: str) -> dict[str, Any]:
    result = {
        metric: {
            "median": float(np.median([row[method][metric] for row in events])),
            "mean": float(np.mean([row[method][metric] for row in events])),
        }
        for metric in ("si_sdr_db", "off_e", "gain", "scale_invariant_distortion")
    }
    if "post_offset_energy_ratio" in events[0][method]:
        result["post_offset_energy_ratio"] = {
            "median": float(np.median([row[method]["post_offset_energy_ratio"] for row in events])),
            "mean": float(np.mean([row[method]["post_offset_energy_ratio"] for row in events])),
        }
    return result


def compare(events: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, Any]:
    delta = np.asarray([
        row[candidate]["si_sdr_db"] - row[baseline]["si_sdr_db"]
        for row in events
    ])
    return {
        "median_db": float(np.median(delta)),
        "mean_db": float(np.mean(delta)),
        "wins_gt_0p1db": int((delta > 0.1).sum()),
        "losses_gt_0p1db": int((delta < -0.1).sum()),
        "wins_gt_1db": int((delta > 1.0).sum()),
        "losses_gt_1db": int((delta < -1.0).sum()),
        "wins_gt_3db": int((delta > 3.0).sum()),
        "losses_gt_3db": int((delta < -3.0).sum()),
    }


@torch.no_grad()
def evaluate(
    model: TemporalHybridSetOwnership,
    authorities: scaled.Authorities,
    maximum_mixtures: int,
    parent: TemporalHybridSetOwnership | None = None,
) -> dict[str, Any]:
    model.eval()
    if parent is not None:
        parent.eval()
    events: list[dict[str, Any]] = []
    closure = 0.0
    complex_closure = 0.0
    magnitude_closure_relative = 0.0
    complex_closure_relative = 0.0
    maximum_mean_complex_delta = 0.0
    silent = 0
    for kind in ("fresh", "recorded"):
        count = len(authorities.ids(kind, "validation"))
        if maximum_mixtures:
            count = min(count, maximum_mixtures)
        for index in range(count):
            record = authorities.record(kind, "validation", index, torch.device("cuda"))
            if len(record["raw"]) == 1:
                if (
                    model.config.exact_mixture_additivity
                    and model.config.magnitude_reference == "mixture"
                    and model.config.phase_mode == "mixture"
                ):
                    estimate = record["mixture"][None, None]
                elif model.config.magnitude_reference == "raw_total":
                    mix_mag = record["mixture"].abs()
                    mix_unit = torch.where(
                        mix_mag > model.config.epsilon,
                        record["mixture"] / mix_mag.clamp_min(model.config.epsilon),
                        torch.ones_like(record["mixture"]),
                    )
                    estimate = (record["raw"].abs() * mix_unit[None])[None]
                else:
                    estimate = record["base"][None]
                closure_value = 0.0
                complex_closure_value = 0.0
                magnitude_closure_relative_value = 0.0
                complex_closure_relative_value = 0.0
                mean_complex_delta_value = 0.0
            else:
                estimate, diagnostics = call(model, record)
                closure_value = float(diagnostics["magnitude_closure_max_abs"])
                complex_closure_value = float(
                    diagnostics.get(
                        "complex_closure_max_abs",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(float("nan")),
                    )
                )
                magnitude_closure_relative_value = float(
                    diagnostics.get(
                        "magnitude_closure_max_relative",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(float("nan")),
                    )
                )
                complex_closure_relative_value = float(
                    diagnostics.get(
                        "complex_closure_max_relative",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(float("nan")),
                    )
                )
                mean_complex_delta_value = float(
                    diagnostics.get(
                        "mean_complex_delta",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(0.0),
                    )
                )
            parent_estimate = None
            if parent is not None:
                parent_estimate = (
                    (
                        record["mixture"][None, None]
                        if parent.config.exact_mixture_additivity
                        and parent.config.magnitude_reference == "mixture"
                        and parent.config.phase_mode == "mixture"
                        else record["base"][None]
                    )
                    if len(record["raw"]) == 1
                    else call(parent, record)[0]
                )
            estimate_wave = scaled.istft(estimate[0], record["length"]).float()
            parent_wave = (
                scaled.istft(parent_estimate[0], record["length"]).float()
                if parent_estimate is not None else None
            )
            base_wave = scaled.istft(record["base"], record["length"]).float()
            target_energy = record["targets_wave"].square().sum(-1)
            valid = record["query_valid"] & (target_energy > 1e-12)
            silent += int((record["query_valid"] & ~valid).sum())
            indices = torch.where(valid)[0]
            if not len(indices):
                continue
            raw_metrics = scaled.metrics(
                record["targets_wave"][indices], record["raw_wave"][indices]
            )
            candidate_metrics = scaled.metrics(
                record["targets_wave"][indices], estimate_wave[indices]
            )
            base_metrics = scaled.metrics(
                record["targets_wave"][indices], base_wave[indices]
            )
            parent_metrics = (
                scaled.metrics(record["targets_wave"][indices], parent_wave[indices])
                if parent_wave is not None else None
            )
            method_waves = {
                "candidate": estimate_wave,
                "magnitude": base_wave,
                "raw": record["raw_wave"],
                **({"parent": parent_wave} if parent_wave is not None else {}),
            }
            positions = torch.arange(record["length"], device=record["offsets"].device)[None]
            post_mask = positions >= record["offsets"][:, None]
            target_energy_safe = record["targets_wave"].double().square().sum(-1).clamp_min(1e-12)
            post_ratios = {
                method: (wave.double().square() * post_mask).sum(-1) / target_energy_safe
                for method, wave in method_waves.items()
            }
            for local_note, note in enumerate(indices.tolist()):
                events.append({
                    "dataset_kind": kind,
                    "mixture_id": record["mixture_id"],
                    "note": int(note),
                    **{
                        method: {
                            **{key: float(metric[key][local_note]) for key in metric},
                            "post_offset_energy_ratio": float(post_ratios[method][note]),
                        }
                        for method, metric in {
                            "candidate": candidate_metrics,
                            "magnitude": base_metrics,
                            "raw": raw_metrics,
                            **({"parent": parent_metrics} if parent_metrics is not None else {}),
                        }.items()
                    },
                })
            closure = max(closure, closure_value)
            if np.isfinite(complex_closure_value):
                complex_closure = max(complex_closure, complex_closure_value)
            if np.isfinite(magnitude_closure_relative_value):
                magnitude_closure_relative = max(
                    magnitude_closure_relative, magnitude_closure_relative_value
                )
            if np.isfinite(complex_closure_relative_value):
                complex_closure_relative = max(
                    complex_closure_relative, complex_closure_relative_value
                )
            if np.isfinite(mean_complex_delta_value):
                maximum_mean_complex_delta = max(
                    maximum_mean_complex_delta, mean_complex_delta_value
                )
    if not events:
        raise ValueError("empty development evaluation")
    methods = ("candidate", "magnitude", "raw", "parent") if parent is not None else ("candidate", "magnitude", "raw")
    comparisons = {
        "candidate_minus_magnitude": compare(events, "candidate", "magnitude"),
        "candidate_minus_raw": compare(events, "candidate", "raw"),
    }
    if parent is not None:
        comparisons.update({
            "candidate_minus_parent": compare(events, "candidate", "parent"),
            "parent_minus_magnitude": compare(events, "parent", "magnitude"),
            "parent_minus_raw": compare(events, "parent", "raw"),
        })
    return {
        "coverage": len(events),
        "silent_targets_excluded": silent,
        "summary": {
            method: summarize(events, method)
            for method in methods
        },
        "comparisons": comparisons,
        "maximum_magnitude_closure_abs": closure,
        "maximum_complex_closure_abs": complex_closure,
        "maximum_magnitude_closure_relative": magnitude_closure_relative,
        "maximum_complex_closure_relative": complex_closure_relative,
        "maximum_mean_complex_delta": maximum_mean_complex_delta,
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument(
        "--mode",
        choices=("canary", "canary_wide", "adapter_full_eval", "screen", "full"),
        required=True,
    )
    parser.add_argument("--fresh-index", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA compute required")
    for path in (args.fresh_index, args.recorded_root, args.proposal_root, args.output):
        scaled.forbid_test_path(path)
    if args.warm_start is not None:
        scaled.forbid_test_path(args.warm_start)
    if sha256(args.fresh_index) != FRESH_SHA256:
        raise ValueError("fresh index hash mismatch")
    if sha256(args.recorded_root / "FROZEN.json") != RECORDED_SHA256:
        raise ValueError("recorded authority hash mismatch")
    if sha256(args.proposal_root / "VALIDATION.json") != CACHE_VALIDATION_SHA256:
        raise ValueError("proposal validation hash mismatch")
    if args.output.exists() and not (args.output / "RESUME.pt").exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    scaled.FRESH_SHA = FRESH_SHA256
    cache = scaled.ProposalCache(args.proposal_root)
    authorities = scaled.Authorities(args.fresh_index, args.recorded_root, cache)
    config = TemporalHybridOwnershipConfig(**ARMS[args.arm])
    if args.arm == "essential6_temporal_score_onoff":
        model_class = ScoreOnOffTemporalOwnership
    elif args.arm == "essential6_temporal_offset_distance":
        model_class = OffsetDistanceTemporalOwnership
    elif args.arm == "essential6_temporal_post_offset_distance":
        model_class = PostOffsetDistanceTemporalOwnership
    elif args.arm == "essential6_temporal_post_offset_sparsemax":
        model_class = PostOffsetSparsemaxResidualOwnership
    elif args.arm == "essential6_temporal_post_offset_attenuation":
        model_class = PostOffsetAttenuationOwnership
    else:
        model_class = TemporalHybridSetOwnership
    parent = None
    if args.arm in OFFSET_ARMS:
        if args.warm_start is None or sha256(args.warm_start) != SELECTED_TEMPORAL_SHA256:
            raise ValueError("offset-distance arm requires the selected Temporal ASO warm start")
        source = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if source.get("step") != 29058 or source.get("contract", {}).get("arm") != "essential6_temporal_mixture":
            raise ValueError("warm-start contract mismatch")
        model = model_class(config).to(device)
        parent = TemporalHybridSetOwnership(config).to(device)
        parent.load_state_dict(source["model"], strict=True)
        target_state = model.state_dict()
        for name, value in target_state.items():
            if name in {
                "post_offset_delta.weight", "post_offset_attenuation.weight"
            }:
                value.zero_()
            elif name == "input.base.weight":
                value.copy_(source["model"]["input.weight"])
            elif name == "input.base.bias":
                value.copy_(source["model"]["input.bias"])
            elif name == "input.offset.weight":
                value.zero_()
            else:
                value.copy_(source["model"][name])
        model.load_state_dict(target_state, strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if args.arm in INPUT_OFFSET_ARMS:
            model.input.offset.weight.requires_grad_(True)
        elif args.arm == "essential6_temporal_post_offset_sparsemax":
            model.post_offset_delta.weight.requires_grad_(True)
        else:
            model.post_offset_attenuation.weight.requires_grad_(True)
        incumbent = parent
    elif args.warm_start is not None:
        raise ValueError("warm start is only valid for the offset-distance arm")
    elif args.arm in {
        "essential7_relative_peak_temporal_mixture",
        "essential7_harmonic_prominence_temporal_mixture",
        "essential7_relative_harmonic_loudness_temporal_mixture",
    }:
        # Match every common initial parameter to the six-channel incumbent;
        # the single new input slice starts at zero, so the two models have
        # exactly identical outputs before training.
        incumbent = TemporalHybridSetOwnership(
            TemporalHybridOwnershipConfig(**ARMS["essential6_temporal_mixture"])
        ).to(device)
        model = model_class(config).to(device)
        enhanced = model.state_dict()
        for name, value in incumbent.state_dict().items():
            if name == "input.weight":
                enhanced[name].zero_()
                enhanced[name][:, :value.shape[1]].copy_(value)
            else:
                enhanced[name].copy_(value)
        model.load_state_dict(enhanced, strict=True)
        del enhanced
    else:
        incumbent = None
        model = model_class(config).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    trainable_parameters = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    schedule, singleton_ids = scaled.make_schedule(authorities, "small_full")
    if args.mode == "canary":
        schedule = schedule[:320]
        maximum_mixtures = 16
    elif args.mode == "canary_wide":
        schedule = schedule[:320]
        maximum_mixtures = 64
    elif args.mode == "adapter_full_eval":
        schedule = schedule[:320]
        maximum_mixtures = 0
    elif args.mode == "screen":
        schedule = schedule[:2000]
        maximum_mixtures = 64
    else:
        maximum_mixtures = 0

    initialization_parity_max_abs = None
    if incumbent is not None:
        parity_kind, parity_index = next(
            (kind, index) for kind, index in schedule
            if len(authorities.record(kind, "train", index, device)["raw"]) >= 2
        )
        parity_record = authorities.record(parity_kind, "train", parity_index, device)
        incumbent.eval(); model.eval()
        with torch.no_grad():
            incumbent_output, _ = call(incumbent, parity_record)
            enhanced_output, _ = call(model, parity_record)
        initialization_parity_max_abs = float(
            (incumbent_output - enhanced_output).abs().amax()
        )
        if initialization_parity_max_abs != 0.0:
            raise ValueError(
                f"matched initialization parity failed: {initialization_parity_max_abs}"
            )
        del incumbent_output, enhanced_output, parity_record
        if args.arm not in OFFSET_ARMS:
            del incumbent

    contract = {
        "schema": "temporal_hybrid_aso_training_v1",
        "arm": args.arm,
        "mode": args.mode,
        "seed": SEED,
        "source_archive": os.environ.get("SOURCE_ARCHIVE"),
        "source_archive_sha256": (
            sha256(Path(os.environ["SOURCE_ARCHIVE"]))
            if os.environ.get("SOURCE_ARCHIVE") else None
        ),
        "steps": len(schedule),
        "model_contract": model.contract(),
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "matched_six_channel_initialization_max_abs": initialization_parity_max_abs,
        "optimizer": {
            "kind": "AdamW", "lr": 1e-4,
            "weight_decay": 0.0 if args.arm in OFFSET_ARMS else 1e-4,
            "clip": 1.0,
        },
        "objective": "stable waveform SI-regret plus oracle ownership CE plus 0.01 TF-TV magnitude-delta penalty",
        "fresh_index": str(args.fresh_index.resolve()),
        "fresh_index_sha256": FRESH_SHA256,
        "recorded_authority": str((args.recorded_root / "FROZEN.json").resolve()),
        "recorded_authority_sha256": RECORDED_SHA256,
        "proposal_root": str(args.proposal_root.resolve()),
        "proposal_validation_sha256": CACHE_VALIDATION_SHA256,
        "preregistration": str(PREREG.resolve()),
        "preregistration_sha256": sha256(PREREG),
        "score_onoff_followup_preregistration": (
            str(ONOFF_PREREG.resolve()) if args.arm == "essential6_temporal_score_onoff" else None
        ),
        "score_onoff_followup_preregistration_sha256": (
            sha256(ONOFF_PREREG) if args.arm == "essential6_temporal_score_onoff" else None
        ),
        "relative_peak_followup_preregistration": (
            str(RELATIVE_PEAK_PREREG.resolve())
            if args.arm == "essential7_relative_peak_temporal_mixture" else None
        ),
        "relative_peak_followup_preregistration_sha256": (
            sha256(RELATIVE_PEAK_PREREG)
            if args.arm == "essential7_relative_peak_temporal_mixture" else None
        ),
        "relative_peak_screen_amendment": (
            str(RELATIVE_PEAK_SCREEN.resolve())
            if args.arm == "essential7_relative_peak_temporal_mixture" and args.mode == "screen"
            else None
        ),
        "relative_peak_screen_amendment_sha256": (
            sha256(RELATIVE_PEAK_SCREEN)
            if args.arm == "essential7_relative_peak_temporal_mixture" and args.mode == "screen"
            else None
        ),
        "harmonic_prominence_preregistration": (
            str(HARMONIC_PROMINENCE_PREREG.resolve())
            if args.arm == "essential7_harmonic_prominence_temporal_mixture" else None
        ),
        "harmonic_prominence_preregistration_sha256": (
            sha256(HARMONIC_PROMINENCE_PREREG)
            if args.arm == "essential7_harmonic_prominence_temporal_mixture" else None
        ),
        "relative_harmonic_loudness_preregistration": (
            str(RELATIVE_HARMONIC_LOUDNESS_PREREG.resolve())
            if args.arm == "essential7_relative_harmonic_loudness_temporal_mixture" else None
        ),
        "relative_harmonic_loudness_preregistration_sha256": (
            sha256(RELATIVE_HARMONIC_LOUDNESS_PREREG)
            if args.arm == "essential7_relative_harmonic_loudness_temporal_mixture" else None
        ),
        "raw_total_transport_preregistration": (
            str(RAW_TOTAL_PREREG.resolve())
            if args.arm == "essential6_temporal_raw_total" else None
        ),
        "raw_total_transport_preregistration_sha256": (
            sha256(RAW_TOTAL_PREREG)
            if args.arm == "essential6_temporal_raw_total" else None
        ),
        "offset_distance_preregistration": (
            str(OFFSET_DISTANCE_PREREG.resolve())
            if args.arm == "essential6_temporal_offset_distance" else None
        ),
        "offset_distance_preregistration_sha256": (
            sha256(OFFSET_DISTANCE_PREREG)
            if args.arm == "essential6_temporal_offset_distance" else None
        ),
        "offset_distance_v1b_amendment": (
            str(OFFSET_DISTANCE_V1B.resolve())
            if args.arm == "essential6_temporal_offset_distance" and args.mode == "canary_wide"
            else None
        ),
        "offset_distance_v1b_amendment_sha256": (
            sha256(OFFSET_DISTANCE_V1B)
            if args.arm == "essential6_temporal_offset_distance" and args.mode == "canary_wide"
            else None
        ),
        "offset_distance_v1c_amendment": (
            str(OFFSET_DISTANCE_V1C.resolve())
            if args.arm == "essential6_temporal_offset_distance" and args.mode == "adapter_full_eval"
            else None
        ),
        "offset_distance_v1c_amendment_sha256": (
            sha256(OFFSET_DISTANCE_V1C)
            if args.arm == "essential6_temporal_offset_distance" and args.mode == "adapter_full_eval"
            else None
        ),
        "post_offset_distance_preregistration": (
            str(POST_OFFSET_DISTANCE_PREREG.resolve())
            if args.arm == "essential6_temporal_post_offset_distance" else None
        ),
        "post_offset_distance_preregistration_sha256": (
            sha256(POST_OFFSET_DISTANCE_PREREG)
            if args.arm == "essential6_temporal_post_offset_distance" else None
        ),
        "post_offset_sparsemax_preregistration": (
            str(POST_OFFSET_SPARSEMAX_PREREG.resolve())
            if args.arm == "essential6_temporal_post_offset_sparsemax" else None
        ),
        "post_offset_sparsemax_preregistration_sha256": (
            sha256(POST_OFFSET_SPARSEMAX_PREREG)
            if args.arm == "essential6_temporal_post_offset_sparsemax" else None
        ),
        "post_offset_sparsemax_v3b_amendment": (
            str(POST_OFFSET_SPARSEMAX_V3B.resolve())
            if args.arm == "essential6_temporal_post_offset_sparsemax"
            and args.mode in {"canary_wide", "adapter_full_eval"} else None
        ),
        "post_offset_sparsemax_v3b_amendment_sha256": (
            sha256(POST_OFFSET_SPARSEMAX_V3B)
            if args.arm == "essential6_temporal_post_offset_sparsemax"
            and args.mode in {"canary_wide", "adapter_full_eval"} else None
        ),
        "post_offset_attenuation_preregistration": (
            str(POST_OFFSET_ATTENUATION_PREREG.resolve())
            if args.arm == "essential6_temporal_post_offset_attenuation" else None
        ),
        "post_offset_attenuation_preregistration_sha256": (
            sha256(POST_OFFSET_ATTENUATION_PREREG)
            if args.arm == "essential6_temporal_post_offset_attenuation" else None
        ),
        "warm_start": str(args.warm_start.resolve()) if args.warm_start is not None else None,
        "warm_start_sha256": sha256(args.warm_start) if args.warm_start is not None else None,
        "resource_amendment": str(AMENDMENT.resolve()),
        "resource_amendment_sha256": sha256(AMENDMENT),
        "activation_checkpointing_amendment": str(AMENDMENT_V1C.resolve()),
        "activation_checkpointing_amendment_sha256": sha256(AMENDMENT_V1C),
        "silent_reference_amendment": str(AMENDMENT_V1D.resolve()),
        "silent_reference_amendment_sha256": sha256(AMENDMENT_V1D),
        "runner_sha256": sha256(Path(__file__)),
        "model_source_sha256": sha256(ROOT / "src/model_exp/temporal_hybrid_set_ownership.py"),
        "score_conditioned_model_source_sha256": (
            sha256(ROOT / "src/model_exp/score_conditioned_temporal_ownership.py")
            if args.arm in {"essential6_temporal_score_onoff", *OFFSET_ARMS} else None
        ),
        "singleton_identity_count": len(singleton_ids),
        "test_data_used": False,
    }
    contract_path = args.output / "CONTRACT.json"
    encoded = json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if contract_path.exists() and contract_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("resume contract mismatch")
    if not contract_path.exists():
        contract_path.write_text(encoded, encoding="utf-8")

    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=1e-4,
        weight_decay=0.0 if args.arm in OFFSET_ARMS else 1e-4,
    )
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
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = train_loss(model, record)
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
                    "kind": kind,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient.detach()),
                    "mean_abs_correction": float(diagnostics["mean_abs_correction"].detach()),
                    "smoothness_penalty": float(diagnostics["smoothness_penalty"].detach()),
                    "mean_raw_phase_probability": float(diagnostics["mean_raw_phase_probability"].detach()),
                }
                history.append(row); print(json.dumps(row), flush=True)
            checkpoints = {len(schedule) // 4, len(schedule) // 2, 3 * len(schedule) // 4, len(schedule)}
            if done in checkpoints or done % 500 == 0 or STOP:
                payload = {
                    "schema": "temporal_hybrid_aso_checkpoint_v1",
                    "contract": contract,
                    "step": done,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                }
                atomic_torch(resume, payload)
                if done in checkpoints:
                    atomic_torch(args.output / f"checkpoint-step-{done:06d}.pt", payload)
            if STOP:
                return 75

        development = evaluate(model, authorities, maximum_mixtures, parent=parent)
        final_checkpoint = args.output / f"checkpoint-step-{len(schedule):06d}.pt"
        if not final_checkpoint.exists():
            atomic_torch(final_checkpoint, {
                "schema": "temporal_hybrid_aso_checkpoint_v1",
                "contract": contract,
                "step": len(schedule),
                "model": model.state_dict(),
                "history": history,
            })
        gates = {
            "all_parameters_finite": all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "all_metrics_finite": all(
                np.isfinite(row[method][metric])
                for row in development["events"]
                for method in (
                    ("candidate", "magnitude", "raw", "parent")
                    if parent is not None else ("candidate", "magnitude", "raw")
                )
                for metric in ("si_sdr_db", "off_e", "gain", "scale_invariant_distortion")
            ),
            "nonempty_coverage": development["coverage"] > 0,
            "magnitude_closure_lte_1e_minus4": development["maximum_magnitude_closure_abs"] <= 1e-4,
            "training_completed": len(history) > 0 and history[-1]["step"] == len(schedule),
            "test_data_unused": True,
        }
        if parent is not None:
            candidate_parent = development["comparisons"]["candidate_minus_parent"]
            candidate_magnitude = development["comparisons"]["candidate_minus_magnitude"]
            parent_magnitude = development["comparisons"]["parent_minus_magnitude"]
            gates.update({
                "offset_initialization_exact": initialization_parity_max_abs == 0.0,
                "offset_paired_median_positive": candidate_parent["median_db"] > 0.0,
                "offset_paired_mean_at_least_minus_0p02db": candidate_parent["mean_db"] >= -0.02,
                "offset_median_off_e_not_worse": development["summary"]["candidate"]["off_e"]["median"] <= development["summary"]["parent"]["off_e"]["median"],
                "offset_median_post_energy_not_worse": development["summary"]["candidate"]["post_offset_energy_ratio"]["median"] <= development["summary"]["parent"]["post_offset_energy_ratio"]["median"],
                "offset_parent_severe_loss_gate": (
                    candidate_parent["losses_gt_3db"] / development["coverage"] <= 0.0025
                    if args.mode == "adapter_full_eval"
                    else candidate_parent["losses_gt_3db"] <= 2
                ),
                "offset_severe_magnitude_losses_not_increased": candidate_magnitude["losses_gt_3db"] <= parent_magnitude["losses_gt_3db"],
            })
            if (
                args.arm == "essential6_temporal_post_offset_sparsemax"
                and args.mode in {"canary_wide", "adapter_full_eval"}
            ):
                gates.pop("offset_median_off_e_not_worse")
                gates["offset_median_off_e_within_0p5pct"] = (
                    development["summary"]["candidate"]["off_e"]["median"]
                    <= 1.005 * development["summary"]["parent"]["off_e"]["median"]
                )
                gates["offset_mean_off_e_improves"] = (
                    development["summary"]["candidate"]["off_e"]["mean"]
                    <= development["summary"]["parent"]["off_e"]["mean"]
                )
            if args.arm == "essential6_temporal_post_offset_attenuation":
                gates.pop("magnitude_closure_lte_1e_minus4")
                gates["nonredistributing_attenuation_contract"] = (
                    model.contract().get("redistribute_removed_energy") is False
                    and model.contract().get("body_invariance")
                    == "output is exactly unchanged before acoustic offset"
                )
                gates["offset_mean_off_e_improves"] = (
                    development["summary"]["candidate"]["off_e"]["mean"]
                    <= development["summary"]["parent"]["off_e"]["mean"]
                )
        result = {
            "schema": "temporal_hybrid_aso_result_v1",
            "status": "passed" if all(gates.values()) else "failed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "device": torch.cuda.get_device_name(),
            "runtime_seconds": time.monotonic() - started,
            "contract": contract,
            "checkpoint": str(final_checkpoint.resolve()),
            "checkpoint_sha256": sha256(final_checkpoint),
            "development": development,
            "gates": gates,
            "test_data_used": False,
        }
        atomic_json(args.output / "result.json", result)
        print(json.dumps({
            "status": result["status"], "arm": args.arm,
            "coverage": development["coverage"],
            "median_si_sdr": development["summary"]["candidate"]["si_sdr_db"]["median"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        }, indent=2), flush=True)
        return 0 if result["status"] == "passed" else 2
    finally:
        authorities.close()


if __name__ == "__main__":
    raise SystemExit(main())
