#!/usr/bin/env python3
"""Train a residual corrector with complete renderer-source holdouts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_corrector import (  # noqa: E402
    DNAComplexResidualCorrector,
    DNAResidualCorrectorConfig,
    dna_residual_losses,
)
from model_exp.literal_residual_canary_store import (  # noqa: E402
    PackedLiteralResidualCanaryCatalog,
)
from model_exp.dna_residual_corpus import FrozenDNAResidualPriorCatalog  # noqa: E402


SAMPLE_RATE = 16_000
TRAINING_WEIGHTS = {
    "perceptual_target": {
        "complex_weight": 0.1,
        "log_magnitude_weight": 1.0,
        "waveform_weight": 0.1,
        "envelope_weight": 0.5,
        "phase_coherence_weight": 0.25,
        "do_no_harm_weight": 0.0,
        "correction_weight": 0.0,
    },
    # Learn the target residual without rewarding identity copying. DNA remains
    # an input/skip connection, while do-no-harm is evaluated only as a gate.
    "free_target": {
        "do_no_harm_weight": 0.0,
        "correction_weight": 0.0,
    },
    "standard": {},
    "standard_guarded": {
        "quiet_region_weight": 0.02,
        "quiet_do_no_harm_weight": 0.01,
        "onset_derivative_weight": 0.005,
        "offset_derivative_weight": 0.005,
        "boundary_do_no_harm_weight": 0.01,
        "tail_energy_weight": 0.005,
        "tail_do_no_harm_weight": 0.005,
    },
    "guarded_curriculum": {
        "quiet_region_weight": 0.02,
        "quiet_do_no_harm_weight": 0.01,
        "onset_derivative_weight": 0.005,
        "offset_derivative_weight": 0.005,
        "boundary_do_no_harm_weight": 0.01,
        "tail_energy_weight": 0.005,
        "tail_do_no_harm_weight": 0.005,
    },
    "tail_boundary": {
        "quiet_region_weight": 0.1,
        "quiet_do_no_harm_weight": 0.2,
        "onset_derivative_weight": 0.1,
        "offset_derivative_weight": 0.1,
        "boundary_do_no_harm_weight": 0.2,
        "tail_energy_weight": 0.1,
        "tail_do_no_harm_weight": 0.2,
        "tail_cvar_weight": 0.1,
    },
    "moderate_tail": {
        "quiet_region_weight": 0.02,
        "quiet_do_no_harm_weight": 0.05,
        "onset_derivative_weight": 0.05,
        "offset_derivative_weight": 0.05,
        "boundary_do_no_harm_weight": 0.05,
        "tail_energy_weight": 0.1,
        "tail_do_no_harm_weight": 0.05,
        "tail_cvar_weight": 0.05,
    },
}
EVALUATION_WEIGHTS = {
    "quiet_region_weight": 1.0,
    "quiet_do_no_harm_weight": 1.0,
    "onset_derivative_weight": 1.0,
    "offset_derivative_weight": 1.0,
    "boundary_do_no_harm_weight": 1.0,
    "tail_energy_weight": 1.0,
    "tail_do_no_harm_weight": 1.0,
    "tail_cvar_weight": 1.0,
}


def stable_source_holdout(source_ids: list[str], fraction: float, seed: int) -> set[str]:
    if not source_ids or not 0.0 < fraction < 1.0:
        raise ValueError("source holdout requires sources and a fraction in (0, 1)")
    ordered = sorted(
        set(source_ids),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    count = min(len(ordered) - 1, max(1, round(len(ordered) * fraction)))
    return set(ordered[:count])


def internal_holdout(
    rows: list[dict[str, Any]], fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, set[str]]:
    sources = {row["source_id"] for row in rows}
    if len(sources) > 1:
        heldout = stable_source_holdout(list(sources), fraction, seed)
        validation = [row for row in rows if row["source_id"] in heldout]
        training = [row for row in rows if row["source_id"] not in heldout]
        return training, validation, "internal_renderer_source_holdout", set()

    heldout_sounds = stable_source_holdout(
        [row["sound_id"] for row in rows], fraction, seed
    )
    validation = [row for row in rows if row["sound_id"] in heldout_sounds]
    validation_parents = {row["score_parent_id"] for row in validation}
    training = [
        row
        for row in rows
        if row["sound_id"] not in heldout_sounds
        and row["score_parent_id"] not in validation_parents
    ]
    return (
        training,
        validation,
        "internal_sound_and_score_parent_holdout",
        heldout_sounds,
    )


def training_weights(arm: str, step: int, total_steps: int) -> dict[str, float]:
    weights = TRAINING_WEIGHTS[arm]
    if arm != "guarded_curriculum":
        return dict(weights)
    progress = step / max(total_steps, 1)
    ramp = min(1.0, max(0.0, (progress - 0.1) / 0.5))
    smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    return {key: value * smooth_ramp for key, value in weights.items()}


def target_complete_window(metadata: dict[str, Any], total: int) -> tuple[int, int]:
    onset = int(metadata["onset_sample"])
    offset = int(metadata["offset_sample"])
    start = max(0, onset - int(0.25 * SAMPLE_RATE))
    stop = min(total, offset + int(0.5 * SAMPLE_RATE))
    if not start <= onset < offset <= stop:
        raise ValueError("target note is not completely retained")
    return start, stop


def pad(values: np.ndarray, samples: int) -> np.ndarray:
    output = np.zeros(samples, dtype=np.float32)
    source = np.asarray(values, dtype=np.float32)
    if len(source) > samples:
        raise ValueError("cannot pad an array into a shorter output")
    output[: len(source)] = source
    return output


def load_raw_rows(
    packed_root: Path, *, include_region_context: bool = False
) -> list[dict[str, Any]]:
    rows = []
    with PackedLiteralResidualCanaryCatalog(packed_root) as catalog:
        items = [catalog.load(index) for index in range(len(catalog))]
        if include_region_context:
            missing = [
                item["sample_id"]
                for item in items
                if item.get("dna_context_sum") is None
                or not bool(item["metadata"].get("complete_dna_context_sum", False))
                or int(item["metadata"].get("dna_context_notes", -1))
                != int(item["metadata"]["context_notes"])
            ]
            if missing:
                raise ValueError(
                    "region context requires complete all-note DNA sums; "
                    f"missing or incomplete for {len(missing)} rows"
                )
        for item in items:
            start, stop = target_complete_window(item["metadata"], len(item["target"]))
            row = {
                "sample_id": item["sample_id"],
                "sound_id": str(item["metadata"]["sound_id"]),
                "source_id": str(item["metadata"]["source_id"]),
                "score_parent_id": str(item["metadata"]["score_parent_id"]),
                "metadata": item["metadata"],
                "mixture": np.ascontiguousarray(item["mixture"][start:stop]),
                "target": np.ascontiguousarray(item["target"][start:stop]),
                "dna": np.ascontiguousarray(item["dna"][start:stop]),
                "valid_samples": stop - start,
                "pitch": int(item["pitch"]),
            }
            if include_region_context:
                all_dna = np.asarray(item["dna_context_sum"], dtype=np.float32)
                row["other_dna"] = np.ascontiguousarray(
                    (all_dna - item["dna"])[start:stop]
                )
                row["unassigned_residual"] = np.ascontiguousarray(
                    (item["mixture"] - all_dna)[start:stop]
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"packed root is empty: {packed_root}")
    return rows


def load_raw_roots(
    packed_roots: list[Path], *, include_region_context: bool = False
) -> list[dict[str, Any]]:
    rows = []
    for packed_root in packed_roots:
        resolved = packed_root.expanduser().resolve()
        loaded = (
            load_raw_rows(resolved, include_region_context=True)
            if include_region_context
            else load_raw_rows(resolved)
        )
        rows.extend(loaded)
        print(
            "DATA_LOAD "
            + json.dumps(
                {"root": str(resolved), "root_rows": len(loaded), "total_rows": len(rows)},
                sort_keys=True,
            ),
            flush=True,
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("packed roots contain duplicate pair identities")
    return rows


def load_frozen_residual_rows(root: Path, *, split: str = "validation") -> list[dict[str, Any]]:
    """Normalize a frozen recorded residual corpus to the canary evaluation contract."""

    rows = []
    with FrozenDNAResidualPriorCatalog(root.expanduser().resolve(), split=split) as catalog:
        for record in catalog.iter_records():
            metadata = dict(record.metadata)
            samples = int(metadata["samples"])
            dna, _ = catalog.load_priors(record, include_rt2=False)
            mixture, target = catalog.load_self_contained_raw_slice(
                record, start=0, count=samples
            )
            start_step = int(metadata["start_step"])
            end_step = int(metadata["end_step"])
            step_span = end_step - start_step
            if step_span < 1 or samples % step_span:
                raise ValueError("recorded residual timeline does not map exactly to samples")
            samples_per_step = samples // step_span
            onset = (int(metadata["onset_step"]) - start_step) * samples_per_step
            offset = (int(metadata["offset_step"]) - start_step) * samples_per_step
            difficulty = dict(metadata.get("difficulty", {}))
            peak_competitors = int(difficulty.get("peak_competitors", 0))
            staggered_competitors = int(difficulty.get("staggered_competitors", 0))
            normalized_metadata = {
                **metadata,
                "mixture_id": str(metadata.get("sample_id", record.asset_id)),
                "score_parent_id": str(metadata["source_parent_id"]),
                "matched_score_family": str(metadata.get("source_instrument", "recorded")),
                "velocity": int(metadata.get("velocity", -1)),
                "onset_sample": onset,
                "offset_sample": offset,
                "target_notes": 1,
                "difficulty": {
                    **difficulty,
                    "peak_competitors": peak_competitors,
                    "staggered_competitors": staggered_competitors,
                },
                "context_notes": peak_competitors + 1,
                "release_context_notes": staggered_competitors,
                "dna_rms": (
                    rms := float(np.sqrt(np.mean(np.square(dna, dtype=np.float64))))
                ),
            }
            target_rms = float(metadata["target_rms"])
            normalized_metadata["dna_to_target_rms_ratio"] = rms / max(target_rms, 1e-12)
            start, stop = target_complete_window(normalized_metadata, samples)
            rows.append(
                {
                    "sample_id": record.asset_id,
                    "sound_id": f"{metadata['source_dataset']}:{metadata.get('source_instrument', 'unknown')}",
                    "source_id": str(metadata["source_dataset"]),
                    "score_parent_id": str(metadata["source_parent_id"]),
                    "metadata": normalized_metadata,
                    "mixture": np.ascontiguousarray(mixture[start:stop]),
                    "target": np.ascontiguousarray(target[start:stop]),
                    "dna": np.ascontiguousarray(dna[start:stop]),
                    "valid_samples": stop - start,
                    "pitch": int(metadata["pitch"]),
                }
            )
    if not rows:
        raise ValueError(f"frozen residual root is empty: {root}")
    return rows


def make_batch(rows: list[dict[str, Any]], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    selected = [rows[int(index)] for index in indices]
    maximum_samples = int(
        math.ceil(max(int(row["valid_samples"]) for row in selected) / 1024) * 1024
    )
    result = {}
    audio_keys = ["mixture", "target", "dna"]
    if "other_dna" in selected[0]:
        if not all("other_dna" in row and "unassigned_residual" in row for row in selected):
            raise ValueError("batch mixes region-context and ordinary rows")
        audio_keys.extend(("other_dna", "unassigned_residual"))
    for key in audio_keys:
        result[key] = torch.as_tensor(
            np.stack([pad(row[key], maximum_samples) for row in selected]), device=device
        )
    for key in ("pitch", "valid_samples"):
        result[key] = torch.as_tensor(
            [row[key] for row in selected], device=device
        )
    return result


@torch.no_grad()
def evaluate(
    model: DNAComplexResidualCorrector,
    rows: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    correction_scale: float = 1.0,
    mixture_mode: str = "normal",
) -> dict[str, float]:
    if mixture_mode not in {"normal", "shuffled", "zero"}:
        raise ValueError(f"unsupported mixture mode: {mixture_mode}")
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    for start in range(0, len(rows), batch_size):
        indices = np.arange(start, min(len(rows), start + batch_size))
        batch = make_batch(rows, indices, device)
        if mixture_mode == "zero":
            batch["mixture"] = torch.zeros_like(batch["mixture"])
        elif mixture_mode == "shuffled":
            partner_indices = (indices + max(1, len(rows) // 2)) % len(rows)
            partner = make_batch(rows, partner_indices, device)["mixture"]
            samples = batch["mixture"].shape[-1]
            if partner.shape[-1] < samples:
                partner = torch.nn.functional.pad(partner, (0, samples - partner.shape[-1]))
            batch["mixture"] = partner[:, :samples]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            estimate, diagnostics = model(
                batch["mixture"], batch["dna"], batch["pitch"].long(),
                valid_samples=batch["valid_samples"].long(),
                other_dna=(
                    batch.get("other_dna") if model.config.use_region_context else None
                ),
                unassigned_residual=(
                    batch.get("unassigned_residual")
                    if model.config.use_region_context
                    else None
                ),
            )
            correction = diagnostics["correction"]
            if correction_scale != 1.0:
                correction = float(correction_scale) * correction
                estimate = batch["dna"] + correction
        losses = dna_residual_losses(
            estimate,
            batch["target"],
            batch["dna"],
            correction,
            valid_samples=batch["valid_samples"].long(),
            **EVALUATION_WEIGHTS,
        )
        count = len(indices)
        seen += count
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + count * float(value.item())
    return {key: value / seen for key, value in totals.items()}


def score(metrics: dict[str, float]) -> float:
    return (
        metrics["complex"]
        + 0.5 * metrics["log_magnitude"]
        + 0.05 * metrics["quiet_do_no_harm"]
        + 0.05 * metrics["boundary_do_no_harm"]
        + 0.02 * metrics["tail_do_no_harm"]
    )


def tail_score(metrics: dict[str, float]) -> float:
    return (
        metrics["complex"]
        + 0.5 * metrics["log_magnitude"]
        + 0.01 * metrics["tail_energy_error_db"]
        + 0.05 * metrics["quiet_do_no_harm"]
        + 0.05 * metrics["boundary_do_no_harm"]
        + 0.02 * metrics["tail_do_no_harm"]
    )


def spectral_score(metrics: dict[str, float]) -> float:
    return metrics["complex"] + 0.5 * metrics["log_magnitude"]


def perceptual_score(metrics: dict[str, float]) -> float:
    return (
        metrics["log_magnitude"]
        + 0.25 * metrics["phase_coherence"]
        + 0.1 * metrics["envelope"]
        + 0.1 * metrics["waveform_l1"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-root", type=Path, action="append", required=True)
    parser.add_argument("--validation-packed-root", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(TRAINING_WEIGHTS), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--curriculum-schedule-steps",
        type=int,
        default=0,
        help="Reference duration for curriculum timing; zero uses --steps.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument(
        "--output-mode",
        choices=(
            "bounded_residual",
            "complex_mask_delta",
            "soft_replace",
            "unassigned_residual_filter",
            "ungated_residual",
        ),
        default="bounded_residual",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--region-context", action="store_true")
    parser.add_argument("--harmonic-features", action="store_true")
    parser.add_argument("--frequency-dilated-blocks", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("source-OOD canary requires CUDA")
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.curriculum_schedule_steps < 0:
        raise ValueError("training limits must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda", 0)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    train_raw = load_raw_roots(
        args.packed_root, include_region_context=args.region_context
    )
    heldout_sound_ids: set[str] = set()
    if args.validation_packed_root is None:
        train_raw, validation_raw, validation_mode, heldout_sound_ids = internal_holdout(
            train_raw, args.holdout_fraction, args.seed
        )
    else:
        validation_mode = "separate_pack_sound_and_score_ood"
        validation_raw = load_raw_roots(
            args.validation_packed_root,
            include_region_context=args.region_context,
        )
    train_rows = train_raw
    validation_rows = validation_raw
    maximum_samples = max(
        int(row["valid_samples"]) for row in train_rows + validation_rows
    )
    train_parents = {row["score_parent_id"] for row in train_rows}
    validation_parents = {row["score_parent_id"] for row in validation_rows}
    if not train_rows or not validation_rows or train_parents & validation_parents:
        raise RuntimeError("source holdout failed to produce parent-disjoint nonempty splits")
    train_sounds = {row["sound_id"] for row in train_rows}
    validation_sounds = {row["sound_id"] for row in validation_rows}
    if train_sounds & validation_sounds:
        raise RuntimeError("training and validation packs share sound identities")
    print(
        "DATA_READY "
        + json.dumps(
            {
                "maximum_samples": maximum_samples,
                "train_pairs": len(train_rows),
                "validation_pairs": len(validation_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    config = DNAResidualCorrectorConfig(
        channels=args.channels,
        blocks=args.blocks,
        output_mode=args.output_mode,
        use_region_context=args.region_context,
        use_harmonic_features=args.harmonic_features,
        use_frequency_dilated_blocks=args.frequency_dilated_blocks,
        sample_rate=SAMPLE_RATE,
    )
    model = DNAComplexResidualCorrector(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = np.random.default_rng(args.seed)
    baseline = evaluate(model, validation_rows, args.batch_size, device)
    print("BASELINE " + json.dumps(baseline, sort_keys=True), flush=True)
    best = {"step": 0, "score": score(baseline), "metrics": baseline}
    best_tail = {"step": 0, "score": tail_score(baseline), "metrics": baseline}
    best_spectral = {
        "step": 0,
        "score": spectral_score(baseline),
        "metrics": baseline,
    }
    best_perceptual = {
        "step": 0,
        "score": perceptual_score(baseline),
        "metrics": baseline,
    }
    best_path = output / "best.pt"
    best_tail_path = output / "best_tail.pt"
    best_spectral_path = output / "best_spectral.pt"
    best_perceptual_path = output / "best_perceptual.pt"
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": 0}, best_path)
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": 0}, best_tail_path)
    torch.save(
        {"model": model.state_dict(), "config": config.__dict__, "step": 0},
        best_spectral_path,
    )
    torch.save(
        {"model": model.state_dict(), "config": config.__dict__, "step": 0},
        best_perceptual_path,
    )
    history = [{"step": 0, "validation": baseline, "score": best["score"]}]

    model.train()
    for step in range(1, args.steps + 1):
        indices = generator.choice(
            len(train_rows), size=args.batch_size, replace=len(train_rows) < args.batch_size
        )
        batch = make_batch(train_rows, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            estimate, diagnostics = model(
                batch["mixture"], batch["dna"], batch["pitch"].long(),
                valid_samples=batch["valid_samples"].long(),
                other_dna=batch.get("other_dna"),
                unassigned_residual=batch.get("unassigned_residual"),
            )
        losses = dna_residual_losses(
            estimate,
            batch["target"],
            batch["dna"],
            diagnostics["correction"],
            valid_samples=batch["valid_samples"].long(),
            **training_weights(
                args.arm,
                step,
                args.curriculum_schedule_steps or args.steps,
            ),
        )
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"nonfinite loss at step {step}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, **{k: float(v.item()) for k, v in losses.items()}}, sort_keys=True), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, validation_rows, args.batch_size, device)
            current_score = score(metrics)
            event = {"step": step, "validation": metrics, "score": current_score}
            history.append(event)
            print("VALIDATION " + json.dumps(event, sort_keys=True), flush=True)
            if current_score < float(best["score"]):
                best = event
                temporary = output / "best.pt.partial"
                torch.save({"model": model.state_dict(), "config": config.__dict__, "step": step}, temporary)
                os.replace(temporary, best_path)
            current_tail_score = tail_score(metrics)
            if current_tail_score < float(best_tail["score"]):
                best_tail = {**event, "score": current_tail_score}
                temporary = output / "best_tail.pt.partial"
                torch.save({"model": model.state_dict(), "config": config.__dict__, "step": step}, temporary)
                os.replace(temporary, best_tail_path)
            current_spectral_score = spectral_score(metrics)
            if current_spectral_score < float(best_spectral["score"]):
                best_spectral = {**event, "score": current_spectral_score}
                temporary = output / "best_spectral.pt.partial"
                torch.save(
                    {"model": model.state_dict(), "config": config.__dict__, "step": step},
                    temporary,
                )
                os.replace(temporary, best_spectral_path)
            current_perceptual_score = perceptual_score(metrics)
            if current_perceptual_score < float(best_perceptual["score"]):
                best_perceptual = {**event, "score": current_perceptual_score}
                temporary = output / "best_perceptual.pt.partial"
                torch.save(
                    {"model": model.state_dict(), "config": config.__dict__, "step": step},
                    temporary,
                )
                os.replace(temporary, best_perceptual_path)
            model.train()

    final_path = output / "final.pt"
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": args.steps}, final_path)
    selected_path = (
        best_perceptual_path
        if args.arm == "perceptual_target"
        else
        best_spectral_path
        if args.arm in {
            "free_target",
            "standard",
            "standard_guarded",
            "guarded_curriculum",
        }
        else best_tail_path
    )
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    listening = output / "listening"
    listening.mkdir()
    selected = sorted(
        validation_rows,
        key=lambda row: (
            -int(row["metadata"]["target_notes"]),
            -float(row["metadata"]["target_rms"]),
            row["sample_id"],
        ),
    )[:16]
    model.eval()
    for index, row in enumerate(selected, start=1):
        batch = make_batch([row], np.asarray([0]), device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            estimate, _ = model(
                batch["mixture"], batch["dna"], batch["pitch"].long(),
                valid_samples=batch["valid_samples"].long(),
                other_dna=batch.get("other_dna"),
                unassigned_residual=batch.get("unassigned_residual"),
            )
        valid = int(row["valid_samples"])
        for label, values in (
            ("mixture", row["mixture"][:valid]),
            ("exact", row["target"][:valid]),
            ("dna", row["dna"][:valid]),
            ("corrected", estimate[0, :valid].float().cpu().numpy()),
        ):
            sf.write(listening / f"{index:02d}_{label}.wav", values, SAMPLE_RATE, subtype="PCM_24")
    report = {
        "schema": "musical_residual_source_ood_canary_v1",
        "status": "completed",
        "claim_boundary": "Small source-held-out canary; not production promotion evidence.",
        "arm": args.arm,
        "curriculum_schedule_steps": args.curriculum_schedule_steps or args.steps,
        "packed_root": (
            str(args.packed_root[0].expanduser().resolve())
            if len(args.packed_root) == 1
            else None
        ),
        "packed_roots": [str(path.expanduser().resolve()) for path in args.packed_root],
        "validation_packed_root": (
            str(args.validation_packed_root[0].expanduser().resolve())
            if args.validation_packed_root is not None
            and len(args.validation_packed_root) == 1
            else None
        ),
        "validation_packed_roots": (
            [str(path.expanduser().resolve()) for path in args.validation_packed_root]
            if args.validation_packed_root is not None
            else None
        ),
        "validation_mode": validation_mode,
        "train_pairs": len(train_rows),
        "validation_pairs": len(validation_rows),
        "train_sources": sorted({row["source_id"] for row in train_rows}),
        "validation_sources": sorted({row["source_id"] for row in validation_rows}),
        "heldout_sound_ids": sorted(heldout_sound_ids),
        "train_sounds": len(train_sounds),
        "validation_sounds": len(validation_sounds),
        "train_score_parents": len(train_parents),
        "validation_score_parents": len(validation_parents),
        "maximum_samples": maximum_samples,
        "steps": args.steps,
        "training_contract": {
            "batch_size": args.batch_size,
            "blocks": args.blocks,
            "channels": args.channels,
            "eval_every": args.eval_every,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "learning_rate": args.lr,
            "model_config": config.__dict__,
            "seed": args.seed,
            "region_context": args.region_context,
            "training_weights_at_final_step": training_weights(
                args.arm,
                args.steps,
                args.curriculum_schedule_steps or args.steps,
            ),
        },
        "baseline": baseline,
        "best": best,
        "best_tail": best_tail,
        "best_spectral": best_spectral,
        "best_perceptual": best_perceptual,
        "listening_checkpoint": str(selected_path),
        "history": history,
        "gpu": torch.cuda.get_device_name(device),
        "source_counts": dict(
            sorted(Counter(row["source_id"] for row in train_rows + validation_rows).items())
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("SOURCE_OOD_CANARY_COMPLETE " + json.dumps({"arm": args.arm, "best": best, "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
