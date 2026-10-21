#!/usr/bin/env python3
"""Train only the merge stage of an over-decompose-then-group separator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_v13_streaming import FrozenV13Contract  # noqa: E402
from model_exp.occurrence_atomic_merger import (  # noqa: E402
    OccurrenceAtomicMerger,
    OccurrenceAtomicMergerConfig,
)
from model_exp.occurrence_synthetic_note_bank import (  # noqa: E402
    CompleteSyntheticBankReader,
    select_complete_mixture_groups,
)
import scripts.train_dna_residual_v13_streaming as synthetic_corpus  # noqa: E402
from scripts.audit_occurrence_atomic_merge_feasibility import (  # noqa: E402
    finite_mean,
    source_metrics,
    stft,
    strict_atom_partition,
)
from scripts.train_occurrence_joint_note_bank import (  # noqa: E402
    _synthetic_bank_to_device,
)
from scripts.train_occurrence_score_remix_ssl import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    file_sha256,
    load_dual_hpss_checkpoint,
    project_input,
)


SCHEMA = "occurrence_atomic_merger_canary_v1"
SAMPLE_RATE = 16_000
INPUT_FEATURES = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--grouped-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--train-groups", type=int, default=32)
    parser.add_argument("--validation-groups", type=int, default=8)
    parser.add_argument("--train-candidate-pool", type=int, default=256)
    parser.add_argument("--validation-candidate-pool", type=int, default=128)
    parser.add_argument("--maximum-true-residual-ratio", type=float, default=1e-3)
    parser.add_argument("--crop-samples", type=int, default=49_152)
    parser.add_argument("--maximum-sources", type=int, default=64)
    parser.add_argument("--query-microbatch", type=int, default=4)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--atom-frequency-bins", type=int, default=3)
    parser.add_argument("--atom-time-frames", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.steps,
        args.eval_every,
        args.train_groups,
        args.validation_groups,
        args.train_candidate_pool,
        args.validation_candidate_pool,
        args.crop_samples,
        args.maximum_sources,
        args.query_microbatch,
        args.n_fft,
        args.atom_frequency_bins,
        args.atom_time_frames,
        args.hidden_channels,
        args.blocks,
    )
    if min(positive) < 1 or args.n_fft % 4:
        raise ValueError("invalid merger canary geometry")
    if (
        args.train_candidate_pool < args.train_groups
        or args.validation_candidate_pool < args.validation_groups
        or not 0 <= args.maximum_true_residual_ratio <= 1
        or args.learning_rate <= 0
        or args.gradient_clip <= 0
    ):
        raise ValueError("invalid merger canary limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def select_examples(
    contract: FrozenV13Contract,
    ordinals: np.ndarray,
    *,
    count: int,
    candidate_pool: int,
    crop_samples: int,
    maximum_sources: int,
    maximum_true_residual_ratio: float,
    seed: int,
    split: str,
    support_mode: str = "scored",
) -> list[dict[str, Any]]:
    groups = select_complete_mixture_groups(
        contract, ordinals, count=candidate_pool, seed=seed
    )
    reader = CompleteSyntheticBankReader(
        contract,
        fixed_samples=crop_samples,
        maximum_sources=maximum_sources,
        support_mode=support_mode,
        cache_size=8,
    )
    selected: list[dict[str, Any]] = []
    try:
        for candidate, (mixture_id, rows) in enumerate(groups):
            try:
                example = reader.load(
                    mixture_id, rows, crop_seed=seed + candidate
                )
            except ValueError:
                continue
            if example["true_residual_ratio"] > maximum_true_residual_ratio:
                continue
            selected.append(example)
            print(
                json.dumps(
                    {
                        "cache_selection": split,
                        "selected": len(selected),
                        "requested": count,
                        "candidate": candidate + 1,
                        "mixture_id": mixture_id,
                        "sources": int(example["targets"].shape[0]),
                        "true_residual_ratio": float(example["true_residual_ratio"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if len(selected) == count:
                break
    finally:
        reader.close()
    if len(selected) != count:
        raise RuntimeError(f"{split} yielded {len(selected)}/{count} complete banks")
    return selected


@torch.inference_mode()
def teacher_evidence(
    backbone: torch.nn.Module,
    bank: dict[str, torch.Tensor],
    *,
    query_microbatch: int,
    input_gate_strength: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    harmonic, percussive = project_input(backbone, bank["mixture"], bank["valid"])
    assignment = bank["assignment"].long()
    valid = bank["valid"].index_select(0, assignment)
    waveforms: list[torch.Tensor] = []
    harmonic_logits: list[torch.Tensor] = []
    percussive_logits: list[torch.Tensor] = []
    for start in range(0, int(assignment.shape[0]), query_microbatch):
        stop = min(start + query_microbatch, int(assignment.shape[0]))
        local = assignment[start:stop]
        with torch.autocast(
            device_type=bank["mixture"].device.type,
            dtype=torch.bfloat16,
            enabled=bank["mixture"].device.type == "cuda",
        ):
            estimate, diagnostics = backbone(
                harmonic.index_select(0, local),
                percussive.index_select(0, local),
                bank["pitch"][start:stop],
                valid[start:stop],
                bank["onset"][start:stop],
                bank["offset"][start:stop],
                input_gate_strength=(
                    None
                    if input_gate_strength is None
                    else input_gate_strength[start:stop]
                ),
            )
        waveforms.append(estimate.float())
        harmonic_logits.append(
            diagnostics["harmonic_branch"]["ownership_logits"].float()
        )
        percussive_logits.append(
            diagnostics["percussive_branch"]["ownership_logits"].float()
        )
    return (
        torch.cat(waveforms).clone(),
        torch.cat(harmonic_logits).clone(),
        torch.cat(percussive_logits).clone(),
    )


def padded_pool(values: torch.Tensor, tile: tuple[int, int]) -> torch.Tensor:
    frequency_tile, time_tile = tile
    frequency, frames = values.shape[-2:]
    padded = F.pad(
        values,
        (0, (-frames) % time_tile, 0, (-frequency) % frequency_tile),
    )
    return F.avg_pool2d(
        padded.flatten(0, -3),
        kernel_size=tile,
        stride=tile,
    ).reshape(*values.shape[:-2], padded.shape[-2] // frequency_tile, padded.shape[-1] // time_tile)


def build_record(
    backbone: torch.nn.Module,
    example: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    *,
    validation: bool,
) -> dict[str, Any]:
    bank = _synthetic_bank_to_device(example, device)
    teacher_waveform, harmonic_logits, percussive_logits = teacher_evidence(
        backbone, bank, query_microbatch=args.query_microbatch
    )
    mixture_waveform = bank["mixture"][0].float()
    target_waveform = bank["targets"].float()
    mixture = stft(mixture_waveform, args.n_fft)
    targets = stft(target_waveform, args.n_fft)
    anchors = stft(teacher_waveform, args.n_fft)
    target_background = mixture - targets.sum(dim=0)
    anchor_background = mixture - anchors.sum(dim=0)
    target_bank = torch.cat((targets, target_background[None]), dim=0)
    anchor_bank = torch.cat((anchors, anchor_background[None]), dim=0)
    sources, frequency, frames = target_bank.shape
    note_sources = sources - 1

    harmonic_logits = F.interpolate(
        harmonic_logits[:, None], size=(frequency, frames), mode="bilinear", align_corners=False
    )[:, 0]
    percussive_logits = F.interpolate(
        percussive_logits[:, None], size=(frequency, frames), mode="bilinear", align_corners=False
    )[:, 0]
    harmonic_logits = torch.cat(
        (harmonic_logits, -harmonic_logits.amax(dim=0, keepdim=True)), dim=0
    )
    percussive_logits = torch.cat(
        (percussive_logits, -percussive_logits.amax(dim=0, keepdim=True)), dim=0
    )

    mixture_power = mixture.abs().square()
    mixture_floor = 0.01 * mixture_power.mean().clamp_min(1e-10)
    phase_ratio = anchor_bank * mixture.conj()[None] / (mixture_power + mixture_floor)[None]
    log_anchor = torch.log1p(100.0 * anchor_bank.abs()) / 5.0
    log_mixture = torch.log1p(100.0 * mixture.abs()) / 5.0
    log_flux = F.pad((log_mixture[:, 1:] - log_mixture[:, :-1]).abs(), (1, 0))
    anchor_share = anchor_bank.abs() / anchor_bank.abs().sum(dim=0, keepdim=True).clamp_min(1e-8)

    frequency_hz = torch.arange(frequency, device=device) * SAMPLE_RATE / args.n_fft
    f0 = 440.0 * torch.pow(
        torch.tensor(2.0, device=device), (bank["pitch"].float() - 69.0) / 12.0
    )
    relative = 12.0 * torch.log2(
        frequency_hz.clamp_min(SAMPLE_RATE / args.n_fft)[None] / f0[:, None]
    )
    relative_clipped = (relative / 48.0).clamp(-1.0, 1.0)
    relative_sine = torch.sin(torch.pi * relative / 6.0)
    relative_cosine = torch.cos(torch.pi * relative / 6.0)
    zero_frequency = torch.zeros((1, frequency), device=device)
    relative_clipped = torch.cat((relative_clipped, zero_frequency), dim=0)
    relative_sine = torch.cat((relative_sine, zero_frequency), dim=0)
    relative_cosine = torch.cat((relative_cosine, zero_frequency), dim=0)

    frame_sample = torch.arange(frames, device=device) * (args.n_fft // 4)
    onset = bank["onset"].long()
    offset = bank["offset"].long()
    active = (frame_sample[None] >= onset[:, None]) & (
        frame_sample[None] < offset[:, None]
    )
    onset_state = active & (frame_sample[None] < onset[:, None] + SAMPLE_RATE // 25)
    held_state = active & ~onset_state
    off_state = ~active
    since = ((frame_sample[None] - onset[:, None]).float() / (2 * SAMPLE_RATE)).clamp(0, 1)
    until = ((offset[:, None] - frame_sample[None]).float() / (2 * SAMPLE_RATE)).clamp(0, 1)
    off_state = torch.cat((off_state, torch.ones((1, frames), dtype=torch.bool, device=device)))
    held_state = torch.cat((held_state, torch.zeros((1, frames), dtype=torch.bool, device=device)))
    onset_state = torch.cat((onset_state, torch.zeros((1, frames), dtype=torch.bool, device=device)))
    since = torch.cat((since, torch.zeros((1, frames), device=device)))
    until = torch.cat((until, torch.zeros((1, frames), device=device)))
    background = torch.zeros((sources, 1, 1), device=device)
    background[-1] = 1.0
    frequency_coordinate = torch.linspace(0, 1, frequency, device=device)[None, :, None]
    time_coordinate = torch.linspace(0, 1, frames, device=device)[None, None, :]

    expanded_log_mixture = log_mixture[None].expand(sources, -1, -1)
    expanded_flux = log_flux[None].expand(sources, -1, -1)
    features = torch.stack(
        (
            (phase_ratio.real / 4.0).clamp(-1, 1),
            (phase_ratio.imag / 4.0).clamp(-1, 1),
            log_anchor,
            expanded_log_mixture,
            (harmonic_logits / 8.0).clamp(-1, 1),
            (percussive_logits / 8.0).clamp(-1, 1),
            relative_clipped[:, :, None].expand(-1, -1, frames),
            relative_sine[:, :, None].expand(-1, -1, frames),
            relative_cosine[:, :, None].expand(-1, -1, frames),
            off_state[:, None].expand(-1, frequency, -1).float(),
            held_state[:, None].expand(-1, frequency, -1).float(),
            onset_state[:, None].expand(-1, frequency, -1).float(),
            since[:, None].expand(-1, frequency, -1),
            until[:, None].expand(-1, frequency, -1),
            background.expand(-1, frequency, frames),
            anchor_share,
            expanded_flux,
            frequency_coordinate.expand(sources, -1, frames),
            time_coordinate.expand(sources, frequency, -1),
        ),
        dim=1,
    )
    if features.shape[1] != INPUT_FEATURES:
        raise RuntimeError("atomic feature contract changed")
    tile = (args.atom_frequency_bins, args.atom_time_frames)
    pooled_features = padded_pool(features, tile)
    target_energy = padded_pool(target_bank.abs().square(), tile)
    labels = target_energy.argmax(dim=0)
    total_energy = target_energy.sum(dim=0)
    energy_weight = (total_energy / total_energy.mean().clamp_min(1e-12)).sqrt()
    source_energy = target_energy.flatten(1).sum(dim=1).clamp_min(1e-12)
    inverse_source = source_energy.rsqrt()
    inverse_source = inverse_source / inverse_source.mean()
    class_weight = inverse_source.index_select(0, labels.flatten()).reshape_as(labels)
    weights = (energy_weight * class_weight).clamp(0.05, 20.0)

    record: dict[str, Any] = {
        "mixture_id": str(example["mixture_id"]),
        "features": pooled_features.to(torch.float16).cpu(),
        "labels": labels.to(torch.int16).cpu(),
        "weights": weights.to(torch.float16).cpu(),
        "sources": note_sources,
        "true_residual_ratio": float(example["true_residual_ratio"]),
    }
    if validation:
        record.update(
            {
                "mixture_spectrum": mixture.cpu(),
                "target_spectrum": targets.cpu(),
                "teacher_spectrum": anchors.cpu(),
                "target_waveform": target_waveform.cpu(),
                "pitches": bank["pitch"].cpu(),
            }
        )
    return record


def expand_atom_labels(
    labels: torch.Tensor,
    *,
    sources: int,
    frequency: int,
    frames: int,
    tile: tuple[int, int],
) -> torch.Tensor:
    frequency_tile, time_tile = tile
    expanded = labels.repeat_interleave(frequency_tile, dim=0).repeat_interleave(
        time_tile, dim=1
    )[:frequency, :frames]
    return F.one_hot(expanded.long(), num_classes=sources).permute(2, 0, 1).bool()


@torch.inference_mode()
def evaluate(
    model: OccurrenceAtomicMerger,
    records: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    model.eval()
    methods: dict[str, list[dict[str, float]]] = {
        "teacher": [],
        "oracle": [],
        "learned_hard": [],
    }
    classification_accuracy: list[float] = []
    tile = (args.atom_frequency_bins, args.atom_time_frames)
    for record in records:
        features = record["features"].to(device).float()
        logits = model(features)
        predicted = logits.argmax(dim=0)
        labels = record["labels"].to(device).long()
        weights = record["weights"].to(device).float()
        classification_accuracy.append(
            float(((predicted == labels).float() * weights).sum() / weights.sum())
        )
        mixture = record["mixture_spectrum"].to(device)
        targets = record["target_spectrum"].to(device)
        teacher = record["teacher_spectrum"].to(device)
        target_waveform = record["target_waveform"].to(device)
        pitches = record["pitches"].to(device)
        target_background = mixture - targets.sum(dim=0)
        target_bank = torch.cat((targets, target_background[None]), dim=0)
        assignment = expand_atom_labels(
            predicted,
            sources=target_bank.shape[0],
            frequency=mixture.shape[-2],
            frames=mixture.shape[-1],
            tile=tile,
        )
        learned_bank = assignment * mixture[None]
        oracle_bank, _ = strict_atom_partition(mixture, target_bank, tile)
        for method, estimate in (
            ("teacher", teacher),
            ("oracle", oracle_bank[:-1]),
            ("learned_hard", learned_bank[:-1]),
        ):
            metrics = source_metrics(
                estimate,
                targets,
                mixture,
                pitches,
                target_waveform=target_waveform,
                n_fft=args.n_fft,
            )
            if method == "learned_hard":
                metrics["mixture_sum_error_max_abs"] = float(
                    (learned_bank.sum(dim=0) - mixture).abs().amax()
                )
            methods[method].append(metrics)
    summary = {
        method: {
            key: finite_mean([row[key] for row in rows]) for key in rows[0]
        }
        for method, rows in methods.items()
    }
    summary["learned_hard"]["weighted_atom_accuracy"] = finite_mean(
        classification_accuracy
    )
    return summary


def main() -> None:
    args = parse_args()
    validate(args)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frozen = FrozenV13Contract.load(args.frozen_manifest)
    train_ordinals, validation_ordinals, view_binding = synthetic_corpus.load_synthetic_ordinal_view(
        args.grouped_view, contract=frozen
    )
    if train_ordinals is None or validation_ordinals is None or view_binding is None:
        raise ValueError("atomic merger requires mixture-disjoint train and validation splits")
    backbone, initializer_provenance = load_dual_hpss_checkpoint(args.initializer, device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    train_examples = select_examples(
        frozen,
        train_ordinals,
        count=args.train_groups,
        candidate_pool=args.train_candidate_pool,
        crop_samples=args.crop_samples,
        maximum_sources=args.maximum_sources,
        maximum_true_residual_ratio=args.maximum_true_residual_ratio,
        seed=args.seed + 1,
        split="train",
    )
    validation_examples = select_examples(
        frozen,
        validation_ordinals,
        count=args.validation_groups,
        candidate_pool=args.validation_candidate_pool,
        crop_samples=args.crop_samples,
        maximum_sources=args.maximum_sources,
        maximum_true_residual_ratio=args.maximum_true_residual_ratio,
        seed=args.seed,
        split="validation",
    )
    cache_started = time.time()
    train_records = []
    for index, example in enumerate(train_examples, start=1):
        train_records.append(
            build_record(backbone, example, device, args, validation=False)
        )
        print(json.dumps({"cache": "train", "complete": index, "total": len(train_examples)}), flush=True)
    validation_records = []
    for index, example in enumerate(validation_examples, start=1):
        validation_records.append(
            build_record(backbone, example, device, args, validation=True)
        )
        print(json.dumps({"cache": "validation", "complete": index, "total": len(validation_examples)}), flush=True)
    cache_elapsed = time.time() - cache_started

    config = OccurrenceAtomicMergerConfig(
        input_features=INPUT_FEATURES,
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
    )
    model = OccurrenceAtomicMerger(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    contract = {
        "schema": SCHEMA,
        "initializer": initializer_provenance,
        "initializer_sha256": file_sha256(args.initializer),
        "backbone_policy": "frozen_teacher_evidence_only",
        "model": model.contract(),
        "view": view_binding,
        "train_mixtures": [record["mixture_id"] for record in train_records],
        "validation_mixtures": [record["mixture_id"] for record in validation_records],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source": {
            "trainer": file_sha256(Path(__file__).resolve()),
            "model": file_sha256(ROOT / "src/model_exp/occurrence_atomic_merger.py"),
        },
    }
    atomic_json(contract, output / "contract.json")
    history: list[dict[str, Any]] = []
    baseline = evaluate(model, validation_records, device, args)
    history.append({"step": 0, "validation": baseline})
    print("EVAL " + json.dumps(history[-1], sort_keys=True), flush=True)
    best_step = 0
    best_nmse = baseline["learned_hard"]["note_complex_nmse"]
    atomic_torch_save(
        {"schema": SCHEMA, "step": 0, "model": model.state_dict(), "config": asdict(config)},
        output / "checkpoints/best.pt",
    )

    train_started = time.time()
    for step in range(1, args.steps + 1):
        record = train_records[int(rng.integers(len(train_records)))]
        features = record["features"].to(device, non_blocking=True).float()
        labels = record["labels"].to(device, non_blocking=True).long()
        weights = record["weights"].to(device, non_blocking=True).float()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        cross_entropy = F.cross_entropy(
            logits[None], labels[None], reduction="none"
        )[0]
        loss = (cross_entropy * weights).sum() / weights.sum().clamp_min(1e-8)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("atomic merger produced non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("atomic merger produced non-finite gradients")
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                        "sources": int(record["sources"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate(model, validation_records, device, args)
            row = {"step": step, "validation": validation}
            history.append(row)
            print("EVAL " + json.dumps(row, sort_keys=True), flush=True)
            candidate_nmse = validation["learned_hard"]["note_complex_nmse"]
            if candidate_nmse < best_nmse:
                best_nmse = candidate_nmse
                best_step = step
                atomic_torch_save(
                    {"schema": SCHEMA, "step": step, "model": model.state_dict(), "config": asdict(config)},
                    output / "checkpoints/best.pt",
                )

    state = torch.load(output / "checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    final = evaluate(model, validation_records, device, args)
    teacher = final["teacher"]
    oracle = final["oracle"]
    learned = final["learned_hard"]
    headroom = teacher["note_complex_nmse"] - oracle["note_complex_nmse"]
    captured = (teacher["note_complex_nmse"] - learned["note_complex_nmse"]) / max(headroom, 1e-12)
    decision = {
        "status": "learned_merging_feasible" if (
            learned["note_complex_nmse"] <= 0.85 * teacher["note_complex_nmse"]
            and learned["pitch_assignment_accuracy"] >= teacher["pitch_assignment_accuracy"] + 0.05
            and learned["octave_margin"] >= 1.10 * teacher["octave_margin"]
            and learned["mixture_sum_error_max_abs"] <= 1e-6
            and captured >= 0.35
        ) else "learned_merger_did_not_capture_oracle_headroom",
        "gates": {
            "complex_nmse_relative_to_teacher_max": 0.85,
            "pitch_accuracy_gain_min": 0.05,
            "octave_margin_relative_to_teacher_min": 1.10,
            "mixture_sum_error_max_abs": 1e-6,
            "oracle_headroom_captured_min": 0.35,
        },
        "best_step": best_step,
        "oracle_headroom_captured": captured,
        "teacher": teacher,
        "oracle": oracle,
        "learned_hard": learned,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "cache_elapsed_sec": cache_elapsed,
        "train_elapsed_sec": time.time() - train_started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "decision": decision,
    }
    atomic_json(result, output / "result.json")
    print("RESULT " + json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
