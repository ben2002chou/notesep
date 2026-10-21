#!/usr/bin/env python3
"""Jointly train a query separator on synthetic and real complete note banks.

Each optimizer update contains:

* the checkpoint's original fully supervised objective on synthetic stems;
* pre-projection bank-sum consistency on the same synthetic mixture;
* pre-projection bank-sum consistency on score-complete real recordings; and
* synthetic calibration of a learned, permutation-equivariant uncertainty
  allocator used by the exact no-residual bank projection.

The model still predicts one queried occurrence.  Complete banks are evaluated
by repeating one mixture across its queries in activation-bounded chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_v13_streaming import FrozenV13Contract  # noqa: E402
from model_exp.occurrence_bank_uncertainty import (  # noqa: E402
    BankUncertaintyConfig,
    LearnedBankUncertainty,
    heteroscedastic_source_nll,
    uncertainty_weighted_bank_projection,
)
from model_exp.occurrence_synthetic_note_bank import (  # noqa: E402
    CompleteSyntheticBankReader,
    select_complete_mixture_groups,
)
import scripts.train_dna_residual_v13_streaming as synthetic_corpus  # noqa: E402
import scripts.train_occurrence_dual_harmonic_residual_separator as supervised  # noqa: E402
from scripts.train_occurrence_score_remix_ssl import (  # noqa: E402
    DeterministicStepBatchSampler,
    MaestroScoreCropDataset,
    _load_configured_real_tracks,
    add_parameter_gradients,
    atomic_json,
    atomic_torch_save,
    collate_score_crops,
    file_sha256,
    load_dual_hpss_checkpoint,
    project_input,
)
from model_exp.occurrence_score_remix_ssl import (  # noqa: E402
    masked_rms,
    score_complete_mixture_closure,
    trainable_parameter_names,
)


SCHEMA = "occurrence_joint_note_bank_training_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--real-corpus-config", type=Path, required=True)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--synthetic-frozen-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-ordinal-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--trainable-scope", choices=("ownership", "heads", "heads_and_film", "full"), default="heads_and_film")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--real-bank-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-bank-weight", type=float, default=0.25)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-learning-rate", type=float, default=3e-4)
    parser.add_argument("--uncertainty-hidden-channels", type=int, default=16)
    parser.add_argument("--real-proposal-batch-size", type=int, default=32)
    parser.add_argument("--real-banks-per-update", type=int, default=2)
    parser.add_argument("--maximum-real-sources", type=int, default=24)
    parser.add_argument("--synthetic-group-pool", type=int, default=8192)
    parser.add_argument("--maximum-synthetic-sources", type=int, default=24)
    parser.add_argument("--maximum-true-residual-ratio", type=float, default=0.01)
    parser.add_argument("--crop-samples", type=int, default=49_152)
    parser.add_argument("--query-microbatch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--model-save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.start_step < 0 or args.start_step >= args.steps:
        raise ValueError("invalid step interval")
    if args.query_microbatch < 1 or args.real_proposal_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if not 1 <= args.real_banks_per_update <= args.real_proposal_batch_size:
        raise ValueError("real bank count exceeds proposal batch")
    if min(args.maximum_real_sources, args.maximum_synthetic_sources) < 1:
        raise ValueError("source capacities must be positive")
    if min(args.real_bank_weight, args.synthetic_bank_weight, args.supervised_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if args.supervised_weight == 0:
        raise ValueError("the production trainer cannot disable original supervision")
    if not 0 <= args.maximum_true_residual_ratio <= 1:
        raise ValueError("invalid synthetic completeness threshold")


def _real_bank_from_batch(
    batch: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor | list[str]]:
    eligible = [
        index
        for index, events in enumerate(batch["score_events"])
        if bool(batch["score_complete"][index])
        and 1 <= len(events) <= args.maximum_real_sources
    ][: args.real_banks_per_update]
    if not eligible:
        raise RuntimeError("real proposal batch contains no complete bank")
    rows = torch.tensor(eligible, dtype=torch.long)
    mixtures = batch["mixture"].index_select(0, rows).to(device, non_blocking=True)
    valid = batch["valid_samples"].index_select(0, rows).to(device, non_blocking=True)
    pitches: list[int] = []
    onsets: list[int] = []
    offsets: list[int] = []
    assignment: list[int] = []
    corpora: list[str] = []
    for local, original in enumerate(eligible):
        corpora.append(str(batch["corpus"][original]))
        for pitch, onset, offset in batch["score_events"][original]:
            pitches.append(int(pitch))
            onsets.append(int(onset))
            offsets.append(int(offset))
            assignment.append(local)
    return {
        "mixture": mixtures,
        "valid": valid,
        "pitch": torch.tensor(pitches, device=device, dtype=torch.long),
        "onset": torch.tensor(onsets, device=device, dtype=torch.long),
        "offset": torch.tensor(offsets, device=device, dtype=torch.long),
        "assignment": torch.tensor(assignment, device=device, dtype=torch.long),
        "corpora": corpora,
    }


def _synthetic_bank_to_device(
    example: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    count = int(example["valid_samples"])
    queries = int(example["targets"].shape[0])
    return {
        "mixture": example["mixture"].to(device, non_blocking=True)[None],
        "targets": example["targets"].to(device, non_blocking=True),
        "pitch": example["pitches"].to(device, non_blocking=True),
        "onset": example["onsets"].to(device, non_blocking=True),
        "offset": example["offsets"].to(device, non_blocking=True),
        "valid": torch.tensor([count], device=device, dtype=torch.long),
        "assignment": torch.zeros(queries, device=device, dtype=torch.long),
    }


def _supervised_synthetic_backward(
    model: torch.nn.Module,
    bank: dict[str, torch.Tensor],
    *,
    chunk_size: int,
    weight: float,
) -> tuple[float, int]:
    queries = int(bank["pitch"].shape[0])
    total = 0.0
    mixture = bank["mixture"].expand(queries, -1)
    valid = bank["valid"].expand(queries)
    for start in range(0, queries, chunk_size):
        stop = min(start + chunk_size, queries)
        fraction = (stop - start) / queries
        metrics = supervised.forward_metrics(
            model,
            {
                "mixture": mixture[start:stop],
                "target": bank["targets"][start:stop],
                "pitch": bank["pitch"][start:stop],
                "valid_samples": valid[start:stop],
                "note_onset_samples": bank["onset"][start:stop],
                "note_offset_samples": bank["offset"][start:stop],
            },
        )
        loss = metrics["loss"] * weight * fraction
        loss.backward()
        total += float(metrics["loss"].detach()) * fraction
    return total, queries


def _forward_bank_chunked(
    model: torch.nn.Module,
    bank: dict[str, torch.Tensor],
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Query a bank while decomposing each unique mixture only once."""
    harmonic, percussive = project_input(model, bank["mixture"], bank["valid"])
    assignment = bank["assignment"]
    valid = bank["valid"].index_select(0, assignment)
    outputs: list[torch.Tensor] = []
    for start in range(0, int(assignment.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(assignment.shape[0]))
        local = assignment[start:stop]
        estimate, _ = model(
            harmonic.index_select(0, local),
            percussive.index_select(0, local),
            bank["pitch"][start:stop],
            valid[start:stop],
            bank["onset"][start:stop],
            bank["offset"][start:stop],
        )
        outputs.append(estimate)
    return torch.cat(outputs, dim=0)


def _streaming_bank_closure_gradients(
    model: torch.nn.Module,
    bank: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Exact bank-closure VJP with cached per-mixture HPSS decomposition."""
    with torch.no_grad():
        estimates = _forward_bank_chunked(model, bank, chunk_size=chunk_size)
        summed = torch.zeros_like(bank["mixture"]).index_add(
            0, bank["assignment"], estimates
        )
    proxy = summed.detach().requires_grad_(True)
    identity = torch.arange(
        bank["mixture"].shape[0], device=bank["mixture"].device, dtype=torch.long
    )
    _, error = score_complete_mixture_closure(
        proxy, identity, bank["mixture"], bank["valid"]
    )
    loss = error.mean() / masked_rms(
        bank["mixture"], bank["valid"]
    ).mean().clamp_min(1e-4)
    cotangent = torch.autograd.grad(loss, proxy)[0].detach()

    harmonic, percussive = project_input(model, bank["mixture"], bank["valid"])
    accumulated: tuple[torch.Tensor | None, ...] = tuple(None for _ in parameters)
    assignment = bank["assignment"]
    indexed_valid = bank["valid"].index_select(0, assignment)
    for start in range(0, int(assignment.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(assignment.shape[0]))
        local = assignment[start:stop]
        estimate, _ = model(
            harmonic.index_select(0, local),
            percussive.index_select(0, local),
            bank["pitch"][start:stop],
            indexed_valid[start:stop],
            bank["onset"][start:stop],
            bank["offset"][start:stop],
        )
        surrogate = (
            estimate * cotangent.index_select(0, local)
        ).sum()
        gradients = torch.autograd.grad(surrogate, parameters, allow_unused=True)
        accumulated = add_parameter_gradients(accumulated, gradients)
    return loss.detach(), accumulated


def _model_bank_estimates(
    model: torch.nn.Module,
    bank: dict[str, torch.Tensor],
    *,
    chunk_size: int,
) -> torch.Tensor:
    with torch.no_grad():
        return _forward_bank_chunked(model, bank, chunk_size=chunk_size).float()


def _add_gradients_to_parameters(
    parameters: list[torch.nn.Parameter],
    gradients: tuple[torch.Tensor | None, ...],
    *,
    weight: float,
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            continue
        contribution = gradient.detach().mul(weight)
        if parameter.grad is None:
            parameter.grad = contribution
        else:
            parameter.grad.add_(contribution)


def _source_checkpoint(
    initializer: Path,
    model: torch.nn.Module,
    uncertainty: LearnedBankUncertainty,
    contract: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    source = torch.load(initializer, map_location="cpu", weights_only=False)
    checkpoint = {
        key: source.get(key)
        for key in ("schema", "global_step", "model_contract", "loss_contract", "corpus_binding")
    }
    checkpoint.update(
        model=model.state_dict(),
        joint_note_bank_contract=contract,
        joint_note_bank_step=step,
        bank_uncertainty=uncertainty.state_dict(),
        bank_uncertainty_contract=uncertainty.contract(),
    )
    return checkpoint


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    model, initializer_provenance = load_dual_hpss_checkpoint(args.initializer, device)
    names, trainable_count = trainable_parameter_names(model, args.trainable_scope)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    uncertainty = LearnedBankUncertainty(
        BankUncertaintyConfig(hidden_channels=args.uncertainty_hidden_channels)
    ).to(device)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    uncertainty_optimizer = torch.optim.AdamW(
        uncertainty.parameters(), lr=args.uncertainty_learning_rate, weight_decay=args.weight_decay
    )
    start_step = args.start_step
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("schema") != SCHEMA:
            raise ValueError("resume state has the wrong schema")
        model.load_state_dict(state["model"], strict=True)
        uncertainty.load_state_dict(state["bank_uncertainty"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        uncertainty_optimizer.load_state_dict(state["uncertainty_optimizer"])
        start_step = int(state["step"])

    configured_tracks, corpus_weights, real_binding = _load_configured_real_tracks(
        args.real_corpus_config
    )
    real_dataset = MaestroScoreCropDataset(
        args.maestro_root,
        split="train",
        crop_samples=args.crop_samples,
        lead_samples=3_200,
        release_samples=4_000,
        max_tracks=0,
        minimum_note_sec=0.05,
        maximum_anchor_polyphony=0,
        seed=args.seed,
        tracks_override=configured_tracks,
    )
    real_sampler = DeterministicStepBatchSampler(
        len(real_dataset),
        batch_size=args.real_proposal_batch_size,
        start_step=start_step,
        stop_step=args.steps,
        seed=args.seed,
        collision_free_queries=real_dataset.queries,
        coverage_groups=real_dataset.query_indices_by_corpus,
        coverage_weights=corpus_weights,
    )
    real_loader = DataLoader(
        real_dataset,
        batch_sampler=real_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_score_crops,
    )

    frozen = FrozenV13Contract.load(args.synthetic_frozen_manifest)
    train_ordinals, validation_ordinals, view_binding = synthetic_corpus.load_synthetic_ordinal_view(
        args.synthetic_ordinal_view, contract=frozen
    )
    if train_ordinals is None or validation_ordinals is None or view_binding is None:
        raise ValueError("synthetic bank training requires a mixture-disjoint ordinal view")
    synthetic_groups = select_complete_mixture_groups(
        frozen, train_ordinals, count=args.synthetic_group_pool, seed=args.seed
    )
    synthetic_reader = CompleteSyntheticBankReader(
        frozen,
        fixed_samples=args.crop_samples,
        maximum_sources=args.maximum_synthetic_sources,
    )

    contract = {
        "schema": SCHEMA,
        "initializer": initializer_provenance,
        "real_corpus_binding": real_binding,
        "synthetic_view": view_binding,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": trainable_count,
        "trainable_parameter_names": names,
        "uncertainty": uncertainty.contract(),
        "objectives": {
            "original_supervised_synthetic": args.supervised_weight,
            "synthetic_preprojection_bank_consistency": args.synthetic_bank_weight,
            "real_preprojection_bank_consistency": args.real_bank_weight,
            "synthetic_uncertainty_calibration": "robust heteroscedastic projected-source Student-t NLL",
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                ROOT / "scripts/train_occurrence_joint_note_bank.py",
                ROOT / "src/model_exp/occurrence_bank_uncertainty.py",
                ROOT / "src/model_exp/occurrence_synthetic_note_bank.py",
            )
        },
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_QOS", "SLURM_JOB_PARTITION")
        },
    }
    atomic_json(contract, output / "contract.json")
    metrics_path = output / "metrics.jsonl"
    preempt_requested = False

    def request_preempt(_signum: int, _frame: Any) -> None:
        nonlocal preempt_requested
        preempt_requested = True

    signal.signal(signal.SIGUSR1, request_preempt)

    def save_state(step: int) -> Path:
        path = output / "checkpoints/state-last.pt"
        atomic_torch_save(
            {
                "schema": SCHEMA,
                "step": step,
                "model": model.state_dict(),
                "bank_uncertainty": uncertainty.state_dict(),
                "optimizer": optimizer.state_dict(),
                "uncertainty_optimizer": uncertainty_optimizer.state_dict(),
                "contract": contract,
            },
            path,
        )
        return path

    started = time.time()
    final_step = start_step
    synthetic_cursor = start_step
    try:
        for step, real_batch in enumerate(real_loader, start=start_step + 1):
            final_step = step
            model.train()
            optimizer.zero_grad(set_to_none=True)
            uncertainty_optimizer.zero_grad(set_to_none=True)

            real_bank = _real_bank_from_batch(real_batch, args, device)
            real_loss, real_gradients = _streaming_bank_closure_gradients(
                model, real_bank, parameters,
                chunk_size=args.query_microbatch,
            )

            synthetic_example = None
            synthetic_attempts = 0
            while synthetic_attempts < len(synthetic_groups):
                mixture_id, rows = synthetic_groups[synthetic_cursor % len(synthetic_groups)]
                synthetic_cursor += 1
                synthetic_attempts += 1
                try:
                    candidate = synthetic_reader.load(mixture_id, rows, crop_seed=args.seed + step)
                except ValueError:
                    continue
                if candidate["true_residual_ratio"] <= args.maximum_true_residual_ratio:
                    synthetic_example = candidate
                    break
            if synthetic_example is None:
                raise RuntimeError("synthetic group pool yielded no complete bank")
            synthetic_bank = _synthetic_bank_to_device(synthetic_example, device)
            synthetic_loss, synthetic_gradients = _streaming_bank_closure_gradients(
                model, synthetic_bank, parameters,
                chunk_size=args.query_microbatch,
            )
            supervised_loss, synthetic_queries = _supervised_synthetic_backward(
                model,
                synthetic_bank,
                chunk_size=args.query_microbatch,
                weight=args.supervised_weight,
            )
            _add_gradients_to_parameters(parameters, real_gradients, weight=args.real_bank_weight)
            _add_gradients_to_parameters(parameters, synthetic_gradients, weight=args.synthetic_bank_weight)

            estimates = _model_bank_estimates(
                model, synthetic_bank, chunk_size=args.query_microbatch
            ).detach()
            log_variance = uncertainty(
                estimates,
                synthetic_bank["mixture"],
                synthetic_bank["pitch"],
                synthetic_bank["onset"],
                synthetic_bank["offset"],
                synthetic_bank["valid"],
                synthetic_bank["assignment"],
            )
            projected, projection_metrics = uncertainty_weighted_bank_projection(
                synthetic_bank["mixture"],
                estimates,
                log_variance,
                synthetic_bank["assignment"],
                synthetic_bank["valid"],
            )
            uncertainty_loss = heteroscedastic_source_nll(
                projected, synthetic_bank["targets"], log_variance
            )
            uncertainty_loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            uncertainty_gradient_norm = torch.nn.utils.clip_grad_norm_(
                uncertainty.parameters(), args.gradient_clip
            )
            optimizer.step()
            uncertainty_optimizer.step()

            row = {
                "step": step,
                "elapsed_sec": time.time() - started,
                "real_bank_loss": float(real_loss),
                "synthetic_bank_loss": float(synthetic_loss),
                "supervised_synthetic_loss": supervised_loss,
                "uncertainty_loss": float(uncertainty_loss.detach()),
                "gradient_norm": float(gradient_norm),
                "uncertainty_gradient_norm": float(uncertainty_gradient_norm),
                "real_queries": int(real_bank["pitch"].shape[0]),
                "synthetic_queries": synthetic_queries,
                "real_corpora": real_bank["corpora"],
                "synthetic_true_residual_ratio": float(synthetic_example["true_residual_ratio"]),
                **{key: float(value.detach()) for key, value in projection_metrics.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step == start_step + 1 or step % args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step % args.save_every == 0 or preempt_requested:
                save_state(step)
            if args.model_save_every and step % args.model_save_every == 0:
                atomic_torch_save(
                    _source_checkpoint(args.initializer, model, uncertainty, contract, step),
                    output / f"checkpoints/model-adapt-{step:06d}.pt",
                )
            if preempt_requested:
                print(f"USR1 checkpoint saved at step {step}", flush=True)
                return
    finally:
        synthetic_reader.close()

    state_path = save_state(final_step)
    final_checkpoint = output / f"checkpoints/model-adapt-{final_step:06d}.pt"
    atomic_torch_save(
        _source_checkpoint(args.initializer, model, uncertainty, contract, final_step),
        final_checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "step": final_step,
        "state": str(state_path.resolve()),
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(final_checkpoint),
        "elapsed_sec": time.time() - started,
    }
    atomic_json(report, output / "report.json")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
