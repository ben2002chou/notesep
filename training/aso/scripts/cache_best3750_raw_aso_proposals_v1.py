#!/usr/bin/env python3
"""Cache raw best3750 note proposals for scaled ASO training.

This deliberately stores the raw symmetric anchor, not the historical
Lifecycle post-allocation.  Fresh packed banks and recorded natural-score
scenes share one immutable packed output contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.fresh_source_complete_data import FreshSourceCompleteDataset  # noqa: E402
from model_exp.packed_blob_shards import PackedBlobShardWriter  # noqa: E402
from model_exp.recorded_natural_score_scene_data import (  # noqa: E402
    RecordedNaturalScoreSceneDataset,
)
from scripts.infer_occurrence_atomic_sparse_allocator_arp_v3 import (  # noqa: E402
    sha256_file,
)
from scripts.train_occurrence_atomic_merger_canary import teacher_evidence  # noqa: E402
from scripts.train_occurrence_score_remix_ssl import (  # noqa: E402
    load_dual_hpss_checkpoint,
)


SCHEMA = "best3750_raw_aso_proposal_cache_v1"
KIND = "best3750_anchor_f16"
CHECKPOINT_SHA256 = "3e8a81d98c76306caf99486c556597f4614dc303c8012a36678b4679e33e7496"
INDEX_SHA256 = "ba5eec95ee976b71832368804c0bb863d733f5342da063d40fab3685e1103544"
RECORDED_FREEZE_SHA256 = "24c30535b171b19a6ce06b29d886aece2f1cd65d7828f4f719c928df5ea534c7"
STOP_REQUESTED = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def contract_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def proposal_id(
    dataset_kind: str, mixture_id: str, source_index: int, checkpoint_sha256: str,
) -> str:
    text = f"{SCHEMA}:{dataset_kind}:{mixture_id}:{source_index}:{checkpoint_sha256}"
    return hashlib.sha256(text.encode()).hexdigest()


def recorded_example(
    dataset: RecordedNaturalScoreSceneDataset, scene_index: int,
) -> dict[str, Any]:
    indices = dataset.scene_batch_indices()[scene_index]
    rows = [dataset[index] for index in indices]
    lengths = {int(row["valid_samples"]) for row in rows}
    scene_ids = {str(row["scene_id"]) for row in rows}
    if len(lengths) != 1 or len(scene_ids) != 1:
        raise ValueError("recorded scene lost aligned geometry")
    mixtures = torch.stack([row["mixture"] for row in rows])
    if not torch.equal(mixtures, mixtures[0:1].expand_as(mixtures)):
        raise ValueError("recorded note rows do not share a mixture")
    targets = torch.stack([row["target"] for row in rows])
    mixture = mixtures[0]
    if float((targets.sum(0) - mixture).abs().max()) > 1e-6:
        raise ValueError("recorded source-complete scene lost closure")
    return {
        "mixture_id": rows[0]["scene_id"],
        "source_parent_id": rows[0]["source_parent_id"],
        "score_parent_id": rows[0]["score_parent_id"],
        "mixture": mixture,
        "targets": targets,
        "pitches": torch.stack([row["pitch"] for row in rows]),
        "onsets": torch.stack([row["note_onset_samples"] for row in rows]),
        "offsets": torch.stack([row["note_offset_samples"] for row in rows]),
        "symbolic_offsets": torch.stack([row["midi_offset_samples"] for row in rows]),
        "valid_samples": len(mixture),
    }


def infer_anchor(
    model: torch.nn.Module,
    example: dict[str, Any],
    device: torch.device,
    query_microbatch: int,
    input_gate_policy: str,
) -> torch.Tensor:
    source_count = int(example["targets"].shape[0])
    samples = int(example["valid_samples"])
    raw_onsets = example["onsets"].to(device)
    raw_offsets = example["offsets"].to(device)
    if input_gate_policy == "full":
        query_onsets, query_offsets = raw_onsets, raw_offsets
    else:
        query_onsets = raw_onsets.clamp(0, max(0, samples - 1))
        query_offsets = torch.maximum(
            raw_offsets.clamp(1, samples), query_onsets + 1
        ).clamp_max(samples)
    bank = {
        "mixture": example["mixture"].to(device)[None],
        "valid": torch.tensor([int(example["valid_samples"])], device=device),
        "assignment": torch.zeros(source_count, dtype=torch.long, device=device),
        "pitch": example["pitches"].to(device),
        "onset": query_onsets,
        "offset": query_offsets,
    }
    input_gate_strength = None
    if input_gate_policy == "lower_member_exact_octave":
        pitch = bank["pitch"]
        onset = raw_onsets
        # Eval-v3 decides the selective gate from its frozen conditioning
        # interval, which is the measured acoustic endpoint for this corpus.
        offset = raw_offsets
        duration = (offset - onset).clamp_min(1)
        lower = pitch[:, None] + 12 == pitch[None, :]
        overlap = (
            torch.minimum(offset[:, None], offset[None, :])
            - torch.maximum(onset[:, None], onset[None, :])
        ).clamp_min(0)
        shorter = torch.minimum(duration[:, None], duration[None, :]).clamp_min(1)
        active = lower & (overlap >= 1_280) & (overlap >= 0.25 * shorter)
        input_gate_strength = active.any(dim=1).to(torch.float32)
    elif input_gate_policy != "full":
        raise ValueError(f"unsupported input gate policy: {input_gate_policy}")
    anchor, _, _ = teacher_evidence(
        model, bank, query_microbatch=query_microbatch,
        input_gate_strength=input_gate_strength,
    )
    if anchor.shape != example["targets"].shape or not bool(torch.isfinite(anchor).all()):
        raise ValueError("raw anchor shape/finite contract failed")
    return anchor.float().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-kind", choices=("fresh", "recorded"), required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument(
        "--expected-index-sha256",
        default=INDEX_SHA256,
        help=(
            "Frozen SHA-256 for --index. The default preserves the original "
            "core-bank contract; expanded-bank launches must pass their "
            "preregistered hash explicitly."
        ),
    )
    parser.add_argument("--recorded-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=CHECKPOINT_SHA256,
        help=(
            "Expected raw-separator checkpoint hash. The default preserves the "
            "historical best3750 cache contract; successor caches must bind this "
            "explicitly to their selected separator."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--expected-mixtures", type=int, required=True)
    parser.add_argument("--expected-proposals", type=int, required=True)
    parser.add_argument("--maximum-mixtures", type=int, default=0)
    parser.add_argument("--crop-samples", type=int, default=49_152)
    parser.add_argument("--maximum-sources", type=int, default=64)
    parser.add_argument("--unit-mixtures", type=int, default=8)
    parser.add_argument("--query-microbatch", type=int, default=4)
    parser.add_argument(
        "--input-gate-policy",
        choices=("full", "lower_member_exact_octave"),
        default="full",
        help="Raw separator gate policy; default preserves historical cache behavior.",
    )
    parser.add_argument("--target-shard-bytes", type=int, default=4 * 1024**3)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA compute node required")
    if not 0 <= args.task_index < args.task_count:
        raise ValueError("invalid task geometry")
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = args.expected_checkpoint_sha256.lower()
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise ValueError("invalid expected checkpoint SHA-256")
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("raw separator checkpoint mismatch")
    historical_contract = checkpoint_sha256 == CHECKPOINT_SHA256
    proposal_waveform_semantics = (
        "raw symmetric_robust_best3750 waveform"
        if historical_contract else "raw symmetric separator waveform"
    )
    proposal_semantics = (
        "raw symmetric_robust_best3750"
        if historical_contract else "raw symmetric separator"
    )
    output_root = args.output_root.expanduser().resolve()
    if Path("/scratch/gilbreth/chou150") not in output_root.parents:
        raise ValueError("output must be a versioned chou150 scratch path")
    if args.dataset_kind == "fresh":
        if args.index is None or args.recorded_root is not None:
            raise ValueError("fresh source requires only --index")
        source_path = args.index.expanduser().resolve()
        if sha256_file(source_path) != args.expected_index_sha256:
            raise ValueError("fresh index mismatch")
        fresh_kwargs = {}
        if not historical_contract:
            fresh_kwargs = {
                "preserve_signed_note_times": True,
            }
        dataset: Any = FreshSourceCompleteDataset(
            source_path, split=args.split, crop_samples=args.crop_samples,
            maximum_sources=args.maximum_sources, **fresh_kwargs,
        )
        source_sha = args.expected_index_sha256
        get_example = lambda index: dataset[index]
    else:
        if args.recorded_root is None or args.index is not None:
            raise ValueError("recorded source requires only --recorded-root")
        source_path = args.recorded_root.expanduser().resolve()
        if sha256_file(source_path / "FROZEN.json") != RECORDED_FREEZE_SHA256:
            raise ValueError("recorded freeze mismatch")
        dataset = RecordedNaturalScoreSceneDataset([source_path], split=args.split)
        source_sha = RECORDED_FREEZE_SHA256
        get_example = lambda index: recorded_example(dataset, index)

    population = len(dataset) if args.dataset_kind == "fresh" else len(dataset.scenes)
    if population != args.expected_mixtures:
        raise ValueError(f"mixture count changed: {population}")
    indices = list(range(population))
    if args.maximum_mixtures:
        indices = indices[: args.maximum_mixtures]
    indices = indices[args.task_index :: args.task_count]
    contract = {
        "schema": SCHEMA,
        "payload_kind": KIND,
        "dataset_kind": args.dataset_kind,
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "split": args.split,
        "task_index": args.task_index,
        "task_count": args.task_count,
        "expected_mixtures": args.expected_mixtures,
        "expected_proposals": args.expected_proposals,
        "maximum_mixtures": args.maximum_mixtures,
        "crop_samples": args.crop_samples,
        "maximum_sources": args.maximum_sources,
        "unit_mixtures": args.unit_mixtures,
        "query_microbatch": args.query_microbatch,
        "input_gate_policy": args.input_gate_policy,
        "isolated_target_input": False,
        "proposal_semantics": proposal_waveform_semantics,
    }
    identity = contract_hash(contract)
    task_root = (
        output_root / args.dataset_kind / args.split / "tasks"
        / f"task-{args.task_index:03d}"
    )
    task_root.mkdir(parents=True, exist_ok=True)
    contract_path = task_root / "CONTRACT.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()).get("contract_sha256") != identity:
            raise ValueError("existing task belongs to a different contract")
    else:
        atomic_json(contract_path, {**contract, "contract_sha256": identity})

    device = torch.device("cuda")
    model, model_info = load_dual_hpss_checkpoint(checkpoint, device)
    if model_info["sha256"] != checkpoint_sha256:
        raise ValueError("loaded checkpoint provenance mismatch")
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    completed: list[dict[str, Any]] = []
    repeat_checked = False
    try:
        for unit_index, unit_start in enumerate(range(0, len(indices), args.unit_mixtures)):
            unit_indices = indices[unit_start : unit_start + args.unit_mixtures]
            final = task_root / "units" / f"unit-{unit_index:05d}"
            if final.exists():
                marker = json.loads((final / "UNIT_COMPLETE.json").read_text())
                if marker.get("contract_sha256") != identity:
                    raise ValueError("completed unit contract mismatch")
                completed.append(marker)
                continue
            restart = os.environ.get("SLURM_RESTART_COUNT", "0")
            partial = task_root / "partial" / f"unit-{unit_index:05d}-r{restart}"
            partial.mkdir(parents=True, exist_ok=False)
            writer = PackedBlobShardWriter(
                partial / "packed",
                target_shard_bytes=args.target_shard_bytes,
                prefix=f"aso-raw-{args.dataset_kind}-{args.task_index:03d}-{unit_index:05d}",
            )
            manifest: list[str] = []
            proposal_count = 0
            for dataset_index in unit_indices:
                example = get_example(dataset_index)
                anchor = infer_anchor(
                    model, example, device, args.query_microbatch,
                    args.input_gate_policy,
                )
                if not repeat_checked:
                    repeated = infer_anchor(
                        model, example, device, args.query_microbatch,
                        args.input_gate_policy,
                    )
                    if not torch.equal(anchor, repeated):
                        raise ValueError("raw best3750 inference is not bit-repeatable")
                    repeat_checked = True
                proposal_ids = []
                for source_index, estimate in enumerate(anchor):
                    sample_id = proposal_id(
                        args.dataset_kind, str(example["mixture_id"]), source_index,
                        checkpoint_sha256,
                    )
                    payload = estimate.numpy().astype("<f2", copy=False)
                    writer.add_bytes(
                        sample_id,
                        KIND,
                        payload.tobytes(),
                        metadata={
                            "mixture_id": str(example["mixture_id"]),
                            "source_index": source_index,
                            "samples": int(payload.size),
                            "dtype": "float16",
                            "proposal_semantics": proposal_semantics,
                        },
                    )
                    proposal_ids.append(sample_id)
                    proposal_count += 1
                manifest.append(json.dumps({
                    "mixture_id": str(example["mixture_id"]),
                    "dataset_index": dataset_index,
                    "sources": len(proposal_ids),
                    "proposal_ids": proposal_ids,
                    "pitches": example["pitches"].tolist(),
                    "onsets": example["onsets"].tolist(),
                    "offsets": example["offsets"].tolist(),
                    "source_parent_id": str(example["source_parent_id"]),
                    "score_parent_id": str(example["score_parent_id"]),
                }, sort_keys=True, separators=(",", ":")))
            writer.finalize({
                "schema": SCHEMA,
                "contract_sha256": identity,
                "payload_kind": KIND,
                "mixtures": len(unit_indices),
                "proposals": proposal_count,
            })
            (partial / "banks.jsonl").write_text("\n".join(manifest) + "\n")
            marker = {
                "schema": SCHEMA,
                "status": "passed",
                "contract_sha256": identity,
                "payload_kind": KIND,
                "unit_index": unit_index,
                "mixtures": len(unit_indices),
                "proposals": proposal_count,
            }
            atomic_json(partial / "UNIT_COMPLETE.json", marker)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, final)
            completed.append(marker)
            atomic_json(task_root / "PROGRESS.json", {
                "status": "running",
                "units": len(completed),
                "mixtures": sum(row["mixtures"] for row in completed),
                "proposals": sum(row["proposals"] for row in completed),
            })
            if STOP_REQUESTED:
                return 75
    finally:
        dataset.close()

    report = {
        "schema": SCHEMA,
        "status": "passed",
        "contract_sha256": identity,
        "payload_kind": KIND,
        "dataset_kind": args.dataset_kind,
        "split": args.split,
        "task_index": args.task_index,
        "units": len(completed),
        "mixtures": sum(row["mixtures"] for row in completed),
        "proposals": sum(row["proposals"] for row in completed),
        ("best3750" if historical_contract else "source_separator"): model_info,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    if report["mixtures"] != len(indices):
        raise ValueError("terminal mixture count changed")
    atomic_json(task_root / "TASK_COMPLETE.json", report)
    atomic_json(task_root / "PROGRESS.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
