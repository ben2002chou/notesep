"""Train NoteGrab on synthetic and recorded note examples."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_losses import dna_residual_losses  # noqa: E402
from model_exp.dna_residual_corpus import FrozenDNAResidualPriorCatalog  # noqa: E402
from model_exp.training_contract import (  # noqa: E402
    FREE_TARGET_LOSS_WEIGHTS,
)
from model_exp.corpus import FrozenV13Contract, sha256_file  # noqa: E402
from model_exp.samplers import (  # noqa: E402
    ResumeSafeAudioBudgetBatchSampler,
    ResumeSafeBatchSampler,
)
from model_exp.streaming_dataset import (  # noqa: E402
    V13StreamingResidualDataset,
)
from model_exp.batching import collate_v13  # noqa: E402
from model_exp.residual_source_data import (  # noqa: E402
    SyntheticOrdinalViewDataset,
    load_synthetic_ordinal_view,
    acoustic_offset_marker_digest,
    load_acoustic_release_samples,
    load_duration_windows,
    DeterministicMixedBatchSampler,
    MixedSyntheticRecordedDataset,
    RecordedValidationDataset,
    load_frozen_residual_rows,
    target_complete_window,
)


CHECKPOINT_SCHEMA = "dna_residual_v13_streaming_checkpoint_v1"
RUN_SCHEMA = "dna_residual_v13_streaming_run_v1"
RECORDED_MIX_SCHEMA = "dna_residual_v13_recorded_mix_v1"
OPTIMIZER_CHOICES = ("adam", "adamw")


def normalized_checkpoint_loss_contract(contract: Any) -> Any:
    """Compare the score-index identity after relocating its filesystem path."""
    from notesep.training_paths import relocated

    normalized = copy.deepcopy(contract)
    if isinstance(normalized, dict):
        gate = normalized.get("decomposition", {}).get("selective_gate")
        if isinstance(gate, dict) and gate.get("score_index"):
            gate["score_index"] = str(relocated(gate["score_index"]))
    return normalized


def normalized_checkpoint_model_contract(contract: Any) -> Any:
    """Normalize additive fields that were implicit in frozen old checkpoints."""

    normalized = copy.deepcopy(contract)
    if (
        isinstance(normalized, dict)
        and normalized.get("kind") == "occurrence_dual_hpss_separator_v1"
        and isinstance(normalized.get("architecture"), dict)
    ):
        normalized["architecture"].setdefault(
            "harmonic_parameterization", "ownership"
        )
        normalized["architecture"].setdefault(
            "percussive_parameterization", "ownership"
        )
    return normalized


def resolve_environment_flag(
    key: str,
    environment: Mapping[str, str] | None = None,
    *,
    default: bool = False,
) -> bool:
    """Resolve an explicit 0/1 environment flag without truthy-string ambiguity."""

    environment = os.environ if environment is None else environment
    raw = environment.get(key)
    if raw is None:
        return default
    if raw not in {"0", "1"}:
        raise ValueError(f"{key} must be '0' or '1'; got {raw!r}")
    return raw == "1"


def resolve_optimizer_name(
    environment: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environment is None else environment
    optimizer_name = environment.get("DNA_V13_OPTIMIZER", "adamw")
    if optimizer_name not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"DNA_V13_OPTIMIZER must be one of {OPTIMIZER_CHOICES}; "
            f"got {optimizer_name!r}"
        )
    return optimizer_name


OPTIMIZER_NAME = resolve_optimizer_name()
SKIP_INITIAL_MONITOR = resolve_environment_flag("DNA_V13_SKIP_INITIAL_MONITOR")
ACTIVE_MODEL_CONTRACT: dict[str, Any] = {}
MODEL_KIND = "notesep"
COMPILE_MODE = os.environ.get("DNA_V13_TORCH_COMPILE", "off")
if COMPILE_MODE != "off":
    raise ValueError("The NoteGrab recipe uses uncompiled training")
ACTIVE_BASE_LOSS_WEIGHTS = dict(FREE_TARGET_LOSS_WEIGHTS)
for key in tuple(ACTIVE_BASE_LOSS_WEIGHTS):
    environment_key = f"DNA_V13_{key.upper()}"
    if environment_key in os.environ:
        ACTIVE_BASE_LOSS_WEIGHTS[key] = float(os.environ[environment_key])
for key, value in ACTIVE_BASE_LOSS_WEIGHTS.items():
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument(
        "--synthetic-ordinal-view",
        type=Path,
        help=(
            "Immutable JSON view with mixture-disjoint train_ordinals and "
            "validation_ordinals into the frozen synthetic split."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-checkpoint",
        type=Path,
        help=(
            "Load model weights from a compatible checkpoint while starting a "
            "new optimizer, step counter, corpus binding, and output directory."
        ),
    )
    parser.add_argument(
        "--initialize-allow-loss-contract-change",
        action="store_true",
        help=(
            "Allow model-only initialization when the architecture contract "
            "matches but the new run intentionally changes its loss contract."
        ),
    )
    parser.add_argument(
        "--initialize-allow-output-carrier-mode-change",
        action="store_true",
        help=(
            "Allow the exact model-only migration from the legacy implicit "
            "encoder_input carrier to an explicit ungated_hpss output carrier."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fixed-samples", type=int, default=61_440)
    parser.add_argument("--duration-index", type=Path)
    parser.add_argument(
        "--priority-index",
        type=Path,
        help="NPZ with local_ordinal rows to upsample in dynamic batching.",
    )
    parser.add_argument(
        "--priority-fraction",
        type=float,
        default=0.0,
        help="Target fraction of presented synthetic examples from --priority-index.",
    )
    parser.add_argument(
        "--priority-family-temperature-alpha",
        type=float,
        default=1.0,
        help="Family-temperature alpha applied only within priority examples.",
    )
    parser.add_argument(
        "--acoustic-offset-index",
        type=Path,
        help="Directory of ordinal-aligned acoustic-offset sidecar shards.",
    )
    parser.add_argument("--audio-budget-samples", type=int, default=0)
    parser.add_argument("--maximum-dynamic-batch-size", type=int, default=128)
    parser.add_argument("--minimum-window-samples", type=int, default=16_384)
    parser.add_argument("--lead-samples", type=int, default=4_000)
    parser.add_argument("--release-samples", type=int, default=12_000)
    parser.add_argument("--window-quantum", type=int, default=4_096)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--reader-cache-size", type=int, default=8)
    parser.add_argument(
        "--recorded-frozen-root",
        type=Path,
        action="append",
        default=[],
        help="Passed recorded residual root; required exactly twice when mixing real rows.",
    )
    parser.add_argument("--real-example-fraction", type=float, default=0.0)
    parser.add_argument(
        "--family-temperature-alpha",
        type=float,
        default=1.0,
        help="Pair-family temperature in [0,1]; 1.0 preserves uniform sampling.",
    )
    parser.add_argument(
        "--monitor-pairs",
        "--validation-pairs",
        dest="monitor_pairs",
        type=int,
        default=2_048,
    )
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument(
        "--keep-checkpoint-every",
        type=int,
        default=0,
        help=(
            "Retain an immutable step-named hard link every N steps; zero disables. "
            "N must be a multiple of --save-every."
        ),
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        help=(
            "Fixed cosine-decay horizon. Defaults to --max-steps; set this to "
            "the parent run horizon for trajectory-matched short canaries."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--wandb-project", default="dna_residual_v13_production")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--monitor-media-every",
        type=int,
        default=int(os.environ.get("DNA_V13_MONITOR_MEDIA_EVERY", "0")),
        help=(
            "Upload the model-specific W&B audio and spectrogram monitor every N "
            "steps; zero disables media logging."
        ),
    )
    parser.add_argument(
        "--monitor-media-on-resume",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DNA_V13_MONITOR_MEDIA_ON_RESUME", "0")
        in {"1", "true", "True"},
        help="Upload one W&B audio/spectrogram monitor immediately after resuming.",
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def validate_recorded_mix_args(args: argparse.Namespace) -> None:
    fraction = float(getattr(args, "real_example_fraction", 0.0))
    roots = [
        Path(root).expanduser().resolve()
        for root in getattr(args, "recorded_frozen_root", [])
    ]
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("--real-example-fraction must be finite and in [0, 1]")
    if fraction == 0.0:
        if roots:
            raise ValueError(
                "--recorded-frozen-root requires a positive --real-example-fraction"
            )
        return
    if len(roots) != 2 or len(set(roots)) != 2:
        raise ValueError(
            "A positive --real-example-fraction requires exactly two distinct "
            "--recorded-frozen-root values"
        )


def load_recorded_training_data(
    roots: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only frozen train rows and verify parent-disjoint validation authority."""

    rows: list[dict[str, Any]] = []
    root_contracts: list[dict[str, Any]] = []
    train_parents: set[str] = set()
    validation_parents: set[str] = set()
    seen_assets: set[str] = set()
    for root_value in roots:
        root = root_value.expanduser().resolve()
        freeze_path = root / "FROZEN.json"
        freeze = _read_json(freeze_path)
        if (
            freeze.get("schema") != "frozen_dna_residual_corpus_v1"
            or freeze.get("status") != "passed"
            or not freeze.get("payload_checksums_verified", False)
            or int(freeze.get("parent_overlap", -1)) != 0
        ):
            raise ValueError(f"Recorded residual root is not frozen-passed: {root}")
        partitions = freeze.get("partitions")
        if not isinstance(partitions, dict) or any(
            int(dict(partitions.get(split, {})).get("joined_notes", 0)) < 1
            for split in ("train", "validation")
        ):
            raise ValueError(
                f"Recorded residual root lacks train/validation rows: {root}"
            )

        with FrozenDNAResidualPriorCatalog(root, split="validation") as catalog:
            root_validation_parents = {
                str(record.metadata["source_parent_id"])
                for record in catalog.iter_records()
            }
        loaded = load_frozen_residual_rows(root, split="train")
        root_train_parents = {str(row["score_parent_id"]) for row in loaded}
        overlap = root_train_parents & root_validation_parents
        if overlap:
            raise ValueError(
                f"Recorded train/validation parent overlap at {root}: "
                f"{sorted(overlap)[:5]}"
            )
        for row in loaded:
            metadata = dict(row["metadata"])
            asset_id = ":".join(
                (
                    str(row["source_id"]),
                    str(row["sample_id"]),
                    f"note-{int(metadata['note_row']):05d}",
                )
            )
            if asset_id in seen_assets:
                raise ValueError(f"Duplicate recorded training asset: {asset_id}")
            seen_assets.add(asset_id)
            rows.append(
                {
                    **row,
                    "training_asset_id": asset_id,
                    "frozen_root": str(root),
                    "frozen_split": "train",
                }
            )
        train_parents.update(root_train_parents)
        validation_parents.update(root_validation_parents)
        root_contracts.append(
            {
                "root": str(root),
                "split": "train",
                "frozen_sha256": sha256_file(freeze_path),
                "index_sha256": str(freeze["index_sha256"]),
                "rows": len(loaded),
                "train_parents": len(root_train_parents),
                "validation_parents": len(root_validation_parents),
                "declared_parent_overlap": int(freeze["parent_overlap"]),
            }
        )
    cross_overlap = train_parents & validation_parents
    if cross_overlap:
        raise ValueError(
            "Recorded roots introduce cross-root train/validation parent overlap: "
            f"{sorted(cross_overlap)[:5]}"
        )
    if not rows:
        raise ValueError("Recorded training mixture is empty")
    return rows, {
        "schema": RECORDED_MIX_SCHEMA,
        "split": "train",
        "target_blind_input_contract": {
            "mixture": "recorded mixture waveform",
            "dna": "selected-note DNA waveform",
            "other_dna": "digital silence",
            "unassigned_residual": "mixture minus selected-note DNA",
            "target_access": "loss only; never an inference input",
        },
        "roots": root_contracts,
        "rows": len(rows),
        "train_parents": len(train_parents),
        "validation_parents": len(validation_parents),
        "train_validation_parent_overlap": 0,
    }


def _cohort_sha256(values: Iterable[str | int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_recorded_validation_data(
    roots: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the immutable recorded validation cohort with exact root bindings."""

    rows: list[dict[str, Any]] = []
    root_contracts: list[dict[str, Any]] = []
    asset_ids: list[str] = []
    seen_assets: set[str] = set()
    for root_value in roots:
        root = root_value.expanduser().resolve()
        freeze_path = root / "FROZEN.json"
        freeze = _read_json(freeze_path)
        if (
            freeze.get("schema") != "frozen_dna_residual_corpus_v1"
            or freeze.get("status") != "passed"
            or not freeze.get("payload_checksums_verified", False)
        ):
            raise ValueError(f"Recorded residual root is not frozen-passed: {root}")
        loaded = load_frozen_residual_rows(root, split="validation")
        if not loaded:
            raise ValueError(f"Recorded validation cohort is empty: {root}")
        root_asset_ids: list[str] = []
        for row in loaded:
            metadata = dict(row["metadata"])
            asset_id = ":".join(
                (
                    str(row["source_id"]),
                    str(row["sample_id"]),
                    f"note-{int(metadata['note_row']):05d}",
                )
            )
            if asset_id in seen_assets:
                raise ValueError(f"Duplicate recorded validation asset: {asset_id}")
            seen_assets.add(asset_id)
            asset_ids.append(asset_id)
            root_asset_ids.append(asset_id)
            rows.append(
                {
                    **row,
                    "training_asset_id": asset_id,
                    "frozen_root": str(root),
                    "frozen_split": "validation",
                }
            )
        root_contracts.append(
            {
                "root": str(root),
                "split": "validation",
                "frozen_sha256": sha256_file(freeze_path),
                "index_sha256": str(freeze["index_sha256"]),
                "count": len(root_asset_ids),
                "cohort_sha256": _cohort_sha256(root_asset_ids),
            }
        )
    return rows, {
        "schema": "dna_residual_v13_recorded_validation_cohort_v1",
        "immutable": True,
        "split": "validation",
        "count": len(rows),
        "cohort_sha256": _cohort_sha256(asset_ids),
        "roots": root_contracts,
    }


def validation_aggregate_75_25(
    synthetic: dict[str, float], recorded: dict[str, float] | None
) -> dict[str, float]:
    """Aggregate each metric by fixed cohort weights, never by row count."""

    if recorded is None:
        return dict(synthetic)
    if synthetic.keys() != recorded.keys():
        raise ValueError("Synthetic and recorded validation metrics differ")
    return {
        key: 0.75 * float(synthetic[key]) + 0.25 * float(recorded[key])
        for key in synthetic
    }


def validation_spectral_score(aggregate: dict[str, float]) -> float:
    return float(aggregate["complex"] + 0.5 * aggregate["log_magnitude"])


def validation_metric_fields(
    synthetic: dict[str, float],
    recorded: dict[str, float] | None,
    aggregate: dict[str, float],
) -> dict[str, float]:
    fields = {
        f"validation/synthetic/{key}": value for key, value in synthetic.items()
    }
    if recorded is not None:
        fields.update(
            {f"validation/recorded/{key}": value for key, value in recorded.items()}
        )
    fields.update(
        {
            f"validation/aggregate_75_25/{key}": value
            for key, value in aggregate.items()
        }
    )
    return fields


def best_checkpoint_selection_metric(recorded_exists: bool) -> str:
    return (
        "validation/aggregate_75_25/spectral"
        if recorded_exists
        else "validation/synthetic/spectral"
    )




def selected_model() -> torch.nn.Module:
    raise RuntimeError("Launch NoteGrab training through train.py")


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_type = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[optimizer_name]
    return optimizer_type(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def active_loss_contract() -> dict[str, Any]:
    raise RuntimeError("Launch NoteGrab training through train.py")


def configure_compilation(model: torch.nn.Module) -> None:
    """The paper recipe uses uncompiled training."""
    del model


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_metrics(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    raise RuntimeError("Launch NoteGrab training through train.py")


def lr_decay_steps(args: argparse.Namespace) -> int:
    value = getattr(args, "lr_decay_steps", None)
    return int(args.max_steps if value is None else value)


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step <= int(args.warmup_steps):
        scale = step / max(1, int(args.warmup_steps))
    else:
        progress = (step - int(args.warmup_steps)) / max(
            1, lr_decay_steps(args) - int(args.warmup_steps)
        )
        scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return float(args.learning_rate) * scale


def sampler_contract(args: argparse.Namespace) -> dict[str, Any]:
    alpha = float(args.family_temperature_alpha)
    dynamic = int(args.audio_budget_samples) > 0
    if not dynamic:
        synthetic_contract = {
            "schema": "dna_residual_v13_sampler_v1",
            "strategy": "uniform_affine" if alpha == 1.0 else "family_temperature",
            "family_key": "sound_family",
            "family_temperature_alpha": alpha,
            "virtual_epoch_pairs": "full_train_split",
            "resume_state": "optimizer_step",
        }
    else:
        synthetic_contract = {
            "schema": "dna_residual_v13_sampler_v2",
            "strategy": "duration_bucket_audio_budget",
            "family_key": "sound_family",
            "family_temperature_alpha": alpha,
            "virtual_epoch_pairs": "step_stateless",
            "resume_state": "optimizer_step",
            "duration_index": str(args.duration_index),
            "audio_budget_samples": int(args.audio_budget_samples),
            "maximum_dynamic_batch_size": int(args.maximum_dynamic_batch_size),
            "minimum_window_samples": int(args.minimum_window_samples),
            "lead_samples": int(args.lead_samples),
            "release_samples": int(args.release_samples),
            "window_quantum": int(args.window_quantum),
        }
        if args.priority_index is not None:
            source = args.priority_index.expanduser().resolve()
            synthetic_contract["priority_sampling"] = {
                "path": str(source),
                "sha256": sha256_file(source),
                "fraction": float(args.priority_fraction),
                "family_temperature_alpha": float(
                    args.priority_family_temperature_alpha
                ),
                "policy": "mixed_within_duration_bucket_family_temperature_v2",
            }
        if args.acoustic_offset_index is not None:
            source = args.acoustic_offset_index.expanduser().resolve()
            synthetic_contract["acoustic_offset_index"] = {
                "path": str(source),
                "shard_markers_sha256": acoustic_offset_marker_digest(source),
                "release_policy": "max(fixed_release,detected_acoustic_release)",
            }
    fraction = float(getattr(args, "real_example_fraction", 0.0))
    synthetic_ordinal_view = getattr(args, "synthetic_ordinal_view", None)
    if synthetic_ordinal_view is not None:
        view = synthetic_ordinal_view.expanduser().resolve()
        synthetic_contract["ordinal_view"] = {
            "path": str(view),
            "sha256": sha256_file(view),
        }
    if fraction == 0.0:
        return synthetic_contract
    return {
        "schema": "dna_residual_v13_sampler_v3",
        "strategy": "deterministic_synthetic_recorded_slot_replacement",
        "real_example_fraction": fraction,
        "recorded_split": "train",
        "recorded_frozen_roots": [
            str(Path(root).expanduser().resolve())
            for root in getattr(args, "recorded_frozen_root", [])
        ],
        "synthetic_sampler": synthetic_contract,
        "resume_state": "optimizer_step",
    }


def localize_family_ordinals(
    family_ordinals: dict[str, np.ndarray],
    view_ordinals: list[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Map corpus-global family membership onto a controlled view's indices."""
    view = np.asarray(view_ordinals, dtype=np.int64)
    if view.ndim != 1 or len(view) < 1 or np.any(view < 0):
        raise ValueError("Controlled view ordinals must be a nonempty nonnegative vector")
    maximum = max(
        int(view.max()),
        max(int(values[-1]) for values in family_ordinals.values() if len(values)),
    )
    inverse = np.full(maximum + 1, -1, dtype=np.int32)
    inverse[view] = np.arange(len(view), dtype=np.int32)
    if np.count_nonzero(inverse >= 0) != len(view):
        raise ValueError("Controlled view ordinals must be unique")

    localized: dict[str, np.ndarray] = {}
    covered = np.zeros(len(view), dtype=np.bool_)
    for family, global_values in sorted(family_ordinals.items()):
        values = np.asarray(global_values, dtype=np.int64)
        local = inverse[values]
        local = np.sort(local[local >= 0]).astype(np.int32, copy=False)
        if len(local):
            if np.any(covered[local]):
                raise ValueError("Localized family ordinals overlap")
            covered[local] = True
            local.setflags(write=False)
            localized[str(family)] = local
    if not bool(np.all(covered)):
        raise ValueError("Localized family ordinals do not cover the controlled view")
    return localized




@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    autocast: Any,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for raw in loader:
        batch = to_device(raw, device)
        with autocast():
            metrics = forward_metrics(model, batch)
        count = int(batch["mixture"].shape[0])
        examples += count
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + count * float(value.detach().float())
    if examples < 1:
        raise ValueError("Validation loader is empty")
    model.train()
    return {key: value / examples for key, value in totals.items()}


def cuda_memory_percent(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    total_memory = torch.cuda.get_device_properties(device).total_memory
    statistics = torch.cuda.memory_stats(device)
    return {
        "gpu_memory_allocated_percent": (
            100.0 * torch.cuda.memory_allocated(device) / total_memory
        ),
        "gpu_memory_reserved_percent": (
            100.0 * torch.cuda.memory_reserved(device) / total_memory
        ),
        "gpu_memory_inactive_split_percent": (
            100.0 * statistics.get("inactive_split_bytes.all.current", 0) / total_memory
        ),
        "gpu_memory_segments": float(statistics.get("segment.all.current", 0)),
        "gpu_memory_allocation_retries": float(statistics.get("num_alloc_retries", 0)),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def retain_numbered_checkpoint(
    latest: Path, *, global_step: int, keep_every: int
) -> Path | None:
    """Hard-link a durable checkpoint without serializing its tensors twice."""
    if keep_every == 0 or global_step % keep_every != 0:
        return None
    retained = latest.with_name(f"step-{global_step:09d}.pt")
    os.link(latest, retained)
    return retained


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    best_score: float,
    examples_presented: int,
    audio_samples_presented: int,
    binding: dict[str, Any],
    args: argparse.Namespace,
    validation_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "global_step": int(global_step),
        "best_score": float(best_score),
        "examples_presented": int(examples_presented),
        "audio_samples_presented": int(audio_samples_presented),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_contract": dict(ACTIVE_MODEL_CONTRACT),
        "loss_contract": active_loss_contract(),
        "sampler_contract": sampler_contract(args),
        "corpus_binding": binding,
        "validation_binding": validation_binding,
        "training_contract": {
            "batch_size": int(args.batch_size),
            "fixed_samples": int(args.fixed_samples),
            "max_steps": int(args.max_steps),
            "lr_decay_steps": lr_decay_steps(args),
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "optimizer_name": OPTIMIZER_NAME,
            "compile_mode": COMPILE_MODE,
        },
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def restore_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    binding: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    validation_binding: dict[str, Any] | None = None,
) -> tuple[int, float, int, int]:
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=False
    )
    expected_training = {
        "batch_size": int(args.batch_size),
        "fixed_samples": int(args.fixed_samples),
        "max_steps": int(args.max_steps),
        "lr_decay_steps": lr_decay_steps(args),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_name": OPTIMIZER_NAME,
        "compile_mode": COMPILE_MODE,
    }
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Resume checkpoint has the wrong schema")
    if normalized_checkpoint_model_contract(
        checkpoint.get("model_contract")
    ) != normalized_checkpoint_model_contract(ACTIVE_MODEL_CONTRACT):
        raise ValueError(
            "Resume checkpoint model contract differs from selected production model"
        )
    if normalized_checkpoint_loss_contract(
        checkpoint.get("loss_contract")
    ) != normalized_checkpoint_loss_contract(active_loss_contract()):
        raise ValueError("Resume checkpoint loss contract differs from free_target")
    if checkpoint.get("sampler_contract") != sampler_contract(args):
        raise ValueError("Resume checkpoint sampler contract changed")
    if checkpoint.get("corpus_binding") != binding:
        raise ValueError("Resume checkpoint is bound to another frozen corpus")
    recorded_validation_exists = bool(
        validation_binding and validation_binding.get("recorded") is not None
    )
    if recorded_validation_exists and checkpoint.get("validation_binding") != validation_binding:
        raise ValueError("Resume checkpoint validation cohort changed")
    actual_training = dict(checkpoint.get("training_contract", {}))
    actual_training.setdefault("compile_mode", "off")
    actual_training.setdefault("optimizer_name", "adamw")
    actual_training.setdefault(
        "lr_decay_steps", int(actual_training.get("max_steps", args.max_steps))
    )
    checkpoint_max_steps = actual_training.pop("max_steps", None)
    requested_max_steps = expected_training.pop("max_steps")
    if (
        isinstance(checkpoint_max_steps, bool)
        or not isinstance(checkpoint_max_steps, int)
        or checkpoint_max_steps > requested_max_steps
        or actual_training != expected_training
    ):
        raise ValueError("Resume checkpoint training geometry changed")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_states" in checkpoint:
        states = checkpoint["cuda_rng_states"]
        if not isinstance(states, (list, tuple)) or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            for state in states
        ):
            raise ValueError("checkpoint CUDA RNG states are invalid")
        # map_location=cuda also moves RNG byte tensors, but PyTorch's CUDA RNG
        # setter requires CPU ByteTensors.
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in states])
    step = int(checkpoint["global_step"])
    examples = int(checkpoint.get("examples_presented", step * int(args.batch_size)))
    audio_samples = int(
        checkpoint.get(
            "audio_samples_presented",
            examples * int(args.fixed_samples),
        )
    )
    return step, float(checkpoint.get("best_score", math.inf)), examples, audio_samples


def initialize_model_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    allow_loss_contract_change: bool = False,
    allow_output_carrier_mode_change: bool = False,
) -> dict[str, Any]:
    """Initialize compatible model weights without inheriting run state."""

    source = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Initialization checkpoint has the wrong schema")
    source_model_contract = normalized_checkpoint_model_contract(
        checkpoint.get("model_contract")
    )
    target_model_contract = normalized_checkpoint_model_contract(
        ACTIVE_MODEL_CONTRACT
    )
    model_contract_match = source_model_contract == target_model_contract
    output_carrier_mode_change = False
    if not model_contract_match and allow_output_carrier_mode_change:
        migrated_source = copy.deepcopy(source_model_contract)
        source_architecture = migrated_source.get("architecture", {})
        target_architecture = target_model_contract.get("architecture", {})
        source_mode = source_architecture.get(
            "output_carrier_mode", "encoder_input"
        )
        target_mode = target_architecture.get(
            "output_carrier_mode", "encoder_input"
        )
        if source_mode == "encoder_input" and target_mode == "ungated_hpss":
            source_architecture["output_carrier_mode"] = target_mode
            # These are descriptive consequences of the carrier switch in the
            # model contract; they do not correspond to parameters or another
            # forward-path migration.
            for section, keys in {
                "inputs": (
                    "harmonic_encoder_evidence",
                    "harmonic_output_carrier",
                    "percussive_output_carrier",
                ),
                "outputs": ("background",),
                "harmonic_gate": ("application",),
            }.items():
                target_section = target_model_contract.get(section)
                if not isinstance(target_section, dict):
                    continue
                source_section = migrated_source.setdefault(section, {})
                if not isinstance(source_section, dict):
                    break
                for key in keys:
                    if key in target_section:
                        source_section[key] = copy.deepcopy(target_section[key])
            output_carrier_mode_change = migrated_source == target_model_contract
    if not model_contract_match and not output_carrier_mode_change:
        compared_source = (
            migrated_source
            if allow_output_carrier_mode_change and "migrated_source" in locals()
            else source_model_contract
        )
        difference = first_contract_difference(
            compared_source, target_model_contract
        )
        raise ValueError(
            "Initialization checkpoint model contract differs after allowed "
            f"migrations: {difference!r}"
        )
    source_loss_contract = checkpoint.get("loss_contract")
    target_loss_contract = active_loss_contract()
    normalized_source_loss = normalized_checkpoint_loss_contract(source_loss_contract)
    normalized_target_loss = normalized_checkpoint_loss_contract(target_loss_contract)
    loss_contract_match = normalized_source_loss == normalized_target_loss
    if not loss_contract_match and not allow_loss_contract_change:
        raise ValueError(
            "Initialization checkpoint loss contract differs: "
            f"{first_contract_difference(normalized_source_loss, normalized_target_loss)!r}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "schema": (
            "dna_residual_v13_model_initialization_loss_change_v1"
            if not loss_contract_match
            else "dna_residual_v13_model_initialization_v1"
        ),
        "checkpoint": str(source),
        "checkpoint_sha256": sha256_file(source),
        "source_global_step": int(checkpoint.get("global_step", 0)),
        "model_only": True,
        "optimizer_reset": True,
        "model_contract_match": model_contract_match,
        "output_carrier_mode_change": (
            {
                "source": "encoder_input",
                "target": "ungated_hpss",
                "parameter_change": False,
            }
            if output_carrier_mode_change
            else None
        ),
        "loss_contract_match": loss_contract_match,
        "source_loss_contract": source_loss_contract,
        "target_loss_contract": target_loss_contract,
    }


def first_contract_difference(
    left: Any, right: Any, path: str = "$"
) -> tuple[str, Any, Any] | None:
    """Return the first deterministic leaf mismatch between two contracts."""

    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}"
            if key not in left:
                return child_path, "<missing>", right[key]
            if key not in right:
                return child_path, left[key], "<missing>"
            difference = first_contract_difference(left[key], right[key], child_path)
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_contract_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, left, right
    return None


def validate_resume_run_contract(
    existing: dict[str, Any], requested: dict[str, Any]
) -> tuple[int, int]:
    """Allow only a monotonic step-budget extension of an existing run."""

    old = json.loads(json.dumps(existing))
    new = json.loads(json.dumps(requested))
    old_arguments = old.get("arguments")
    new_arguments = new.get("arguments")
    if not isinstance(old_arguments, dict) or not isinstance(new_arguments, dict):
        difference = first_contract_difference(old, new)
        detail = "unknown difference" if difference is None else repr(difference)
        raise ValueError(f"Resume run contract changed: {detail}")
    # Resume restores the full checkpoint, including its initialization.
    if new.get("initialization") is None and old.get("initialization") is not None:
        new["initialization"] = old["initialization"]
        new["trained_from_scratch"] = old.get("trained_from_scratch", False)
    # These diagnostic-only arguments were added after older checkpoints were
    # written. Their disabled defaults preserve the original training contract.
    for key, disabled_default in {
        "monitor_media_every": 0,
        "monitor_media_on_resume": False,
    }.items():
        if key not in old_arguments and new_arguments.get(key) == disabled_default:
            new_arguments.pop(key)
    # This permission is consumed only while constructing a new model-only
    # initialization.  A true resume must not repeat --initialize-checkpoint,
    # so the parser necessarily reports False even when the original run used
    # the permission.  The immutable ``initialization`` binding above retains
    # the source checkpoint and loss-contract provenance.
    old_arguments.pop("initialize_allow_loss_contract_change", None)
    new_arguments.pop("initialize_allow_loss_contract_change", None)
    old_arguments.pop("initialize_allow_output_carrier_mode_change", None)
    new_arguments.pop("initialize_allow_output_carrier_mode_change", None)
    old_max_steps = old_arguments.pop("max_steps", None)
    new_max_steps = new_arguments.pop("max_steps", None)
    if (
        isinstance(old_max_steps, bool)
        or not isinstance(old_max_steps, int)
        or isinstance(new_max_steps, bool)
        or not isinstance(new_max_steps, int)
        or new_max_steps < old_max_steps
        or old != new
    ):
        difference = first_contract_difference(old, new)
        detail = "unknown difference" if difference is None else repr(difference)
        raise ValueError(f"Resume run contract changed: {detail}")
    return old_max_steps, new_max_steps


def json_safe_argument(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_safe_argument(item) for item in value]
    return value




def load_priority_ordinals(
    path: Path | None,
    *,
    train_pairs: int,
    ordinal_view_binding: dict[str, Any] | None,
) -> np.ndarray | None:
    """Load a hash-bound local-ordinal priority index fail-closed."""

    if path is None:
        return None
    source = path.expanduser().resolve()
    report = _read_json(source.with_suffix(".json"))
    if report.get("schema") != "occurrence_octave_ambiguity_index_v1":
        raise ValueError("Priority index has the wrong schema")
    if report.get("output_sha256") != sha256_file(source):
        raise ValueError("Priority index hash does not match its report")
    expected_pairs = int(report.get("view_train_pairs", report.get("train_pairs", -1)))
    if expected_pairs != train_pairs:
        raise ValueError("Priority index does not align with the training view")
    if (
        ordinal_view_binding is not None
        and report.get("ordinal_view_sha256") != ordinal_view_binding["sha256"]
    ):
        raise ValueError("Priority index is bound to another ordinal view")
    with np.load(source, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != "occurrence_octave_ambiguity_index_v1":
            raise ValueError("Priority NPZ has the wrong schema")
        values = np.asarray(payload["local_ordinal"], dtype=np.int64)
    if (
        values.ndim != 1
        or len(values) < 1
        or int(values.min()) < 0
        or int(values.max()) >= train_pairs
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("Priority index local ordinals are invalid")
    values.setflags(write=False)
    return values




def append_metric(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resume is not None and args.initialize_checkpoint is not None:
        raise ValueError("--resume and --initialize-checkpoint are mutually exclusive")
    if args.initialize_allow_loss_contract_change and args.initialize_checkpoint is None:
        raise ValueError(
            "--initialize-allow-loss-contract-change requires --initialize-checkpoint"
        )
    if (
        args.initialize_allow_output_carrier_mode_change
        and args.initialize_checkpoint is None
    ):
        raise ValueError(
            "--initialize-allow-output-carrier-mode-change requires "
            "--initialize-checkpoint"
        )
    validate_recorded_mix_args(args)
    positive = (
        args.max_steps,
        args.batch_size,
        args.prefetch_factor,
        args.reader_cache_size,
        args.monitor_pairs,
        args.eval_every,
        args.save_every,
        args.log_every,
    )
    if min(map(int, positive)) < 1 or int(args.num_workers) < 0:
        raise ValueError(
            "Training, loader, logging, and checkpoint limits must be positive"
        )
    if int(args.keep_checkpoint_every) < 0:
        raise ValueError("--keep-checkpoint-every must be non-negative")
    if int(args.monitor_media_every) < 0:
        raise ValueError("--monitor-media-every must be non-negative")
    if int(args.keep_checkpoint_every) % int(args.save_every) != 0:
        raise ValueError("--keep-checkpoint-every must be a multiple of --save-every")
    if lr_decay_steps(args) < int(args.warmup_steps):
        raise ValueError("--lr-decay-steps must be at least --warmup-steps")
    if not 0.0 <= float(args.family_temperature_alpha) <= 1.0:
        raise ValueError("--family-temperature-alpha must be in [0, 1]")
    if not 0.0 <= float(args.priority_fraction) <= 1.0:
        raise ValueError("--priority-fraction must be in [0, 1]")
    if not 0.0 <= float(args.priority_family_temperature_alpha) <= 1.0:
        raise ValueError("--priority-family-temperature-alpha must be in [0, 1]")
    dynamic = int(args.audio_budget_samples) > 0
    if dynamic:
        if args.duration_index is None:
            raise ValueError("--duration-index is required with --audio-budget-samples")
        if (
            min(
                int(args.maximum_dynamic_batch_size),
                int(args.minimum_window_samples),
                int(args.window_quantum),
            )
            < 1
        ):
            raise ValueError("Invalid dynamic batching limits")
    elif int(args.fixed_samples) < 1024:
        raise ValueError("--fixed-samples must be at least 1024")
    if args.priority_index is not None and not dynamic:
        raise ValueError("--priority-index currently requires dynamic batching")
    if float(args.priority_fraction) > 0.0 and args.priority_index is None:
        raise ValueError("--priority-fraction requires --priority-index")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Production V13 training requires an available CUDA device")

    contract = FrozenV13Contract.load(args.frozen_manifest)
    acoustic_release_samples = load_acoustic_release_samples(
        args.acoustic_offset_index, pairs=int(contract.split_pairs["train"])
    )
    synthetic_binding = contract.checkpoint_binding()
    train_ordinals, validation_ordinals, ordinal_view_binding = (
        load_synthetic_ordinal_view(args.synthetic_ordinal_view, contract=contract)
    )
    training_pair_count = (
        int(contract.split_pairs["train"])
        if train_ordinals is None
        else int(len(train_ordinals))
    )
    priority_ordinals = load_priority_ordinals(
        args.priority_index,
        train_pairs=training_pair_count,
        ordinal_view_binding=ordinal_view_binding,
    )
    if ordinal_view_binding is not None:
        synthetic_binding = {
            **synthetic_binding,
            "ordinal_view": ordinal_view_binding,
        }
    recorded_rows: list[dict[str, Any]] = []
    recorded_contract: dict[str, Any] | None = None
    recorded_validation_rows: list[dict[str, Any]] = []
    recorded_validation_binding: dict[str, Any] | None = None
    if float(args.real_example_fraction) > 0.0:
        recorded_rows, recorded_contract = load_recorded_training_data(
            [Path(root) for root in args.recorded_frozen_root]
        )
        binding = {
            "schema": "dna_residual_v13_mixed_corpus_binding_v1",
            "synthetic": synthetic_binding,
            "recorded": recorded_contract,
            "real_example_fraction": float(args.real_example_fraction),
        }
        recorded_roots = [Path(root) for root in args.recorded_frozen_root]
        frozen_paths = [root.expanduser().resolve() / "FROZEN.json" for root in recorded_roots]
        if all(path.is_file() for path in frozen_paths):
            recorded_validation_rows, recorded_validation_binding = (
                load_recorded_validation_data(recorded_roots)
            )
        elif any(path.exists() for path in frozen_paths):
            raise ValueError("Recorded validation roots are only partially available")
    else:
        binding = synthetic_binding
    synthetic_validation_count = int(contract.split_pairs["train"])
    if validation_ordinals is None:
        monitor_count = min(int(args.monitor_pairs), synthetic_validation_count)
        monitor_indices = np.linspace(
            0, synthetic_validation_count - 1, monitor_count, dtype=np.int64
        ).tolist()
    else:
        monitor_count = min(int(args.monitor_pairs), len(validation_ordinals))
        positions = np.linspace(
            0, len(validation_ordinals) - 1, monitor_count, dtype=np.int64
        )
        monitor_indices = validation_ordinals[positions].tolist()
    synthetic_validation_binding = {
        "schema": "dna_residual_v13_synthetic_validation_cohort_v1",
        "immutable": True,
        "split": "validation" if validation_ordinals is not None else "train_monitor",
        "count": len(monitor_indices),
        "cohort_sha256": _cohort_sha256(monitor_indices),
        "source_binding": synthetic_binding,
    }
    validation_binding = {
        "schema": "dna_residual_v13_validation_binding_v1",
        "aggregate_weights": {"synthetic": 0.75, "recorded": 0.25},
        "synthetic": synthetic_validation_binding,
        "recorded": recorded_validation_binding,
    }
    output = args.output_dir.expanduser().resolve()
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(f"Resume output directory does not exist: {output}")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = selected_model().to(device)
    configure_compilation(model)
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=OPTIMIZER_NAME,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    global_step = 0
    best_score = math.inf
    examples_presented = 0
    audio_samples_presented = 0
    initialization_binding: dict[str, Any] | None = None
    if args.resume is not None:
        global_step, best_score, examples_presented, audio_samples_presented = (
            restore_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                binding=binding,
                args=args,
                device=device,
                validation_binding=validation_binding,
            )
        )
    elif args.initialize_checkpoint is not None:
        initialization_binding = initialize_model_checkpoint(
            args.initialize_checkpoint,
            model=model,
            device=device,
            allow_loss_contract_change=args.initialize_allow_loss_contract_change,
            allow_output_carrier_mode_change=(
                args.initialize_allow_output_carrier_mode_change
            ),
        )
    if global_step >= int(args.max_steps):
        raise ValueError("Resume checkpoint already reached --max-steps")

    run_contract = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "trained_from_scratch": initialization_binding is None and not bool(
            ACTIVE_MODEL_CONTRACT.get("initialization_checkpoint")
        ),
        "initialization": initialization_binding,
        "compile_mode": COMPILE_MODE,
        "skip_initial_monitor": SKIP_INITIAL_MONITOR,
        "model_contract": ACTIVE_MODEL_CONTRACT,
        "loss_contract": active_loss_contract(),
        "sampler_contract": sampler_contract(args),
        "corpus_binding": binding,
        "recorded_training_contract": recorded_contract,
        "validation_binding": validation_binding,
        "arguments": {
            key: json_safe_argument(value)
            for key, value in vars(args).items()
            if key not in {"resume", "initialize_checkpoint", "keep_checkpoint_every"}
        },
    }
    contract_path = output / "contract.json"
    if args.resume is None:
        contract_path.write_text(
            json.dumps(run_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        existing = _read_json(contract_path)
        previous_max_steps, requested_max_steps = validate_resume_run_contract(
            existing, run_contract
        )
        append_metric(
            output / "continuations.jsonl",
            {
                "schema": "dna_residual_v13_continuation_v1",
                "checkpoint": str(args.resume.expanduser().resolve()),
                "checkpoint_sha256": sha256_file(args.resume),
                "resume_step": global_step,
                "previous_max_steps": previous_max_steps,
                "requested_max_steps": requested_max_steps,
                "lr_decay_steps": lr_decay_steps(args),
            },
        )

    synthetic_train_dataset = V13StreamingResidualDataset(
        contract,
        split="train",
        fixed_samples=0 if dynamic else args.fixed_samples,
        reader_cache_size=args.reader_cache_size,
        preserve_complete_note=dynamic,
        lead_samples=args.lead_samples if dynamic else 0,
        release_samples=args.release_samples if dynamic else 0,
        acoustic_release_samples=acoustic_release_samples if dynamic else None,
        minimum_window_samples=args.minimum_window_samples if dynamic else 1024,
        window_quantum=args.window_quantum if dynamic else 1,
    )
    train_dataset: Dataset[dict[str, Any]]
    synthetic_training_view: Dataset[dict[str, Any]] = (
        synthetic_train_dataset
        if train_ordinals is None
        else SyntheticOrdinalViewDataset(synthetic_train_dataset, train_ordinals)
    )
    if recorded_contract is None:
        train_dataset = synthetic_training_view
    else:
        train_dataset = MixedSyntheticRecordedDataset(
            synthetic_training_view, recorded_rows
        )
    monitor_dataset = V13StreamingResidualDataset(
        contract,
        split="train",
        fixed_samples=0 if dynamic else args.fixed_samples,
        reader_cache_size=args.reader_cache_size,
        preserve_complete_note=dynamic,
        lead_samples=args.lead_samples if dynamic else 0,
        release_samples=args.release_samples if dynamic else 0,
        acoustic_release_samples=acoustic_release_samples if dynamic else None,
        minimum_window_samples=args.minimum_window_samples if dynamic else 1024,
        window_quantum=args.window_quantum if dynamic else 1,
    )
    monitor_subset = Subset(monitor_dataset, monitor_indices)
    family_ordinals = None
    if (
        float(args.family_temperature_alpha) != 1.0
        or (
            priority_ordinals is not None
            and float(args.priority_family_temperature_alpha) != 1.0
        )
    ):
        family_ordinals = contract.load_family_ordinals("train")
        if train_ordinals is not None:
            family_ordinals = localize_family_ordinals(
                family_ordinals, train_ordinals
            )
    if dynamic:
        windows = load_duration_windows(
            args.duration_index,
            pairs=len(synthetic_train_dataset),
            binding=synthetic_binding,
            args=args,
            acoustic_release_samples=acoustic_release_samples,
        )
        if train_ordinals is not None:
            windows = windows[train_ordinals]
        synthetic_sampler: Iterable[list[Any]] = ResumeSafeAudioBudgetBatchSampler(
            windows,
            audio_budget_samples=args.audio_budget_samples,
            maximum_batch_size=args.maximum_dynamic_batch_size,
            start_step=global_step,
            stop_step=args.max_steps,
            seed=args.seed,
            family_temperature_alpha=args.family_temperature_alpha,
            family_ordinals=family_ordinals,
            priority_ordinals=priority_ordinals,
            priority_fraction=args.priority_fraction,
            priority_family_temperature_alpha=(
                args.priority_family_temperature_alpha
            ),
        )
    else:
        synthetic_sampler = ResumeSafeBatchSampler(
            (
                len(synthetic_train_dataset)
                if train_ordinals is None
                else len(train_dataset)
            ),
            batch_size=args.batch_size,
            start_step=global_step,
            stop_step=args.max_steps,
            seed=args.seed,
            family_temperature_alpha=args.family_temperature_alpha,
            family_ordinals=family_ordinals,
        )
    sampler: Iterable[list[Any]]
    if recorded_contract is None:
        sampler = synthetic_sampler
    else:
        sampler = DeterministicMixedBatchSampler(
            synthetic_sampler,
            recorded_window_samples=train_dataset.recorded_window_samples,
            real_fraction=float(args.real_example_fraction),
            fixed_samples=int(args.fixed_samples),
            seed=int(args.seed),
            start_step=global_step,
        )
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
        "collate_fn": collate_v13,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_kwargs)
    validation_workers = min(4, int(args.num_workers))
    validation_kwargs: dict[str, Any] = {
        "num_workers": validation_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": validation_workers > 0,
        "collate_fn": collate_v13,
    }
    if validation_workers > 0:
        validation_kwargs["prefetch_factor"] = 2
    monitor_loader = DataLoader(
        monitor_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **validation_kwargs,
    )
    recorded_validation_loader = (
        DataLoader(
            RecordedValidationDataset(recorded_validation_rows),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **validation_kwargs,
        )
        if recorded_validation_rows
        else None
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if use_bf16
        else nullcontext
    )
    metrics_path = output / "metrics.jsonl"
    started = time.monotonic()

    def evaluate_validation() -> tuple[dict[str, float], dict[str, float] | None, dict[str, float]]:
        synthetic = evaluate(model, monitor_loader, device=device, autocast=autocast)
        recorded = (
            evaluate(model, recorded_validation_loader, device=device, autocast=autocast)
            if recorded_validation_loader is not None
            else None
        )
        return synthetic, recorded, validation_aggregate_75_25(synthetic, recorded)

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output.name,
            id=args.wandb_run_id,
            resume="allow",
            mode=args.wandb_mode,
            config=run_contract,
        )

    if (
        (global_step > 0 or initialization_binding is not None or not SKIP_INITIAL_MONITOR)
        and wandb_run is not None
        and bool(args.monitor_media_on_resume)
    ):
        media_builder = globals().get("build_monitor_media")
        if callable(media_builder):
            media = media_builder(
                model,
                monitor_loader,
                device=device,
                autocast=autocast,
                wandb=wandb,
            )
            wandb_run.log(media, step=global_step)
            print(f"MONITOR_MEDIA step={global_step} uploaded=true", flush=True)

    if global_step == 0 and not SKIP_INITIAL_MONITOR:
        monitor_started = time.monotonic()
        memory_before = cuda_memory_percent(device)
        synthetic_validation, recorded_validation, aggregate_validation = evaluate_validation()
        validation = validation_metric_fields(
            synthetic_validation, recorded_validation, aggregate_validation
        )
        event = {
            "kind": "monitor",
            "step": 0,
            "monitor_seconds": time.monotonic() - monitor_started,
            **{f"before_{key}": value for key, value in memory_before.items()},
            **{
                f"after_{key}": value
                for key, value in cuda_memory_percent(device).items()
            },
            **validation,
        }
        append_metric(metrics_path, event)
        print("MONITOR " + json.dumps(event, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(
                {
                    **{f"monitor/{key}": value for key, value in synthetic_validation.items()},
                    **validation,
                },
                step=0,
            )

    model.train()
    loss_ema: dict[str, float] = {}
    previous_step_finished = time.monotonic()
    for raw in train_loader:
        step_started = time.monotonic()
        loader_wait_seconds = step_started - previous_step_finished
        batch = to_device(raw, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        next_step = global_step + 1
        current_lr = learning_rate(next_step, args)
        for group in optimizer.param_groups:
            # Preserve explicit parameter-group multipliers under the shared
            # warmup/decay schedule. Groups without the key remain bit-for-bit
            # on the historical single-LR contract.
            group["lr"] = current_lr * float(group.get("lr_multiplier", 1.0))
        optimizer.zero_grad(set_to_none=True)
        training_step_hook = globals().get("set_training_step")
        if callable(training_step_hook):
            training_step_hook(next_step, int(args.seed))
        with autocast():
            losses = forward_metrics(model, batch)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Non-finite free_target loss at step {next_step}")
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {next_step}")
        optimizer.step()
        global_step = next_step
        step_seconds = time.monotonic() - step_started
        batch_examples = int(batch["mixture"].shape[0])
        padded_samples = int(batch["mixture"].numel())
        valid_samples = int(batch["valid_samples"].sum())
        examples_presented += batch_examples
        audio_samples_presented += valid_samples
        for key, value in losses.items():
            scalar = float(value.detach().float())
            loss_ema[key] = (
                scalar if key not in loss_ema else 0.98 * loss_ema[key] + 0.02 * scalar
            )

        if global_step == 1 or global_step % int(args.log_every) == 0:
            family_counts: dict[str, int] = {}
            for family in raw["sound_family"]:
                family_counts[str(family)] = family_counts.get(str(family), 0) + 1
            event = {
                "kind": "train",
                "step": global_step,
                "lr": current_lr,
                "grad_norm": float(grad_norm.detach().float()),
                "seconds_per_step": step_seconds,
                "loader_wait_seconds": loader_wait_seconds,
                "examples_per_second": batch_examples / max(step_seconds, 1e-9),
                "valid_audio_seconds_per_second": (
                    valid_samples / 16_000 / max(step_seconds, 1e-9)
                ),
                "padded_audio_seconds_per_second": (
                    padded_samples / 16_000 / max(step_seconds, 1e-9)
                ),
                "batch_examples": batch_examples,
                "batch_padded_samples": padded_samples,
                "batch_valid_samples": valid_samples,
                "batch_real_examples": sum(
                    str(value) == "recorded_real_train" for value in raw["stratum"]
                ),
                "batch_real_fraction": sum(
                    str(value) == "recorded_real_train" for value in raw["stratum"]
                )
                / batch_examples,
                "elapsed_seconds": time.monotonic() - started,
                "examples_presented": examples_presented,
                "equivalent_corpus_passes": examples_presented / len(train_dataset),
                "audio_hours_presented": audio_samples_presented / 16_000 / 3_600,
                "batch_ordinals": [int(value) for value in raw["ordinal"].tolist()],
                "batch_family_counts": family_counts,
                "batch_hard_fraction": float(raw["hard_case"].float().mean()),
                "batch_valid_samples_min": int(raw["valid_samples"].min()),
                "batch_valid_samples_max": int(raw["valid_samples"].max()),
                **{f"ema_{key}": value for key, value in loss_ema.items()},
                **{key: float(value.detach().float()) for key, value in losses.items()},
            }
            if device.type == "cuda":
                event.update(cuda_memory_percent(device))
                total_memory = torch.cuda.get_device_properties(device).total_memory
                event["gpu_peak_memory_allocated_percent"] = (
                    100.0 * torch.cuda.max_memory_allocated(device) / total_memory
                )
            append_metric(metrics_path, event)
            print("TRAIN " + json.dumps(event, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"train/{key}": value
                        for key, value in event.items()
                        if key not in {"kind", "step"}
                    },
                    step=global_step,
                )

        if global_step % int(args.eval_every) == 0 or global_step == int(
            args.max_steps
        ):
            monitor_started = time.monotonic()
            memory_before = cuda_memory_percent(device)
            synthetic_validation, recorded_validation, aggregate_validation = (
                evaluate_validation()
            )
            validation = validation_metric_fields(
                synthetic_validation, recorded_validation, aggregate_validation
            )
            event = {
                "kind": "monitor",
                "step": global_step,
                "monitor_seconds": time.monotonic() - monitor_started,
                **{f"before_{key}": value for key, value in memory_before.items()},
                **{
                    f"after_{key}": value
                    for key, value in cuda_memory_percent(device).items()
                },
                **validation,
            }
            append_metric(metrics_path, event)
            print("MONITOR " + json.dumps(event, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{
                            f"monitor/{key}": value
                            for key, value in synthetic_validation.items()
                        },
                        **validation,
                    },
                    step=global_step,
                )
                media_every = int(args.monitor_media_every)
                media_builder = globals().get("build_monitor_media")
                if (
                    media_every > 0
                    and global_step % media_every == 0
                    and callable(media_builder)
                ):
                    try:
                        media = media_builder(
                            model,
                            monitor_loader,
                            device=device,
                            autocast=autocast,
                            wandb=wandb,
                        )
                        wandb_run.log(media, step=global_step)
                        print(
                            f"MONITOR_MEDIA step={global_step} uploaded=true keys={len(media)}",
                            flush=True,
                        )
                    except Exception as error:
                        print(
                            f"MONITOR_MEDIA_WARNING {type(error).__name__}: {error}",
                            flush=True,
                        )
            score = validation_spectral_score(aggregate_validation)
            if score < best_score:
                best_score = score
                atomic_torch_save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        best_score=best_score,
                        examples_presented=examples_presented,
                        audio_samples_presented=audio_samples_presented,
                        binding=binding,
                        args=args,
                        validation_binding=validation_binding,
                    ),
                    checkpoint_dir / "best_monitor_spectral.pt",
                )

        if global_step % int(args.save_every) == 0 or global_step == int(
            args.max_steps
        ):
            checkpoint_started = time.monotonic()
            latest_checkpoint = checkpoint_dir / "latest.pt"
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    best_score=best_score,
                    examples_presented=examples_presented,
                    audio_samples_presented=audio_samples_presented,
                    binding=binding,
                    args=args,
                    validation_binding=validation_binding,
                ),
                latest_checkpoint,
            )
            retained_checkpoint = retain_numbered_checkpoint(
                latest_checkpoint,
                global_step=global_step,
                keep_every=int(args.keep_checkpoint_every),
            )
            checkpoint_event = {
                "kind": "checkpoint",
                "step": global_step,
                "checkpoint_seconds": time.monotonic() - checkpoint_started,
                "retained_checkpoint": (
                    str(retained_checkpoint) if retained_checkpoint is not None else ""
                ),
                **cuda_memory_percent(device),
            }
            append_metric(metrics_path, checkpoint_event)
            print(
                "CHECKPOINT " + json.dumps(checkpoint_event, sort_keys=True), flush=True
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"runtime/{key}": value
                        for key, value in checkpoint_event.items()
                        if key not in {"kind", "step"}
                    },
                    step=global_step,
                )
        # Release step-local tensors without forcing CUDA to discard reusable
        # allocator blocks. The allocator can then reuse or release those
        # blocks according to the next variable-length batch's requirements.
        del batch, losses, grad_norm
        previous_step_finished = time.monotonic()
    train_dataset.close()
    monitor_dataset.close()
    wandb_url = str(getattr(wandb_run, "url", "")) if wandb_run is not None else ""
    if wandb_run is not None:
        wandb_run.finish()
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best_monitor_spectral.pt"
    report = {
        "schema": "dna_residual_v13_streaming_report_v1",
        "status": "completed",
        "trained_from_scratch": not bool(
            ACTIVE_MODEL_CONTRACT.get("initialization_checkpoint")
        ),
        "compile_mode": COMPILE_MODE,
        "global_step": global_step,
        "examples_presented": examples_presented,
        "audio_samples_presented": audio_samples_presented,
        "corpus_passes": examples_presented / len(train_dataset),
        "best_monitor_score": best_score,
        "best_checkpoint_selection_metric": best_checkpoint_selection_metric(
            recorded_validation_binding is not None
        ),
        "monitor_is_training_distribution": recorded_contract is None,
        "monitor_distribution": "synthetic_train_only",
        "monitor_is_promotion_evidence": False,
        "monitor_pairs": monitor_count,
        "model_contract": ACTIVE_MODEL_CONTRACT,
        "loss_contract": active_loss_contract(),
        "sampler_contract": sampler_contract(args),
        "corpus_binding": binding,
        "recorded_training_contract": recorded_contract,
        "validation_binding": validation_binding,
        "latest_checkpoint": str(latest),
        "latest_checkpoint_sha256": sha256_file(latest),
        "best_monitor_checkpoint": str(best),
        "best_monitor_checkpoint_sha256": sha256_file(best),
        "wandb_url": wandb_url,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(output / "report.json", report)
    print(
        "TRAINING_COMPLETE "
        + json.dumps(
            {"step": global_step, "best_score": best_score, "output": str(output)},
            sort_keys=True,
        ),
        flush=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value
