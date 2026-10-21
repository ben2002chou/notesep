#!/usr/bin/env python3
"""Train the frozen symmetric separator with exact real-score event clusters.

This is a data-adapter overlay only.  It reuses the frozen selective symmetric
model, HPSS losses, score conditioning, optimizer, and scheduler.  Unlike the
older cross-instrument overlay it reads the native ``notes`` schema emitted by
the real-score event-cluster builder, so no legacy transposition/loop metadata
is silently assumed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
SELECTIVE_ROOT = Path(os.environ["OCCURRENCE_SELECTIVE_SOURCE_ROOT"]).resolve()
FACTORIAL_ROOT = Path(os.environ["OCCURRENCE_FACTORIAL_SOURCE_ROOT"]).resolve()
sys.path[:0] = [str(SELECTIVE_ROOT / "src"), str(SELECTIVE_ROOT), str(FACTORIAL_ROOT / "src"), str(FACTORIAL_ROOT), str(ROOT / "src"), str(ROOT)]
import model_exp  # noqa: E402,F401
from model_exp.recorded_natural_score_scene_data import CompositeSyntheticSceneDataset, RecordedNaturalScoreSceneDataset  # noqa: E402


selective_path = SELECTIVE_ROOT / "scripts/train_occurrence_directional_selective_gate_v1.py"
spec = importlib.util.spec_from_file_location("scns_eventcluster_selective", selective_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen selective trainer: {selective_path}")
selective = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selective)
base = selective.base

_TRAIN_SCENES: "TelemetryEventClusterDataset | None" = None
_REALIZED = {"optimizer_steps": 0, "scene_counts": {2: 0, 4: 0}, "row_counts": {2: 0, 4: 0}}


def canonical_instrument(value: str) -> str:
    key = "_".join(value.strip().lower().replace("-", " ").split())
    return {
        "contrabass": "double_bass", "doublebass": "double_bass", "bass": "bass_guitar",
        "electric_bass": "bass_guitar", "alto_saxophone": "saxophone", "alto_sax": "saxophone",
        "horn": "french_horn", "french_horn": "french_horn",
    }.get(key, key)


class TelemetryEventClusterDataset(RecordedNaturalScoreSceneDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        row["stratum"] = f"recorded_real_{self.split}"
        row["recorded_scene_k"] = int(row["scene_k"])
        row["recorded_timing_view"] = "real_score_event_cluster"
        row["recorded_source_view"] = "exact_native_raw_or_sample"
        return row


def cohort_hash(dataset: TelemetryEventClusterDataset) -> str:
    rows = [f"{scene.root_index}:{scene.scene_id}:{scene.split}:K{scene.k}" for scene in dataset.scenes]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def binding(dataset: TelemetryEventClusterDataset, split: str) -> dict[str, Any]:
    counts = {f"K{k}": sum(scene.k == k for scene in dataset.scenes) for k in (2, 4)}
    return {
        "schema": "scns_eventcluster_training_binding_v1", "split": split,
        "scene_sampling_unit": False, "model_rows": "score-addressed isolated-note targets within complete source mixtures",
        "scene_k_counts": counts, "scenes": len(dataset.scenes), "target_rows": len(dataset),
        "cohort_sha256": cohort_hash(dataset), "timing_view": "real_score_event_cluster",
        "endpoint": "acoustic_offset_exact_native_recording", "dna_conditioning": False, "residual_conditioning": False,
        "roots": [{"path": str(root), "freeze_sha256": dataset.freezes[index]["freeze_sha256"]} for index, root in enumerate(dataset.roots)],
    }


def load_training(roots: list[Path]) -> tuple[TelemetryEventClusterDataset, dict[str, Any]]:
    global _TRAIN_SCENES
    dataset = TelemetryEventClusterDataset(roots, split="train")
    _TRAIN_SCENES = dataset
    return dataset, binding(dataset, "train")


def load_validation(roots: list[Path]) -> tuple[TelemetryEventClusterDataset, dict[str, Any]]:
    override = os.environ.get("OCCURRENCE_EVENTCLUSTER_VALIDATION_ROOT", "")
    dataset = TelemetryEventClusterDataset([Path(override)] if override else roots, split="validation")
    return dataset, binding(dataset, "validation")


class MixedDataset(CompositeSyntheticSceneDataset):
    """Legacy mixed-loader compatibility for source-complete scene rows."""

    def __init__(self, synthetic: Dataset[Any], scenes: TelemetryEventClusterDataset) -> None:
        super().__init__(synthetic, scenes)
        # The frozen trainer passes this vector to its sampler constructor.
        # It is not used by SourceDiverseWithinBatchSampler, which inspects
        # complete-scene lengths directly, but retaining it makes the adapter
        # conform to the base trainer's explicit mixed-data interface.
        self.recorded_window_samples = np.asarray(
            [scene.samples for scene in scenes.scenes for _ in range(scene.k)],
            dtype=np.int64,
        )
        if self.recorded_window_samples.size != len(scenes) or np.any(self.recorded_window_samples < 1):
            raise ValueError("event-cluster recorded windows must be positive and row-aligned")


@dataclass(frozen=True)
class RowRef:
    row_index: int
    samples: int
    k: int
    instrument: str
    source_parent: str


class SourceDiverseWithinBatchSampler(Sampler[list[Any]]):
    """Replace a fixed fraction of rows, never whole steps or score scenes."""

    def __init__(self, base_sampler: Iterable[list[Any]], *, recorded_window_samples: np.ndarray,
                 real_fraction: float, fixed_samples: int, seed: int, start_step: int) -> None:
        del recorded_window_samples
        if _TRAIN_SCENES is None:
            raise RuntimeError("recorded event-cluster scenes were not loaded")
        minimum = float(os.environ.get("SCNS_MINIMUM_REAL_EXAMPLE_FRACTION", "0.20"))
        maximum = float(os.environ.get("SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION", str(1.0 / 3.0)))
        if not minimum <= float(real_fraction) <= maximum + 1e-9:
            raise ValueError(
                "REAL_EXAMPLE_FRACTION outside declared bounds: "
                f"fraction={real_fraction}, minimum={minimum}, maximum={maximum}"
            )
        self.base_sampler, self.scenes = base_sampler, _TRAIN_SCENES
        self.fraction, self.fixed_samples, self.seed, self.start_step = float(real_fraction), int(fixed_samples), int(seed), int(start_step)
        self.output = Path(os.environ.get("OUTPUT", ".")).expanduser().resolve()
        self.rows = self._rows()
        parent_sets: dict[str, set[str]] = {}
        for row in self.rows:
            parent_sets.setdefault(row.instrument, set()).add(row.source_parent)
        self.family_weight = {
            instrument: min(3.0, math.sqrt(len(parents)))
            for instrument, parents in parent_sets.items()
        }
        self.family_use: Counter[str] = Counter()
        self.parent_use: Counter[str] = Counter()

    def _rows(self) -> list[RowRef]:
        rows: list[RowRef] = []
        for scene in self.scenes.scenes:
            record = self.scenes.readers[scene.root_index].record(scene.scene_id, scene.record_kind)
            notes = list(record.metadata.get("notes", ()))
            if len(notes) != scene.k:
                raise ValueError(f"note metadata mismatch: {scene.scene_id}")
            for note_index, note in enumerate(notes):
                rows.append(RowRef(scene.first_row + note_index, scene.samples, scene.k,
                                   canonical_instrument(str(note["recorded_instrument"])), str(note["source_parent_id"])))
        if not rows:
            raise RuntimeError("no event-cluster training rows")
        return rows

    def __len__(self) -> int:
        return len(self.base_sampler)  # type: ignore[arg-type]

    def _rank(self, value: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{self.seed}:{value}".encode()).digest()[:8], "little")

    def _count(self, step: int, batch_size: int) -> int:
        exact = self.fraction * batch_size
        return math.floor(exact) + int(self._rank(f"round:{step}") / 2**64 < exact - math.floor(exact))

    def __iter__(self) -> Iterator[list[Any]]:
        for step, batch in enumerate(self.base_sampler, start=self.start_step):
            # Source-complete real scenes must never be shortened merely to fit
            # a synthetic duration bucket.  Small synthetic buckets therefore
            # contribute no real rows; log that shortfall explicitly rather
            # than failing the loader or silently truncating a long tail.
            eligible_slots = [
                slot for slot, synthetic_index in enumerate(batch)
                if any(
                    row.samples <= (
                        int(synthetic_index[1])
                        if isinstance(synthetic_index, tuple)
                        else self.fixed_samples
                    )
                    for row in self.rows
                )
            ]
            requested_count = self._count(step, len(batch))
            count = min(requested_count, len(eligible_slots))
            slots = set(sorted(
                eligible_slots,
                key=lambda slot: self._rank(f"slot:{step}:{slot}"),
            )[:count])
            used: set[int] = set(); selected: list[RowRef] = []; output: list[Any] = []
            for slot, synthetic_index in enumerate(batch):
                if slot not in slots:
                    output.append(("synthetic", synthetic_index)); continue
                requested = int(synthetic_index[1]) if isinstance(synthetic_index, tuple) else self.fixed_samples
                eligible = [row for row in self.rows if row.samples <= requested and row.row_index not in used]
                if not eligible:
                    raise RuntimeError(f"no event-cluster target fits batch length={requested}, step={step}")
                eligible.sort(key=lambda row: (
                    self.family_use[row.instrument] / self.family_weight[row.instrument],
                    self.parent_use[row.source_parent], self._rank(f"row:{step}:{slot}:{row.row_index}"),
                ))
                row = eligible[0]; used.add(row.row_index); selected.append(row)
                self.family_use[row.instrument] += 1; self.parent_use[row.source_parent] += 1
                output.append(("scene", row.row_index))
            authority = self.output / "eventcluster_within_batch_yields.jsonl"
            authority.parent.mkdir(parents=True, exist_ok=True)
            with authority.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "schema": "scns_eventcluster_within_batch_yield_v1", "candidate_optimizer_step": step + 1,
                    "requested_recorded_fraction": self.fraction, "batch_rows": len(output), "recorded_rows": len(selected),
                    "requested_recorded_rows": requested_count, "recorded_eligible_slots": len(eligible_slots),
                    "recorded_instruments": sorted({row.instrument for row in selected}),
                    "recorded_source_parents": sorted({row.source_parent for row in selected}), "recorded_k": [row.k for row in selected],
                }, sort_keys=True) + "\n")
            yield output


_base_collate = base.collate_v13
def telemetry_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = _base_collate(rows)
    recorded = torch.tensor([str(row.get("stratum", "")) == "recorded_real_train" for row in rows])
    output["telemetry_recorded_mask"] = recorded
    output["telemetry_recorded_k"] = torch.tensor([int(row.get("recorded_scene_k", 0)) for row in rows])
    output["telemetry_real_score"] = torch.tensor([row.get("recorded_timing_view") == "real_score_event_cluster" for row in rows])
    return output


_base_forward_metrics = base.forward_metrics
def forward_metrics(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    metrics = _base_forward_metrics(model, batch)
    dtype, device = metrics["loss"].dtype, metrics["loss"].device
    mask = batch["telemetry_recorded_mask"].to(device).bool()
    denominator = mask.sum().clamp_min(1).to(dtype)
    k = batch["telemetry_recorded_k"].to(device)
    metrics["telemetry_eventcluster_row_fraction"] = mask.float().mean()
    metrics["telemetry_eventcluster_real_score_fraction"] = batch["telemetry_real_score"].to(device).float().mean()
    for value in (2, 4):
        metrics[f"telemetry_eventcluster_K{value}_row_fraction"] = ((k == value) & mask).sum().to(dtype) / denominator
    if model.training and bool(mask.any()):
        _REALIZED["optimizer_steps"] += 1
        for value in (2, 4):
            _REALIZED["row_counts"][value] += int(((k == value) & mask).sum())
    return metrics


_base_append_metric = base.append_metric
def append_metric(path: Path, event: dict[str, Any]) -> None:
    if event.get("kind") == "train":
        event["realized_eventcluster_optimizer_steps"] = _REALIZED["optimizer_steps"]
        event["realized_eventcluster_rows_by_K"] = {f"K{k}": _REALIZED["row_counts"][k] for k in (2, 4)}
    _base_append_metric(path, event)


class ValidationDataset(Dataset[dict[str, Any]]):
    def __init__(self, scenes: TelemetryEventClusterDataset) -> None: self.scenes = scenes
    def __len__(self) -> int: return len(self.scenes)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.scenes[index]


def validate_mix_args(args: Any) -> None:
    roots = [Path(value).expanduser().resolve() for value in getattr(args, "recorded_frozen_root", [])]
    fraction = float(getattr(args, "real_example_fraction", 0))
    minimum = float(os.environ.get("SCNS_MINIMUM_REAL_EXAMPLE_FRACTION", "0.20"))
    maximum = float(os.environ.get("SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION", str(1.0 / 3.0)))
    if not len(roots) == 1 or not minimum <= fraction <= maximum + 1e-9:
        raise ValueError(
            "event-cluster adapter requires exactly one root and a fraction within "
            f"[{minimum}, {maximum}], got {fraction}"
        )


_base_contract = base.sampler_contract
def sampler_contract(args: Any) -> dict[str, Any]:
    return _base_contract(args) | {
        "strategy": "source_diverse_eventcluster_rows_within_batch_v1", "allocation_unit": "model_rows_within_batch",
        "recorded_fraction_bounds": [.20, 1.0 / 3.0], "scene_schema": "recorded_natural_score_scene_freeze_v1",
        "source_policy": "exact_native_pitch_closest_duration_no_waveform_transform", "endpoint": "acoustic_endpoint",
        "source_balance": "family_then_physical_parent", "no_legacy_cross_instrument_events_schema": True,
    }


base.load_recorded_training_data = load_training
base.load_recorded_validation_data = load_validation
base.MixedSyntheticRecordedDataset = MixedDataset
base.DeterministicMixedBatchSampler = SourceDiverseWithinBatchSampler
base.RecordedValidationDataset = ValidationDataset
base.validate_recorded_mix_args = validate_mix_args
base.collate_v13 = telemetry_collate
base.forward_metrics = forward_metrics
base.append_metric = append_metric
base.sampler_contract = sampler_contract


if __name__ == "__main__":
    base.main()
