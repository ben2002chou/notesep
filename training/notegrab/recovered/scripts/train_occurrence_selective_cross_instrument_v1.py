#!/usr/bin/env python3
"""Minimal selective-gated trainer overlay for exact-note natural scenes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
SELECTIVE_ROOT = Path(os.environ.get(
    "OCCURRENCE_SELECTIVE_SOURCE_ROOT",
    "/depot/yunglu/data/ben/code/DNA/scns_directional_selective_gate_v2_20260903",
)).resolve()
FACTORIAL_ROOT = Path(os.environ.get(
    "OCCURRENCE_FACTORIAL_SOURCE_ROOT",
    "/depot/yunglu/data/ben/code/DNA/model_scns_factorial_2x2_v1_20260902",
)).resolve()
sys.path[:0] = [str(SELECTIVE_ROOT / "src"), str(SELECTIVE_ROOT),
                str(FACTORIAL_ROOT / "src"), str(FACTORIAL_ROOT),
                str(ROOT / "src"), str(ROOT)]
import model_exp  # noqa: E402,F401
from model_exp.cross_instrument_exact_note_scene_data import (  # noqa: E402
    SCENE_KIND, CompositeSyntheticSceneDataset, CrossInstrumentExactNoteSceneDataset,
)

selective_path = SELECTIVE_ROOT / "scripts/train_occurrence_directional_selective_gate_v1.py"
spec = importlib.util.spec_from_file_location("frozen_selective_cross_instrument", selective_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load selective trainer: {selective_path}")
selective = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selective)
base = selective.base

_TRAIN_SCENES: "TelemetryCrossInstrumentDataset | None" = None
_REALIZED_EXPOSURE = {
    "recorded_optimizer_steps": 0,
    "scenes_by_k": {2: 0, 3: 0, 4: 0},
    "rows_by_k": {2: 0, 3: 0, 4: 0},
}
K_POLICY = os.environ.get("OCCURRENCE_CROSS_INSTRUMENT_K_POLICY", "mixed")
if K_POLICY not in {"k2_only", "mixed"}:
    raise ValueError("OCCURRENCE_CROSS_INSTRUMENT_K_POLICY must be k2_only or mixed")


def mixed_k_schedule() -> list[int]:
    """Prefix-matched finite K schedule for the declared training budget."""
    from model_exp.recorded_exposure_schedule import recorded_schedule
    return [k for _, k in recorded_schedule(
        int(os.environ.get('MAX_STEPS', '500')),
        os.environ.get('REAL_EXAMPLE_FRACTION', '.25'))]


MIXED_K_SCHEDULE = mixed_k_schedule()
CROP_POLICY = os.environ.get("OCCURRENCE_RECORDED_CROP_POLICY", "whole_scene")
if CROP_POLICY not in {"whole_scene", "target_relative"}:
    raise ValueError("unknown recorded crop policy")


class TelemetryCrossInstrumentDataset(CrossInstrumentExactNoteSceneDataset):
    """Preserve base recorded detection while carrying explicit scene labels."""
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        if CROP_POLICY == "target_relative":
            from model_exp.recorded_note_crop import crop_recorded_note
            row = crop_recorded_note(row)
        row["stratum"] = "recorded_real_train" if self.split == "train" else "recorded_real_validation"
        row["recorded_scene_k"] = int(row["scene_k"])
        row["recorded_timing_view"] = "natural_score"
        row["recorded_source_view"] = str(row["view"])
        return row


def _cohort_hash(dataset: CrossInstrumentExactNoteSceneDataset) -> str:
    rows = [f"{x.root_index}:{x.scene_id}:{x.split}:K{x.k}" for x in dataset.scenes]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def _binding(dataset: TelemetryCrossInstrumentDataset, split: str) -> dict[str, Any]:
    counts = {f"K{k}": sum(x.k == k for x in dataset.scenes) for k in (2, 3, 4)}
    return {
        "schema": "cross_instrument_selective_training_binding_v1", "split": split,
        "scene_sampling_unit": True, "model_rows": "all_K_exact_isolated_note_targets",
        "scene_k_counts": counts, "scenes": len(dataset.scenes), "target_rows": len(dataset),
        "cohort_sha256": _cohort_hash(dataset), "timing_view": "natural_score",
        "source_views": sorted({str(x["view"]) for x in dataset.freezes}),
        "endpoint": "acoustic_offset_including_release", "dna_conditioning": False,
        "residual_conditioning": False,
        "roots": [{"path": str(root), "freeze_sha256": dataset.freezes[i]["freeze_sha256"]}
                  for i, root in enumerate(dataset.roots)],
    }


def load_training(roots: list[Path]) -> tuple[TelemetryCrossInstrumentDataset, dict[str, Any]]:
    global _TRAIN_SCENES
    dataset = TelemetryCrossInstrumentDataset(roots, split="train")
    _TRAIN_SCENES = dataset
    return dataset, _binding(dataset, "train")


def load_validation(roots: list[Path]) -> tuple[TelemetryCrossInstrumentDataset, dict[str, Any]]:
    override = os.environ.get("OCCURRENCE_CROSS_INSTRUMENT_VALIDATION_ROOT", "")
    dataset = TelemetryCrossInstrumentDataset([Path(override)] if override else roots, split="validation")
    return dataset, _binding(dataset, "validation")


class MixedDataset(CompositeSyntheticSceneDataset):
    def __init__(self, synthetic: Dataset[Any], scenes: TelemetryCrossInstrumentDataset) -> None:
        super().__init__(synthetic, scenes)
        self.recorded_window_samples = np.asarray([x.samples for x in scenes.scenes], dtype=np.int64)


class SceneSampler(Sampler[list[Any]]):
    """Replace complete steps with same-K, indivisible complete scenes."""
    def __init__(self, base_sampler: Iterable[list[Any]], *, recorded_window_samples: np.ndarray,
                 real_fraction: float, fixed_samples: int, seed: int, start_step: int) -> None:
        del recorded_window_samples, fixed_samples
        if _TRAIN_SCENES is None:
            raise RuntimeError("training scenes must be loaded before sampler construction")
        self.base_sampler, self.scenes = base_sampler, _TRAIN_SCENES
        self.fraction, self.seed, self.start_step = float(real_fraction), int(seed), int(start_step)
        if not 0 < self.fraction < 1: raise ValueError("recorded fraction must be in (0,1)")
        self.max_rows = int(os.environ.get("MAXIMUM_DYNAMIC_BATCH_SIZE", "192"))
        self.audio_budget = int(os.environ.get("AUDIO_BUDGET_SAMPLES", "3200000"))
        self.by_k = {k: [i for i, x in enumerate(self.scenes.scenes) if x.k == k] for k in (2, 3, 4)}
        if any(not self.by_k[k] for k in self.by_k): raise ValueError("K2/K3/K4 are required")
        output = Path(os.environ.get("OUTPUT", ".")).expanduser().resolve()
        self.yield_authority = output / "recorded_scene_batches_yielded.jsonl"

    def __len__(self) -> int: return len(self.base_sampler)  # type: ignore[arg-type]

    def _rank(self, value: str) -> str:
        return hashlib.sha256(f"{self.seed}:{value}".encode()).hexdigest()

    def __iter__(self) -> Iterator[list[Any]]:
        for step, batch in enumerate(self.base_sampler, start=self.start_step):
            if int((step + 1) * self.fraction) == int(step * self.fraction):
                yield [("synthetic", x) for x in batch]; continue
            intervention_ordinal = int((step + 1) * self.fraction) - 1
            if K_POLICY == "k2_only":
                selected_k = 2
            else:
                if intervention_ordinal >= len(MIXED_K_SCHEDULE):
                    # Do not silently extend the declared training budget.
                    raise RuntimeError("mixed-K schedule exhausted")
                selected_k = MIXED_K_SCHEDULE[intervention_ordinal]
            output, used, cursor = [], set(), 0
            while len(output) < self.max_rows:
                k = selected_k
                ranked = sorted(self.by_k[k], key=lambda i: self._rank(f"{step}:{cursor}:{i}"))
                choice = next((i for i in ranked if i not in used), None)
                if choice is None: break
                scene = self.scenes.scenes[choice]
                prospective_rows = len(output) + scene.k
                prospective_length = max([self.scenes.scenes[i].samples for i in used] + [scene.samples])
                if prospective_rows > self.max_rows or prospective_rows * prospective_length > self.audio_budget: break
                used.add(choice)
                output.extend(("scene", row) for row in range(scene.first_row, scene.first_row + scene.k))
                cursor += 1
            if not output: raise RuntimeError("no complete cross-instrument scene fits recorded step")
            scene_counts = {k: sum(self.scenes.scenes[i].k == k for i in used) for k in (2, 3, 4)}
            row_counts = {k: k * scene_counts[k] for k in (2, 3, 4)}
            event = {
                "schema": "cross_instrument_scene_batch_yielded_v1",
                "candidate_optimizer_step": step + 1,
                "recorded_intervention_ordinal": intervention_ordinal,
                "K_policy": K_POLICY,
                "selected_step_K": selected_k,
                "realized_optimizer_step": False,
                "scene_counts_by_K": {f"K{k}": scene_counts[k] for k in (2, 3, 4)},
                "row_counts_by_K": {f"K{k}": row_counts[k] for k in (2, 3, 4)},
                "timing_view": "natural_score",
                "source_views": sorted({str(self.scenes.freezes[i]["view"]) for i in range(len(self.scenes.freezes))}),
            }
            self.yield_authority.parent.mkdir(parents=True, exist_ok=True)
            with self.yield_authority.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            yield output


_selective_collate = base.collate_v13


def telemetry_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = _selective_collate(rows)
    recorded = torch.tensor([str(x.get("stratum", "")) == "recorded_real_train" for x in rows])
    output["telemetry_recorded_mask"] = recorded
    output["telemetry_recorded_k"] = torch.tensor([int(x.get("recorded_scene_k", 0)) for x in rows])
    output["telemetry_timing_natural"] = torch.tensor([x.get("recorded_timing_view") == "natural_score" for x in rows])
    output["telemetry_source_production"] = torch.tensor([x.get("recorded_source_view") == "production" for x in rows])
    output["telemetry_source_research"] = torch.tensor([x.get("recorded_source_view") == "research" for x in rows])
    scenes = {(str(x.get("scene_id", "")), int(x.get("recorded_scene_k", 0)))
              for x in rows if x.get("recorded_scene_k")}
    for k in (2, 3, 4):
        output[f"telemetry_scene_K{k}_count"] = torch.tensor(sum(value == k for _, value in scenes))
        output[f"telemetry_row_K{k}_count"] = torch.tensor(sum(int(x.get("recorded_scene_k", 0)) == k for x in rows))
    return output


_selective_forward_metrics = base.forward_metrics


def forward_metrics(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    metrics = _selective_forward_metrics(model, batch)
    dtype, device = metrics["loss"].dtype, metrics["loss"].device
    recorded = batch.get("telemetry_recorded_mask", torch.zeros(len(batch["pitch"]), device=device)).to(device).bool()
    denominator = recorded.sum().clamp_min(1).to(dtype)
    k = batch.get("telemetry_recorded_k", torch.zeros_like(batch["pitch"])).to(device)
    metrics.update({
        "telemetry_recorded_real_train_fraction": recorded.float().mean(),
        "telemetry_recorded_K2_row_fraction": ((k == 2) & recorded).sum().to(dtype) / denominator,
        "telemetry_recorded_K3_row_fraction": ((k == 3) & recorded).sum().to(dtype) / denominator,
        "telemetry_recorded_K4_row_fraction": ((k == 4) & recorded).sum().to(dtype) / denominator,
        "telemetry_recorded_natural_timing_fraction": batch["telemetry_timing_natural"].to(device).float().mean(),
        "telemetry_recorded_production_view_fraction": batch["telemetry_source_production"].to(device).float().mean(),
        "telemetry_recorded_research_view_fraction": batch["telemetry_source_research"].to(device).float().mean(),
    })
    for value in (2, 3, 4):
        metrics[f"telemetry_recorded_K{value}_scene_count"] = batch[f"telemetry_scene_K{value}_count"].to(device, dtype=dtype)
        metrics[f"telemetry_recorded_K{value}_row_count"] = batch[f"telemetry_row_K{value}_count"].to(device, dtype=dtype)
    # This in-memory accumulator is only made authoritative by
    # ``append_metric`` below, which the parent loop calls after optimizer.step.
    if model.training and bool(recorded.any()):
        _REALIZED_EXPOSURE["recorded_optimizer_steps"] += 1
        for value in (2, 3, 4):
            _REALIZED_EXPOSURE["scenes_by_k"][value] += int(batch[f"telemetry_scene_K{value}_count"])
            _REALIZED_EXPOSURE["rows_by_k"][value] += int(batch[f"telemetry_row_K{value}_count"])
    return metrics


_base_append_metric = base.append_metric


def append_metric(path: Path, event: dict[str, Any]) -> None:
    """Commit exposure only from a parent train event emitted after optimizer.step."""
    if event.get("kind") == "train":
        event.update({
            "realized_recorded_optimizer_steps": _REALIZED_EXPOSURE["recorded_optimizer_steps"],
            "realized_recorded_scene_counts_by_K": {
                f"K{k}": _REALIZED_EXPOSURE["scenes_by_k"][k] for k in (2, 3, 4)
            },
            "realized_recorded_row_counts_by_K": {
                f"K{k}": _REALIZED_EXPOSURE["rows_by_k"][k] for k in (2, 3, 4)
            },
        })
    _base_append_metric(path, event)
    if event.get("kind") == "train":
        authority = path.parent / "recorded_scene_exposure.jsonl"
        authority_event = {
            "schema": "cross_instrument_realized_scene_exposure_v1",
            "committed_after_optimizer_step": True,
            "global_optimizer_step": int(event["step"]),
            "recorded_optimizer_steps_this_process": _REALIZED_EXPOSURE["recorded_optimizer_steps"],
            "scene_counts_by_K_this_process": event["realized_recorded_scene_counts_by_K"],
            "row_counts_by_K_this_process": event["realized_recorded_row_counts_by_K"],
            "timing_view": "natural_score",
        }
        with authority.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(authority_event, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())


class SceneValidationDataset(Dataset[dict[str, Any]]):
    def __init__(self, scenes: TelemetryCrossInstrumentDataset) -> None: self.scenes = scenes
    def __len__(self) -> int: return len(self.scenes)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.scenes[index]


def validate_mix_args(args: Any) -> None:
    roots = [Path(x).expanduser().resolve() for x in getattr(args, "recorded_frozen_root", [])]
    fraction = float(getattr(args, "real_example_fraction", 0))
    # Keep the zero-recorded arm on this exact overlay so the four-arm causal
    # comparison does not confound corpus composition with a different trainer.
    if fraction == 0 and not roots:
        return
    if not 0 < fraction < 1 or len(roots) != 1:
        raise ValueError(
            "cross-instrument overlay requires zero roots at fraction 0, or "
            "one root with recorded fraction in (0,1)"
        )


_base_sampler_contract = base.sampler_contract
def sampler_contract(args: Any) -> dict[str, Any]:
    return _base_sampler_contract(args) | {
        "strategy": "complete_cross_instrument_scene_replacement_v1",
        "scene_sampling_unit": True,
        "K_policy": K_POLICY,
        "recorded_crop_policy": CROP_POLICY,
        "recorded_batch_selection_geometry": "original_full_scene_lengths_for_identical_draws",
        "mixed_K_step_schedule": (
            {f"K{k}": MIXED_K_SCHEDULE.count(k) for k in (2, 3, 4)}
            if K_POLICY == "mixed" else {"K2": len(MIXED_K_SCHEDULE), "K3": 0, "K4": 0}
        ),
        "mixed_K_schedule_sha256": hashlib.sha256(
            bytes(MIXED_K_SCHEDULE)
        ).hexdigest(),
        "timing_view": "natural_score", "endpoint": "acoustic_offset_including_release",
        "telemetry": "recorded_real_train plus K/timing/source-view fractions",
        "yielded_batch_authority": "OUTPUT/recorded_scene_batches_yielded.jsonl; non-authoritative pre-optimizer sampler output",
        "realized_exposure_authority": "OUTPUT/recorded_scene_exposure.jsonl; cumulative commits only after optimizer.step train events",
    }


base.load_recorded_training_data = load_training
base.load_recorded_validation_data = load_validation
base.MixedSyntheticRecordedDataset = MixedDataset
base.DeterministicMixedBatchSampler = SceneSampler
base.RecordedValidationDataset = SceneValidationDataset
base.validate_recorded_mix_args = validate_mix_args
base.sampler_contract = sampler_contract
base.collate_v13 = telemetry_collate
base.forward_metrics = forward_metrics
base.append_metric = append_metric


if __name__ == "__main__": base.main()
