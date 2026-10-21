#!/usr/bin/env python3
"""SCNS-v3 score-bound rows cropped around each complete acoustic target.

This overlay enforces one acoustic-endpoint crop/conditioning contract across
synthetic and recorded rows. It retains the v2 source sampler, synthetic
ordering, model, losses, optimizer, and augmentation probabilities. No target
acoustic endpoint is truncated and held-at-entry notes remain held.
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import sys
from typing import Any

import library_balanced_sampling as v2
from model_exp.recorded_note_crop import crop_recorded_note, target_crop_geometry


_SyntheticWithFullScore = v2.trackbound.v1.base.V13StreamingResidualDataset


def _overlapping_score_event(
    pitch: int, onset: int, offset: int, samples: int
) -> dict[str, int] | None:
    """Clip the right edge but preserve held-at-entry negative onsets."""
    onset = int(onset)
    offset = min(int(samples), int(offset))
    if offset <= 0 or onset >= int(samples) or onset >= offset or not 0 <= int(pitch) < 128:
        return None
    return {"pitch": int(pitch), "onset_sample": onset, "offset_sample": offset}


# The frozen full-score adapter historically clamped negative onsets to zero,
# despite the model explicitly supporting negative-onset HELD state. Patch the
# function in its owning module for this canonical data contract only.
_full_score_globals = _SyntheticWithFullScore.__getitem__.__globals__
_full_score_globals["_clipped_event"] = _overlapping_score_event


def _load_canonical_augmentation():
    path = Path(__file__).resolve().parents[1] / "src/model_exp/occurrence_audio_augmentation.py"
    spec = importlib.util.spec_from_file_location(
        "model_exp.occurrence_audio_augmentation_canonical", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical occurrence augmentation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.apply_occurrence_augmentation


v2.trackbound.v1.selective.apply_occurrence_augmentation = _load_canonical_augmentation()


class AcousticEndpointSyntheticDataset(_SyntheticWithFullScore):
    """Expose the measured acoustic endpoint as the target offset everywhere."""

    def __getitem__(self, ordinal: int | tuple[int, int]) -> dict[str, Any]:
        output = super().__getitem__(ordinal)
        raw_ordinal = int(ordinal[0] if isinstance(ordinal, tuple) else ordinal)
        if self.acoustic_release_samples is None:
            raise ValueError("canonical acoustic-offset training requires the release index")
        release = int(self.acoustic_release_samples[raw_ordinal])
        if release < 0:
            raise ValueError("negative measured acoustic release")
        midi_offset = int(output["note_offset_samples"])
        acoustic_offset = midi_offset + release
        valid = int(output["valid_samples"])
        if not int(output["note_onset_samples"]) < acoustic_offset <= valid:
            raise ValueError(
                f"synthetic acoustic endpoint outside retained crop: offset={acoustic_offset}, valid={valid}"
            )
        output["midi_offset_samples"] = output["note_offset_samples"].clone()
        output["note_offset_samples"] = output["note_offset_samples"].new_tensor(
            acoustic_offset
        )
        # Full-score events are materialized only when a score-dependent input
        # gate is active. Ungated factorial arms still need the identical
        # acoustic target endpoint, but do not consume a full-score event list.
        if "full_score_events" in output:
            events = []
            for item in output["full_score_events"]:
                event = dict(item)
                if bool(event.get("is_target")):
                    event["midi_offset_sample"] = midi_offset
                    event["offset_sample"] = acoustic_offset
                    event["duration_samples"] = max(
                        1, acoustic_offset - int(event["onset_sample"])
                    )
                events.append(event)
            if sum(bool(event.get("is_target")) for event in events) != 1:
                raise ValueError("synthetic row must retain exactly one target event")
            output["full_score_events"] = events
        output["canonical_target_offset"] = "measured_acoustic_endpoint"
        return output


class TargetCroppedEventClusterDataset(v2.trackbound.v1.TelemetryEventClusterDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = crop_recorded_note(super().__getitem__(index))
        # Keep the semantic timing label consumed by frozen telemetry.  Crop
        # geometry is recorded separately by ``crop_recorded_note``.
        return row


def load_training(roots: list[Path]) -> tuple[TargetCroppedEventClusterDataset, dict[str, Any]]:
    dataset = TargetCroppedEventClusterDataset(roots, split="train")
    v2.trackbound.v1._TRAIN_SCENES = dataset
    binding = v2.trackbound.v1.binding(dataset, "train")
    binding.update({
        "input_geometry": "historical_variable_window_midi_anchor_acoustic_release_4000_2560_min16384_q4096",
        "full_acoustic_target_retained": True,
        "target_conditioning_offset": "exact_measured_acoustic_endpoint",
        "held_at_crop_entry": "negative_onset_not_fabricated_onset_zero",
    })
    return dataset, binding


def load_validation(roots: list[Path]) -> tuple[TargetCroppedEventClusterDataset, dict[str, Any]]:
    """Evaluate recorded rows under the same target-relative geometry as training."""
    override = os.environ.get("OCCURRENCE_EVENTCLUSTER_VALIDATION_ROOT", "")
    dataset = TargetCroppedEventClusterDataset(
        [Path(override)] if override else roots,
        split="validation",
    )
    binding = v2.trackbound.v1.binding(dataset, "validation")
    binding.update({
        "input_geometry": "historical_variable_window_midi_anchor_acoustic_release_4000_2560_min16384_q4096",
        "full_acoustic_target_retained": True,
        "target_conditioning_offset": "exact_measured_acoustic_endpoint",
        "held_at_crop_entry": "negative_onset_not_fabricated_onset_zero",
        "train_validation_geometry_match": True,
    })
    return dataset, binding


class TargetCroppedLibraryBalancedSampler(v2.LibraryBalancedInterleavedSampler):
    """Use the actual target-crop length in promoted-batch budgeting."""

    def _rows(self) -> list[Any]:
        rows = []
        scenes = v2.trackbound.v1._TRAIN_SCENES
        if scenes is None:
            raise RuntimeError("target-cropped SCNS-v3 scenes were not loaded")
        for scene in scenes.scenes:
            record = scenes.readers[scene.root_index].record(scene.scene_id, scene.record_kind)
            metadata = record.metadata
            notes = list(metadata.get("notes", ()))
            if len(notes) != scene.k:
                raise ValueError(f"note metadata mismatch: {scene.scene_id}")
            sample_rate = int(metadata["sample_rate"])
            for note_index, note in enumerate(notes):
                onset_seconds = note.get("onset_seconds", note.get("midi_onset_seconds"))
                if onset_seconds is None:
                    raise ValueError(f"target event has no onset: {scene.scene_id}")
                onset = round(float(onset_seconds) * sample_rate)
                offset = round(float(note["acoustic_offset_seconds"]) * sample_rate)
                midi_offset = round(float(note["midi_offset_seconds"]) * sample_rate)
                _, crop_samples = target_crop_geometry(
                    scene.samples, onset, offset, midi_offset
                )
                library = str(note.get("source_library", "")).strip()
                parent = str(note.get("source_parent_id", "")).strip()
                if not library or not parent:
                    raise ValueError("SCNS-v3 target crop requires source library and parent")
                rows.append(v2.trackbound.LibraryRowRef(
                    scene.first_row + note_index,
                    crop_samples,
                    scene.k,
                    v2.canonical_instrument(str(note["recorded_instrument"])),
                    library,
                    parent,
                ))
        if not rows:
            raise RuntimeError("no target-cropped SCNS-v3 rows")
        return rows


def sampler_contract(args: Any) -> dict[str, Any]:
    return v2.sampler_contract(args) | {
        "strategy": "scns_v3_target_cropped_library_balanced_promoted_batch_v1",
        "recorded_input_geometry": "historical_variable_window_midi_anchor_acoustic_release_4000_2560_min16384_q4096",
        "batch_budget_length": "exact_target_crop_samples",
        "full_acoustic_target_retained": True,
        "target_conditioning_offset_all_source_families": "exact_measured_acoustic_endpoint",
        "held_at_crop_entry": "negative_onset_not_fabricated_onset_zero",
        "scene_mixture_crop": "same_crop_applied_to_mixture_target_and_full_score_coordinates",
        "causal_scope": "recorded_input_geometry_and_resulting_effective_exposure",
        "recorded_fraction_bounds": [
            float(os.environ.get("SCNS_MINIMUM_REAL_EXAMPLE_FRACTION", args.real_example_fraction)),
            float(os.environ.get("SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION", args.real_example_fraction)),
        ],
    }


v2.trackbound.v1.base.load_recorded_training_data = load_training
v2.trackbound.v1.base.load_recorded_validation_data = load_validation
v2.trackbound.v1.base.DeterministicMixedBatchSampler = TargetCroppedLibraryBalancedSampler
v2.trackbound.v1.base.sampler_contract = sampler_contract
v2.trackbound.v1.base.V13StreamingResidualDataset = AcousticEndpointSyntheticDataset


if __name__ == "__main__":
    os.environ.setdefault("SCNS_REQUIRE_ALL_16_RECORDED_FAMILIES", "0")
    v2.trackbound.v1.base.main()
