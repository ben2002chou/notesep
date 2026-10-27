"""Read-only, DNA-free adapter for packed recorded natural-score scenes."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from notesep.shards import PackedBlobShardReader
from notesep.training_paths import relocated

SCENE_KIND = "recorded_natural_score_exact_stems_v1"
SCENE_KINDS_BY_FREEZE_SCHEMA = {
    "recorded_natural_score_scene_freeze_v1": SCENE_KIND,
    "recorded_paired_random_timing_freeze_v2": SCENE_KIND,
    "cross_instrument_natural_score_exact_note_freeze_v1": "cross_instrument_natural_score_exact_note_scene_v1",
}
FREEZE_SCHEMAS = {
    "recorded_natural_score_scene_freeze_v1",
    "recorded_paired_random_timing_freeze_v2",
    "cross_instrument_natural_score_exact_note_freeze_v1",
}


def sha256_file(path: Path) -> str:
    path = relocated(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SceneRef:
    root_index: int
    scene_id: str
    record_kind: str
    split: str
    k: int
    first_row: int
    samples: int
    canonical_key: str


class RecordedNaturalScoreSceneDataset(Dataset[dict[str, Any]]):
    """Expose one full-mixture target row per stem without DNA or residuals.

    Rows from one scene remain adjacent, and ``scene_batch_indices`` exposes the
    corresponding complete-scene batch. The adapter never crops a scene and
    retains the full list of score events on every row.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        split: str,
        comparison_roots: Sequence[str | Path] = (),
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.split = split
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        if not self.roots or len(set(self.roots)) != len(self.roots):
            raise ValueError("scene roots must be nonempty and distinct")
        self.readers: list[PackedBlobShardReader] = []
        self.freezes: list[dict[str, Any]] = []
        self.scenes: list[SceneRef] = []
        self.row_to_scene_note: list[tuple[int, int]] = []
        comparison_lengths: dict[str, int] = {}
        for comparison_root in comparison_roots:
            with PackedBlobShardReader(
                Path(comparison_root).expanduser().resolve()
            ) as comparison:
                for record in comparison.iter_records(SCENE_KIND):
                    metadata = record.metadata
                    if str(metadata["split"]) != split:
                        continue
                    key = str(metadata.get("paired_parent_scene_id", record.sample_id))
                    comparison_lengths[key] = int(metadata["num_samples"])
        seen_scene_ids: set[tuple[int, str]] = set()
        for root_index, root in enumerate(self.roots):
            freeze_path = root / "FROZEN.json"
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if (
                freeze.get("schema") not in FREEZE_SCHEMAS
                or freeze.get("status") != "passed"
            ):
                raise ValueError(f"scene root is not frozen-passed: {root}")
            record_kind = SCENE_KINDS_BY_FREEZE_SCHEMA[str(freeze["schema"])]
            manifest = dict(freeze["manifest"])
            if sha256_file(relocated(manifest["path"])) != str(manifest["sha256"]):
                raise ValueError(f"scene manifest checksum mismatch: {root}")
            if sha256_file(root / "manifest.sqlite") != str(
                freeze["packed_index"]["sha256"]
            ):
                raise ValueError(f"packed index checksum mismatch: {root}")
            reader = PackedBlobShardReader(root)
            self.readers.append(reader)
            self.freezes.append(freeze | {"freeze_sha256": sha256_file(freeze_path)})
            for record in reader.iter_records(record_kind):
                metadata = record.metadata
                if str(metadata["split"]) != split:
                    continue
                k = int(metadata["k"])
                notes = metadata.get("notes", metadata.get("events", ()))
                if k < 1 or len(notes) != k:
                    raise ValueError(f"invalid scene cardinality: {record.sample_id}")
                identity = (root_index, record.sample_id)
                if identity in seen_scene_ids:
                    raise ValueError(f"duplicate scene identity: {identity}")
                seen_scene_ids.add(identity)
                canonical_key = str(
                    metadata.get("paired_parent_scene_id", record.sample_id)
                )
                common_samples = max(
                    int(metadata["num_samples"]),
                    int(comparison_lengths.get(canonical_key, metadata["num_samples"])),
                )
                first_row = 0
                self.scenes.append(
                    SceneRef(
                        root_index,
                        record.sample_id,
                        record_kind,
                        split,
                        k,
                        first_row,
                        common_samples,
                        canonical_key,
                    )
                )
        self.scenes.sort(key=lambda scene: scene.canonical_key)
        rebuilt: list[SceneRef] = []
        for scene in self.scenes:
            scene_index = len(rebuilt)
            first_row = len(self.row_to_scene_note)
            rebuilt.append(
                SceneRef(
                    scene.root_index,
                    scene.scene_id,
                    scene.record_kind,
                    scene.split,
                    scene.k,
                    first_row,
                    scene.samples,
                    scene.canonical_key,
                )
            )
            self.row_to_scene_note.extend(
                (scene_index, note_index) for note_index in range(scene.k)
            )
        self.scenes = rebuilt
        if not self.scenes:
            raise ValueError(f"no {split} scenes found")
        self._cached_key: tuple[int, str] | None = None
        self._cached_stems: np.ndarray | None = None
        self._cached_metadata: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.row_to_scene_note)

    def _load_scene(self, scene_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        scene = self.scenes[int(scene_index)]
        key = (scene.root_index, scene.scene_id)
        if self._cached_key == key:
            assert self._cached_stems is not None and self._cached_metadata is not None
            return self._cached_stems, self._cached_metadata
        reader = self.readers[scene.root_index]
        record = reader.record(scene.scene_id, scene.record_kind)
        payload = reader.read_bytes(scene.scene_id, scene.record_kind)
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError(f"packed payload checksum mismatch: {scene.scene_id}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            stems = np.asarray(archive["stems"], dtype=np.float32)
        metadata = dict(record.metadata)
        native_samples = int(metadata["num_samples"])
        if stems.shape != (scene.k, native_samples) or not np.isfinite(stems).all():
            raise ValueError(f"invalid stems: {scene.scene_id}")
        native_mixture = np.sum(stems, axis=0, dtype=np.float32)
        if (
            hashlib.sha256(np.ascontiguousarray(native_mixture).tobytes()).hexdigest()
            != metadata["mixture_sha256"]
        ):
            raise ValueError(
                f"native mixture closure checksum mismatch: {scene.scene_id}"
            )
        if scene.samples > native_samples:
            stems = np.pad(stems, ((0, 0), (0, scene.samples - native_samples)))
        self._cached_key, self._cached_stems, self._cached_metadata = (
            key,
            stems,
            metadata,
        )
        return stems, metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_index, note_index = self.row_to_scene_note[int(index)]
        scene = self.scenes[scene_index]
        stems, metadata = self._load_scene(scene_index)
        mixture = np.sum(stems, axis=0, dtype=np.float32)
        metadata_notes = metadata.get("notes", metadata.get("events"))
        if metadata_notes is None:
            raise ValueError(f"scene has no note events: {scene.scene_id}")
        note = dict(metadata_notes[note_index])
        sample_rate = int(metadata["sample_rate"])
        events: list[dict[str, Any]] = []
        for event_index, value in enumerate(metadata_notes):
            event = dict(value)
            onset_seconds = event.get("onset_seconds", event.get("midi_onset_seconds"))
            if onset_seconds is None:
                raise ValueError(f"event has no onset: {scene.scene_id}")
            onset = round(float(onset_seconds) * sample_rate)
            offset = round(float(event["acoustic_offset_seconds"]) * sample_rate)
            midi_offset = round(float(event["midi_offset_seconds"]) * sample_rate)
            events.append(
                {
                    "pitch": int(event["rendered_pitch"]),
                    "onset_sample": int(onset),
                    "offset_sample": int(offset),
                    "midi_offset_sample": int(midi_offset),
                    "is_target": int(event_index == note_index),
                    "velocity": int(event.get("score_velocity", 64)),
                    "duration_samples": max(1, int(offset - onset)),
                }
            )
        if sum(int(event["is_target"]) for event in events) != 1:
            raise ValueError("scene row must mark exactly one target score event")
        zero = torch.zeros_like(torch.from_numpy(mixture))
        note_onset_seconds = note.get("onset_seconds", note.get("midi_onset_seconds"))
        if note_onset_seconds is None:
            raise ValueError(f"target event has no onset: {scene.scene_id}")
        source_parent_id = note.get(
            "source_parent_id", metadata.get("source_parent_id")
        )
        instrument = note.get("recorded_instrument", metadata.get("instrument"))
        score_parent_id = metadata.get(
            "score_parent_id", metadata.get("score_piece_parent_id")
        )
        if source_parent_id is None or instrument is None or score_parent_id is None:
            raise ValueError(f"target event lacks source identity: {scene.scene_id}")
        return {
            "mixture": torch.from_numpy(mixture.copy()),
            "target": torch.from_numpy(np.ascontiguousarray(stems[note_index]).copy()),
            # Compatibility placeholders are required by the frozen V13
            # collator but ignored by the selective model contract.  They do
            # not introduce a DNA or residual dependency.
            "dna": zero.clone(),
            "other_dna": zero.clone(),
            "unassigned_residual": zero.clone(),
            "pitch": torch.tensor(int(note["rendered_pitch"]), dtype=torch.long),
            "valid_samples": torch.tensor(len(mixture), dtype=torch.long),
            "note_onset_samples": torch.tensor(
                round(float(note_onset_seconds) * sample_rate), dtype=torch.long
            ),
            "note_offset_samples": torch.tensor(
                round(float(note["acoustic_offset_seconds"]) * sample_rate),
                dtype=torch.long,
            ),
            "midi_offset_samples": torch.tensor(
                round(float(note["midi_offset_seconds"]) * sample_rate),
                dtype=torch.long,
            ),
            "ordinal": torch.tensor(-(int(index) + 1), dtype=torch.long),
            "hard_case": torch.tensor(1, dtype=torch.long),
            "pair_id": f"recorded-scene::{scene.scene_id}::note-{note_index:02d}",
            "sound_id": f"recorded-scene::{scene.scene_id}",
            "sound_family": str(instrument),
            "source_group": str(source_parent_id),
            "stratum": f"recorded_natural_scene_{self.split}_K{scene.k}",
            "scene_id": scene.scene_id,
            "scene_index": scene_index,
            "scene_note_index": note_index,
            "scene_k": scene.k,
            "full_score_events": tuple(events),
            "all_score_events": tuple(events),
            "score_parent_id": str(score_parent_id),
            "source_parent_id": str(source_parent_id),
            "instrument": str(instrument),
            "split": self.split,
            "frozen_root": str(self.roots[scene.root_index]),
            "conditioning_contract": "full_scene_score_events_no_dna_no_residual",
        }

    def scene_batch_indices(self) -> list[list[int]]:
        return [
            list(range(scene.first_row, scene.first_row + scene.k))
            for scene in self.scenes
        ]

    def close(self) -> None:
        for reader in self.readers:
            reader.close()
