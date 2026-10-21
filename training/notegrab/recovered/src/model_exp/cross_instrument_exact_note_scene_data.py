"""Read-only training adapter for cross-instrument exact-note scene packs."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from notesep.training_paths import relocated
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .packed_blob_shards import PackedBlobShardReader
from .recorded_natural_score_scene_data import (
    CompleteSceneBatchSampler,
    CompositeSyntheticSceneDataset,
    SceneRef,
    collate_complete_scene_rows,
    collate_selective_scene_rows,
    sha256_file,
)


SCENE_KIND = "cross_instrument_natural_score_exact_note_scene_v1"
FREEZE_SCHEMA = "cross_instrument_natural_score_exact_note_freeze_v1"


class CrossInstrumentExactNoteSceneDataset(Dataset[dict[str, Any]]):
    """Expose K exact isolated-note rows while sampling complete scenes.

    Index construction is metadata-only. Packed NPZ bytes are fetched lazily,
    verified, and cached one scene at a time.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        split: str,
        expected_freeze_sha256: dict[str, str] | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.split = split
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        if not self.roots or len(set(self.roots)) != len(self.roots):
            raise ValueError("roots must be nonempty and distinct")
        self.readers: list[PackedBlobShardReader] = []
        self.freezes: list[dict[str, Any]] = []
        staged: list[SceneRef] = []
        seen: set[tuple[int, str]] = set()
        for root_index, root in enumerate(self.roots):
            freeze_path = root / "FROZEN.json"
            freeze_hash = sha256_file(freeze_path)
            if expected_freeze_sha256 is not None:
                expected = expected_freeze_sha256.get(str(root))
                if expected is None or expected != freeze_hash:
                    raise ValueError(f"freeze checksum mismatch: {root}")
            freeze = json.loads(freeze_path.read_text())
            if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != "passed":
                raise ValueError(f"root is not frozen-passed: {root}")
            manifest = freeze.get("manifest", {})
            packed_index = freeze.get("packed_index", {})
            if sha256_file(relocated(manifest["path"])) != manifest["sha256"]:
                raise ValueError(f"scene manifest checksum mismatch: {root}")
            if sha256_file(root / "manifest.sqlite") != packed_index["sha256"]:
                raise ValueError(f"packed index checksum mismatch: {root}")
            reader = PackedBlobShardReader(root)
            self.readers.append(reader)
            self.freezes.append(freeze | {"freeze_sha256": freeze_hash})
            for record in reader.iter_records(SCENE_KIND):
                meta = dict(record.metadata)
                if str(meta["split"]) != split:
                    continue
                self._validate_metadata(record.sample_id, meta)
                identity = (root_index, record.sample_id)
                if identity in seen:
                    raise ValueError(f"duplicate scene: {identity}")
                seen.add(identity)
                staged.append(SceneRef(root_index, record.sample_id, SCENE_KIND, split,
                                       int(meta["k"]), 0, int(meta["num_samples"]),
                                       record.sample_id))
        staged.sort(key=lambda x: (x.canonical_key, x.root_index))
        self.scenes: list[SceneRef] = []
        self.row_to_scene_note: list[tuple[int, int]] = []
        for scene in staged:
            first_row = len(self.row_to_scene_note)
            self.scenes.append(SceneRef(scene.root_index, scene.scene_id, scene.record_kind,
                                        scene.split, scene.k, first_row, scene.samples,
                                        scene.canonical_key))
            scene_index = len(self.scenes) - 1
            self.row_to_scene_note.extend((scene_index, stem) for stem in range(scene.k))
        if not self.scenes:
            raise ValueError(f"no {split} scenes")
        self._cache_key: tuple[int, str] | None = None
        self._cache_stems: np.ndarray | None = None
        self._cache_meta: dict[str, Any] | None = None

    @staticmethod
    def _validate_metadata(scene_id: str, meta: dict[str, Any]) -> None:
        k, sr, samples = int(meta["k"]), int(meta["sample_rate"]), int(meta["num_samples"])
        events = list(meta.get("events", ()))
        if k not in {2, 3, 4} or len(events) != k or sr <= 0 or samples <= 0:
            raise ValueError(f"invalid cardinality/timing: {scene_id}")
        instruments = {str(x["recorded_instrument"]) for x in events}
        tracks = {str(x["score_track_parent_id"]) for x in events}
        parents = {str(x["source_parent_id"]) for x in events}
        if len(instruments) != k or len(tracks) != k or len(parents) != k:
            raise ValueError(f"scene is not source-complete cross-instrument: {scene_id}")
        for event in events:
            onset = round(float(event["midi_onset_seconds"]) * sr)
            midi_offset = round(float(event["midi_offset_seconds"]) * sr)
            acoustic_offset = round(float(event["acoustic_offset_seconds"]) * sr)
            if not (0 <= onset < midi_offset <= acoustic_offset <= samples):
                raise ValueError(f"invalid local acoustic endpoint: {scene_id}")
            if int(event["rendered_pitch"]) != int(event["pitch"]) + int(meta["transposition_semitones"]):
                raise ValueError(f"invalid exact-pitch binding: {scene_id}")
        if meta.get("closure_contract") != "mixture_is_float32_sum_of_stored_stems":
            raise ValueError(f"unknown closure contract: {scene_id}")

    def __len__(self) -> int:
        return len(self.row_to_scene_note)

    def _load_scene(self, scene_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        scene = self.scenes[int(scene_index)]
        key = (scene.root_index, scene.scene_id)
        if self._cache_key == key:
            assert self._cache_stems is not None and self._cache_meta is not None
            return self._cache_stems, self._cache_meta
        reader = self.readers[scene.root_index]
        record = reader.record(scene.scene_id, SCENE_KIND)
        payload = reader.read_bytes(scene.scene_id, SCENE_KIND)
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError(f"payload checksum mismatch: {scene.scene_id}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            stems = np.asarray(archive["stems"], dtype=np.float32)
        meta = dict(record.metadata)
        if stems.shape != (scene.k, scene.samples) or not np.isfinite(stems).all():
            raise ValueError(f"invalid packed stems: {scene.scene_id}")
        if np.any(np.max(np.abs(stems), axis=1) <= 1e-5):
            raise ValueError(f"silent packed stem: {scene.scene_id}")
        mixture = np.sum(stems, axis=0, dtype=np.float32)
        if hashlib.sha256(np.ascontiguousarray(mixture).tobytes()).hexdigest() != meta["mixture_sha256"]:
            raise ValueError(f"exact mixture closure mismatch: {scene.scene_id}")
        self._cache_key, self._cache_stems, self._cache_meta = key, stems, meta
        return stems, meta

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_index, stem_index = self.row_to_scene_note[int(index)]
        scene = self.scenes[scene_index]
        stems, meta = self._load_scene(scene_index)
        mixture = np.sum(stems, axis=0, dtype=np.float32)
        events = []
        for event_index, raw in enumerate(meta["events"]):
            onset = round(float(raw["midi_onset_seconds"]) * int(meta["sample_rate"]))
            midi_offset = round(float(raw["midi_offset_seconds"]) * int(meta["sample_rate"]))
            acoustic_offset = round(float(raw["acoustic_offset_seconds"]) * int(meta["sample_rate"]))
            events.append({
                "pitch": int(raw["rendered_pitch"]), "onset_sample": onset,
                "offset_sample": acoustic_offset, "midi_offset_sample": midi_offset,
                "is_target": int(event_index == stem_index), "velocity": int(raw["velocity"]),
                "duration_samples": max(1, midi_offset - onset),
            })
        target_event = events[stem_index]
        target_meta = dict(meta["events"][stem_index])
        zero = torch.zeros_like(torch.from_numpy(mixture))
        return {
            "mixture": torch.from_numpy(mixture.copy()),
            "target": torch.from_numpy(np.ascontiguousarray(stems[stem_index]).copy()),
            "dna": zero.clone(), "other_dna": zero.clone(), "unassigned_residual": zero.clone(),
            "pitch": torch.tensor(target_event["pitch"], dtype=torch.long),
            "valid_samples": torch.tensor(scene.samples, dtype=torch.long),
            "note_onset_samples": torch.tensor(target_event["onset_sample"], dtype=torch.long),
            "note_offset_samples": torch.tensor(target_event["offset_sample"], dtype=torch.long),
            "midi_offset_samples": torch.tensor(target_event["midi_offset_sample"], dtype=torch.long),
            "ordinal": torch.tensor(-(int(index) + 1), dtype=torch.long),
            "hard_case": torch.tensor(1, dtype=torch.long),
            "pair_id": f"cross-instrument::{scene.scene_id}::stem-{stem_index:02d}",
            "sound_id": str(target_meta["source_id"]),
            "sound_family": str(target_meta["recorded_instrument"]),
            "source_group": str(target_meta["source_parent_id"]),
            "stratum": f"cross_instrument_{meta['view']}_{self.split}_K{scene.k}",
            "scene_id": scene.scene_id, "scene_index": scene_index,
            "scene_note_index": stem_index, "scene_k": scene.k,
            "full_score_events": tuple(events), "all_score_events": tuple(events),
            "score_parent_id": str(meta["score_piece_parent_id"]),
            "source_parent_id": str(target_meta["source_parent_id"]),
            "score_track_parent_id": str(target_meta["score_track_parent_id"]),
            "instrument": str(target_meta["recorded_instrument"]),
            "source_lane": str(target_meta["source_lane"]),
            "split": self.split, "view": str(meta["view"]), "scene_k_value": scene.k,
            "transposition_semitones": int(meta["transposition_semitones"]),
            "timing": {
                "midi_onset_seconds": float(target_meta["midi_onset_seconds"]),
                "midi_offset_seconds": float(target_meta["midi_offset_seconds"]),
                "acoustic_offset_seconds": float(target_meta["acoustic_offset_seconds"]),
            },
            "frozen_root": str(self.roots[scene.root_index]),
            "conditioning_contract": "full_local_score_exact_isolated_note_no_dna_no_residual",
        }

    def scene_batch_indices(self) -> list[list[int]]:
        return [list(range(x.first_row, x.first_row + x.k)) for x in self.scenes]

    def io_stats(self) -> dict[str, int]:
        stats = [reader.io_stats() for reader in self.readers]
        return {"pread_calls": sum(x["pread_calls"] for x in stats),
                "pread_bytes": sum(x["pread_bytes"] for x in stats)}

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


__all__ = [
    "CrossInstrumentExactNoteSceneDataset", "CompleteSceneBatchSampler",
    "CompositeSyntheticSceneDataset", "collate_complete_scene_rows",
    "collate_selective_scene_rows",
]
