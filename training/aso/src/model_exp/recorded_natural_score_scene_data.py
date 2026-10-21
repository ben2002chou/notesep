"""Read-only, DNA-free adapter for packed recorded natural-score scenes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from notesep.training_paths import relocated
import random
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .packed_blob_shards import PackedBlobShardReader


SCENE_KIND = "recorded_natural_score_exact_stems_v1"
SCENE_KINDS_BY_FREEZE_SCHEMA = {
    "recorded_natural_score_scene_freeze_v1": SCENE_KIND,
    "recorded_paired_random_timing_freeze_v2": SCENE_KIND,
    "cross_instrument_natural_score_exact_note_freeze_v1":
        "cross_instrument_natural_score_exact_note_scene_v1",
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
            with PackedBlobShardReader(Path(comparison_root).expanduser().resolve()) as comparison:
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
            if freeze.get("schema") not in FREEZE_SCHEMAS or freeze.get("status") != "passed":
                raise ValueError(f"scene root is not frozen-passed: {root}")
            record_kind = SCENE_KINDS_BY_FREEZE_SCHEMA[str(freeze["schema"])]
            manifest = dict(freeze["manifest"])
            if sha256_file(Path(manifest["path"]).resolve()) != str(manifest["sha256"]):
                raise ValueError(f"scene manifest checksum mismatch: {root}")
            if sha256_file(root / "manifest.sqlite") != str(freeze["packed_index"]["sha256"]):
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
                canonical_key = str(metadata.get("paired_parent_scene_id", record.sample_id))
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
                    scene.root_index, scene.scene_id, scene.record_kind,
                    scene.split, scene.k,
                    first_row, scene.samples, scene.canonical_key,
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
        if hashlib.sha256(np.ascontiguousarray(native_mixture).tobytes()).hexdigest() != metadata["mixture_sha256"]:
            raise ValueError(f"native mixture closure checksum mismatch: {scene.scene_id}")
        if scene.samples > native_samples:
            stems = np.pad(stems, ((0, 0), (0, scene.samples - native_samples)))
        self._cached_key, self._cached_stems, self._cached_metadata = key, stems, metadata
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
        source_parent_id = note.get("source_parent_id", metadata.get("source_parent_id"))
        instrument = note.get("recorded_instrument", metadata.get("instrument"))
        score_parent_id = metadata.get("score_parent_id", metadata.get("score_piece_parent_id"))
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
            "note_onset_samples": torch.tensor(round(float(note_onset_seconds) * sample_rate), dtype=torch.long),
            "note_offset_samples": torch.tensor(round(float(note["acoustic_offset_seconds"]) * sample_rate), dtype=torch.long),
            "midi_offset_samples": torch.tensor(round(float(note["midi_offset_seconds"]) * sample_rate), dtype=torch.long),
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
        return [list(range(scene.first_row, scene.first_row + scene.k)) for scene in self.scenes]

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


class CompleteSceneBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: RecordedNaturalScoreSceneDataset, *, shuffle: bool, seed: int = 0) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset.scenes)

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.dataset.scene_batch_indices()
        if self.shuffle:
            random.Random(self.seed).shuffle(batches)
        yield from batches


def collate_complete_scene_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows or len({str(row["scene_id"]) for row in rows}) != 1:
        raise ValueError("a complete-scene batch must contain exactly one scene")
    if len(rows) != int(rows[0]["scene_k"]):
        raise ValueError("complete-scene batch is missing note rows")
    ordered = sorted(rows, key=lambda row: int(row["scene_note_index"]))
    mixture = ordered[0]["mixture"]
    if any(not torch.equal(row["mixture"], mixture) for row in ordered[1:]):
        raise ValueError("per-note rows disagree on full mixture")
    targets = torch.stack([row["target"] for row in ordered])
    if float((targets.sum(dim=0) - mixture).abs().max()) > 1e-6:
        raise ValueError("scene targets do not close exactly to mixture")
    return {
        "mixture": mixture,
        "targets": targets,
        "pitches": torch.stack([row["pitch"] for row in ordered]),
        "note_onset_samples": torch.stack([row["note_onset_samples"] for row in ordered]),
        "note_offset_samples": torch.stack([row["note_offset_samples"] for row in ordered]),
        "valid_samples": ordered[0]["valid_samples"],
        "scene_id": ordered[0]["scene_id"],
        "all_score_events": ordered[0]["all_score_events"],
        "score_parent_id": ordered[0]["score_parent_id"],
        "source_parent_id": ordered[0]["source_parent_id"],
        "instrument": ordered[0]["instrument"],
        "split": ordered[0]["split"],
        "conditioning_contract": ordered[0]["conditioning_contract"],
    }


def collate_selective_scene_rows(
    rows: Sequence[dict[str, Any]],
    *,
    base_collate: Any,
) -> dict[str, Any]:
    """Flatten one complete scene into the frozen selective batch contract."""

    collate_complete_scene_rows(rows)  # structural and exact-closure audit
    from .occurrence_full_score_data import collate_full_score

    return collate_full_score(list(rows), base_collate)


class CompositeSyntheticSceneDataset(Dataset[dict[str, Any]]):
    """Dispatch tagged indices to a synthetic dataset or complete-scene rows."""

    def __init__(self, synthetic: Dataset[Any], scenes: RecordedNaturalScoreSceneDataset) -> None:
        self.synthetic = synthetic
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.synthetic) + len(self.scenes)

    @staticmethod
    def _right_pad(row: dict[str, Any], samples: int) -> dict[str, Any]:
        """Pad waveforms without changing the sampled crop or valid region."""

        output = dict(row)
        waveform_keys = (
            "mixture", "target", "dna", "other_dna", "unassigned_residual",
        )
        for key in waveform_keys:
            value = output[key]
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{key} must be a one-dimensional tensor")
            if len(value) > samples:
                raise ValueError(f"promoted batch length truncates {key}")
            if len(value) < samples:
                output[key] = torch.nn.functional.pad(value, (0, samples - len(value)))
        output["batch_padded_samples"] = torch.tensor(samples, dtype=torch.long)
        return output

    def __getitem__(self, index: Any) -> dict[str, Any]:
        if isinstance(index, tuple) and index and index[0] == "scene":
            row = self.scenes[int(index[1])]
            return self._right_pad(row, int(index[2])) if len(index) >= 3 else row
        if isinstance(index, tuple) and index and index[0] == "synthetic":
            row = self.synthetic[index[1]]
            return self._right_pad(row, int(index[2])) if len(index) >= 3 else row
        raise ValueError("composite dataset requires a tagged sampler index")

    def close(self) -> None:
        close = getattr(self.synthetic, "close", None)
        if callable(close):
            close()
        self.scenes.close()


class CompleteSceneReplacementSampler(Sampler[list[Any]]):
    """Interleave whole recorded-scene optimizer steps with untouched synth.

    On a recorded step, uniformly selected scenes are packed under the exact
    row and audio-sample budgets. On every other step, the original synthetic
    batch is yielded byte/order-identically. Scene selection is uniform before
    K policy, so scenes with more stems do not receive extra probability.
    """

    def __init__(
        self,
        base_sampler: Iterable[list[Any]],
        scenes: RecordedNaturalScoreSceneDataset,
        *,
        recorded_step_fraction: float,
        seed: int,
        start_step: int = 0,
        maximum_rows: int = 192,
        scene_k_policy: str = "k2_only",
        audio_budget_samples: int | None = None,
    ) -> None:
        if not 0.0 < recorded_step_fraction < 1.0:
            raise ValueError("recorded_step_fraction must be in (0,1)")
        if scene_k_policy not in {"k2_only", "k2_k4_80_20"}:
            raise ValueError("scene_k_policy must be k2_only or k2_k4_80_20")
        self.base_sampler = base_sampler
        self.scenes = scenes
        self.recorded_step_fraction = float(recorded_step_fraction)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.maximum_rows = int(maximum_rows)
        self.scene_k_policy = scene_k_policy
        self.audio_budget_samples = (
            None if audio_budget_samples is None else int(audio_budget_samples)
        )
        self._by_k = {
            k: [index for index, scene in enumerate(scenes.scenes) if scene.k == k]
            for k in (2, 4)
        }
        if not self._by_k[2] or (scene_k_policy == "k2_k4_80_20" and not self._by_k[4]):
            raise ValueError("requested scene K policy lacks eligible scenes")

    def __len__(self) -> int:
        return len(self.base_sampler)  # type: ignore[arg-type]

    def _rank(self, key: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{self.seed}:{key}".encode()).digest()[:8], "little")

    def _uniform(self, key: str) -> float:
        return self._rank(key) / float(2**64)

    def __iter__(self) -> Iterator[list[Any]]:
        for step, batch in enumerate(self.base_sampler, start=self.start_step):
            # Low-discrepancy thresholding gives exactly floor/ceil(f*N)
            # recorded steps over every prefix, without Bernoulli variance.
            before = int(step * self.recorded_step_fraction)
            after = int((step + 1) * self.recorded_step_fraction)
            if after == before:
                yield [("synthetic", value) for value in batch]
                continue
            audio_budget = self.audio_budget_samples
            if audio_budget is None:
                audio_budget = sum(
                    int(value[1]) if isinstance(value, tuple) else 0 for value in batch
                )
            if audio_budget <= 0:
                raise ValueError("base batch does not expose its audio-sample budget")
            output: list[Any] = []
            used: set[int] = set()
            cursor = 0
            while len(output) < self.maximum_rows:
                block_k = [2] if self.scene_k_policy == "k2_only" else [2, 2, 2, 2, 4]
                block: list[int] = []
                prospective_used = set(used)
                for block_offset, k in enumerate(block_k):
                    pool = self._by_k[k]
                    ranked = sorted(
                        pool,
                        key=lambda value: self._rank(
                            f"scene:{step}:{cursor + block_offset}:{value}"
                        ),
                    )
                    eligible = [value for value in ranked if value not in prospective_used]
                    if not eligible:
                        block = []
                        break
                    block.append(eligible[0])
                    prospective_used.add(eligible[0])
                if not block:
                    break
                block_rows = sum(self.scenes.scenes[value].k for value in block)
                existing_lengths = [
                    self.scenes.scenes[
                        self.scenes.row_to_scene_note[int(value[1])][0]
                    ].samples
                    for value in output
                ]
                block_lengths = [
                    self.scenes.scenes[value].samples
                    for value in block
                ]
                padded_cost = max(existing_lengths + block_lengths) * (
                    len(output) + block_rows
                )
                if (
                    padded_cost > audio_budget
                    or len(output) + block_rows > self.maximum_rows
                ):
                    break
                used = prospective_used
                for scene_index in block:
                    scene = self.scenes.scenes[scene_index]
                    output.extend(
                        ("scene", row)
                        for row in range(scene.first_row, scene.first_row + scene.k)
                    )
                cursor += len(block)
            if not output:
                minimum_cost = min(
                    scene.k * scene.samples
                    for scene in self.scenes.scenes
                    if scene.k == 2
                )
                raise RuntimeError(
                    "no complete recorded scene fits the optimizer-step budget: "
                    f"audio_budget={audio_budget} maximum_rows={self.maximum_rows} "
                    f"minimum_K2_cost={minimum_cost} base_batch_head={batch[:2]}"
                )
            for scene_index in used:
                scene = self.scenes.scenes[scene_index]
                present = sum(
                    int(value[1]) in range(scene.first_row, scene.first_row + scene.k)
                    for value in output
                )
                if present not in {0, scene.k}:
                    raise RuntimeError("recorded optimizer step contains a partial scene")
            yield output
