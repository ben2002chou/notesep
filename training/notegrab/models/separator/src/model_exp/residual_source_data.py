"""Read complete residual examples for the NoteGrab training recipe."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset

from model_exp.corpus import FrozenV13Contract, sha256_file
from model_exp.dna_residual_corpus import FrozenDNAResidualPriorCatalog

SAMPLE_RATE = 16_000


def target_complete_window(
    metadata: dict[str, Any], total: int
) -> tuple[int, int]:
    onset = int(metadata["onset_sample"])
    offset = int(metadata["offset_sample"])
    start = max(0, onset - int(0.25 * SAMPLE_RATE))
    stop = min(total, offset + int(0.5 * SAMPLE_RATE))
    if not start <= onset < offset <= stop:
        raise ValueError("target note is not completely retained")
    return start, stop


def load_frozen_residual_rows(
    root: Path, *, split: str = "validation"
) -> list[dict[str, Any]]:
    """Load recorded notes in the format used by NoteGrab."""

    rows = []
    with FrozenDNAResidualPriorCatalog(
        root.expanduser().resolve(), split=split
    ) as catalog:
        for record in catalog.iter_records():
            metadata = dict(record.metadata)
            samples = int(metadata["samples"])
            dna, _ = catalog.load_priors(record, include_rt2=False)
            mixture, target = catalog.load_self_contained_raw_slice(
                record, start=0, count=samples
            )
            start_step = int(metadata["start_step"])
            end_step = int(metadata["end_step"])
            step_span = end_step - start_step
            if step_span < 1 or samples % step_span:
                raise ValueError(
                    "recorded residual timeline does not map exactly to samples"
                )
            samples_per_step = samples // step_span
            onset = (
                int(metadata["onset_step"]) - start_step
            ) * samples_per_step
            offset = (
                int(metadata["offset_step"]) - start_step
            ) * samples_per_step
            difficulty = dict(metadata.get("difficulty", {}))
            peak_competitors = int(difficulty.get("peak_competitors", 0))
            staggered_competitors = int(
                difficulty.get("staggered_competitors", 0)
            )
            normalized_metadata = {
                **metadata,
                "mixture_id": str(metadata.get("sample_id", record.asset_id)),
                "score_parent_id": str(metadata["source_parent_id"]),
                "matched_score_family": str(
                    metadata.get("source_instrument", "recorded")
                ),
                "velocity": int(metadata.get("velocity", -1)),
                "onset_sample": onset,
                "offset_sample": offset,
                "target_notes": 1,
                "difficulty": {
                    **difficulty,
                    "peak_competitors": peak_competitors,
                    "staggered_competitors": staggered_competitors,
                },
                "context_notes": peak_competitors + 1,
                "release_context_notes": staggered_competitors,
                "dna_rms": (
                    rms := float(
                        np.sqrt(np.mean(np.square(dna, dtype=np.float64)))
                    )
                ),
            }
            target_rms = float(metadata["target_rms"])
            normalized_metadata["dna_to_target_rms_ratio"] = rms / max(
                target_rms, 1e-12
            )
            start, stop = target_complete_window(normalized_metadata, samples)
            rows.append(
                {
                    "sample_id": record.asset_id,
                    "sound_id": (
                        f"{metadata['source_dataset']}:"
                        f"{metadata.get('source_instrument', 'unknown')}"
                    ),
                    "source_id": str(metadata["source_dataset"]),
                    "score_parent_id": str(metadata["source_parent_id"]),
                    "metadata": normalized_metadata,
                    "mixture": np.ascontiguousarray(mixture[start:stop]),
                    "target": np.ascontiguousarray(target[start:stop]),
                    "dna": np.ascontiguousarray(dna[start:stop]),
                    "valid_samples": stop - start,
                    "pitch": int(metadata["pitch"]),
                }
            )
    if not rows:
        raise ValueError(f"frozen residual root is empty: {root}")
    return rows


class MixedSyntheticRecordedDataset(Dataset[dict[str, Any]]):
    """Mix synthetic examples with recorded notes during training."""

    def __init__(
        self,
        synthetic: Dataset[dict[str, Any]],
        recorded_rows: list[dict[str, Any]],
    ) -> None:
        self.synthetic = synthetic
        self.recorded_rows = recorded_rows
        self.recorded_window_samples = np.asarray(
            [int(row["valid_samples"]) for row in recorded_rows], dtype=np.int64
        )
        if len(recorded_rows) < 1 or np.any(self.recorded_window_samples < 1):
            raise ValueError("Recorded training rows must have positive lengths")

    def __len__(self) -> int:
        return len(self.synthetic)

    def __getitem__(self, index: tuple[Any, ...]) -> dict[str, Any]:
        if not isinstance(index, tuple) or not index:
            raise TypeError("Mixed dataset indices must identify synthetic or recorded")
        source = str(index[0])
        if source == "synthetic" and len(index) == 2:
            return self.synthetic[index[1]]
        if source != "recorded" or len(index) != 3:
            raise ValueError(f"Invalid mixed dataset index: {index!r}")
        row_index, requested_samples = int(index[1]), int(index[2])
        row = self.recorded_rows[row_index]
        valid_samples = int(row["valid_samples"])
        if requested_samples < valid_samples:
            raise ValueError("Recorded complete-note window exceeds its sampled slot")
        metadata = dict(row["metadata"])
        total = int(metadata["samples"])
        start, stop = target_complete_window(metadata, total)
        if stop - start != valid_samples:
            raise ValueError("Recorded row crop no longer matches frozen metadata")
        onset = int(metadata["onset_sample"]) - start
        offset = int(metadata["offset_sample"]) - start
        if not 0 <= onset < offset <= valid_samples:
            raise ValueError("Recorded lifecycle bounds are invalid after cropping")

        arrays = {
            key: np.asarray(row[key], dtype=np.float32)
            for key in ("mixture", "target", "dna")
        }
        if any(
            value.shape != (valid_samples,) or not np.isfinite(value).all()
            for value in arrays.values()
        ):
            raise ValueError("Recorded training waveforms are invalid")
        padded = {
            key: np.pad(value, (0, requested_samples - valid_samples))
            for key, value in arrays.items()
        }
        mixture = torch.from_numpy(padded["mixture"].astype(np.float32, copy=False))
        dna = torch.from_numpy(padded["dna"].astype(np.float32, copy=False))
        return {
            "mixture": mixture,
            "target": torch.from_numpy(padded["target"].astype(np.float32, copy=False)),
            "dna": dna,
            "other_dna": torch.zeros_like(dna),
            "unassigned_residual": mixture - dna,
            "pitch": torch.tensor(int(row["pitch"]), dtype=torch.long),
            "valid_samples": torch.tensor(valid_samples, dtype=torch.long),
            "note_onset_samples": torch.tensor(onset, dtype=torch.long),
            "note_offset_samples": torch.tensor(offset, dtype=torch.long),
            "ordinal": torch.tensor(-(row_index + 1), dtype=torch.long),
            "hard_case": torch.tensor(1, dtype=torch.long),
            "pair_id": str(row["training_asset_id"]),
            "sound_family": str(metadata.get("matched_score_family", "recorded")),
            "source_group": f"recorded_real:{row['source_id']}",
            "stratum": "recorded_real_train",
        }

    def close(self) -> None:
        close = getattr(self.synthetic, "close", None)
        if callable(close):
            close()

class RecordedValidationDataset(Dataset[dict[str, Any]]):
    """Read recorded validation notes without random cropping or padding."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._adapter = MixedSyntheticRecordedDataset(Subset([], []), rows)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._adapter.recorded_rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        samples = int(self._adapter.recorded_window_samples[index])
        row = self._adapter[("recorded", index, samples)]
        row["stratum"] = "recorded_real_validation"
        return row

class DeterministicMixedBatchSampler(Sampler[list[tuple[Any, ...]]]):
    """Replace a deterministic fraction of synthetic slots with fitting real rows."""

    def __init__(
        self,
        base_sampler: Iterable[list[Any]],
        *,
        recorded_window_samples: np.ndarray,
        real_fraction: float,
        fixed_samples: int,
        seed: int,
        start_step: int,
    ) -> None:
        self.base_sampler = base_sampler
        self.recorded_window_samples = np.asarray(
            recorded_window_samples, dtype=np.int64
        )
        self.real_fraction = float(real_fraction)
        self.fixed_samples = int(fixed_samples)
        self.seed = int(seed)
        self.start_step = int(start_step)
        if (
            self.recorded_window_samples.ndim != 1
            or len(self.recorded_window_samples) < 1
            or np.any(self.recorded_window_samples < 1)
        ):
            raise ValueError("Recorded window lengths must be a positive vector")
        if not 0.0 < self.real_fraction <= 1.0:
            raise ValueError("Mixed sampler real fraction must be in (0, 1]")

    def __len__(self) -> int:
        return len(self.base_sampler)  # type: ignore[arg-type]

    def _uniform(self, key: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "little") / float(2**64)

    def __iter__(self) -> Iterable[list[tuple[Any, ...]]]:
        for step, batch in enumerate(self.base_sampler, start=self.start_step):
            exact = self.real_fraction * len(batch)
            real_count = int(math.floor(exact))
            if self._uniform(f"round:{step}") < exact - real_count:
                real_count += 1
            ranked_slots = sorted(
                range(len(batch)), key=lambda slot: self._uniform(f"slot:{step}:{slot}")
            )
            real_slots = set(ranked_slots[:real_count])
            used_real: set[int] = set()
            output: list[tuple[Any, ...]] = []
            for slot, synthetic_index in enumerate(batch):
                if slot not in real_slots:
                    output.append(("synthetic", synthetic_index))
                    continue
                requested = (
                    int(synthetic_index[1])
                    if isinstance(synthetic_index, tuple)
                    else self.fixed_samples
                )
                eligible = np.flatnonzero(
                    self.recorded_window_samples <= requested
                ).tolist()
                eligible = [index for index in eligible if index not in used_real]
                if not eligible:
                    raise ValueError(
                        "No unused recorded complete-note row fits sampled window "
                        f"of {requested} samples"
                    )
                selected = eligible[
                    int(self._uniform(f"row:{step}:{slot}") * len(eligible))
                    % len(eligible)
                ]
                used_real.add(selected)
                output.append(("recorded", selected, requested))
            yield output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_duration_windows(
    path: Path,
    *,
    pairs: int,
    binding: dict[str, Any],
    args: argparse.Namespace,
    acoustic_release_samples: np.ndarray | None = None,
) -> np.ndarray:
    source = path.expanduser().resolve()
    contract_path = (
        source / "CONTRACT.json"
        if source.is_dir()
        else source.with_name("CONTRACT.json")
    )
    index_path = source / "duration_geometry.npy" if source.is_dir() else source
    index_contract = _read_json(contract_path)
    expected = {
        "schema": "literal_residual_v13_duration_geometry_v1",
        "status": "passed",
        "immutable": True,
        "split": "train",
        "corpus_id": str(binding["corpus_id"]),
        "rows": pairs,
        "frozen_manifest_sha256": str(binding["frozen_sha256"]),
        "train_split_manifest_sha256": str(binding["split_manifest_sha256s"]["train"]),
    }
    mismatches = {
        key: (index_contract.get(key), value)
        for key, value in expected.items()
        if index_contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Duration index contract changed: {mismatches}")
    array_contract = index_contract.get("array")
    if not isinstance(array_contract, dict) or sha256_file(
        index_path
    ) != array_contract.get("sha256"):
        raise ValueError("Duration index array hash mismatch")
    payload = np.load(index_path, mmap_mode="r", allow_pickle=False)
    required = {"ordinal", "total_samples", "onset_sample", "offset_sample"}
    names = set(payload.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(f"Duration index lacks {sorted(required - names)}")
    if payload.shape != (pairs,) or not np.array_equal(
        payload["ordinal"], np.arange(pairs, dtype=payload["ordinal"].dtype)
    ):
        raise ValueError("Duration index does not align one-to-one with train ordinals")
    total = np.asarray(payload["total_samples"], dtype=np.int64)
    onset = np.asarray(payload["onset_sample"], dtype=np.int64)
    offset = np.asarray(payload["offset_sample"], dtype=np.int64)
    if any(array.shape != (pairs,) for array in (total, onset, offset)):
        raise ValueError("Duration index does not align one-to-one with train ordinals")
    if (
        np.any(total < 1024)
        or np.any(onset < 0)
        or np.any(offset < onset)
        or np.any(offset > total)
    ):
        raise ValueError("Duration index contains invalid note geometry")
    release = np.full(pairs, int(args.release_samples), dtype=np.int64)
    if acoustic_release_samples is not None:
        if acoustic_release_samples.shape != (pairs,):
            raise ValueError("Acoustic release index does not align with duration index")
        release = np.maximum(release, acoustic_release_samples)
    natural = np.maximum(
        int(args.minimum_window_samples),
        offset - onset + int(args.lead_samples) + release,
    )
    quantum = int(args.window_quantum)
    windows = np.minimum(total, ((natural + quantum - 1) // quantum) * quantum)
    windows = np.maximum(windows, offset - onset).astype(np.int64, copy=False)
    if np.any(windows > int(args.audio_budget_samples)):
        raise ValueError(
            "Audio budget is smaller than at least one complete-note window"
        )
    windows.setflags(write=False)
    return windows

def acoustic_offset_marker_digest(source: Path) -> str:
    markers = sorted(source.glob("acoustic_offsets_*_of_*.json"))
    if not markers:
        raise FileNotFoundError(f"No acoustic-offset markers under {source}")
    digest = hashlib.sha256()
    for marker in markers:
        digest.update(marker.name.encode())
        digest.update(bytes.fromhex(sha256_file(marker)))
    return digest.hexdigest()

def load_acoustic_release_samples(source: Path | None, *, pairs: int) -> np.ndarray | None:
    """Load and strictly validate completed ordinal-aligned acoustic releases."""

    if source is None:
        return None
    root = source.expanduser().resolve()
    markers = sorted(root.glob("acoustic_offsets_*_of_*.json"))
    if not markers:
        raise FileNotFoundError(f"No acoustic-offset markers under {root}")
    result = np.full(pairs, -1, dtype=np.int32)
    expected_shards: int | None = None
    for marker_path in markers:
        marker = _read_json(marker_path)
        if marker.get("schema") != "synthetic_acoustic_offset_sidecar_shard_v1":
            raise ValueError(f"Wrong acoustic-offset schema: {marker_path}")
        shard_id = int(marker["shard_id"])
        num_shards = int(marker["num_shards"])
        if expected_shards is None:
            expected_shards = num_shards
        if num_shards != expected_shards:
            raise ValueError("Acoustic-offset shard counts disagree")
        array_path = root / str(marker["array"])
        payload = np.load(array_path, mmap_mode="r", allow_pickle=False)
        begin, end = int(marker["begin"]), int(marker["end"])
        if shard_id < 0 or end - begin != len(payload) or int(marker["count"]) != len(payload):
            raise ValueError(f"Invalid acoustic-offset shard geometry: {marker_path}")
        expected = np.arange(begin, end, dtype=payload["ordinal"].dtype)
        if not np.array_equal(payload["ordinal"], expected):
            raise ValueError(f"Acoustic-offset ordinals changed: {array_path}")
        release = np.asarray(payload["release_samples"], dtype=np.int64)
        if np.any(release < 0) or np.any(release > np.iinfo(np.int32).max):
            raise ValueError(f"Invalid acoustic releases: {array_path}")
        result[begin:end] = release.astype(np.int32)
    if expected_shards != len(markers) or np.any(result < 0):
        raise ValueError("Acoustic-offset index is incomplete")
    result.setflags(write=False)
    return result


def load_synthetic_ordinal_view(
    path: Path | None,
    *,
    contract: FrozenV13Contract,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    """Load the training and validation row selections."""

    if path is None:
        return None, None, None
    source = path.expanduser().resolve()
    raw = _read_json(source)
    if raw.get("schema") != "occurrence_synthetic_ordinal_view_v1":
        raise ValueError("Synthetic ordinal view has the wrong schema")
    expected_binding = contract.checkpoint_binding()
    if raw.get("frozen_sha256") != expected_binding["frozen_sha256"]:
        raise ValueError("Synthetic ordinal view is bound to another freeze")

    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        values = raw.get(f"{split}_ordinals")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Synthetic ordinal view has no {split} ordinals")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"Synthetic ordinal view {split} ordinals are not integers")
        array = np.asarray(values, dtype=np.int64)
        if len(np.unique(array)) != len(array):
            raise ValueError(f"Synthetic ordinal view repeats {split} ordinals")
        if int(array.min()) < 0 or int(array.max()) >= int(
            contract.split_pairs["train"]
        ):
            raise ValueError(f"Synthetic ordinal view {split} ordinal is out of range")
        arrays[split] = array
    overlap = np.intersect1d(arrays["train"], arrays["validation"])
    if len(overlap):
        raise ValueError("Synthetic ordinal view train/validation rows overlap")
    if int(raw.get("mixture_overlap", -1)) != 0:
        raise ValueError("Synthetic ordinal view does not certify mixture disjointness")
    declared = raw.get("counts")
    observed = {key: int(len(value)) for key, value in arrays.items()}
    if declared != observed:
        raise ValueError(
            f"Synthetic ordinal view counts changed: {declared!r} != {observed!r}"
        )
    binding = {
        "schema": str(raw["schema"]),
        "path": str(source),
        "sha256": sha256_file(source),
        "counts": observed,
        "mixture_overlap": 0,
    }
    return arrays["train"], arrays["validation"], binding


class SyntheticOrdinalViewDataset(Dataset[dict[str, Any]]):
    """Map local sampler indices, including dynamic-window tuples, to ordinals."""

    def __init__(
        self,
        dataset: Dataset,
        ordinals: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.ordinals = np.asarray(ordinals, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.ordinals))

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            local, requested_samples = map(int, index)
            return self.dataset[(int(self.ordinals[local]), requested_samples)]
        return self.dataset[int(self.ordinals[int(index)])]

    def close(self) -> None:
        self.dataset.close()
