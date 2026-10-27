"""Read complete residual examples for the NoteGrab training recipe."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from model_exp.corpus import FrozenV13Contract, sha256_file











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
