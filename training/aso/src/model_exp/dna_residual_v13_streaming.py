"""Fail-closed, lazy access to the frozen V13 musical residual corpus."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .packed_blob_shards import PackedBlobShardReader


FREEZE_SCHEMA = "literal_residual_musical_production_freeze_v1"
SPLIT_INDEX_SCHEMA = "literal_residual_v13_split_index_v2"
IDENTITY_SCHEMA = "literal_residual_v13_identity_manifest_v1"
EXPECTED_PAIRS = 2_000_000
MODEL_CONTRACT = {
    "n_fft": 512,
    "hop_length": 128,
    "channels": 96,
    "blocks": 8,
    "pitch_embed_dim": 16,
    "use_rt2_prior": False,
    "use_region_context": True,
    "maximum_correction_scale": 2.0,
    "output_mode": "ungated_residual",
    "soft_replace_gate_bias": -12.0,
    "sample_rate": 16_000,
    "use_harmonic_features": False,
    "use_frequency_dilated_blocks": False,
}
def _environment_loss_weight(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


FREE_TARGET_LOSS_WEIGHTS = {
    "complex_weight": _environment_loss_weight("DNA_V13_COMPLEX_WEIGHT", 1.0),
    "complex_error_cap": _environment_loss_weight("DNA_V13_COMPLEX_CAP", 0.0),
    "log_magnitude_weight": _environment_loss_weight("DNA_V13_LOGMAG_WEIGHT", 0.5),
    "waveform_weight": _environment_loss_weight("DNA_V13_WAVEFORM_WEIGHT", 0.2),
    "envelope_weight": _environment_loss_weight("DNA_V13_ENVELOPE_WEIGHT", 0.2),
    "phase_coherence_weight": _environment_loss_weight("DNA_V13_PHASE_WEIGHT", 0.0),
    "do_no_harm_weight": 0.0,
    "correction_weight": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _bound_path(root: Path, value: str | Path) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Frozen binding escapes corpus root: {path}")
    return path


def _descriptor(
    root: Path, value: object, *, label: str
) -> tuple[Path, str, int | None]:
    if not isinstance(value, dict):
        raise ValueError(f"FROZEN.json lacks a bound {label} descriptor")
    try:
        path = _bound_path(root, str(value["path"]))
        expected_hash = str(value["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label} descriptor") from error
    if len(expected_hash) != 64 or sha256_file(path) != expected_hash:
        raise ValueError(f"Frozen {label} hash mismatch: {path}")
    count = value.get("pairs")
    return path, expected_hash, None if count is None else int(count)


def _sqlite_metadata(path: Path) -> dict[str, Any]:
    uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            return {
                str(key): json.loads(value)
                for key, value in connection.execute(
                    "SELECT key, value_json FROM metadata ORDER BY key"
                )
            }
    except (sqlite3.DatabaseError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid frozen split index: {path}") from error


@dataclass(frozen=True)
class FrozenV13Contract:
    frozen_path: Path
    frozen_sha256: str
    root: Path
    corpus_id: str
    identity_path: Path
    identity_file_sha256: str
    identity_sha256: str
    packed_roots_manifest: Path
    split_paths: Mapping[str, Path]
    split_sha256s: Mapping[str, str]
    split_pairs: Mapping[str, int]
    split_family_counts: Mapping[str, Mapping[str, int]]
    packed_roots: frozenset[Path]

    @classmethod
    def load(cls, frozen_path: str | Path) -> "FrozenV13Contract":
        path = Path(frozen_path).expanduser().resolve()
        if path.name != "FROZEN.json":
            raise ValueError("--frozen-manifest must explicitly name FROZEN.json")
        freeze = _load_object(path)
        if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != "passed":
            raise ValueError(f"V13 corpus is not frozen and passed: {path}")
        if int(freeze.get("pairs", -1)) != EXPECTED_PAIRS:
            raise ValueError(
                f"Production V13 requires exactly {EXPECTED_PAIRS} pairs; "
                f"freeze declares {freeze.get('pairs')!r}"
            )
        if not (
            bool(freeze.get("complete_dna_context_sum", False))
            and bool(freeze.get("complete_dna_context_residual", False))
        ):
            raise ValueError("V13 freeze lacks complete region-context assets")
        root = path.parent
        corpus_id = str(freeze.get("corpus_id", ""))
        if not corpus_id:
            raise ValueError("V13 freeze lacks corpus_id")

        packed_manifest = _bound_path(root, str(freeze.get("packed_roots_manifest", "")))
        packed_hash = str(freeze.get("packed_roots_manifest_sha256", ""))
        if not packed_manifest.is_file() or sha256_file(packed_manifest) != packed_hash:
            raise ValueError("V13 packed-roots manifest is absent or changed")
        packed_roots: set[Path] = set()
        for line in packed_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            packed_roots.add(_bound_path(root, str(row["packed_root"])))
        if not packed_roots:
            raise ValueError("V13 packed-roots manifest is empty")

        identity_path, identity_file_hash, _ = _descriptor(
            root, freeze.get("identity_manifest"), label="identity manifest"
        )
        identities = _load_object(identity_path)
        if (
            identities.get("schema") != IDENTITY_SCHEMA
            or identities.get("status") != "passed"
        ):
            raise ValueError("V13 identity manifest is not passed")
        identity_sha = str(identities.get("identity_sha256", ""))
        if len(identity_sha) != 64:
            raise ValueError("V13 identity manifest lacks identity_sha256")

        bindings = freeze.get("split_manifests")
        if not isinstance(bindings, dict):
            raise ValueError("FROZEN.json lacks split_manifests")
        split_paths: dict[str, Path] = {}
        split_hashes: dict[str, str] = {}
        split_pairs: dict[str, int] = {}
        split_family_counts: dict[str, dict[str, int]] = {}
        for split in ("train",):
            split_path, split_hash, declared_pairs = _descriptor(
                root, bindings.get(split), label=f"{split} split manifest"
            )
            metadata = _sqlite_metadata(split_path)
            expected = {
                "schema": SPLIT_INDEX_SCHEMA,
                "status": "passed",
                "split": split,
                "corpus_id": corpus_id,
                "identity_sha256": identity_sha,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Invalid {split} split metadata: {mismatches}")
            count = int(metadata.get("pairs", -1))
            if count < 1 or (declared_pairs is not None and count != declared_pairs):
                raise ValueError(f"Invalid {split} split pair count")
            uri = f"file:{quote(str(split_path))}?mode=ro&immutable=1"
            try:
                with sqlite3.connect(uri, uri=True) as connection:
                    columns = {
                        str(row[1]) for row in connection.execute("PRAGMA table_info(pairs)")
                    }
                    required = {
                        "ordinal",
                        "pair_id",
                        "mixture_id",
                        "packed_root",
                        "sound_family",
                        "source_group",
                        "hard_case",
                        "stratum",
                    }
                    if not required.issubset(columns):
                        raise ValueError(
                            f"{split} pair index lacks columns {sorted(required - columns)}"
                        )
                    observed = connection.execute(
                        "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM pairs"
                    ).fetchone()
                    observed_family_counts = {
                        str(family): int(family_count)
                        for family, family_count in connection.execute(
                            "SELECT sound_family, COUNT(*) FROM pairs "
                            "GROUP BY sound_family ORDER BY sound_family"
                        )
                    }
                    observed_source_group_counts = {
                        str(group): int(group_count)
                        for group, group_count in connection.execute(
                            "SELECT source_group, COUNT(*) FROM pairs "
                            "GROUP BY source_group ORDER BY source_group"
                        )
                    }
                    observed_stratum_counts = {
                        str(stratum): int(stratum_count)
                        for stratum, stratum_count in connection.execute(
                            "SELECT stratum, COUNT(*) FROM pairs "
                            "GROUP BY stratum ORDER BY stratum"
                        )
                    }
                    observed_hard_case_counts = {
                        str(int(hard_case)): int(hard_count)
                        for hard_case, hard_count in connection.execute(
                            "SELECT hard_case, COUNT(*) FROM pairs "
                            "GROUP BY hard_case ORDER BY hard_case"
                        )
                    }
            except sqlite3.DatabaseError as error:
                raise ValueError(f"Invalid {split} pair table") from error
            if int(observed[0]) != count or int(observed[1]) != 0 or int(observed[2]) != count - 1:
                raise ValueError(f"{split} pair ordinals are not complete and contiguous")
            declared_family_counts = metadata.get("family_counts")
            if not isinstance(declared_family_counts, dict):
                raise ValueError(f"{split} split metadata lacks family_counts")
            family_counts = {
                str(family): int(family_count)
                for family, family_count in declared_family_counts.items()
            }
            if observed_family_counts != family_counts or sum(family_counts.values()) != count:
                raise ValueError(f"{split} split family counts are inconsistent")
            for label, metadata_key, observed_counts in (
                ("source-group", "source_group_counts", observed_source_group_counts),
                ("stratum", "stratum_counts", observed_stratum_counts),
                ("hard-case", "hard_case_counts", observed_hard_case_counts),
            ):
                declared_counts = metadata.get(metadata_key)
                if not isinstance(declared_counts, dict):
                    raise ValueError(f"{split} split metadata lacks {metadata_key}")
                normalized_counts = {
                    str(key): int(value) for key, value in declared_counts.items()
                }
                if normalized_counts != observed_counts or sum(normalized_counts.values()) != count:
                    raise ValueError(f"{split} split {label} counts are inconsistent")
            split_paths[split] = split_path
            split_hashes[split] = split_hash
            split_pairs[split] = count
            split_family_counts[split] = family_counts
        if split_pairs.get("train") != EXPECTED_PAIRS:
            raise ValueError("Training split index does not cover exactly 2M pairs")
        return cls(
            frozen_path=path,
            frozen_sha256=sha256_file(path),
            root=root,
            corpus_id=corpus_id,
            identity_path=identity_path,
            identity_file_sha256=identity_file_hash,
            identity_sha256=identity_sha,
            packed_roots_manifest=packed_manifest,
            split_paths=split_paths,
            split_sha256s=split_hashes,
            split_pairs=split_pairs,
            split_family_counts=split_family_counts,
            packed_roots=frozenset(packed_roots),
        )

    def checkpoint_binding(self) -> dict[str, Any]:
        return {
            "frozen_manifest": str(self.frozen_path),
            "frozen_sha256": self.frozen_sha256,
            "corpus_id": self.corpus_id,
            "identity_manifest": str(self.identity_path),
            "identity_manifest_sha256": self.identity_file_sha256,
            "identity_sha256": self.identity_sha256,
            "split_manifest_sha256s": dict(self.split_sha256s),
            "split_pairs": dict(self.split_pairs),
        }

    def load_family_ordinals(self, split: str = "train") -> dict[str, np.ndarray]:
        """Load the opt-in family lookup using one bounded int32 array per family."""
        if split not in self.split_paths:
            raise ValueError(f"Unsupported V13 split: {split}")
        path = self.split_paths[split]
        uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
        output: dict[str, np.ndarray] = {}
        with sqlite3.connect(uri, uri=True) as connection:
            for family, expected in sorted(self.split_family_counts[split].items()):
                values = np.fromiter(
                    (
                        int(row[0])
                        for row in connection.execute(
                            "SELECT ordinal FROM pairs WHERE sound_family = ? "
                            "ORDER BY ordinal",
                            (family,),
                        )
                    ),
                    dtype=np.int32,
                    count=int(expected),
                )
                if len(values) != int(expected):
                    raise ValueError(f"Family ordinal count changed for {family!r}")
                values.setflags(write=False)
                output[family] = values
        if sum(map(len, output.values())) != self.split_pairs[split]:
            raise ValueError(f"{split} family ordinals do not cover the split")
        return output


class V13StreamingResidualDataset(Dataset[dict[str, Any]]):
    """Map-style split index with bounded, worker-local packed readers."""

    def __init__(
        self,
        contract: FrozenV13Contract,
        *,
        split: str,
        fixed_samples: int = 61_440,
        reader_cache_size: int = 8,
        preserve_complete_note: bool = False,
        lead_samples: int = 0,
        release_samples: int = 0,
        acoustic_release_samples: np.ndarray | None = None,
        minimum_window_samples: int = 1024,
        window_quantum: int = 1,
    ) -> None:
        if split not in contract.split_paths:
            raise ValueError(f"Unsupported V13 split: {split}")
        self.contract = contract
        self.split = split
        self.fixed_samples = int(fixed_samples)
        self.reader_cache_size = int(reader_cache_size)
        self.preserve_complete_note = bool(preserve_complete_note)
        self.lead_samples = int(lead_samples)
        self.release_samples = int(release_samples)
        self.acoustic_release_samples = acoustic_release_samples
        self.minimum_window_samples = int(minimum_window_samples)
        self.window_quantum = int(window_quantum)
        if self.fixed_samples < 0 or self.reader_cache_size < 1:
            raise ValueError("Invalid streaming dataset limits")
        if self.fixed_samples and self.fixed_samples < 1024:
            raise ValueError("Fixed windows must contain at least 1024 samples")
        if min(self.lead_samples, self.release_samples) < 0:
            raise ValueError("Note context cannot be negative")
        if self.minimum_window_samples < 1024 or self.window_quantum < 1:
            raise ValueError("Invalid variable-window geometry")
        if self.acoustic_release_samples is not None:
            values = np.asarray(self.acoustic_release_samples)
            if values.shape != (len(self),) or np.any(values < 0):
                raise ValueError("Acoustic release index does not align with the split")
        self._pid = os.getpid()
        self._connection: sqlite3.Connection | None = None
        self._readers: OrderedDict[Path, PackedBlobShardReader] = OrderedDict()

    def __len__(self) -> int:
        return int(self.contract.split_pairs[self.split])

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_readers"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

    def _reset_after_fork(self) -> None:
        self.close()
        self._pid = os.getpid()

    def _db(self) -> sqlite3.Connection:
        if self._pid != os.getpid():
            self._reset_after_fork()
        if self._connection is None:
            path = self.contract.split_paths[self.split]
            uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def _reader(self, root: Path) -> PackedBlobShardReader:
        if self._pid != os.getpid():
            self._reset_after_fork()
        reader = self._readers.get(root)
        if reader is not None:
            self._readers.move_to_end(root)
            return reader
        reader = PackedBlobShardReader(root, require_complete=True)
        self._readers[root] = reader
        while len(self._readers) > self.reader_cache_size:
            _, old = self._readers.popitem(last=False)
            old.close()
        return reader

    @staticmethod
    def _crop_start(metadata: Mapping[str, Any], total: int, fixed: int) -> int:
        if total <= fixed:
            return 0
        onset = int(metadata.get("onset_sample", -1))
        offset = int(metadata.get("offset_sample", -1))
        if not 0 <= onset <= offset <= total:
            raise ValueError("Target note has invalid bounds")
        if offset - onset > fixed:
            choices = (onset, (onset + offset - fixed) // 2, offset - fixed)
            digest = hashlib.sha256(str(metadata.get("pair_id", "")).encode()).digest()
            return max(0, min(choices[digest[0] % len(choices)], total - fixed))
        minimum = max(0, offset - fixed)
        maximum = min(onset, total - fixed)
        if minimum > maximum:
            raise ValueError("No fixed window contains the complete target note")
        digest = hashlib.sha256(str(metadata.get("pair_id", "")).encode()).digest()
        return minimum + int.from_bytes(digest[:8], "little") % (maximum - minimum + 1)

    def _variable_window(
        self,
        metadata: Mapping[str, Any],
        total: int,
        requested_samples: int = 0,
        acoustic_release_samples: int | None = None,
    ) -> tuple[int, int]:
        onset = int(metadata.get("onset_sample", -1))
        offset = int(metadata.get("offset_sample", -1))
        if not 0 <= onset <= offset <= total:
            raise ValueError("Target note has invalid bounds")
        release = (
            self.release_samples
            if acoustic_release_samples is None
            else max(self.release_samples, int(acoustic_release_samples))
        )
        natural = max(
            self.minimum_window_samples,
            offset - onset + self.lead_samples + release,
        )
        desired = max(natural, int(requested_samples))
        desired = min(total, int(math.ceil(desired / self.window_quantum)) * self.window_quantum)
        desired = max(offset - onset, desired)
        minimum = max(0, offset + release - desired)
        maximum = min(max(0, onset - self.lead_samples), total - desired)
        if minimum > maximum:
            minimum = max(0, min(onset, total - desired))
            maximum = min(max(minimum, offset - desired), total - desired)
        start = max(0, min((minimum + maximum) // 2, total - desired))
        if not start <= onset <= offset <= start + desired:
            raise ValueError("Variable window truncated the target note")
        return start, desired

    def __getitem__(self, ordinal: int | tuple[int, int]) -> dict[str, Any]:
        requested_samples = 0
        if isinstance(ordinal, tuple):
            ordinal, requested_samples = map(int, ordinal)
        ordinal = int(ordinal)
        if not 0 <= ordinal < len(self):
            raise IndexError(ordinal)
        acoustic_release = (
            None
            if self.acoustic_release_samples is None
            else int(self.acoustic_release_samples[ordinal])
        )
        row = self._db().execute(
            """
            SELECT pair_id, mixture_id, packed_root, sound_family, source_group,
                   hard_case, stratum
            FROM pairs
            WHERE ordinal = ?
            """,
            (ordinal,),
        ).fetchone()
        if row is None:
            raise IndexError(ordinal)
        pair_id, mixture_id = str(row[0]), str(row[1])
        packed_root = _bound_path(self.contract.root, str(row[2]))
        if packed_root not in self.contract.packed_roots:
            raise ValueError(f"Split index names an unfrozen packed root: {packed_root}")
        reader = self._reader(packed_root)
        target_record = reader.record(pair_id, "target_f32")
        dna_record = reader.record(pair_id, "dna_f32")
        mixture_record = reader.record(mixture_id, "mixture_f32")
        context_sum_record = reader.record(mixture_id, "dna_context_sum_f32")
        context_residual_record = reader.record(
            mixture_id, "dna_context_residual_f32"
        )
        metadata = target_record.metadata
        if dna_record.metadata != metadata:
            raise ValueError(f"Target/DNA metadata mismatch: {pair_id}")
        if str(metadata.get("pair_id", "")) != pair_id:
            raise ValueError(f"Pair identity mismatch: {pair_id}")
        if str(metadata.get("mixture_id", "")) != mixture_id:
            raise ValueError(f"Mixture identity mismatch: {pair_id}")
        if not bool(metadata.get("complete_dna_context_sum", False)):
            raise ValueError(f"Incomplete DNA region context: {pair_id}")
        mixture_metadata = mixture_record.metadata
        if not (
            context_sum_record.metadata == mixture_metadata
            and context_residual_record.metadata == mixture_metadata
            and bool(mixture_metadata.get("complete_dna_context_sum", False))
            and bool(mixture_metadata.get("complete_dna_context_residual", False))
        ):
            raise ValueError(f"Incomplete DNA mixture context: {mixture_id}")
        sample_rate = int(metadata.get("sample_rate", 0))
        total = int(metadata.get("samples", -1))
        if sample_rate <= 0 or total <= 0:
            raise ValueError(f"Invalid V13 sample geometry: {pair_id}")
        expected_bytes = total * np.dtype("<f4").itemsize
        if target_record.length_bytes != expected_bytes or dna_record.length_bytes != expected_bytes:
            raise ValueError(f"V13 pair length mismatch: {pair_id}")
        if requested_samples:
            note_samples = int(metadata.get("offset_sample", -1)) - int(
                metadata.get("onset_sample", -1)
            )
            if requested_samples < note_samples:
                raise ValueError("Sampler-requested window would truncate the target note")
            start, window_samples = self._variable_window(
                metadata,
                total,
                requested_samples=requested_samples,
                acoustic_release_samples=acoustic_release,
            )
            if window_samples > requested_samples:
                raise ValueError("Duration index understates the required note window")
        elif self.preserve_complete_note or self.fixed_samples == 0:
            start, window_samples = self._variable_window(
                metadata, total, acoustic_release_samples=acoustic_release
            )
        else:
            start = self._crop_start(metadata, total, self.fixed_samples)
            window_samples = self.fixed_samples
        count = min(total - start, window_samples)
        arrays = {
            "mixture": reader.read_array_slice(
                mixture_id, "mixture_f32", dtype="<f4", start=start, count=count
            ),
            "target": reader.read_array_slice(
                pair_id, "target_f32", dtype="<f4", start=start, count=count
            ),
            "dna": reader.read_array_slice(
                pair_id, "dna_f32", dtype="<f4", start=start, count=count
            ),
            "context_sum": reader.read_array_slice(
                mixture_id, "dna_context_sum_f32", dtype="<f4", start=start, count=count
            ),
            "unassigned_residual": reader.read_array_slice(
                mixture_id,
                "dna_context_residual_f32",
                dtype="<f4",
                start=start,
                count=count,
            ),
        }
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Non-finite V13 waveform: {pair_id}")
        padded: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            destination = np.zeros(window_samples, dtype=np.float32)
            destination[:count] = value
            padded[key] = destination
        other_dna = padded.pop("context_sum") - padded["dna"]
        pitch = int(metadata.get("pitch", -1))
        if not 0 <= pitch < 128:
            raise ValueError(f"Invalid target pitch: {pair_id}")
        output = {key: torch.from_numpy(value) for key, value in padded.items()}
        output["other_dna"] = torch.from_numpy(np.ascontiguousarray(other_dna))
        output.update(
            pitch=torch.tensor(pitch, dtype=torch.long),
            valid_samples=torch.tensor(count, dtype=torch.long),
            note_onset_samples=torch.tensor(
                max(0, min(count, int(metadata["onset_sample"]) - start)),
                dtype=torch.long,
            ),
            note_offset_samples=torch.tensor(
                max(0, min(count, int(metadata["offset_sample"]) - start)),
                dtype=torch.long,
            ),
            ordinal=torch.tensor(ordinal, dtype=torch.long),
            pair_id=pair_id,
            mixture_id=mixture_id,
            sound_family=str(row[3]),
            source_group=str(row[4]),
            hard_case=torch.tensor(int(row[5]), dtype=torch.long),
            stratum=str(row[6]),
        )
        return output

    def cache_info(self) -> dict[str, int]:
        return {"packed_readers": len(self._readers)}

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


class ResumeSafeBatchSampler(Sampler[list[int]]):
    """Resume-safe uniform or opt-in family-temperature batch sampling."""

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        start_step: int,
        stop_step: int,
        seed: int,
        family_temperature_alpha: float = 1.0,
        family_ordinals: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        self.family_temperature_alpha = float(family_temperature_alpha)
        if min(self.dataset_size, self.batch_size) < 1:
            raise ValueError("Dataset and batch sizes must be positive")
        if not 0 <= self.start_step <= self.stop_step:
            raise ValueError("Invalid resume step interval")
        if not 0.0 <= self.family_temperature_alpha <= 1.0:
            raise ValueError("family temperature alpha must be in [0, 1]")
        self._family_ordinals: tuple[np.ndarray, ...] = ()
        self._family_names: tuple[str, ...] = ()
        self._family_quotas: tuple[int, ...] = ()
        self._family_cumulative: tuple[int, ...] = ()
        if self.family_temperature_alpha != 1.0:
            if not family_ordinals:
                raise ValueError("Family ordinals are required when alpha != 1.0")
            normalized = {
                str(family): np.asarray(ordinals, dtype=np.int32)
                for family, ordinals in family_ordinals.items()
            }
            if any(len(ordinals) < 1 for ordinals in normalized.values()):
                raise ValueError("Family ordinal arrays must be non-empty")
            seen = np.zeros(self.dataset_size, dtype=np.bool_)
            observed = 0
            for ordinals in normalized.values():
                if (
                    int(ordinals[0]) < 0
                    or int(ordinals[-1]) >= self.dataset_size
                    or np.any(ordinals[1:] <= ordinals[:-1])
                    or np.any(seen[ordinals])
                ):
                    raise ValueError("Family ordinals must partition the dataset exactly")
                seen[ordinals] = True
                observed += len(ordinals)
            if observed != self.dataset_size or not bool(np.all(seen)):
                raise ValueError("Family ordinals must partition the dataset exactly")
            self._family_names = tuple(sorted(normalized))
            self._family_ordinals = tuple(
                normalized[family] for family in self._family_names
            )
            self._family_quotas = self._temperature_quotas(
                tuple(map(len, self._family_ordinals)), self.family_temperature_alpha
            )
            cumulative = np.cumsum(self._family_quotas, dtype=np.int64)
            self._family_cumulative = tuple(map(int, cumulative))

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _permutation(self, epoch: int) -> tuple[int, int]:
        digest = hashlib.sha256(f"{self.seed}:{epoch}".encode()).digest()
        multiplier = max(1, int.from_bytes(digest[:8], "little") % self.dataset_size)
        while math.gcd(multiplier, self.dataset_size) != 1:
            multiplier = (multiplier + 1) % self.dataset_size or 1
        offset = int.from_bytes(digest[8:16], "little") % self.dataset_size
        return multiplier, offset

    def _keyed_permutation(self, size: int, key: str) -> tuple[int, int]:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        multiplier = max(1, int.from_bytes(digest[:8], "little") % size)
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size or 1
        offset = int.from_bytes(digest[8:16], "little") % size
        return multiplier, offset

    @staticmethod
    def _temperature_quotas(counts: tuple[int, ...], alpha: float) -> tuple[int, ...]:
        total = sum(counts)
        weights = np.power(np.asarray(counts, dtype=np.float64), alpha)
        exact = weights * (total / float(weights.sum()))
        quotas = np.floor(exact).astype(np.int64)
        remainder = total - int(quotas.sum())
        order = sorted(range(len(counts)), key=lambda index: (-(exact[index] - quotas[index]), index))
        for index in order[:remainder]:
            quotas[index] += 1
        if int(quotas.sum()) != total or np.any(quotas < 1):
            raise ValueError("Invalid family-temperature quotas")
        return tuple(map(int, quotas))

    def _family_sample(self, cursor: int) -> int:
        epoch, position = divmod(cursor, self.dataset_size)
        multiplier, offset = self._keyed_permutation(
            self.dataset_size, f"family-slots:{epoch}"
        )
        slot = (multiplier * position + offset) % self.dataset_size
        family_index = bisect_right(self._family_cumulative, slot)
        previous = 0 if family_index == 0 else self._family_cumulative[family_index - 1]
        local_slot = slot - previous
        ordinals = self._family_ordinals[family_index]
        cycle, local_position = divmod(local_slot, len(ordinals))
        inner_multiplier, inner_offset = self._keyed_permutation(
            len(ordinals),
            f"family:{self._family_names[family_index]}:{epoch}:{cycle}",
        )
        return int(ordinals[(inner_multiplier * local_position + inner_offset) % len(ordinals)])

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            batch = []
            for cursor in range(step * self.batch_size, (step + 1) * self.batch_size):
                if self.family_temperature_alpha == 1.0:
                    epoch, position = divmod(cursor, self.dataset_size)
                    multiplier, offset = self._permutation(epoch)
                    batch.append((multiplier * position + offset) % self.dataset_size)
                else:
                    batch.append(self._family_sample(cursor))
            yield batch


class ResumeSafeAudioBudgetBatchSampler(Sampler[list[tuple[int, int]]]):
    """Stateless duration-bucket sampling with a fixed padded-audio budget."""

    def __init__(
        self,
        window_samples: Sequence[int] | np.ndarray,
        *,
        audio_budget_samples: int,
        maximum_batch_size: int,
        start_step: int,
        stop_step: int,
        seed: int,
        family_temperature_alpha: float = 1.0,
        family_ordinals: Mapping[str, np.ndarray] | None = None,
        priority_ordinals: Sequence[int] | np.ndarray | None = None,
        priority_fraction: float = 0.0,
        priority_family_temperature_alpha: float | None = None,
    ) -> None:
        windows = np.asarray(window_samples, dtype=np.int64)
        if windows.ndim != 1 or len(windows) < 1 or np.any(windows < 1024):
            raise ValueError("window_samples must be a nonempty vector of valid lengths")
        self.window_samples = windows
        self.audio_budget_samples = int(audio_budget_samples)
        self.maximum_batch_size = int(maximum_batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        self.family_temperature_alpha = float(family_temperature_alpha)
        self.priority_fraction = float(priority_fraction)
        self.priority_family_temperature_alpha = (
            self.family_temperature_alpha
            if priority_family_temperature_alpha is None
            else float(priority_family_temperature_alpha)
        )
        if min(self.audio_budget_samples, self.maximum_batch_size) < 1:
            raise ValueError("Dynamic batch limits must be positive")
        if not 0 <= self.start_step <= self.stop_step:
            raise ValueError("Invalid resume step interval")
        if not 0.0 <= self.family_temperature_alpha <= 1.0:
            raise ValueError("family temperature alpha must be in [0, 1]")
        if not 0.0 <= self.priority_fraction <= 1.0:
            raise ValueError("priority fraction must be in [0, 1]")
        if not 0.0 <= self.priority_family_temperature_alpha <= 1.0:
            raise ValueError("priority family temperature alpha must be in [0, 1]")

        priority = np.zeros(len(windows), dtype=np.bool_)
        if priority_ordinals is not None:
            values = np.asarray(priority_ordinals, dtype=np.int64)
            if values.ndim != 1 or len(values) < 1:
                raise ValueError("priority ordinals must be a nonempty vector")
            if int(values.min()) < 0 or int(values.max()) >= len(windows):
                raise ValueError("priority ordinal is out of range")
            if len(np.unique(values)) != len(values):
                raise ValueError("priority ordinals must be unique")
            priority[values] = True
        if self.priority_fraction > 0.0 and not bool(priority.any()):
            raise ValueError("positive priority fraction requires priority ordinals")
        if self.priority_fraction < 1.0 and bool(priority.all()):
            raise ValueError("priority fraction below one requires non-priority rows")
        if self.priority_fraction == 0.0:
            priority[:] = False

        if family_ordinals is None:
            family_ordinals = {"all": np.arange(len(windows), dtype=np.int32)}
        covered = np.zeros(len(windows), dtype=np.bool_)
        cells: dict[int, dict[bool, list[tuple[str, np.ndarray]]]] = {}
        for family, ordinals_value in sorted(family_ordinals.items()):
            ordinals = np.asarray(ordinals_value, dtype=np.int32)
            if len(ordinals) < 1 or np.any(covered[ordinals]):
                raise ValueError("Family ordinals must partition dynamic windows")
            covered[ordinals] = True
            family_windows = windows[ordinals]
            family_priority = priority[ordinals]
            for is_priority in np.unique(family_priority):
                group = ordinals[family_priority == is_priority]
                group_windows = windows[group]
                for bucket in np.unique(group_windows):
                    selected = group[group_windows == bucket]
                    cells.setdefault(int(bucket), {}).setdefault(
                        bool(is_priority), []
                    ).append((str(family), selected))
        if not bool(np.all(covered)):
            raise ValueError("Family ordinals must cover dynamic windows")
        group_fractions = {
            False: 1.0 - self.priority_fraction,
            True: self.priority_fraction,
        }
        self._buckets = tuple(sorted(cells))
        if not self._buckets:
            raise ValueError("priority policy leaves no sampleable rows")
        self._cells = {
            bucket: {
                group: tuple(values) for group, values in groups.items()
            }
            for bucket, groups in cells.items()
        }
        self._cell_cumulative: dict[tuple[bool, int], np.ndarray] = {}
        group_count_totals: dict[bool, int] = {}
        self._bucket_sizes: dict[int, int] = {}
        self._group_counts: dict[tuple[bool, int], int] = {}
        for bucket in self._buckets:
            total = 0
            for group, group_cells in self._cells[bucket].items():
                alpha = (
                    self.priority_family_temperature_alpha
                    if group
                    else self.family_temperature_alpha
                )
                weights = np.power(
                    np.asarray(
                        [len(values) for _, values in group_cells], dtype=np.float64
                    ),
                    alpha,
                )
                key = (group, bucket)
                self._cell_cumulative[key] = np.cumsum(weights)
                count = sum(len(values) for _, values in group_cells)
                self._group_counts[key] = count
                group_count_totals[group] = group_count_totals.get(group, 0) + count
                total += count
            self._bucket_sizes[bucket] = min(
                self.maximum_batch_size,
                max(1, self.audio_budget_samples // bucket),
                total,
            )
        bucket_weights = []
        self._priority_fraction_by_bucket: dict[int, float] = {}
        for bucket in self._buckets:
            masses = {}
            for group in (False, True):
                key = (group, bucket)
                masses[group] = (
                    group_fractions[group]
                    * self._group_counts.get(key, 0)
                    / max(group_count_totals.get(group, 0), 1)
                )
            mass = masses[False] + masses[True]
            if mass <= 0.0:
                raise ValueError("duration bucket has no requested sampling mass")
            self._priority_fraction_by_bucket[bucket] = masses[True] / mass
            # Use the full mixed-bucket capacity. This preserves the padded
            # audio budget instead of fragmenting rare durations by priority.
            bucket_weights.append(mass / self._bucket_sizes[bucket])
        self._bucket_cumulative = np.cumsum(np.asarray(bucket_weights, dtype=np.float64))

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _uniform(self, key: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "little") / float(2**64)

    def _permuted_position(self, size: int, key: str, position: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        multiplier = max(1, int.from_bytes(digest[:8], "little") % size)
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size or 1
        offset = int.from_bytes(digest[8:16], "little") % size
        return (multiplier * int(position) + offset) % size

    @staticmethod
    def _weighted_index(cumulative: np.ndarray, unit: float) -> int:
        return min(len(cumulative) - 1, int(np.searchsorted(cumulative, unit * cumulative[-1])))

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for step in range(self.start_step, self.stop_step):
            bucket_index = self._weighted_index(
                self._bucket_cumulative, self._uniform(f"bucket:{step}")
            )
            bucket = self._buckets[bucket_index]
            batch_size = self._bucket_sizes[bucket]
            desired_priority = self._priority_fraction_by_bucket[bucket] * batch_size
            priority_count = int(math.floor(desired_priority))
            if self._uniform(f"priority-round:{step}") < desired_priority - priority_count:
                priority_count += 1
            priority_available = self._group_counts.get((True, bucket), 0)
            ordinary_available = self._group_counts.get((False, bucket), 0)
            priority_count = min(priority_count, priority_available)
            priority_count = max(priority_count, batch_size - ordinary_available)
            group_slots = [True] * priority_count + [False] * (batch_size - priority_count)
            if len(group_slots) > 1:
                group_slots = [
                    group_slots[index]
                    for index in sorted(
                        range(len(group_slots)),
                        key=lambda index: self._uniform(
                            f"group-order:{step}:{index}"
                        ),
                    )
                ]
            batch: list[tuple[int, int]] = []
            family_positions: dict[tuple[bool, int], int] = {}
            used_ordinals: set[int] = set()
            for slot, is_priority in enumerate(group_slots):
                key = (is_priority, bucket)
                cells = self._cells[bucket][is_priority]
                cumulative = self._cell_cumulative[key]
                cell_index = self._weighted_index(
                    cumulative,
                    self._uniform(f"family:{int(is_priority)}:{step}:{slot}"),
                )
                family, ordinals = cells[cell_index]
                position_key = (is_priority, cell_index)
                local_position = family_positions.get(position_key, 0)
                if local_position >= len(ordinals):
                    available = [
                        index
                        for index, (_, values) in enumerate(cells)
                        if family_positions.get((is_priority, index), 0) < len(values)
                    ]
                    cell_index = available[
                        int(
                            self._uniform(
                                f"fallback:{int(is_priority)}:{step}:{slot}"
                            )
                            * len(available)
                        )
                        % len(available)
                    ]
                    family, ordinals = cells[cell_index]
                    position_key = (is_priority, cell_index)
                    local_position = family_positions.get(position_key, 0)
                family_positions[position_key] = local_position + 1
                position = self._permuted_position(
                    len(ordinals),
                    f"ordinal:{int(is_priority)}:{family}:{bucket}:{step}",
                    local_position,
                )
                ordinal = int(ordinals[position])
                if ordinal in used_ordinals:
                    raise RuntimeError("Dynamic sampler produced a duplicate ordinal")
                used_ordinals.add(ordinal)
                batch.append((ordinal, bucket))
            yield batch


def collate_v13(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "mixture",
        "target",
        "dna",
        "other_dna",
        "unassigned_residual",
        "pitch",
        "valid_samples",
        "note_onset_samples",
        "note_offset_samples",
        "ordinal",
        "hard_case",
    )
    maximum = max(int(row["mixture"].shape[-1]) for row in batch)
    waveform_keys = {"mixture", "target", "dna", "other_dna", "unassigned_residual"}
    output: dict[str, Any] = {}
    for key in keys:
        if key == "note_onset_samples":
            values = [row.get(key, torch.zeros_like(row["valid_samples"])) for row in batch]
        elif key == "note_offset_samples":
            values = [row.get(key, row["valid_samples"]) for row in batch]
        else:
            values = [row[key] for row in batch]
        if key in waveform_keys:
            padded = [torch.nn.functional.pad(value, (0, maximum - value.shape[-1])) for value in values]
            output[key] = torch.stack(padded)
        else:
            output[key] = torch.stack(values)
    for key in ("pair_id", "sound_family", "source_group", "stratum"):
        output[key] = [str(row[key]) for row in batch]
    if all("mixture_id" in row for row in batch):
        output["mixture_id"] = [str(row["mixture_id"]) for row in batch]
    return output
