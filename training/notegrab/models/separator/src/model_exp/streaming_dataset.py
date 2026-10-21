"""Read score-addressed examples from the frozen training corpus."""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import Dataset

from .corpus import FrozenV13Contract, _bound_path
from .packed_blob_shards import PackedBlobShardReader


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
                raise ValueError(
                    "Acoustic release index does not align with the split"
                )
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
            digest = hashlib.sha256(
                str(metadata.get("pair_id", "")).encode()
            ).digest()
            return max(0, min(choices[digest[0] % len(choices)], total - fixed))
        minimum = max(0, offset - fixed)
        maximum = min(onset, total - fixed)
        if minimum > maximum:
            raise ValueError(
                "No fixed window contains the complete target note"
            )
        digest = hashlib.sha256(
            str(metadata.get("pair_id", "")).encode()
        ).digest()
        return minimum + int.from_bytes(digest[:8], "little") % (
            maximum - minimum + 1
        )

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
        desired = min(
            total,
            int(math.ceil(desired / self.window_quantum)) * self.window_quantum,
        )
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
        row = (
            self._db()
            .execute(
                """
            SELECT pair_id, mixture_id, packed_root, sound_family, source_group,
                   hard_case, stratum
            FROM pairs
            WHERE ordinal = ?
            """,
                (ordinal,),
            )
            .fetchone()
        )
        if row is None:
            raise IndexError(ordinal)
        pair_id, mixture_id = str(row[0]), str(row[1])
        packed_root = _bound_path(self.contract.root, str(row[2]))
        if packed_root not in self.contract.packed_roots:
            raise ValueError(
                f"Split index names an unfrozen packed root: {packed_root}"
            )
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
            and bool(
                mixture_metadata.get("complete_dna_context_residual", False)
            )
        ):
            raise ValueError(f"Incomplete DNA mixture context: {mixture_id}")
        sample_rate = int(metadata.get("sample_rate", 0))
        total = int(metadata.get("samples", -1))
        if sample_rate <= 0 or total <= 0:
            raise ValueError(f"Invalid V13 sample geometry: {pair_id}")
        expected_bytes = total * np.dtype("<f4").itemsize
        if (
            target_record.length_bytes != expected_bytes
            or dna_record.length_bytes != expected_bytes
        ):
            raise ValueError(f"V13 pair length mismatch: {pair_id}")
        if requested_samples:
            note_samples = int(metadata.get("offset_sample", -1)) - int(
                metadata.get("onset_sample", -1)
            )
            if requested_samples < note_samples:
                raise ValueError(
                    "Sampler-requested window would truncate the target note"
                )
            start, window_samples = self._variable_window(
                metadata,
                total,
                requested_samples=requested_samples,
                acoustic_release_samples=acoustic_release,
            )
            if window_samples > requested_samples:
                raise ValueError(
                    "Duration index understates the required note window"
                )
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
                mixture_id,
                "dna_context_sum_f32",
                dtype="<f4",
                start=start,
                count=count,
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
