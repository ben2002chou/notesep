"""Streaming loader for fresh packed source-complete note banks."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from notesep.training_paths import relocated
import sqlite3
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import Dataset

from notesep.shards import PackedBlobShardReader


# A crop-edge fragment is still part of the exact additive source bank, but it
# is not automatically a useful occurrence query.  In particular, a note that
# began before the crop and ends within a few milliseconds of its left edge can
# contain only an inaudible endpoint/tail.  Keep such stems for closure while
# withholding normalized per-note supervision and evaluation authority.
QUERY_EDGE_GUARD_SAMPLES = 640  # 40 ms at the fixed 16 kHz bank rate.
MINIMUM_QUERY_HELD_SAMPLES = 1_600  # 100 ms of held-note evidence.


def crop_query_eligible(
    relative_onset: int, relative_offset: int, samples: int
) -> bool:
    """Return whether a cropped stem has enough score support to be a query."""
    onset = int(relative_onset)
    offset = int(relative_offset)
    count = int(samples)
    trusted_onset = 0 <= onset < max(0, count - QUERY_EDGE_GUARD_SAMPLES)
    held_start = max(0, onset)
    held_stop = min(count, offset)
    held_samples = max(0, held_stop - held_start)
    return trusted_onset or held_samples >= MINIMUM_QUERY_HELD_SAMPLES


class FreshSourceCompleteDataset(Dataset[dict[str, Any]]):
    """Load a fixed crop and every acoustically nonzero note in that crop."""

    def __init__(
        self,
        index_path: Path,
        *,
        split: str,
        crop_samples: int,
        maximum_sources: int = 64,
        seed: int = 20260824,
        reader_cache_size: int = 8,
        preserve_signed_note_times: bool = False,
        require_acoustic_offsets: bool = False,
    ) -> None:
        super().__init__()
        self.index_path = index_path.expanduser().resolve()
        self.split = str(split)
        self.crop_samples = int(crop_samples)
        self.maximum_sources = int(maximum_sources)
        self.seed = int(seed)
        self.reader_cache_size = int(reader_cache_size)
        self.preserve_signed_note_times = bool(preserve_signed_note_times)
        self.require_acoustic_offsets = bool(require_acoustic_offsets)
        self.epoch = 0
        self._connection: sqlite3.Connection | None = None
        self._readers: OrderedDict[Path, PackedBlobShardReader] = OrderedDict()
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("unknown fresh-bank split")
        if min(self.crop_samples, self.maximum_sources, self.reader_cache_size) < 1:
            raise ValueError("invalid fresh-bank loader geometry")
        if self.require_acoustic_offsets:
            authority_path = self.index_path.parent / "AUTHORITY.json"
            authority = json.loads(authority_path.read_text(encoding="utf-8"))
            if authority.get("rt2_offset_semantics") != "measured_acoustic_endpoint":
                raise ValueError(
                    "joint training requires measured acoustic RT2 offsets; "
                    f"index authority is {authority_path}"
                )
        uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            self.mixture_ids = [
                str(row[0])
                for row in connection.execute(
                    "SELECT mixture_id FROM banks WHERE split=? ORDER BY mixture_id",
                    (self.split,),
                )
            ]
        if not self.mixture_ids:
            raise ValueError(f"empty fresh-bank split: {self.split}")

    def __len__(self) -> int:
        return len(self.mixture_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def _database(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def _reader(self, root_text: str) -> PackedBlobShardReader:
        root = relocated(root_text)
        reader = self._readers.get(root)
        if reader is None:
            reader = PackedBlobShardReader(root, require_complete=True)
            self._readers[root] = reader
            while len(self._readers) > self.reader_cache_size:
                _, stale = self._readers.popitem(last=False)
                stale.close()
        else:
            self._readers.move_to_end(root)
        return reader

    def _crop_start(
        self,
        mixture_id: str,
        bank: dict[str, Any],
        crop_variant: int = 0,
    ) -> int:
        total = int(bank["samples"])
        count = min(total, self.crop_samples)
        anchors = [note for note in bank["notes"] if note["original_target"]]
        if not anchors:
            anchors = list(bank["notes"])
        # Variant zero is the long-standing fixed-crop identity used by all
        # packed teacher/lifecycle caches. Preserve it byte-for-byte; only
        # additional dense-sampler variants extend the hash namespace.
        crop_identity = (
            f"{self.seed}:{self.epoch}:{mixture_id}"
            if int(crop_variant) == 0
            else f"{self.seed}:{self.epoch}:{int(crop_variant)}:{mixture_id}"
        )
        digest = hashlib.sha256(crop_identity.encode()).digest()
        anchor = anchors[int.from_bytes(digest[:8], "little") % len(anchors)]
        midpoint = (int(anchor["onset_sample"]) + int(anchor["offset_sample"])) // 2
        maximum = max(0, total - count)
        return max(0, min(midpoint - count // 2, maximum))

    def describe_crop(self, index: int | tuple[int, int]) -> dict[str, Any]:
        """Return score/acoustic crop geometry without reading packed audio.

        Dense relation curricula previously discovered an invalid K or absent
        octave relation only after reading every note waveform in a bank.  A
        single rejected crop could therefore perform dozens of random packed
        reads.  This metadata view is deliberately only a prefilter: the full
        loader remains the authority for nonzero stems and exact additivity.
        """
        if isinstance(index, tuple):
            mixture_index, crop_variant = map(int, index)
        else:
            mixture_index, crop_variant = int(index), 0
        mixture_id = self.mixture_ids[mixture_index]
        row = (
            self._database()
            .execute(
                "SELECT source_parent_id,renderer,bank_json FROM banks "
                "WHERE mixture_id=?",
                (mixture_id,),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(mixture_id)
        source_parent_id, renderer, bank_json = row
        bank = json.loads(str(bank_json))
        count = min(int(bank["samples"]), self.crop_samples)
        start = self._crop_start(mixture_id, bank, crop_variant)
        stop = start + count
        notes = []
        for note in bank["notes"]:
            onset = int(note["onset_sample"])
            offset = max(
                int(note.get("acoustic_offset_sample", note["offset_sample"])),
                int(note.get("symbolic_offset_sample", note["offset_sample"])),
                int(note["offset_sample"]),
            )
            if onset < stop and offset > start:
                notes.append(note)
        return {
            "mixture_id": mixture_id,
            "source_parent_id": str(source_parent_id),
            "renderer": str(renderer),
            "pitches": torch.tensor(
                [int(note["pitch"]) for note in notes], dtype=torch.long
            ),
            "onsets": torch.tensor(
                [int(note["onset_sample"]) - start for note in notes],
                dtype=torch.long,
            ),
            "offsets": torch.tensor(
                [
                    int(note.get("acoustic_offset_sample", note["offset_sample"]))
                    - start
                    for note in notes
                ],
                dtype=torch.long,
            ),
            "samples": count,
        }

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        # A retrying sampler may request another deterministic note-anchored
        # window from the same bank.  This matters for dense curricula: trying
        # more mixture IDs while exposing only one fixed window per mixture
        # leaves many valid high-K regions unreachable.
        if isinstance(index, tuple):
            mixture_index, crop_variant = map(int, index)
        else:
            mixture_index, crop_variant = int(index), 0
        mixture_id = self.mixture_ids[mixture_index]
        row = (
            self._database()
            .execute(
                "SELECT source_parent_id,score_parent_id,packed_root,bank_json "
                "FROM banks WHERE mixture_id=?",
                (mixture_id,),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(mixture_id)
        source_parent_id, score_parent_id, packed_root, bank_json = row
        bank = json.loads(str(bank_json))
        reader = self._reader(str(packed_root))
        total = int(bank["samples"])
        count = min(total, self.crop_samples)
        start = self._crop_start(mixture_id, bank, crop_variant)
        mixture = reader.read_array_slice(
            str(bank["mixture_asset"]["sample_id"]),
            str(bank["mixture_asset"]["kind"]),
            dtype="<f4",
            start=start,
            count=count,
        )
        notes: list[tuple[dict[str, Any], np.ndarray]] = []
        for metadata in bank["notes"]:
            if (
                self.require_acoustic_offsets
                and "acoustic_offset_sample" not in metadata
            ):
                raise ValueError(
                    f"bank {mixture_id} lacks measured acoustic RT2 offsets"
                )
            waveform = reader.read_array_slice(
                str(metadata["sample_id"]),
                str(metadata["kind"]),
                dtype="<f4",
                start=start,
                count=count,
            )
            if np.any(waveform != 0.0):
                notes.append((metadata, waveform))
        if not notes or len(notes) > self.maximum_sources:
            raise ValueError(f"crop has invalid active K={len(notes)} for {mixture_id}")
        targets = np.stack([waveform for _, waveform in notes]).astype(np.float32)
        mixture = np.asarray(mixture, dtype=np.float32)
        additive_error = float(np.max(np.abs(targets.sum(axis=0) - mixture)))
        if additive_error > 2e-6:
            raise ValueError(f"fresh crop lost additivity: {additive_error}")
        return {
            "mixture_id": mixture_id,
            "source_parent_id": str(source_parent_id),
            "score_parent_id": str(score_parent_id),
            "mixture": torch.from_numpy(mixture),
            "targets": torch.from_numpy(targets),
            # `notes` contains exactly the stems with nonzero samples in the
            # selected waveform crop.  Keep this explicit rather than asking
            # downstream hierarchy code to reconstruct acoustic activity from
            # score endpoints: a held/reverberant stem can be active even
            # when its symbolic onset is before the crop.
            "source_valid": torch.ones(len(notes), dtype=torch.bool),
            "query_valid": torch.tensor(
                [
                    crop_query_eligible(
                        int(metadata["onset_sample"]) - start,
                        int(
                            metadata.get(
                                "acoustic_offset_sample", metadata["offset_sample"]
                            )
                        )
                        - start,
                        count,
                    )
                    for metadata, _ in notes
                ],
                dtype=torch.bool,
            ),
            # Preserve the legacy/frozen query-membership contract separately
            # when acoustic offsets are present.  An acoustic release may make
            # an edge fragment valid as source-bank context, but changing the
            # scored request set would confound endpoint-only ablations.
            "query_valid_symbolic": torch.tensor(
                [
                    crop_query_eligible(
                        int(metadata["onset_sample"]) - start,
                        int(
                            metadata.get(
                                "symbolic_offset_sample", metadata["offset_sample"]
                            )
                        )
                        - start,
                        count,
                    )
                    for metadata, _ in notes
                ],
                dtype=torch.bool,
            ),
            "pitches": torch.tensor(
                [int(metadata["pitch"]) for metadata, _ in notes], dtype=torch.long
            ),
            "onsets": torch.tensor(
                [
                    (
                        int(metadata["onset_sample"]) - start
                        if self.preserve_signed_note_times
                        else max(0, min(count, int(metadata["onset_sample"]) - start))
                    )
                    for metadata, _ in notes
                ],
                dtype=torch.long,
            ),
            "offsets": torch.tensor(
                [
                    (
                        int(
                            metadata.get(
                                "acoustic_offset_sample", metadata["offset_sample"]
                            )
                        )
                        - start
                        if self.preserve_signed_note_times
                        else max(
                            0,
                            min(
                                count,
                                int(
                                    metadata.get(
                                        "acoustic_offset_sample",
                                        metadata["offset_sample"],
                                    )
                                )
                                - start,
                            ),
                        )
                    )
                    for metadata, _ in notes
                ],
                dtype=torch.long,
            ),
            "symbolic_offsets": torch.tensor(
                [
                    (
                        int(
                            metadata.get(
                                "symbolic_offset_sample", metadata["offset_sample"]
                            )
                        )
                        - start
                        if self.preserve_signed_note_times
                        else max(
                            0,
                            min(
                                count,
                                int(
                                    metadata.get(
                                        "symbolic_offset_sample",
                                        metadata["offset_sample"],
                                    )
                                )
                                - start,
                            ),
                        )
                    )
                    for metadata, _ in notes
                ],
                dtype=torch.long,
            ),
            "valid_samples": count,
            "additive_error_max_abs": additive_error,
        }


def collate_fresh_source_complete(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty fresh bank")
    batch = len(records)
    samples = max(int(record["mixture"].numel()) for record in records)
    sources = max(int(record["targets"].shape[0]) for record in records)
    mixture = torch.zeros(batch, samples)
    targets = torch.zeros(batch, sources, samples)
    pitches = torch.zeros(batch, sources, dtype=torch.long)
    onsets = torch.zeros(batch, sources, dtype=torch.long)
    offsets = torch.zeros(batch, sources, dtype=torch.long)
    symbolic_offsets = torch.zeros(batch, sources, dtype=torch.long)
    source_valid = torch.zeros(batch, sources, dtype=torch.bool)
    query_valid = torch.zeros(batch, sources, dtype=torch.bool)
    valid_samples = torch.zeros(batch, dtype=torch.long)
    for batch_index, record in enumerate(records):
        sample_count = int(record["mixture"].numel())
        source_count = int(record["targets"].shape[0])
        mixture[batch_index, :sample_count] = record["mixture"]
        targets[batch_index, :source_count, :sample_count] = record["targets"]
        pitches[batch_index, :source_count] = record["pitches"]
        onsets[batch_index, :source_count] = record["onsets"]
        offsets[batch_index, :source_count] = record["offsets"]
        symbolic_offsets[batch_index, :source_count] = record["symbolic_offsets"]
        source_valid[batch_index, :source_count] = True
        if not bool(record["source_valid"].all()):
            raise ValueError(
                "FreshSourceCompleteDataset records must contain only "
                "acoustically nonzero source stems"
            )
        query_valid[batch_index, :source_count] = record["query_valid"]
        valid_samples[batch_index] = sample_count
    return {
        "mixture_ids": [record["mixture_id"] for record in records],
        "source_parent_ids": [record["source_parent_id"] for record in records],
        "score_parent_ids": [record["score_parent_id"] for record in records],
        "mixture": mixture,
        "targets": targets,
        "pitches": pitches,
        "onsets": onsets,
        "offsets": offsets,
        "symbolic_offsets": symbolic_offsets,
        "source_valid": source_valid,
        "query_valid": query_valid,
        "valid_samples": valid_samples,
    }


__all__ = [
    "FreshSourceCompleteDataset",
    "MINIMUM_QUERY_HELD_SAMPLES",
    "QUERY_EDGE_GUARD_SAMPLES",
    "collate_fresh_source_complete",
    "crop_query_eligible",
]
