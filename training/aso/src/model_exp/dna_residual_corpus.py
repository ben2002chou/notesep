"""Frozen, streaming access to paired DNA and RT2 note priors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote
import hashlib
import sqlite3

import numpy as np

from .dna_residual_store import DNA_KIND, MIXTURE_KIND, RT2_KIND, TARGET_KIND
from .packed_blob_shards import PackedBlobShardReader


@dataclass(frozen=True)
class ResidualPriorRecord:
    ordinal: int
    asset_id: str
    split: str
    dna_store: str
    rt2_store: str | None
    dna_sha256: str
    rt2_sha256: str | None
    metadata: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenDNAResidualPriorCatalog:
    """Read one immutable paired-prior index with bounded open shard readers."""

    def __init__(self, root: str | Path, *, split: str, max_open_shards: int = 8) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.max_open_shards = int(max_open_shards)
        if self.split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if self.max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self.freeze_path = self.root / "FROZEN.json"
        self.index_path = self.root / "index.sqlite"
        freeze = json.loads(self.freeze_path.read_text(encoding="utf-8"))
        if freeze.get("status") != "passed":
            raise ValueError(f"Residual corpus is not frozen-passed: {self.freeze_path}")
        if sha256_file(self.index_path) != freeze.get("index_sha256"):
            raise ValueError("Residual corpus index checksum mismatch")
        self.freeze = freeze
        self._pid = os.getpid()
        self._connection: sqlite3.Connection | None = None
        self._readers: OrderedDict[str, PackedBlobShardReader] = OrderedDict()
        bounds = self._db().execute(
            "SELECT MIN(ordinal), MAX(ordinal), COUNT(*) FROM paired_assets WHERE split = ?",
            (self.split,),
        ).fetchone()
        self._first_ordinal = int(bounds[0]) if bounds[0] is not None else 0
        self._count = int(bounds[2])

    def _reset_after_fork(self) -> None:
        if self._connection is not None:
            self._connection.close()
        for reader in self._readers.values():
            reader.close()
        self._connection = None
        self._readers = OrderedDict()
        self._pid = os.getpid()

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            self._reset_after_fork()

    def _db(self) -> sqlite3.Connection:
        self._ensure_process()
        if self._connection is None:
            uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def _reader(self, path: str) -> PackedBlobShardReader:
        self._ensure_process()
        reader = self._readers.pop(path, None)
        if reader is None:
            reader = PackedBlobShardReader(path)
        self._readers[path] = reader
        while len(self._readers) > self.max_open_shards:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def close(self) -> None:
        self._ensure_process()
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "FrozenDNAResidualPriorCatalog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self._count

    @staticmethod
    def _decode(row: tuple[Any, ...]) -> ResidualPriorRecord:
        return ResidualPriorRecord(
            ordinal=int(row[0]), asset_id=str(row[1]), split=str(row[2]),
            dna_store=str(row[3]), rt2_store=None if row[4] is None else str(row[4]),
            dna_sha256=str(row[5]), rt2_sha256=None if row[6] is None else str(row[6]),
            metadata=json.loads(row[7]),
        )

    def record(self, asset_id: str) -> ResidualPriorRecord:
        row = self._db().execute(
            "SELECT ordinal, asset_id, split, dna_store, rt2_store, dna_sha256, "
            "rt2_sha256, metadata_json FROM paired_assets "
            "WHERE split = ? AND asset_id = ?",
            (self.split, str(asset_id)),
        ).fetchone()
        if row is None:
            raise KeyError((self.split, str(asset_id)))
        return self._decode(row)

    def record_at(self, position: int) -> ResidualPriorRecord:
        position = int(position)
        if not 0 <= position < self._count:
            raise IndexError(position)
        row = self._db().execute(
            "SELECT ordinal, asset_id, split, dna_store, rt2_store, dna_sha256, "
            "rt2_sha256, metadata_json FROM paired_assets "
            "WHERE split = ? AND ordinal = ?",
            (self.split, self._first_ordinal + position),
        ).fetchone()
        if row is None:
            raise IndexError(position)
        return self._decode(row)

    def iter_records(self) -> Iterator[ResidualPriorRecord]:
        rows = self._db().execute(
            "SELECT ordinal, asset_id, split, dna_store, rt2_store, dna_sha256, "
            "rt2_sha256, metadata_json FROM paired_assets "
            "WHERE split = ? ORDER BY ordinal",
            (self.split,),
        )
        for row in rows:
            yield self._decode(row)

    def load_priors(
        self, record: ResidualPriorRecord | str, *, include_rt2: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        resolved = self.record(record) if isinstance(record, str) else record
        samples = int(resolved.metadata["samples"])
        dna = self._reader(resolved.dna_store).read_array_slice(
            resolved.asset_id, DNA_KIND, dtype="<f4", count=samples
        )
        rt2 = None
        if include_rt2:
            if resolved.rt2_store is None:
                raise ValueError("RT2 prior requested from a DNA-only frozen corpus")
            rt2 = self._reader(resolved.rt2_store).read_array_slice(
                resolved.asset_id, RT2_KIND, dtype="<f4", count=samples
            )
        return dna, rt2

    def load_prior_slice(
        self,
        record: ResidualPriorRecord | str,
        *,
        start: int,
        count: int,
        include_rt2: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        resolved = self.record(record) if isinstance(record, str) else record
        start = int(start)
        count = int(count)
        samples = int(resolved.metadata["samples"])
        if start < 0 or count < 1 or start + count > samples:
            raise ValueError(f"Invalid prior slice start={start} count={count} samples={samples}")
        dna = self._reader(resolved.dna_store).read_array_slice(
            resolved.asset_id, DNA_KIND, dtype="<f4", start=start, count=count
        )
        rt2 = None
        if include_rt2:
            if resolved.rt2_store is None:
                raise ValueError("RT2 prior requested from a DNA-only frozen corpus")
            rt2 = self._reader(resolved.rt2_store).read_array_slice(
                resolved.asset_id, RT2_KIND, dtype="<f4", start=start, count=count
            )
        return dna, rt2

    def load_self_contained_raw_slice(
        self,
        record: ResidualPriorRecord | str,
        *,
        start: int,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read aligned mixture and exact-target audio stored beside a DNA prior."""

        resolved = self.record(record) if isinstance(record, str) else record
        if not bool(resolved.metadata.get("self_contained_raw", False)):
            raise ValueError(f"Residual record has no self-contained raw audio: {resolved.asset_id}")
        start = int(start)
        count = int(count)
        samples = int(resolved.metadata["samples"])
        if start < 0 or count < 1 or start + count > samples:
            raise ValueError(
                f"Invalid raw slice start={start} count={count} samples={samples}"
            )
        reader = self._reader(resolved.dna_store)
        mixture = reader.read_array_slice(
            resolved.asset_id,
            MIXTURE_KIND,
            dtype="<f4",
            start=start,
            count=count,
        )
        target = reader.read_array_slice(
            resolved.asset_id,
            TARGET_KIND,
            dtype="<f4",
            start=start,
            count=count,
        )
        return mixture, target
