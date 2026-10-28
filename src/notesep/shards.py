"""Seekable immutable blob shards with a compact SQLite index.

The payload is not compressed by this container. Assets such as FLAC and NPZ
remain in their native compressed formats and can seek within a bounded view of
the shard. This avoids one inode per asset without forcing a worker to inflate
an entire shard or source performance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from notesep.training_paths import relocated
import sqlite3
from typing import Any, Callable, Iterable
from urllib.parse import quote

import numpy as np


logger = logging.getLogger(__name__)

PACKED_BLOB_SCHEMA = "packed_blob_shards_v1"


@dataclass(frozen=True)
class PackedBlobRecord:
    sample_id: str
    kind: str
    shard: str
    offset_bytes: int
    length_bytes: int
    sha256: str
    metadata: dict[str, Any]


class _PreadView(io.RawIOBase):
    """Independent seek cursor over a byte range using positional reads."""

    def __init__(
        self,
        fd: int,
        offset_bytes: int,
        length_bytes: int,
        on_read: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        self._fd = int(fd)
        self._offset = int(offset_bytes)
        self._length = int(length_bytes)
        self._position = 0
        self._on_read = on_read
        if self._offset < 0 or self._length < 0:
            raise ValueError("Blob offsets and lengths must be nonnegative")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._checkClosed()
        if whence == os.SEEK_SET:
            position = int(offset)
        elif whence == os.SEEK_CUR:
            position = self._position + int(offset)
        elif whence == os.SEEK_END:
            position = self._length + int(offset)
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the beginning of a blob")
        self._position = min(position, self._length)
        return self._position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        self._checkClosed()
        remaining = self._length - self._position
        if remaining <= 0:
            return 0
        view = memoryview(buffer).cast("B")
        requested = min(len(view), remaining)
        data = os.pread(self._fd, requested, self._offset + self._position)
        count = len(data)
        view[:count] = data
        self._position += count
        if self._on_read is not None:
            self._on_read(count)
        return count

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        remaining = self._length - self._position
        if size is None or int(size) < 0:
            requested = remaining
        else:
            requested = min(int(size), remaining)
        if requested <= 0:
            return b""
        data = os.pread(self._fd, requested, self._offset + self._position)
        self._position += len(data)
        if self._on_read is not None:
            self._on_read(len(data))
        return data


class PackedBlobShardWriter:
    """Stream assets into bounded shards and publish the index atomically."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        target_shard_bytes: int = 4 * 1024**3,
        prefix: str = "assets",
        copy_buffer_bytes: int = 8 * 1024**2,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_shard_bytes = int(target_shard_bytes)
        self.prefix = str(prefix)
        self.copy_buffer_bytes = int(copy_buffer_bytes)
        if self.target_shard_bytes < 1 or self.copy_buffer_bytes < 1:
            raise ValueError("Shard and copy-buffer sizes must be positive")
        if not self.prefix or "/" in self.prefix:
            raise ValueError("prefix must be a nonempty filename component")

        self.index_path = self.output_dir / "manifest.sqlite"
        self._partial_index = self.output_dir / ".manifest.sqlite.partial"
        complete_path = self.output_dir / "COMPLETE.json"
        conflicts = [
            path
            for path in (self.index_path, self._partial_index, complete_path)
            if path.exists()
        ]
        conflicts.extend(self.output_dir.glob(f"{self.prefix}-*.blob.partial"))
        if conflicts:
            raise FileExistsError(f"Refusing to overwrite packed store: {conflicts[0]}")

        self._connection = sqlite3.connect(self._partial_index)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.executescript(
            """
            CREATE TABLE assets (
                sample_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                shard TEXT NOT NULL,
                offset_bytes INTEGER NOT NULL,
                length_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (sample_id, kind)
            ) WITHOUT ROWID;
            CREATE INDEX assets_kind_sample ON assets(kind, sample_id);
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        self._handle: io.BufferedWriter | None = None
        self._partial_shard: Path | None = None
        self._final_shard: Path | None = None
        self._shard_index = -1
        self._offset = 0
        self._asset_count = 0
        self._payload_bytes = 0
        self._kind_counts: Counter[str] = Counter()
        self._asset_keys: set[tuple[str, str]] = set()
        self._shards: list[dict[str, Any]] = []
        self._finalized = False

    def _require_unique_asset(self, sample_id: str, kind: str) -> None:
        key = (str(sample_id), str(kind))
        if key in self._asset_keys:
            raise ValueError(f"Duplicate packed asset: {key!r}")
        self._asset_keys.add(key)

    def _open_next_shard(self) -> None:
        self._shard_index += 1
        name = f"{self.prefix}-{self._shard_index:05d}.blob"
        self._final_shard = self.output_dir / name
        self._partial_shard = self.output_dir / f"{name}.partial"
        if self._final_shard.exists() or self._partial_shard.exists():
            raise FileExistsError(f"Refusing to overwrite {self._final_shard}")
        self._handle = self._partial_shard.open("xb")
        self._offset = 0

    def _close_current_shard(self) -> None:
        if self._handle is None:
            return
        assert self._partial_shard is not None and self._final_shard is not None
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.replace(self._partial_shard, self._final_shard)
        self._shards.append(
            {
                "shard": self._final_shard.name,
                "bytes": int(self._final_shard.stat().st_size),
            }
        )
        self._handle = None

    def _prepare_asset(self, length_bytes: int) -> tuple[str, int]:
        length_bytes = int(length_bytes)
        if length_bytes < 0:
            raise ValueError("Asset size cannot be negative")
        if self._handle is None:
            self._open_next_shard()
        if self._offset and self._offset + length_bytes > self.target_shard_bytes:
            self._close_current_shard()
            self._open_next_shard()
        assert self._final_shard is not None
        return self._final_shard.name, self._offset

    def _publish_record(
        self,
        *,
        sample_id: str,
        kind: str,
        shard: str,
        offset_bytes: int,
        length_bytes: int,
        sha256: str,
        metadata: dict[str, Any] | None,
    ) -> PackedBlobRecord:
        sample_id = str(sample_id)
        kind = str(kind)
        if not sample_id or not kind:
            raise ValueError("sample_id and kind must be nonempty")
        metadata_payload = dict(metadata or {})
        try:
            metadata_json = json.dumps(
                metadata_payload, sort_keys=True, separators=(",", ":")
            )
        except TypeError as error:
            raise TypeError("Packed blob metadata must be JSON serializable") from error
        try:
            self._connection.execute(
                "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    kind,
                    shard,
                    int(offset_bytes),
                    int(length_bytes),
                    sha256,
                    metadata_json,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"Duplicate packed asset: {(sample_id, kind)!r}"
            ) from error
        self._asset_count += 1
        self._payload_bytes += int(length_bytes)
        self._kind_counts[kind] += 1
        return PackedBlobRecord(
            sample_id=sample_id,
            kind=kind,
            shard=shard,
            offset_bytes=int(offset_bytes),
            length_bytes=int(length_bytes),
            sha256=sha256,
            metadata=metadata_payload,
        )

    def add_path(
        self,
        sample_id: str,
        kind: str,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> PackedBlobRecord:
        self._require_unique_asset(sample_id, kind)
        source = Path(path).expanduser().resolve()
        length = int(source.stat().st_size)
        shard, offset = self._prepare_asset(length)
        assert self._handle is not None
        digest = hashlib.sha256()
        copied = 0
        with source.open("rb") as input_handle:
            while block := input_handle.read(self.copy_buffer_bytes):
                self._handle.write(block)
                digest.update(block)
                copied += len(block)
        if copied != length:
            raise IOError(f"Asset changed while packing: {source}")
        self._offset += copied
        return self._publish_record(
            sample_id=sample_id,
            kind=kind,
            shard=shard,
            offset_bytes=offset,
            length_bytes=copied,
            sha256=digest.hexdigest(),
            metadata=metadata,
        )

    def add_bytes(
        self,
        sample_id: str,
        kind: str,
        payload: bytes | bytearray | memoryview,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> PackedBlobRecord:
        self._require_unique_asset(sample_id, kind)
        view = memoryview(payload).cast("B")
        shard, offset = self._prepare_asset(len(view))
        assert self._handle is not None
        self._handle.write(view)
        self._offset += len(view)
        return self._publish_record(
            sample_id=sample_id,
            kind=kind,
            shard=shard,
            offset_bytes=offset,
            length_bytes=len(view),
            sha256=hashlib.sha256(view).hexdigest(),
            metadata=metadata,
        )

    def finalize(self, metadata: dict[str, Any] | None = None) -> Path:
        if self._finalized:
            raise RuntimeError("Packed blob writer was already finalized")
        if self._asset_count < 1:
            raise ValueError("Cannot finalize an empty packed blob store")
        self._close_current_shard()
        summary = {
            "schema": PACKED_BLOB_SCHEMA,
            "assets": self._asset_count,
            "payload_bytes": self._payload_bytes,
            "kind_counts": dict(sorted(self._kind_counts.items())),
            "shards": self._shards,
            **dict(metadata or {}),
        }
        for key, value in summary.items():
            self._connection.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                (str(key), json.dumps(value, sort_keys=True)),
            )
        self._connection.commit()
        self._connection.execute("PRAGMA optimize")
        self._connection.close()
        with self._partial_index.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self._partial_index, self.index_path)

        complete_partial = self.output_dir / ".COMPLETE.json.partial"
        complete_path = self.output_dir / "COMPLETE.json"
        with complete_partial.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(complete_partial, complete_path)
        self._finalized = True
        return self.index_path


class PackedBlobShardReader:
    """Read bounded assets without extracting them from their shard."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        require_complete: bool = True,
        buffer_bytes: int = 64 * 1024,
    ) -> None:
        self._pid = os.getpid()
        self._connection: sqlite3.Connection | None = None
        self._fds: dict[str, int] = {}
        self._pread_calls = 0
        self._pread_bytes = 0
        path = relocated(manifest)
        self.index_path = path / "manifest.sqlite" if path.is_dir() else path
        self.root = self.index_path.parent
        self.buffer_bytes = int(buffer_bytes)
        if self.buffer_bytes < 1:
            raise ValueError("buffer_bytes must be positive")
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if require_complete and not (self.root / "COMPLETE.json").is_file():
            raise FileNotFoundError(self.root / "COMPLETE.json")

    def _reset_after_fork(self) -> None:
        if self._connection is not None:
            self._connection.close()
        for fd in self._fds.values():
            os.close(fd)
        self._connection = None
        self._fds = {}
        self._pid = os.getpid()
        self._pread_calls = 0
        self._pread_bytes = 0

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            self._reset_after_fork()

    def _db(self) -> sqlite3.Connection:
        self._ensure_process()
        if self._connection is None:
            uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def close(self) -> None:
        self._ensure_process()
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds = {}

    def __enter__(self) -> "PackedBlobShardReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_fds"] = {}
        state["_pid"] = os.getpid()
        state["_pread_calls"] = 0
        state["_pread_bytes"] = 0
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, sqlite3.Error):
            logger.warning("Could not close a packed audio shard", exc_info=True)

    def __len__(self) -> int:
        row = self._db().execute("SELECT COUNT(*) FROM assets").fetchone()
        return int(row[0])

    def metadata(self) -> dict[str, Any]:
        return {
            str(key): json.loads(value)
            for key, value in self._db().execute(
                "SELECT key, value_json FROM metadata ORDER BY key"
            )
        }

    def list_sample_ids(self, kind: str | None = None) -> list[str]:
        if kind is None:
            rows = self._db().execute(
                "SELECT DISTINCT sample_id FROM assets ORDER BY sample_id"
            )
        else:
            rows = self._db().execute(
                "SELECT sample_id FROM assets WHERE kind = ? ORDER BY sample_id",
                (str(kind),),
            )
        return [str(row[0]) for row in rows]

    def iter_records(self, kind: str | None = None) -> Iterable[PackedBlobRecord]:
        query = (
            "SELECT sample_id, kind, shard, offset_bytes, length_bytes, sha256, "
            "metadata_json FROM assets"
        )
        parameters: tuple[str, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            parameters = (str(kind),)
        query += " ORDER BY sample_id, kind"
        for row in self._db().execute(query, parameters):
            yield PackedBlobRecord(
                sample_id=str(row[0]),
                kind=str(row[1]),
                shard=str(row[2]),
                offset_bytes=int(row[3]),
                length_bytes=int(row[4]),
                sha256=str(row[5]),
                metadata=json.loads(row[6]),
            )

    def record(self, sample_id: str, kind: str) -> PackedBlobRecord:
        row = (
            self._db()
            .execute(
                """
            SELECT sample_id, kind, shard, offset_bytes, length_bytes, sha256,
                   metadata_json
            FROM assets WHERE sample_id = ? AND kind = ?
            """,
                (str(sample_id), str(kind)),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError((str(sample_id), str(kind)))
        return PackedBlobRecord(
            sample_id=str(row[0]),
            kind=str(row[1]),
            shard=str(row[2]),
            offset_bytes=int(row[3]),
            length_bytes=int(row[4]),
            sha256=str(row[5]),
            metadata=json.loads(row[6]),
        )

    def _fd(self, shard: str) -> int:
        self._ensure_process()
        fd = self._fds.get(shard)
        if fd is None:
            path = self.root / shard
            fd = os.open(path, os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def _count_read(self, size: int) -> None:
        self._pread_calls += 1
        self._pread_bytes += int(size)

    def open_asset(self, sample_id: str, kind: str) -> io.BufferedReader:
        record = self.record(sample_id, kind)
        raw = _PreadView(
            self._fd(record.shard),
            record.offset_bytes,
            record.length_bytes,
            self._count_read,
        )
        return io.BufferedReader(raw, buffer_size=self.buffer_bytes)

    def read_bytes(self, sample_id: str, kind: str) -> bytes:
        with self.open_asset(sample_id, kind) as handle:
            return handle.read()

    def read_array_slice(
        self,
        sample_id: str,
        kind: str,
        *,
        dtype: str | np.dtype,
        start: int = 0,
        count: int | None = None,
    ) -> np.ndarray:
        """Read one exact 1-D array range without materializing the asset."""

        record = self.record(sample_id, kind)
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.itemsize < 1:
            raise ValueError("dtype must have a positive item size")
        if record.length_bytes % resolved_dtype.itemsize:
            raise ValueError(
                f"Asset {(sample_id, kind)!r} has {record.length_bytes} bytes, "
                f"not divisible by dtype {resolved_dtype}"
            )
        total = record.length_bytes // resolved_dtype.itemsize
        start = int(start)
        if start < 0 or start > total:
            raise IndexError(f"Array start {start} is outside [0, {total}]")
        resolved_count = total - start if count is None else int(count)
        if resolved_count < 0 or start + resolved_count > total:
            raise IndexError(
                f"Array range [{start}, {start + resolved_count}) exceeds {total}"
            )
        byte_count = resolved_count * resolved_dtype.itemsize
        payload = os.pread(
            self._fd(record.shard),
            byte_count,
            record.offset_bytes + start * resolved_dtype.itemsize,
        )
        self._count_read(len(payload))
        if len(payload) != byte_count:
            raise IOError(
                f"Short packed read for {(sample_id, kind)!r}: "
                f"expected {byte_count}, got {len(payload)}"
            )
        return np.frombuffer(payload, dtype=resolved_dtype).copy()

    def read_verified_array(
        self,
        sample_id: str,
        kind: str,
        *,
        dtype: str | np.dtype,
    ) -> np.ndarray:
        """Read, checksum, and decode one complete 1-D array in one payload pass."""

        record = self.record(sample_id, kind)
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.itemsize < 1:
            raise ValueError("dtype must have a positive item size")
        if record.length_bytes % resolved_dtype.itemsize:
            raise ValueError(
                f"Asset {(sample_id, kind)!r} has {record.length_bytes} bytes, "
                f"not divisible by dtype {resolved_dtype}"
            )
        payload = os.pread(
            self._fd(record.shard),
            record.length_bytes,
            record.offset_bytes,
        )
        self._count_read(len(payload))
        if len(payload) != record.length_bytes:
            raise IOError(
                f"Short packed read for {(sample_id, kind)!r}: "
                f"expected {record.length_bytes}, got {len(payload)}"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record.sha256:
            raise ValueError(
                f"Packed asset checksum failure for {(sample_id, kind)!r}: "
                f"expected {record.sha256}, got {digest}"
            )
        return np.frombuffer(payload, dtype=resolved_dtype).copy()

    def load_npz_arrays(
        self,
        sample_id: str,
        kind: str,
        names: Iterable[str] | None = None,
    ) -> dict[str, np.ndarray]:
        with self.open_asset(sample_id, kind) as handle:
            with np.load(handle, allow_pickle=False) as packed:
                selected = packed.files if names is None else list(names)
                missing = [name for name in selected if name not in packed.files]
                if missing:
                    raise KeyError(f"Packed NPZ {(sample_id, kind)!r} lacks {missing}")
                return {name: packed[name] for name in selected}

    def verify_asset(self, sample_id: str, kind: str) -> bool:
        record = self.record(sample_id, kind)
        digest = hashlib.sha256()
        with self.open_asset(sample_id, kind) as handle:
            while block := handle.read(8 * 1024**2):
                digest.update(block)
        return digest.hexdigest() == record.sha256

    def io_stats(self, *, reset: bool = False) -> dict[str, int]:
        stats = {
            "pread_calls": int(self._pread_calls),
            "pread_bytes": int(self._pread_bytes),
        }
        if reset:
            self._pread_calls = 0
            self._pread_bytes = 0
        return stats
