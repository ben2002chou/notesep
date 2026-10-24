"""Stage checksum-bound retired packed indexes, keeping canonical audio read-only."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zipfile

from notesep.training_paths import relocated


class PackedMetadataOverlay:
    def __init__(self, archive, manifest, manifest_sha256, output, *, blob_paths=None):
        manifest = Path(manifest)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha256:
            raise ValueError("Retired metadata catalogue changed")
        self.catalogue = json.loads(manifest.read_text())
        self.records = {record["path"]: record for record in self.catalogue["files"]}
        if len(self.records) != len(self.catalogue["files"]):
            raise ValueError("Duplicate catalogue source paths")
        self.roots = [
            Path(root).expanduser().absolute() for root in self.catalogue["roots"]
        ]
        self.local_roots = [relocated(root) for root in self.roots]
        self.output = Path(output).resolve()
        self.output.mkdir(exist_ok=False)
        self.archive_path = Path(archive).resolve()
        self.archive = zipfile.ZipFile(self.archive_path)
        self.archive_pid = os.getpid()
        self.process_output = self.output
        names = self.archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        self.staged = {}
        self.blob_paths = blob_paths or {}

    def _isolate_worker(self):
        """Forked readers must not share seek offsets or staging directories."""
        pid = os.getpid()
        if pid != self.archive_pid:
            self.archive.close()
            self.archive = zipfile.ZipFile(self.archive_path)
            self.archive_pid = pid
            self.process_output = self.output / f"worker-{pid}"
            self.process_output.mkdir(exist_ok=False)
            self.staged = {}

    def resolve(self, original):
        incoming = Path(original).expanduser().absolute()
        matches = []
        for root, local_root in zip(self.roots, self.local_roots):
            if incoming.is_relative_to(root):
                matches.append(
                    (
                        root / incoming.relative_to(root),
                        local_root / incoming.relative_to(root),
                        root,
                    )
                )
            elif incoming.is_relative_to(local_root):
                matches.append(
                    (root / incoming.relative_to(local_root), incoming, root)
                )
        if len(matches) != 1:
            raise ValueError("Packed root outside admitted catalogue")
        original, local, root = matches[0]
        if (local / "manifest.sqlite").is_file() and (
            local / "COMPLETE.json"
        ).is_file():
            return local
        self._isolate_worker()
        if str(original) in self.staged:
            return Path(self.staged[str(original)]["overlay"])
        relative = original.relative_to(root)
        staged = self.process_output / relative
        staged.mkdir(parents=True, exist_ok=False)
        checksums = {}
        for name in ["manifest.sqlite", "COMPLETE.json"]:
            record = self.records[str(original / name)]
            member = self.archive.getinfo(record["member"])
            if member.file_size != record["size"]:
                raise ValueError("Archived metadata size differs")
            data = self.archive.read(member)  # ZipFile checks this member's CRC.
            digest = hashlib.sha256(data).hexdigest()
            if digest != record["sha256"]:
                raise ValueError("Archived metadata member SHA differs")
            with (staged / name).open("xb") as stream:
                stream.write(data)
            checksums[name] = digest
        with sqlite3.connect(
            (staged / "manifest.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
        ) as connection:
            shards = [
                row[0]
                for row in connection.execute("SELECT DISTINCT shard FROM assets")
            ]
        links = {}
        for name in shards:
            if Path(name).name != name or name in {"manifest.sqlite", "COMPLETE.json"}:
                raise ValueError("Unsafe archived shard name")
            key = str(original.relative_to(root) / name)
            source = Path(self.blob_paths[key]) if self.blob_paths else local / name
            if not source.is_file():
                raise FileNotFoundError(source)
            (staged / name).symlink_to(source)
            links[name] = {
                "canonical_blob": str(source),
                "bytes": source.stat().st_size,
            }
        self.staged[str(original)] = {
            "overlay": str(staged),
            "metadata_sha256": checksums,
            "read_only_blob_links": links,
        }
        return staged

    def install(self, streaming):
        original_reader = streaming.PackedBlobShardReader

        def reader(root, *args, **kwargs):
            return original_reader(self.resolve(root), *args, **kwargs)

        streaming.PackedBlobShardReader = reader
        return original_reader

    def close(self):
        self.archive.close()
