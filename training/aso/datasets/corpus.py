"""Validate the frozen training corpus and its split indexes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from notesep.training_paths import relocated

import numpy as np

FREEZE_SCHEMA = "literal_residual_musical_production_freeze_v1"
SPLIT_INDEX_SCHEMA = "literal_residual_v13_split_index_v2"
IDENTITY_SCHEMA = "literal_residual_v13_identity_manifest_v1"
EXPECTED_PAIRS = 2_000_000


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
    path = relocated(raw if raw.is_absolute() else root / raw)
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

        packed_manifest = _bound_path(
            root, str(freeze.get("packed_roots_manifest", ""))
        )
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
                        str(row[1])
                        for row in connection.execute("PRAGMA table_info(pairs)")
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
            if (
                int(observed[0]) != count
                or int(observed[1]) != 0
                or int(observed[2]) != count - 1
            ):
                raise ValueError(
                    f"{split} pair ordinals are not complete and contiguous"
                )
            declared_family_counts = metadata.get("family_counts")
            if not isinstance(declared_family_counts, dict):
                raise ValueError(f"{split} split metadata lacks family_counts")
            family_counts = {
                str(family): int(family_count)
                for family, family_count in declared_family_counts.items()
            }
            if (
                observed_family_counts != family_counts
                or sum(family_counts.values()) != count
            ):
                raise ValueError(f"{split} split family counts are inconsistent")
            for label, metadata_key, observed_counts in (
                (
                    "source-group",
                    "source_group_counts",
                    observed_source_group_counts,
                ),
                ("stratum", "stratum_counts", observed_stratum_counts),
                ("hard-case", "hard_case_counts", observed_hard_case_counts),
            ):
                declared_counts = metadata.get(metadata_key)
                if not isinstance(declared_counts, dict):
                    raise ValueError(f"{split} split metadata lacks {metadata_key}")
                normalized_counts = {
                    str(key): int(value) for key, value in declared_counts.items()
                }
                if (
                    normalized_counts != observed_counts
                    or sum(normalized_counts.values()) != count
                ):
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
