"""Download and verify a versioned NoteSep dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tempfile
from pathlib import Path, PurePosixPath


def dataset_files(manifest: dict) -> list[tuple[PurePosixPath, int, str]]:
    if manifest.get("schema") not in {
        "notesep_dataset_files_v1",
        "notesep_dataset_files_v2",
    }:
        raise ValueError("Unknown dataset manifest")
    result = []
    seen = set()
    for entry in manifest["files"]:
        name = PurePosixPath(entry["path"])
        if (
            name.is_absolute()
            or not name.parts
            or any(part in (".", "..") for part in name.parts)
            or str(name) != entry["path"]
            or name in seen
        ):
            raise ValueError(f"Invalid dataset path: {entry['path']}")
        size, digest = int(entry["bytes"]), entry["sha256"]
        if (
            size < 0
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError(f"Invalid dataset checksum: {name}")
        seen.add(name)
        result.append((name, size, digest))
    if not result:
        raise ValueError("Empty dataset manifest")
    return result


def restore(manifest: dict, output: Path) -> None:
    """Restore compressed shards without changing their original audio bytes."""
    destinations = set()
    encoded_paths = {row["path"] for row in manifest["files"]}
    for entry in manifest["files"]:
        if entry.get("encoding", "identity") == "identity":
            continue
        if entry["encoding"] != "gzip":
            raise ValueError(f"Unsupported dataset compression: {entry['encoding']}")
        decoded = {
            "schema": "notesep_dataset_files_v1",
            "files": [
                {
                    "path": entry["decoded_path"],
                    "bytes": entry["decoded_bytes"],
                    "sha256": entry["decoded_sha256"],
                }
            ],
        }
        name, size, digest = dataset_files(decoded)[0]
        if str(name) in destinations or str(name) in encoded_paths:
            raise ValueError(f"Conflicting restored dataset path: {name}")
        destinations.add(str(name))
        target = output.joinpath(*name.parts)
        if not target.resolve().is_relative_to(output.resolve()) or target.is_symlink():
            raise ValueError(f"Invalid restored dataset path: {name}")
        if target.exists():
            verify(decoded, output)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".partial",
            delete=False,
        )
        temporary = Path(temporary_file.name)
        current = hashlib.sha256()
        restored_bytes = 0
        try:
            with (
                temporary_file as writer,
                gzip.open(output / entry["path"], "rb") as reader,
            ):
                for block in iter(lambda: reader.read(8 << 20), b""):
                    current.update(block)
                    restored_bytes += len(block)
                    writer.write(block)
            if restored_bytes != size or current.hexdigest() != digest:
                raise ValueError(f"Restored dataset checksum mismatch: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def verify(manifest: dict, output: Path) -> int:
    rows = dataset_files(manifest)
    for name, size, digest in rows:
        path = output.joinpath(*name.parts)
        if (
            not path.resolve().is_relative_to(output.resolve())
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise ValueError(f"Missing or changed dataset file: {name}")
        current = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                current.update(block)
        if current.hexdigest() != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    return len(rows)


def write_path_map(manifest: dict, output: Path) -> Path | None:
    """Write the relocation map bundled with a verified dataset."""
    locations = manifest.get("path_map")
    if locations is None:
        return None
    if not isinstance(locations, dict) or not locations:
        raise ValueError("Dataset path_map must name its source roots")
    if any(
        name == PurePosixPath("path-map.json") for name, _, _ in dataset_files(manifest)
    ):
        raise ValueError("Dataset files cannot replace the generated path-map.json")
    output = output.resolve()
    mapping = {}
    for original, relative in locations.items():
        if not isinstance(original, str) or not isinstance(relative, str):
            raise ValueError("Dataset path_map entries must be strings")
        old = Path(original)
        new = PurePosixPath(relative)
        if not old.is_absolute() or ".." in old.parts:
            raise ValueError(f"Invalid source root: {original}")
        if (
            new.is_absolute()
            or not new.parts
            or ".." in new.parts
            or str(new) != relative
        ):
            raise ValueError(f"Invalid dataset root: {relative}")
        destination = output.joinpath(*new.parts).resolve()
        if not destination.is_relative_to(output) or not destination.exists():
            raise ValueError(f"Missing dataset root: {relative}")
        mapping[str(old)] = str(destination)
    path = output / "path-map.json"
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    return path


def prepare(
    output: Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    manifest_path: Path | None = None,
) -> tuple[int, Path | None]:
    if manifest_path is None and (not repo_id or not revision):
        raise ValueError("Provide --manifest, or both --repo-id and --revision")
    if repo_id and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("--revision must be a 40-character dataset commit")
    output = output.expanduser().resolve()
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
    else:
        from huggingface_hub import hf_hub_download, snapshot_download

        output.mkdir(parents=True, exist_ok=True)
        manifest_file = hf_hub_download(
            repo_id,
            "manifest.json",
            repo_type="dataset",
            revision=revision,
            local_dir=output,
        )
        manifest = json.loads(Path(manifest_file).read_text())
        paths = [str(name) for name, _, _ in dataset_files(manifest)]
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=output,
            allow_patterns=paths,
        )
    count = verify(manifest, output)
    restore(manifest, output)
    path_map = write_path_map(manifest, output)
    return count, path_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", help="Hugging Face dataset repository")
    parser.add_argument("--revision", help="Immutable dataset commit")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Verify an existing local dataset against this manifest",
    )
    args = parser.parse_args()
    try:
        count, path_map = prepare(
            args.output,
            repo_id=args.repo_id,
            revision=args.revision,
            manifest_path=args.manifest,
        )
    except ValueError as error:
        parser.error(str(error))
    output = args.output.expanduser().resolve()
    print(f"Verified {count} dataset files in {output}")
    if path_map is not None:
        print(f"Training path map: {path_map}")


if __name__ == "__main__":
    main()
