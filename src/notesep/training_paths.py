"""Relocate frozen training paths without editing the saved contracts."""

import json
from pathlib import Path

_mapping = ()


def configure(path=None):
    global _mapping
    if path is None:
        _mapping = ()
        return

    document = Path(path).expanduser().resolve()
    values = json.loads(document.read_text())
    if not isinstance(values, dict):
        raise ValueError("Path map must be an old-prefix to new-folder object")

    rows = []
    for old, new in values.items():
        if not isinstance(new, str) or not new:
            raise ValueError("Destination must be a nonempty path")
        original = Path(old)
        if not original.is_absolute() or ".." in original.parts:
            raise ValueError(
                "Original prefix must be an absolute normalized path"
            )
        destination = Path(new).expanduser()
        if not destination.is_absolute():
            destination = document.parent / destination
        rows.append((original, destination.resolve()))

    if len({old for old, _ in rows}) != len(rows):
        raise ValueError("Duplicate normalized path prefix")
    _mapping = tuple(
        sorted(rows, key=lambda row: len(row[0].parts), reverse=True)
    )


def relocated(path):
    original = Path(path).expanduser().absolute()
    for prefix, destination in _mapping:
        if original.is_relative_to(prefix):
            relative = original.relative_to(prefix)
            if ".." in relative.parts:
                raise ValueError("Mapped path traversal is forbidden")
            return (destination / relative).resolve()
    return original.resolve()
