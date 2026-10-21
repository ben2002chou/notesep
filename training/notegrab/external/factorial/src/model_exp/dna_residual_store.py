"""Shared identity and window rules for production DNA residual priors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DNA_KIND = "dna_note_float32"
RT2_KIND = "rt2_note_float32"
MIXTURE_KIND = "mixture_note_window_float32"
TARGET_KIND = "target_note_window_float32"


def note_asset_id(sample_id: str, note_row: int) -> str:
    return f"{sample_id}::note_{int(note_row):05d}"


def note_window_steps(
    *,
    onset_step: int,
    offset_step: int,
    codec_start_step: int,
    codec_end_step: int,
    valid_end_step: int,
    lead_frames: int = 1,
    release_frames: int = 1,
    frame_step: int = 4,
) -> tuple[int, int]:
    """Return one absolute interval shared by target, DNA, and RT2 priors."""

    start = max(
        0,
        min(int(onset_step), int(codec_start_step))
        - int(lead_frames) * int(frame_step),
    )
    end = min(
        int(valid_end_step),
        max(int(offset_step), int(codec_end_step))
        + int(release_frames) * int(frame_step),
    )
    if end <= start:
        raise ValueError(f"Invalid residual window [{start}, {end})")
    return start, end


def float32_payload(values: np.ndarray) -> memoryview:
    array = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
    return memoryview(array).cast("B")


def rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {resolved}")
    return rows


def assigned_rows(
    rows: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    shard_index = int(shard_index)
    num_shards = int(num_shards)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]
