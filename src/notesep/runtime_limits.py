"""Helpers for reserving realistic, score-sized ZeroGPU inference slots."""

from __future__ import annotations

import json


def separation_gpu_duration(
    session: str, notes_json: str, apply_aso: bool = True
) -> int:
    """Estimate a per-call GPU maximum from the requested note-bank size.

    The measured 12 s, 24-note workload completed in under 19 s. The
    reservation grows for denser scores, but avoids exhausting a free
    visitor's quota before inference can start.
    """
    del session
    if not isinstance(notes_json, str) or len(notes_json) > 16_384:
        return 60
    try:
        notes = json.loads(notes_json)
    except (TypeError, ValueError):
        return 60
    if not isinstance(notes, list) or not 1 <= len(notes) <= 96:
        return 90
    count = len(notes)
    return min(60, 20 + 2 * count) if count <= 48 else min(120, count + 15)
