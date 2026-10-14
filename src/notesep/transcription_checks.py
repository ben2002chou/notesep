"""Compare an editable draft with an independent AMT model's note events.

Cross-model agreement is an uncertainty cue, not ground truth: AMT systems can
share omissions and pitch/time errors, and can disagree on legitimate events.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


ONSET_TOLERANCE_SECONDS = 0.15
_INVALID_COST = 1_000_000.0
_PITCH_NAMES = ("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")


def _validate(notes: list[dict], label: str) -> list[dict]:
    if not isinstance(notes, list):
        raise ValueError(f"{label} must be a list of note events")
    checked = []
    for index, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError(f"{label} note {index + 1} is invalid")
        pitch, start, end = note.get("pitch"), note.get("start"), note.get("end")
        if (not isinstance(pitch, int) or isinstance(pitch, bool) or not 21 <= pitch <= 108
                or not isinstance(start, (int, float)) or not math.isfinite(start)
                or not isinstance(end, (int, float)) or not math.isfinite(end)
                or start < 0 or end <= start):
            raise ValueError(f"{label} note {index + 1} has invalid pitch or timing")
        checked.append({**note, "pitch": pitch, "start": float(start), "end": float(end),
                        "id": str(note.get("id", index))})
    return checked


def _candidate(note: dict) -> dict:
    pitch = note["pitch"]
    return {"id": note["id"], "pitch": pitch,
            "pitchName": f"{_PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}",
            "start": round(note["start"], 3), "end": round(note["end"], 3)}


def compare_transcriptions(primary: list[dict], independent: list[dict],
                           primary_provider: str, independent_provider: str) -> dict:
    """One-to-one pitch/onset match; return disagreements for cautious review."""
    primary = _validate(primary, "Current score")
    independent = _validate(independent, "Independent draft")
    n, m = len(primary), len(independent)
    costs = np.full((n, m), _INVALID_COST, dtype=np.float64)
    for i, first in enumerate(primary):
        for j, second in enumerate(independent):
            onset_error = abs(first["start"] - second["start"])
            if first["pitch"] == second["pitch"] and onset_error <= ONSET_TOLERANCE_SECONDS:
                duration_scale = max(first["end"] - first["start"],
                                     second["end"] - second["start"], 0.1)
                duration_error = abs((first["end"] - first["start"])
                                     - (second["end"] - second["start"])) / duration_scale
                costs[i, j] = onset_error / ONSET_TOLERANCE_SECONDS + 0.05 * duration_error

    matches = []
    matched_primary, matched_independent = set(), set()
    if n and m:
        rows, columns = linear_sum_assignment(costs)
        for i, j in zip(rows.tolist(), columns.tolist()):
            if costs[i, j] >= _INVALID_COST:
                continue
            matched_primary.add(i)
            matched_independent.add(j)
            matches.append({"primaryId": primary[i]["id"],
                            "independentId": independent[j]["id"],
                            "onsetErrorMs": round(1000 * abs(primary[i]["start"]
                                                               - independent[j]["start"]))})

    primary_only = [_candidate(note) for index, note in enumerate(primary)
                    if index not in matched_primary]
    independent_only = [_candidate(note) for index, note in enumerate(independent)
                        if index not in matched_independent]
    precision = len(matches) / n if n else (1.0 if not m else 0.0)
    recall = len(matches) / m if m else (1.0 if not n else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if not n and not m:
        summary = "Neither transcription returned usable notes; do not run separation yet."
    elif primary_only or independent_only:
        summary = (f"Cross-check: {len(matches)} of {n} current notes match the independent draft "
                   f"within {int(ONSET_TOLERANCE_SECONDS * 1000)} ms; "
                   f"{len(primary_only)} appear only in the current score and "
                   f"{len(independent_only)} only in the second draft. "
                   "Review these events against the spectrogram; model disagreement is not proof of an error.")
    else:
        summary = (f"Cross-check: all {n} note events match the independent draft within "
                   f"{int(ONSET_TOLERANCE_SECONDS * 1000)} ms. Agreement cannot rule out shared omissions or errors.")

    return {
        "primaryProvider": primary_provider,
        "independentProvider": independent_provider,
        "primaryCount": n,
        "independentCount": m,
        "matchedCount": len(matches),
        "agreementF1": round(f1, 3),
        "onsetToleranceMs": int(ONSET_TOLERANCE_SECONDS * 1000),
        "matches": matches,
        "primaryOnly": primary_only,
        "independentOnly": independent_only,
        "notes": [{"id": note["id"], "flags": ["model_disagreement"]}
                  for index, note in enumerate(primary) if index not in matched_primary],
        "summary": summary,
    }
