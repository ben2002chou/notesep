"""Measure additive waveform reconstruction; this is not a score-quality check."""

from __future__ import annotations

import numpy as np


def separated_coverage(
    mixture: np.ndarray, note_bank: np.ndarray, rate: int = 16000
) -> dict:
    """Report substantial audible reconstruction gaps without altering audio.

    A small mixture residual does not establish correct note ownership or pitch;
    callers should present this separately from note-level audio checks.
    """
    if mixture.ndim != 1 or note_bank.ndim != 2 or note_bank.shape[1] != len(mixture):
        raise ValueError("Mixture and note-bank shapes do not match")
    if rate <= 0 or not np.isfinite(mixture).all() or not np.isfinite(note_bank).all():
        raise ValueError("Expected finite audio and a positive sample rate")
    estimate = note_bank.sum(axis=0, dtype=np.float64)
    source = mixture.astype(np.float64)
    residual = source - estimate
    total_energy = float(np.dot(source, source))
    error_pct = 100 * float(np.dot(residual, residual)) / max(total_energy, 1e-20)
    global_rms = np.sqrt(total_energy / max(len(source), 1))
    audible_floor = max(0.0002, 0.18 * global_rms)
    hop = max(1, rate // 10)
    missing = []
    active_start = None
    for first in range(0, len(source), hop):
        last = min(len(source), first + hop)
        original = source[first:last]
        difference = residual[first:last]
        original_energy = float(np.dot(original, original))
        original_rms = np.sqrt(original_energy / (last - first))
        absent = (
            original_rms >= audible_floor
            and float(np.dot(difference, difference)) / max(original_energy, 1e-20)
            > 0.5
        )
        if absent and active_start is None:
            active_start = first
        if not absent and active_start is not None:
            if first - active_start >= rate * 0.2:
                missing.append(
                    {
                        "start": round(active_start / rate, 2),
                        "end": round(first / rate, 2),
                    }
                )
            active_start = None
    if active_start is not None and len(source) - active_start >= rate * 0.2:
        missing.append(
            {
                "start": round(active_start / rate, 2),
                "end": round(len(source) / rate, 2),
            }
        )
    return {"errorEnergyPct": round(error_pct, 2), "uncoveredRegions": missing}
