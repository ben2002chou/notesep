"""Conservative audio checks for editable score drafts and separated notes.

These checks are review cues, not a substitute for listening or ground truth.
They use pitch-local harmonic evidence and onset energy changes, not the ASO
mixture residual (which can look small even when note ownership is wrong).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import stft


_MIN_PITCH = 21
_MAX_PITCH = 108
_FFT = 4096
_HOP = 160  # 10 ms at the hosted 16 kHz rate


def _pitch_energy(
    audio: np.ndarray, rate: int, pitches=None
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic energy indexed by MIDI pitch and centered time frame."""
    audio = audio.astype(np.float32, copy=False)
    if len(audio) < _FFT:
        audio = np.pad(audio, (0, _FFT - len(audio)))
    _, times, spectrum = stft(
        audio,
        fs=rate,
        nperseg=_FFT,
        noverlap=_FFT - _HOP,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    bin_hz = rate / _FFT
    energy = np.zeros((_MAX_PITCH + 1, len(times)), dtype=np.float32)
    for pitch in (
        range(_MIN_PITCH, _MAX_PITCH + 1) if pitches is None else sorted(set(pitches))
    ):
        fundamental = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        total = np.zeros(len(times), dtype=np.float32)
        weight_sum = 0.0
        harmonic = 1
        while fundamental * harmonic < rate * 0.47:
            center = int(round(fundamental * harmonic / bin_hz))
            # One neighboring FFT bin accommodates slight tuning/drift. The
            # several partials keep low notes distinguishable at this NFFT.
            radius = max(1, int(round(center * 0.0035)))
            peak = magnitude[
                max(0, center - radius) : min(len(magnitude), center + radius + 1)
            ].max(axis=0)
            weight = 1.0 / harmonic
            total += peak * peak * weight
            weight_sum += weight
            harmonic += 1
        if weight_sum:
            energy[pitch] = total / weight_sum
    return times, energy


def _active_score_pitches(
    notes: list[dict], first: float, last: float, excluded_index: int
) -> set[int]:
    active = set()
    for index, note in enumerate(notes):
        if index == excluded_index:
            continue
        if note["start"] < last and note["end"] > first:
            active.add(note["pitch"])
    return active


def _flags_for_notes(
    times: np.ndarray,
    energy: np.ndarray,
    notes: list[dict],
    rate: int,
    *,
    separated: bool = False,
) -> list[dict]:
    checks = []
    log_energy = np.log(energy + 1e-12)
    for index, note in enumerate(notes):
        pitch = note["pitch"]
        onset, offset = float(note["start"]), float(note["end"])
        flags: list[str] = []

        # Compare a note with nearby pitches only when the score does not
        # already contain that simultaneous pitch. This avoids penalizing
        # real chords for having multiple fundamentals at once.
        first = min(offset, onset + (0.04 if offset - onset > 0.10 else 0.0))
        last = min(offset, max(first + 0.04, onset + 0.35))
        frame_mask = (times >= first) & (times <= last)
        frames = np.flatnonzero(frame_mask)
        if not len(frames):
            frames = np.flatnonzero((times >= onset) & (times <= offset))
        neighbor_pitches = [
            candidate
            for candidate in range(
                max(_MIN_PITCH, pitch - 5), min(_MAX_PITCH, pitch + 5) + 1
            )
            if candidate != pitch
        ]
        occupied = _active_score_pitches(notes, first, last, index)
        neighbor_pitches = [
            candidate for candidate in neighbor_pitches if candidate not in occupied
        ]
        pitch_margin_db = None
        if len(frames) and neighbor_pitches:
            target = energy[pitch, frames]
            alternative = np.max(energy[neighbor_pitches][:, frames], axis=0)
            margins = 10.0 * np.log10((target + 1e-12) / (alternative + 1e-12))
            pitch_margin_db = float(np.median(margins))
            if pitch_margin_db < -3.0:
                flags.append("weak_pitch_evidence")

        # Locate a clear, pitch-specific energy rise near the proposed onset.
        # STFT centering smears attacks by about 0.1 s, so only flag a strong
        # displaced attack beyond a deliberately broad 0.22 s tolerance.
        attack_offset_ms = None
        attack_strength = 0.0
        if onset >= 0.08 and offset - onset >= 0.10:
            step = max(2, round(0.05 * rate / _HOP))
            frame_indices = np.arange(step, len(times) - step)
            candidate = frame_indices[
                (times[frame_indices] >= onset - 0.42)
                & (times[frame_indices] <= onset + 0.42)
            ]
            if len(candidate):
                rise = (
                    log_energy[pitch, candidate + step]
                    - log_energy[pitch, candidate - step]
                )
                best = int(np.argmax(rise))
                attack_strength = float(rise[best])
                attack_offset_ms = int(round((times[candidate[best]] - onset) * 1000))
                if (
                    attack_offset_ms > 100 or attack_offset_ms < -220
                ) and attack_strength >= math.log(3.0):
                    flags.append("displaced_attack")

        # Exact duplicate notes and heavy same-pitch overlaps are usually
        # doubled detections; do not silently let them proceed as separate
        # NoteSep targets.
        for other_index, other in enumerate(notes):
            if other_index == index or other["pitch"] != pitch:
                continue
            overlap = min(offset, float(other["end"])) - max(
                onset, float(other["start"])
            )
            shorter = min(offset - onset, float(other["end"]) - float(other["start"]))
            if overlap >= max(0.08, 0.5 * shorter):
                flags.append("same_pitch_overlap")
                break

        checks.append(
            {
                "id": str(note.get("id", index)),
                "pitchMarginDb": None
                if pitch_margin_db is None
                else round(pitch_margin_db, 1),
                "attackOffsetMs": attack_offset_ms,
                "flags": flags,
            }
        )
    return checks


def _summary(checks: list[dict], *, separated: bool = False) -> dict:
    flag_counts = {
        "weakPitchEvidence": sum(
            "weak_pitch_evidence" in item["flags"] for item in checks
        ),
        "displacedAttack": sum("displaced_attack" in item["flags"] for item in checks),
        "samePitchOverlap": sum(
            "same_pitch_overlap" in item["flags"] for item in checks
        ),
        "weakTargetPitch": sum("weak_target_pitch" in item["flags"] for item in checks),
        "nearSilentOutput": sum(
            "near_silent_output" in item["flags"] for item in checks
        ),
        "otherScorePitchDominates": sum(
            "other_score_pitch_dominates" in item["flags"] for item in checks
        ),
    }
    flagged = [item for item in checks if item["flags"]]
    if not checks:
        text = "No notes to check."
    elif flagged:
        details = []
        if not separated and flag_counts["weakPitchEvidence"]:
            details.append(
                f"{flag_counts['weakPitchEvidence']} with weak pitch evidence"
            )
        if not separated and flag_counts["displacedAttack"]:
            details.append(f"{flag_counts['displacedAttack']} with a displaced attack")
        if not separated and flag_counts["samePitchOverlap"]:
            details.append(
                f"{flag_counts['samePitchOverlap']} with a same-pitch overlap"
            )
        if flag_counts["weakTargetPitch"]:
            details.append(
                f"{flag_counts['weakTargetPitch']} with low target-pitch clarity"
            )
        if flag_counts["nearSilentOutput"]:
            details.append(f"{flag_counts['nearSilentOutput']} nearly silent")
        if flag_counts["otherScorePitchDominates"]:
            details.append(
                f"{flag_counts['otherScorePitchDominates']} dominated by another concurrent score pitch"
            )
        subject = "separated notes" if separated else "score notes"
        text = (
            f"Audio check: review {len(flagged)} of {len(checks)} {subject} "
            f"({'; '.join(details)}). These are warnings, not proof of an error. "
            "The check only evaluates notes already listed; it cannot find omissions."
        )
    else:
        if separated:
            text = (
                f"Output check found no silent, low target-pitch, or strong competing-pitch estimates among {len(checks)} notes. "
                "Still audition the notes; this check cannot certify separation quality."
            )
        else:
            text = (
                f"No clear pitch, attack, or duplicate-note conflict was found among "
                f"the {len(checks)} listed notes. Missing notes are not assessed; "
                "check score coverage against the spectrogram or run the other-model cross-check."
            )
    return {
        "reviewCount": len(flagged),
        "counts": flag_counts,
        "summary": text,
        "notes": checks,
    }


def audit_score_audio(audio: np.ndarray, notes: list[dict], rate: int = 16000) -> dict:
    """Find clear pitch, onset, and duplicate-note conflicts in an editable draft."""
    if audio.ndim != 1 or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError("Expected finite mono audio and a positive sample rate")
    if not notes:
        return _summary([])
    for note in notes:
        if (
            not isinstance(note, dict)
            or note.get("pitch") not in range(_MIN_PITCH, _MAX_PITCH + 1)
            or not 0
            <= float(note["start"])
            < float(note["end"])
            <= len(audio) / rate + 0.001
        ):
            raise ValueError("Invalid note supplied to audio check")
    times, energy = _pitch_energy(audio, rate)
    return _summary(_flags_for_notes(times, energy, notes, rate))


def audit_separated_notes(
    audio_bank: np.ndarray, notes: list[dict], rate: int = 16000
) -> dict:
    """Check that each returned stem is finite, audible, and pitch-specific."""
    if (
        audio_bank.ndim != 2
        or audio_bank.shape[0] != len(notes)
        or not np.isfinite(audio_bank).all()
        or rate <= 0
    ):
        raise ValueError("Expected one finite separated waveform per note")
    checks = []
    for index, (audio, note) in enumerate(zip(audio_bank, notes)):
        pitch = int(note["pitch"])
        first, last = float(note["start"]), min(float(note["end"]), len(audio) / rate)
        inspect_last = min(last, first + 0.40)
        competing_pitches = sorted(
            {
                int(other["pitch"])
                for other_index, other in enumerate(notes)
                if other_index != index
                and int(other["pitch"]) != pitch
                and float(other["start"]) < inspect_last
                and float(other["end"]) > first
            }
        )
        # Inspect a short attack/body crop per note; keeping the audit bounded
        # matters for 96-note GPU jobs and avoids another full-bank STFT pass.
        crop_first = max(0, int(round(first * rate)))
        crop_last = min(len(audio), int(round(inspect_last * rate)))
        crop = audio[crop_first:crop_last]
        pitch_window = [
            pitch,
            *(
                candidate
                for candidate in range(
                    max(_MIN_PITCH, pitch - 5), min(_MAX_PITCH, pitch + 5) + 1
                )
                if candidate != pitch
            ),
            *competing_pitches,
        ]
        times, energy = _pitch_energy(crop, rate, pitch_window)
        first_local, last_local = (
            max(0.0, first - crop_first / rate),
            max(0.0, last - crop_first / rate),
        )
        frames = np.flatnonzero((times >= first_local) & (times <= last_local))
        flags = []
        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio[crop_first : max(crop_first + 1, crop_last)],
                        dtype=np.float64,
                    )
                )
            )
        )
        if rms < 1e-6:
            flags.append("near_silent_output")
        margin_db = None
        competing_margin_db = None
        dominant_competing_pitch = None
        if len(frames):
            neighbors = [
                candidate
                for candidate in range(
                    max(_MIN_PITCH, pitch - 5), min(_MAX_PITCH, pitch + 5) + 1
                )
                if candidate != pitch
            ]
            target = float(np.median(energy[pitch, frames]))
            alternative = (
                float(np.max(np.median(energy[neighbors][:, frames], axis=1)))
                if neighbors
                else 0.0
            )
            margin_db = 10.0 * math.log10((target + 1e-12) / (alternative + 1e-12))
            if margin_db < -8.0 and rms >= 1e-6:
                flags.append("weak_target_pitch")
            if competing_pitches:
                competing_energy = [
                    float(np.median(energy[candidate, frames]))
                    for candidate in competing_pitches
                ]
                strongest = int(np.argmax(competing_energy))
                competing_margin_db = 10.0 * math.log10(
                    (target + 1e-12) / (competing_energy[strongest] + 1e-12)
                )
                dominant_competing_pitch = competing_pitches[strongest]
                if competing_margin_db < -8.0 and rms >= 1e-6:
                    flags.append("other_score_pitch_dominates")
        checks.append(
            {
                "id": str(note.get("id", index)),
                "pitchMarginDb": None if margin_db is None else round(margin_db, 1),
                "competingPitch": dominant_competing_pitch,
                "competingPitchMarginDb": None
                if competing_margin_db is None
                else round(competing_margin_db, 1),
                "rms": round(rms, 7),
                "flags": flags,
            }
        )
    return _summary(checks, separated=True)
