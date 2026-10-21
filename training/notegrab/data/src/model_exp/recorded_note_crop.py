"""Target-relative crop contract for recorded-note training rows."""
from __future__ import annotations

import math
from typing import Any

import torch


def target_crop_geometry(
    total: int,
    onset: int,
    acoustic_offset: int,
    midi_offset: int | None = None,
) -> tuple[int, int]:
    """Return historical variable-window geometry for a recorded target.

    Synthetic rows anchor duration at MIDI note-off and retain the larger of
    the configured release floor and the measured acoustic release. Recorded
    rows expose both endpoints explicitly, so reproduce that contract while
    still requiring the complete acoustic target to fit.
    """
    offset = int(acoustic_offset)
    midi = offset if midi_offset is None else int(midi_offset)
    if total < 1:
        raise ValueError("recorded example has no audio samples")
    if not 0 <= onset < offset <= total:
        raise ValueError(
            f"target timing lies outside audio: total={total}, onset={onset}, offset={offset}"
        )
    if not onset < midi <= total:
        raise ValueError(
            f"MIDI timing lies outside audio: total={total}, onset={onset}, midi_offset={midi}"
        )
    release = max(2560, offset - midi)
    required_end = max(offset, midi + release)
    size = min(
        total,
        math.ceil(max(16384, required_end - onset + 4000) / 4096) * 4096,
    )
    size = max(offset - onset, size)
    low = max(0, required_end - size)
    high = min(max(0, onset-4000), total-size)
    if low > high:
        low = max(0, min(onset, total-size))
        high = min(max(low, offset-size), total-size)
    start = max(0, min((low+high)//2, total-size))
    if not start <= onset <= offset <= start+size:
        raise ValueError('recorded crop truncates target')
    return start, size


def crop_recorded_note(row: dict[str, Any]) -> dict[str, Any]:
    """Translate audio and score together, preserving the full acoustic note.

    Uses the existing synthetic variable-window bounds: 4000 lead, 2560
    release floor, minimum 16384 samples, 4096 quantum. Scene sampling remains
    unchanged; this function acts only after a target row has been selected.
    """
    total = int(row['valid_samples'])
    onset, offset = int(row['note_onset_samples']), int(row['note_offset_samples'])
    midi_offset = int(row.get('midi_offset_samples', offset))
    start, size = target_crop_geometry(total, onset, offset, midi_offset)
    out = dict(row)
    for key in ('mixture', 'target', 'dna', 'other_dna', 'unassigned_residual'):
        if key in row:
            if row[key].shape != (total,):
                raise ValueError(f'invalid waveform shape for {key}')
            out[key] = row[key][start:start+size].clone()
    out['valid_samples'] = torch.tensor(size, dtype=torch.long)
    for key in ('note_onset_samples', 'note_offset_samples', 'midi_offset_samples'):
        out[key] = row[key] - start
    for key in ('full_score_events', 'all_score_events'):
        events = []
        for item in row[key]:
            original_onset = int(item['onset_sample'])
            original_offset = int(item['offset_sample'])
            # Do not carry unrelated scene events into a target-relative crop.
            # Downstream speed/detune augmentation clamps valid score events to
            # the crop; retaining non-overlapping events would therefore turn
            # them into artificial notes at sample zero or the final sample.
            if original_offset <= start or original_onset >= start + size:
                continue
            event = dict(item)
            # Preserve a negative onset for a note already active at crop
            # entry. The RT2 contract interprets this as HELD, whereas zero
            # would fabricate a fresh onset/attack at the crop boundary.
            event['onset_sample'] = original_onset - start
            event['offset_sample'] = min(size, original_offset - start)
            if event['offset_sample'] <= event['onset_sample']:
                raise ValueError('overlapping score event collapsed during crop')
            if 'midi_offset_sample' in event:
                event['midi_offset_sample'] = min(
                    size,
                    max(event['onset_sample'] + 1, int(event['midi_offset_sample']) - start),
                )
            events.append(event)
        if not any(bool(event.get('is_target')) for event in events):
            raise ValueError('target score event was lost during recorded crop')
        out[key] = tuple(events)
    out['recorded_crop_start_samples'] = start
    out['recorded_original_samples'] = total
    out['recorded_crop_policy'] = (
        'historical_variable_window_midi_anchor_acoustic_release_4000_2560_min16384_q4096'
    )
    # Keep scene metadata and original timing as provenance; the model consumes
    # sample coordinates above, not the provenance-only timing dictionary.
    return out
