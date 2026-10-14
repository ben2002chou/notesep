"""Small, dependency-light MIDI reader used by the hosted editing workflow."""
from __future__ import annotations

from collections import defaultdict, deque
import io

import mido

MAX_MIDI_BYTES = 3_000_000
MAX_SCORE_NOTES = 512
MAX_SOURCE_MIDI_NOTES = 50_000

GM_PROGRAMS = (
    "Acoustic Grand Piano", "Bright Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone",
    "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ",
    "Church Organ", "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass", "Fingered Bass",
    "Picked Bass", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1",
    "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
    "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1",
    "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs",
    "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2", "Soprano Sax",
    "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon",
    "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle",
    "Shakuhachi", "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)",
    "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)",
    "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)",
    "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)",
    "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)",
    "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)",
    "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums",
    "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring",
    "Helicopter", "Applause", "Gunshot",
)


def midi_note_event_count(data: bytes) -> int:
    """Count non-percussion positive-velocity note-on events before editor filtering."""
    if not data or len(data) > MAX_MIDI_BYTES:
        raise ValueError("MIDI file is empty or too large")
    midi = mido.MidiFile(file=io.BytesIO(data))
    return sum(1 for track in midi.tracks for message in track
               if message.type == "note_on" and message.velocity > 0 and message.channel != 9)


def midi_notes(data: bytes, duration: float, *, max_notes: int | None = MAX_SCORE_NOTES) -> list[dict]:
    """Decode note events while preserving MIDI track/program as editable parts."""
    if not data or len(data) > MAX_MIDI_BYTES:
        raise ValueError("MIDI file is empty or too large")
    midi = mido.MidiFile(file=io.BytesIO(data))

    events = []
    track_names = {}
    for track_index, track in enumerate(midi.tracks):
        absolute_ticks = 0
        for order, message in enumerate(track):
            absolute_ticks += message.time
            if message.type == "track_name":
                track_names[track_index] = message.name.strip()
            priority = 0 if message.type == "set_tempo" else 1
            events.append((absolute_ticks, priority, track_index, order, message))
    events.sort(key=lambda item: item[:4])

    tempo, last_tick, seconds = 500_000, 0, 0.0
    programs: dict[tuple[int, int], int] = defaultdict(int)
    active: dict[tuple[int, int, int], deque] = defaultdict(deque)
    notes = []
    part_names: dict[tuple[int, int, int], str] = {}
    for ticks, _, track_index, _, message in events:
        seconds += mido.tick2second(ticks - last_tick, midi.ticks_per_beat, tempo)
        last_tick = ticks
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if not hasattr(message, "channel") or message.channel == 9:
            continue
        part_key = (track_index, message.channel, programs[(track_index, message.channel)])
        if message.type == "program_change":
            programs[(track_index, message.channel)] = message.program
            continue
        if message.type == "note_on" and message.velocity > 0:
            active[(track_index, message.channel, message.note)].append(
                (seconds, programs[(track_index, message.channel)]))
            continue
        if message.type not in ("note_off", "note_on"):
            continue
        key = (track_index, message.channel, message.note)
        if not active[key]:
            continue
        start, program = active[key].popleft()
        end = min(seconds, duration)
        if not (21 <= message.note <= 108 and start < duration and end - start >= .04):
            continue
        part_key = (track_index, message.channel, program)
        track_name = track_names.get(track_index, "")
        fallback = GM_PROGRAMS[program] if 0 <= program < len(GM_PROGRAMS) else f"MIDI program {program}"
        part = track_name or fallback
        part_names[part_key] = part
        notes.append({"id": str(len(notes)), "event": len(notes), "pitch": message.note,
                      "start": round(start, 4), "end": round(end, 4),
                      "partId": f"{track_index}:{message.channel}:{program}", "part": part})
        if max_notes is not None and len(notes) > max_notes:
            raise ValueError(f"The score has more than {max_notes} notes. Upload a shorter excerpt or a single instrument part.")

    # Disambiguate repeated track names without losing readable GM instrument names.
    duplicate_labels = defaultdict(list)
    for key, label in part_names.items():
        duplicate_labels[label].append(key)
    label_by_key = {}
    for label, keys in duplicate_labels.items():
        for key in keys:
            label_by_key[key] = f"{label} · track {key[0] + 1}" if len(keys) > 1 else label
    for note in notes:
        key = tuple(map(int, note["partId"].split(":")))
        note["part"] = label_by_key[key]

    notes.sort(key=lambda note: (note["start"], note["pitch"], note["partId"]))
    for index, note in enumerate(notes):
        note["id"] = str(index)
        note["event"] = index
    return notes


def midi_notes_in_window(data: bytes, source_duration: float,
                         window_start: float, window_duration: float) -> list[dict]:
    """Clip full-recording MIDI events to one edit window and shift them to local time."""
    window_end = window_start + window_duration
    notes = midi_notes(data, source_duration, max_notes=MAX_SOURCE_MIDI_NOTES)
    selected = []
    for note in notes:
        if note["end"] <= window_start or note["start"] >= window_end:
            continue
        clipped_start = max(note["start"], window_start) - window_start
        clipped_end = min(note["end"], window_end) - window_start
        if clipped_end - clipped_start < .04:
            continue
        selected.append({**note, "start": round(clipped_start, 4),
                         "end": round(clipped_end, 4)})
    if len(selected) > MAX_SCORE_NOTES:
        raise ValueError(f"The selected window contains more than {MAX_SCORE_NOTES} MIDI notes. Shorten the window or choose one instrument part.")
    selected.sort(key=lambda note: (note["start"], note["pitch"], note["partId"]))
    for index, note in enumerate(selected):
        note["id"] = str(index)
        note["event"] = index
    return selected


def score_notes_in_window(notes: list[dict], source_duration: float,
                          window_start: float, window_duration: float) -> list[dict]:
    """Select a full-recording JSON score using Studio's window clipping policy."""
    import math
    if not isinstance(notes, list) or len(notes) > MAX_SOURCE_MIDI_NOTES:
        raise ValueError("Expected a bounded list of score notes")
    selected = []
    identifiers = set()
    for index, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError("Invalid score note")
        pitch, start, end = note.get('pitch'), note.get('start'), note.get('end')
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 21 <= pitch <= 108:
            raise ValueError(f"Note {index + 1} has an invalid MIDI pitch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (start, end)):
            raise ValueError(f"Note {index + 1} has invalid timing")
        if not 0 <= start < end <= source_duration + .001:
            raise ValueError(f"Note {index + 1} lies outside the source recording")
        identifier = str(note.get('id', index))
        if identifier in identifiers:
            raise ValueError("Note IDs must be unique")
        identifiers.add(identifier)
        if end <= window_start or start >= window_start + window_duration:
            continue
        left = max(start, window_start) - window_start
        right = min(end, window_start + window_duration) - window_start
        if right - left < .04:
            continue
        selected.append({**note, 'id': identifier, 'start': left, 'end': right})
    if len(selected) > 96:
        raise ValueError("The edit window contains more than 96 notes; select a shorter window or one part")
    return selected
