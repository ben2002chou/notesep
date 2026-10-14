from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notesep.score_io import midi_note_event_count, midi_notes, midi_notes_in_window


class MidiScoreTests(unittest.TestCase):
    def test_counts_raw_events_filtered_by_editor_minimum_duration(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=24))
        track.append(mido.Message("note_on", channel=0, note=64, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=96))
        track.append(mido.Message("note_on", channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=96))
        midi.tracks.append(track)
        output = io.BytesIO()
        midi.save(file=output)

        data = output.getvalue()
        self.assertEqual(midi_note_event_count(data), 2)
        self.assertEqual([note["pitch"] for note in midi_notes(data, duration=2)], [64])

    def test_keeps_instrument_parts_and_absolute_timing(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        tempo = mido.MidiTrack()
        tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        midi.tracks.append(tempo)

        piano = mido.MidiTrack()
        piano.append(mido.MetaMessage("track_name", name="Piano", time=0))
        piano.append(mido.Message("program_change", channel=0, program=0, time=0))
        piano.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
        piano.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
        midi.tracks.append(piano)

        guitar = mido.MidiTrack()
        guitar.append(mido.MetaMessage("track_name", name="Guitar", time=0))
        guitar.append(mido.Message("program_change", channel=1, program=26, time=0))
        guitar.append(mido.Message("note_on", channel=1, note=64, velocity=80, time=240))
        guitar.append(mido.Message("note_off", channel=1, note=64, velocity=0, time=480))
        midi.tracks.append(guitar)

        output = io.BytesIO()
        midi.save(file=output)
        notes = midi_notes(output.getvalue(), duration=3)

        self.assertEqual([(n["pitch"], n["part"], n["start"], n["end"]) for n in notes], [
            (60, "Piano", 0.0, 0.5),
            (64, "Guitar", 0.25, 0.75),
        ])
        self.assertNotEqual(notes[0]["partId"], notes[1]["partId"])

    def test_does_not_reject_dense_midi_before_user_can_choose_a_part(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="Piano", time=0))
        for _ in range(30):
            track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
            track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=48))
        midi.tracks.append(track)
        output = io.BytesIO()
        midi.save(file=output)

        notes = midi_notes(output.getvalue(), duration=10)
        self.assertEqual(len(notes), 30)
        self.assertTrue(all(note["part"] == "Piano" for note in notes))

    def test_crops_and_shifts_full_recording_midi_to_edit_window(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="Piano", time=0))
        track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=480))
        track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=960))
        track.append(mido.Message("note_on", channel=0, note=64, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=1920))
        track.append(mido.Message("note_on", channel=0, note=67, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=0, note=67, velocity=0, time=480))
        midi.tracks.append(track)
        output = io.BytesIO()
        midi.save(file=output)

        notes = midi_notes_in_window(output.getvalue(), source_duration=5,
                                     window_start=1, window_duration=2)
        self.assertEqual([(note["pitch"], note["start"], note["end"]) for note in notes], [
            (60, 0.0, 0.5),
            (64, 0.5, 2.0),
        ])

    def test_long_midi_is_capped_after_window_crop_not_before(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="Piano", time=0))
        for _ in range(600):
            track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=48))
            track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=48))
        midi.tracks.append(track)
        output = io.BytesIO()
        midi.save(file=output)

        with self.assertRaisesRegex(ValueError, "more than 512 notes"):
            midi_notes(output.getvalue(), duration=61)
        notes = midi_notes_in_window(output.getvalue(), source_duration=61,
                                     window_start=30, window_duration=2)
        self.assertGreater(len(notes), 1)
        self.assertLessEqual(len(notes), 512)
        self.assertTrue(all(0 <= note["start"] < note["end"] <= 2 for note in notes))


if __name__ == "__main__":
    unittest.main()
