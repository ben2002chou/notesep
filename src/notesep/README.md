# Python API

`NoteSep` runs separation; `NoteBank` holds the resulting waveforms and edits.

```python
import soundfile as sf
from notesep import NoteSep, NoteBank

model = NoteSep(device="auto")
notes = model.separate("recording.wav", "score.mid")
notes.export("separated")

# Retune the first note by two semitones and delay it by 50 milliseconds.
note_id = str(notes.notes[0]["id"])
edits = {note_id: {"pitchShift": 2, "delta": 0.05}}
sf.write("edited.wav", notes.render(edits), notes.sample_rate)

# Projects preserve the note waveforms so editing needs no model reload.
reopened = NoteBank.load("separated/project.json")
```

`device="auto"` selects CUDA, then Apple MPS, then CPU. The released pipeline
runs NoteGrab on that device and ASO on CPU.

## Audio and score inputs

Use a 1.1–12 second excerpt, or select one with `window_start` and
`window_seconds`. Stereo input is averaged to mono and resampled to 16 kHz.
By default, score times refer to the full recording. Use
`score_timeline="window"` if a score already starts at the selected passage.

The score can be a MIDI path, a JSON path, or a list of note dictionaries:

```python
score = [{"id": "0", "pitch": 60, "start": 0.25, "end": 0.80}]
notes = model.separate("recording.wav", score)
```

`pitch` is a MIDI number from 21 to 108; `start` and `end` are seconds. IDs must be
unique. Inference accepts 1–96 notes per passage. Each note must have positive
duration and cover at least one audio sample. The caller supplies the score;
separation does not transcribe or correct it.

`notes.stems` is a float32 array shaped `[notes, samples]`. Every row shares the
same time origin and length. `notes.raw_stems` contains the estimates before
ASO; `notes.notes` gives their pitches and times within the selected passage.

## Editing and export

Edits are keyed by the string note ID. Omitted controls keep their defaults.

| Control | Meaning | Default |
| --- | --- | --- |
| `pitchShift` | Pitch change in semitones, from −12 to 12. | `0` |
| `delta` | Time shift in seconds; positive moves the note later. | `0` |
| `stretch` | Duration multiplier, from 0.5 to 2. | `1` |
| `gainDb` | Gain in decibels, from −60 to 24. | `0` |
| `mute` | Omit the note from the mix. | `False` |

`render()` sums the note waveforms. An unedited sum may differ from the input
recording because the model estimates each note separately. Pitch and time edits
use librosa; the browser editor uses SoundTouch, so edited audio can differ
slightly between the two.

`export(directory, edits=edits)` saves the stems, score, project, and an edited
mix. The directory must be empty. `NoteBank.load()` restores a saved project;
pass its `saved_edits` to `render()` to apply the stored controls.

## Trained checkpoints

Pass `weights="outputs/weights"` to `NoteSep` to use your trained
`separator.pt` and `aso.pt`. Omitting it loads the published paper models.
Exports record the checksums of the files actually used.

## Files

| File or folder | Purpose |
| --- | --- |
| `__init__.py` | Exposes `NoteSep` and `NoteBank`. |
| `api.py` | Separation, note editing, and project loading/export. |
| `cli.py` | Command-line arguments and command dispatch. |
| `inference.py` | Load checkpoints, estimate notes, and apply ASO. |
| `assets.py` | Download and verify the released checkpoints. |
| `models/` | [Network implementations](models/README.md). |
| `training_window.py` | Match the note crop geometry used during training. |
| `aso_onset_support.py` | Limit raw proposals to the score onset region before ASO. |
| `audio_windows.py` | Decode a passage from a longer recording. |
| `score_io.py` | Read MIDI and convert score times to a selected passage. |
| `baselines.py` | Score-informed NMF separation. |
| `studio.py`, `studio_static/` | Local editor server, player, piano roll, and example audio. |
| `public_transcription.py` | Request draft scores from the configured transcription service. |
| `transcription_checks.py` | Check transcription size and timing before display. |
| `score_audio_checks.py` | Detect score/audio timing and coverage problems for the editor. |
| `quality_checks.py` | Summarize clipping, silence, and output levels. |
| `runtime_limits.py` | Bound the editor's audio-processing thread count. |
| `project_archive.py` | Save and reopen editor projects as ZIP files. |
| `source_archive.py` | Serve the editor source as a downloadable archive. |
| `legal/` | License texts included with that source archive. |
| `training_data.py` | Download and verify training or benchmark data, restore compressed shards, and write local paths. |
| `training_paths.py` | Resolve stored dataset locations on the current machine. |
| `training_resume.py` | Check that an ASO resume uses the same data and recipe. |
