# NoteSep

NoteSep separates a music recording into individual notes when you give it an aligned score. You can correct the score against the audio, listen to each separated note, then retune, move, mute, or export the notes as audio tracks.

[Paper](https://arxiv.org/abs/2609.29071) · [Checkpoints](https://huggingface.co/ben2002chou/notesep) · [Listen to examples](https://benschou.com/notesep/)

This repository includes the Python API, command-line tool, local piano-roll editor, and our score-informed NMF baseline. The example below runs without a neural checkpoint.

## Try the example

Use Python 3.10 or newer. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[baseline]"

notesep baseline src/notesep/studio_static/examples/mixture.wav \
  --score src/notesep/studio_static/examples/score.mid \
  --output example-notes --iterations 1000
notesep render example-notes/project.json --output edited.wav
```

`example-notes/` contains one WAV per note, their sum, and a project you can reopen. `edited.wav` matches the sum until you make an edit.

## Edit your own music

To run the neural model locally, install the editor and download the [paper checkpoint pair](https://huggingface.co/ben2002chou/notesep):

```sh
python -m pip install -e ".[studio]"
notesep download
notesep verify
notesep studio
```

In Studio, upload a WAV, FLAC, MP3, or AIFF recording and an aligned MIDI file, or ask for a draft transcription. Fix wrong or missing notes in the piano roll before separating. You can then audition and edit the note audio, download an edited mix, or save the stems and project for later.

If you already have a corrected score, you can work from the command line:

```sh
notesep separate recording.wav --score corrected.mid --output separated
notesep render separated/project.json --output edited.wav
```

Use `--window-start` and `--window-seconds` to select a 1.1–12 second passage from a longer recording. Scores can be MIDI or JSON; by default, their times refer to the full recording. See `notesep separate --help` for the remaining options.

<details>
<summary>Python API and output files</summary>

## Python API

```python
from notesep import NoteSep
import soundfile as sf

bank = NoteSep(device="auto").separate("recording.wav", "corrected.mid")
bank.export("separated")

note_id = str(bank.notes[0]["id"])
edited = bank.render({note_id: {"pitchShift": 2, "delta": 0.05}})
sf.write("edited.wav", edited, bank.sample_rate)
```

`pitchShift` is in semitones and `delta` is in seconds. You can also change `stretch` and `gainDb`, or set `mute`. The render is a direct sum of the note estimates.

## Outputs

Each project contains aligned 16 kHz mono WAVs (`note-0000-midi-<pitch>.wav`, etc.), `sum.wav`, `score.json`, and `project.json`. Neural projects may also contain `raw-*.wav` estimates from before joint separation. You can import the note WAVs into a DAW or reopen the project to keep editing.

</details>

<details>
<summary>Training and repository map</summary>

Paper training uses separate datasets and initialization checkpoints. The
training guide shows how to download a dataset snapshot and run each recipe:
[`training/README.md`](training/README.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/notesep/` | Python API, command-line tools, Studio, and inference support |
| `src/model_exp/` | Model architecture and checkpoint-loading code used by inference |
| `configs/` | Hydra training recipes and their stage settings |
| `training/` | Paper training launchers, data readers, samplers, and source checks |
| `train.py` | Selects and launches a configured training recipe |
| `pyproject.toml` | Package metadata and optional dependency groups |
| `THIRD_PARTY_NOTICES.md` | Licenses and notices for included third-party code |
| `CITATION.cff` | Citation metadata for the paper and software |
| `LICENSE` | License for this repository |
| `LICENSING.md` | Licensing notes for the repository and bundled assets |
| `.gitignore` | Excludes local environments, caches, and generated files from Git |

</details>

## Citation

If you use NoteSep in research, please cite the [paper](https://arxiv.org/abs/2609.29071) or use [CITATION.cff](CITATION.cff). The code is [AGPL-3.0-only](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) are included.
