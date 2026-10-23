# NoteSep

NoteSep separates a music recording into individual notes when you give it an aligned score. You can correct the score against the audio, listen to each separated note, then retune, move, mute, or export the notes as audio tracks.

[Paper](https://arxiv.org/abs/2609.29071) · [Inference checkpoints](https://huggingface.co/ben2002chou/notesep) · [Listen to examples](https://benschou.com/notesep/)

This repository includes the Python API, command-line tool, local piano-roll editor, and our score-informed NMF baseline. The example below runs without a neural checkpoint.

## Try the example

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.10 or newer. From the repository root:

```sh
uv sync --extra baseline

uv run notesep baseline src/notesep/studio_static/examples/mixture.wav \
  --score src/notesep/studio_static/examples/score.mid \
  --output example-notes --iterations 1000
uv run notesep render example-notes/project.json --output edited.wav
```

`example-notes/` contains one WAV per note, their sum, and a project you can reopen. `edited.wav` matches the sum until you make an edit.

## Checkpoints

Studio and command-line inference use the released separator and ASO checkpoint
pair. Install the Studio dependencies, then download the pair and check its
file hashes:

```sh
uv sync --extra studio
uv run notesep download
uv run notesep verify
```

## Inference

Start the editor:

```sh
uv run notesep studio
```

In Studio, upload a WAV, FLAC, MP3, or AIFF recording and an aligned MIDI file, or ask for a draft transcription. Fix wrong or missing notes in the piano roll before separating. You can then audition and edit the note audio, download an edited mix, or save the stems and project for later.

If you already have a corrected score, you can work from the command line:

```sh
uv run notesep separate recording.wav --score corrected.mid --output separated
uv run notesep render separated/project.json --output edited.wav
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

## Training

The paper has two training recipes: NoteGrab and ASO. Install the training
dependencies and download the NoteGrab starting checkpoint:

```sh
uv sync --extra train
uv run python training/download_initial_checkpoint.py --output checkpoints
```

NoteGrab's paper recipe starts from `checkpoints/0441537c.pt`, saved at training
step 250. Download it only when reproducing training; normal inference uses the
checkpoint pair above. Then launch NoteGrab with the prepared data package:

```sh
uv run python train.py stage=notegrab run.dir=outputs/notegrab \
  data.root=/path/to/notegrab-data \
  data.initialization=checkpoints/0441537c.pt
```

ASO trains from scratch and needs a fresh index, recorded data, proposal data,
and the proposal validation hash:

```sh
uv run python train.py stage=aso run.dir=outputs/aso \
  data.fresh_index=/path/to/fresh-index.json \
  data.recorded_root=/path/to/recorded-data \
  data.proposal_root=/path/to/proposal-data \
  training.proposal_validation_sha256=<validated-sha256>
```

The [training guide](training/README.md) explains how the data inputs are
checked and how to download a dataset package.

<details>
<summary>Repository layout</summary>

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/notesep/` | Python API, command-line tools, Studio, and inference support |
| `src/notesep/models/` | Released model architecture and checkpoint-loading code |
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
