# NoteSep

Score-informed note separation: turn a recording and its aligned MIDI score into
individual note waveforms.

[Paper](https://arxiv.org/abs/2609.29071) · [Checkpoints](https://huggingface.co/ben2002chou/notesep) · [Audio examples](https://benschou.com/notesep/)

NoteSep uses **NoteGrab** to estimate each note, then **ASO** to refine the estimates
together. This repository contains the training recipes, pretrained inference,
and our score-informed NMF baseline.

## Installation

Use Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run these commands from the repository root:

```sh
uv sync
uv run notesep download
```

The download command retrieves and checks both inference checkpoints. They are
cached in `~/.cache/notesep/weights/paper-table1/`. Set `NOTESEP_WEIGHTS` to use
another directory containing `separator.pt` and `aso.pt`.

## Inference

Run the bundled example:

```sh
uv run notesep separate src/notesep/studio_static/examples/mixture.wav \
  --score src/notesep/studio_static/examples/score.mid --output separated
uv run notesep render separated/project.json --output reconstructed.wav
```

For your own recording, replace the audio and MIDI paths. The score supplies each
note's pitch, onset, and offset; it should already be aligned to the recording.
The output directory contains one WAV per note, `sum.wav`, and `project.json`.
The WAVs are aligned, mono, and sampled at 16 kHz.

For a longer recording, choose a passage:

```sh
uv run notesep separate recording.wav --score score.mid --output passage \
  --window-start 30 --window-seconds 12
```

Choose a 1.1–12 second passage with at most 96 notes. MIDI times refer to the full
recording. JSON scores are also supported; see the [Python API and score format](src/notesep/README.md).

## Training

Install the training dependencies:

```sh
uv sync --extra train
```

**NoteGrab:** download the data and train from scratch. The command runs the
30,000-update parent, 250-update adaptation, and 5,000-update continuation.

```sh
uv run notesep download-data --repo-id ben2002chou/notesep-training-data --output data/notegrab
uv run python train.py stage=notegrab run.dir=outputs/notegrab data.root=data/notegrab
```

The paper uses the continuation checkpoint at step 2,500:
`outputs/notegrab/continuation/training/checkpoints/step-000002500.pt`.

**ASO:** download its data, generate estimates with your trained NoteGrab, then train:

```sh
uv run notesep download-data --repo-id ben2002chou/notesep-aso-data --output data/aso
uv run python training/prepare_aso.py --data-root data/aso --output data/proposals \
  --checkpoint outputs/notegrab/continuation/training/checkpoints/step-000002500.pt
uv run python train.py stage=aso run.dir=outputs/aso \
  data.root=data/aso data.proposal_root=data/proposals \
  data.separator=outputs/notegrab/continuation/training/checkpoints/step-000002500.pt
```

Training uses CUDA and [Hydra configs](configs/README.md). Both datasets are public downloads. See the [training guide](training/README.md)
for disk requirements, data layouts and checkpoint outputs.

## Datasets

Training and evaluation are separate downloads. NoteGrab uses mixture/target
pairs ([data](https://huggingface.co/datasets/ben2002chou/notesep-training-data)); ASO uses complete groups of isolated notes
([data](https://huggingface.co/datasets/ben2002chou/notesep-aso-data)). The [training guide](training/README.md)
describes their layouts and setup.

[SCNS-Eval and SCNS-Edit](https://huggingface.co/datasets/ben2002chou/notesep-benchmarks)
share 144 recordings and 3,473 isolated note references:

```sh
uv run notesep download-data --repo-id ben2002chou/notesep-benchmarks --output data/benchmark
```

`evaluation.json` describes note requests and scoring crops. `editing.json`
contains the fixed gain, pitch, timing and tuning edits. Each `pieces/` folder
has its mixture, aligned score and isolated-note archive. The public benchmark
includes 11 replacement recordings; the released checkpoints score 7.44 dB
median SI-SDR on this version. Keep this benchmark out of training.

To evaluate the paper checkpoints on SCNS-Eval:

```sh
uv run python -m notesep.evaluate data/benchmark --output results/evaluation.json
```

The output contains per-note scores and the summary metrics from the paper.
For PHENICX and Bach10, see the [instrument evaluation instructions](src/notesep/README.md#instrument-evaluation).
To evaluate your own checkpoints, pass `--weights` with a directory containing
`separator.pt` and `aso.pt`.

## Editing notes

The optional Studio provides an audio player and piano-roll editor:

```sh
uv sync --extra studio
uv run notesep studio
```

Upload audio and a MIDI score, or request a draft transcription. Correct the
score before separating, then listen to, move, retune, mute, or export the notes.
The [Python API](src/notesep/README.md) also supports editing and project export.

<details>
<summary>NMF baseline</summary>

This baseline uses the score to guide nonnegative matrix factorization and needs
no neural checkpoint:

```sh
uv sync --extra baseline
uv run notesep baseline src/notesep/studio_static/examples/mixture.wav \
  --score src/notesep/studio_static/examples/score.mid \
  --output nmf-notes --iterations 1000
```

It writes the same project format as neural inference.
On the public SCNS-Eval benchmark, this baseline scores 1.84 dB median SI-SDR.

</details>

## Code layout

| Path | Contents |
| --- | --- |
| [`train.py`](train.py) | Select and launch a training recipe. |
| [`configs/`](configs/README.md) | Hydra settings for NoteGrab and ASO. |
| [`training/`](training/README.md) | Training loops, datasets, and losses. |
| [`src/notesep/`](src/notesep/README.md) | Inference, Python API, CLI, and editor. |
| [`src/notesep/models/`](src/notesep/models/README.md) | Network layers shared by training and inference. |
| `pyproject.toml`, `uv.lock` | Dependencies, package settings, and locked environment. |
| `.python-version` | Python 3.11 selected by uv. |
| `.gitignore` | Local data, environments, and generated output exclusions. |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | Code license and notices for bundled code. |
| `CITATION.cff` | Paper citation. |

## Citation

If you use NoteSep, please cite the [paper](https://arxiv.org/abs/2609.29071)
or use [CITATION.cff](CITATION.cff). The code is [AGPL-3.0-only](LICENSE).
