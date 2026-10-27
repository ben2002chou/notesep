# Training

NoteSep has two paper recipes: NoteGrab and ASO. The data is separate from the
code repository. Install the training dependencies before downloading data or
running a recipe:

```sh
uv sync --extra train
```

The training extra pins PyTorch 2.3.0 and the numerical libraries used by the
verified runs. uv selects Python 3.11 from `.python-version`.

## Prepare the data

Download and restore the [NoteGrab data](https://huggingface.co/datasets/ben2002chou/notesep-training-data):

```sh
uv run notesep download-data --repo-id ben2002chou/notesep-training-data --output data/notegrab
```

Setup checks the files, restores the audio and creates a path map so the loader
can find the data wherever you store it.

The release package groups the data by its role:

```text
notesep-data/
├── manifest.json
├── synthetic/
│   ├── index.sqlite       # Two million mixture/target training rows
│   ├── shards/            # Audio blobs, distributed as .blob.gz
│   ├── timing/            # Duration and acoustic note-end information
│   ├── metadata.zip       # Per-shard lookup tables, opened by the loader
│   ├── sampling-parent.json # Original parent row selection
│   ├── sampling-adaptation.json # Native-library adaptation rows
│   ├── render-methods.npz    # Instrument-library labels for adaptation
│   └── sampling.json        # Later continuation row selection
├── recorded/
│   └── shards/            # Eight recorded-scene shards with their indexes
└── example/               # One unpacked mixture, target, reference and labels
```

The setup command checks the download and restores `.blob.gz` files to their
original float32 audio bytes. Compressed data occupies about 446 GB; the
restored training data needs about 2.39 TB. Allow space for both during setup.
The small `example/` folder is for listening and inspection; it is not a
source-complete ASO training scene. ASO uses its own data and proposal cache.

## Run NoteGrab

```sh
uv run python train.py stage=notegrab run.dir=outputs/notegrab data.root=data/notegrab
```

The launcher runs three stages in order. Each later stage loads the preceding
model weights and starts a new optimizer.

| Stage | Updates | Learning rate | Data selection |
| --- | ---: | ---: | --- |
| Parent | 30,000 | 0.0002 | Synthetic mixtures, `sampling-parent.json`. |
| Adaptation | 250 | 0.000002 | Native instrument libraries, `sampling-adaptation.json`. |
| Continuation | 5,000 | 0.000002 | Balanced synthetic and recorded crops, `sampling.json`. |

The continuation retains step 2,500, which is the paper's selected checkpoint.
It also finishes the 5,000-update budget. Checkpoints and logs are grouped under
`outputs/notegrab/parent`, `adaptation`, and `continuation`.
`outputs/notegrab/result.json` gives the selected checkpoint path.

To resume an interrupted run, repeat the command with `run.resume=true`.
Completed stages are skipped; the interrupted stage restores its model,
optimizer, random states and step count.

## Run ASO

ASO learns to refine complete groups of NoteGrab estimates. Its
[data download](https://huggingface.co/datasets/ben2002chou/notesep-aso-data)
is about 71 GB; restored data occupies about 470 GB. Allow roughly 550 GB for setup.

```sh
uv run notesep download-data --repo-id ben2002chou/notesep-aso-data --output data/aso
uv run python training/prepare_aso.py --data-root data/aso --output data/proposals \
  --checkpoint outputs/notegrab/continuation/training/checkpoints/step-000002500.pt
uv run python train.py stage=aso run.dir=outputs/aso \
  data.root=data/aso data.proposal_root=data/proposals \
  data.separator=outputs/notegrab/continuation/training/checkpoints/step-000002500.pt
```

Estimate generation runs on a CUDA GPU and resumes completed units. It preserves
the original NoteGrab query times. ASO then reads the separate measured acoustic
endpoints for training. All cached estimates are checked before use; the cache
must come from the checkpoint supplied to ASO training. Estimates cover only the
training and validation selections.

ASO runs 29,058 updates. It saves `RESUME.pt` for automatic resume and
`checkpoint-step-029058.pt` at the final step. Repeating its command resumes an
interrupted run.

## Use the trained models

Place the selected NoteGrab and final ASO checkpoints in one folder:

```sh
mkdir -p outputs/weights
cp outputs/notegrab/continuation/training/checkpoints/step-000002500.pt outputs/weights/separator.pt
cp outputs/aso/checkpoint-step-029058.pt outputs/weights/aso.pt
uv run notesep separate src/notesep/studio_static/examples/mixture.wav \
  --score src/notesep/studio_static/examples/score.mid \
  --weights outputs/weights --output outputs/separated
```

In Python, use `NoteSep(weights="outputs/weights")`.

## Implementation

Start at `train.py`. [Hydra configs](../configs/README.md) select the inputs and
output directory. `notegrab/parent_contract.json`, `adaptation_contract.json`
and `training_contract.json` specify the three NoteGrab stages.
`notegrab/environment.json` supplies their shared model and loss settings.

The parent uses the duration-budget sampler. Adaptation balances native
instrument libraries through `native_sampling.py`. The continuation composes
`event_cluster_data` → `track_bound_sampling` → `library_balanced_sampling` →
`target_cropped_training` → `parent_balanced_training`.
These modules form one continuation trainer. ASO's [implementation guide](aso/README.md)
describes its dataset, loss and training loop.

## Files

| File or folder | Purpose |
| --- | --- |
| `train_notegrab.py` | Launches the NoteGrab training recipe. |
| `validate_notegrab.py` | Checks the saved NoteGrab recipe and runtime components. |
| `stage_metadata.py` | Installs packed metadata while reading audio shards in place. |
| `prepare_aso.py` | Generates and checks NoteGrab estimates for ASO. |
| `train_aso.py` | Validates ASO inputs and launches its training recipe. |
| `run_aso_training.py` | Applies and records the ASO numerical runtime settings. |
| `aso/` | ASO model, data, and training implementation. |
| `notegrab/` | NoteGrab data readers, losses, and training settings. Shared model layers are in `src/notesep/models/`. |
