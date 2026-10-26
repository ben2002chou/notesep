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
│   └── sampling.json      # Training and validation row selection
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

The original NoteGrab parent run trained for 30,000 updates. The starting
checkpoint below comes from that parent followed by a further 250-update stage.
This recipe loads those model weights, starts a new optimizer and runs 5,000
additional updates. It reproduces the continuation stage, not the original
30,000-update parent run. Download the starting checkpoint:

```sh
uv run python training/download_initial_checkpoint.py --output checkpoints
```

Then run the recipe with the data package and checkpoint:

```sh
uv run python train.py stage=notegrab run.dir=outputs/notegrab \
  data.root=data/notegrab \
  data.initialization=checkpoints/initialization.pt
```

## Run ASO

ASO trains from scratch using complete mixtures, isolated targets and NoteGrab
estimates. Its [data download](https://huggingface.co/datasets/ben2002chou/notesep-aso-data) is about 71 GB; restored data occupies about 470 GB.
Allow roughly 550 GB of free disk space for setup. Download and restore the
data, then generate the estimates:

```sh
uv run notesep download
uv run notesep download-data --repo-id ben2002chou/notesep-aso-data --output data/aso
uv run python training/prepare_aso.py --data-root data/aso --output data/proposals
uv run python train.py stage=aso run.dir=outputs/aso \
  data.root=data/aso data.proposal_root=data/proposals
```

Estimate generation uses the released NoteGrab checkpoint downloaded by
`notesep download`. It runs on a CUDA GPU, resumes completed units, and checks
every stored estimate before training can use the cache. The data package
contains both the NoteGrab query index and ASO's measured acoustic endpoint
index; setup selects each automatically. Estimates are generated for the train
and validation selections only.

Use `uv run python train.py --cfg job` to inspect the selected settings. Data
files are checked against the manifests in the package.

## Outputs and implementation

The NoteGrab continuation runs for 5,000 additional updates. It keeps the paper endpoint at
`training/checkpoints/step-000002500.pt` and the final checkpoint at
`training/checkpoints/latest.pt` under `run.dir`, along with the resolved recipe
and training log. ASO writes `RESUME.pt` for automatic resume,
intermediate `checkpoint-step-*.pt` files, and development results under
`run.dir`. Use a new output directory for each NoteGrab run.

Start reading the code at `train.py` in the repository root. It selects one of
the two launchers below. ASO's [implementation guide](aso/README.md) describes
its dataset, loss, and training loop. NoteGrab's saved recipe is in
`notegrab/training_contract.json`; `notegrab/environment.json` contains its
numerical and sampling settings.

NoteGrab's modules apply the sampling and crop steps in this order:
`event_cluster_data` → `track_bound_sampling` → `library_balanced_sampling` →
`target_cropped_training` → `parent_balanced_training`. These are composed into
one trainer by the launcher. They are not separate training recipes.

## Files

| File or folder | Purpose |
| --- | --- |
| `train_notegrab.py` | Launches the NoteGrab training recipe. |
| `validate_notegrab.py` | Checks the saved NoteGrab recipe and runtime components. |
| `stage_metadata.py` | Installs packed metadata while reading audio shards in place. |
| `download_initial_checkpoint.py` | Downloads and verifies the NoteGrab starting checkpoint. |
| `prepare_aso.py` | Generates and checks NoteGrab estimates for ASO. |
| `train_aso.py` | Validates ASO inputs and launches its training recipe. |
| `run_aso_training.py` | Applies and records the ASO numerical runtime settings. |
| `aso/` | ASO model, data, and training implementation. |
| `notegrab/` | NoteGrab data readers, losses, and training settings. Shared model layers are in `src/notesep/models/`. |
