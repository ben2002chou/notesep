# Training

NoteSep has two paper recipes: NoteGrab and ASO. The data is separate from the
code repository. Install the training dependencies before downloading data or
running a recipe:

```sh
uv sync --extra train
```

## Prepare the data

Download a dataset snapshot from its Hugging Face repository at an immutable
commit. Replace the example repository and commit with the dataset release
details:

```sh
uv run notesep download-data --output /path/to/notegrab-data \
  --repo-id ORGANIZATION/DATASET --revision DATASET_COMMIT
```

The downloader checks each file against the package manifest. For data already
on disk, pass its manifest with `--manifest /path/to/manifest.json`.

## Run NoteGrab

NoteGrab starts from the model checkpoint saved at training step 250. Download
that starting checkpoint:

```sh
uv run python training/download_initial_checkpoint.py --output /path/to/checkpoints
```

Then run the recipe with the data package and checkpoint:

```sh
uv run python train.py stage=notegrab run.dir=outputs/notegrab \
  data.root=/path/to/notegrab-data \
  data.initialization=/path/to/checkpoints/0441537c.pt
```

## Run ASO

ASO trains from scratch. It needs a fresh index, recorded data, proposal data,
and the proposal cache from the validated training package:

```sh
uv run python train.py stage=aso run.dir=outputs/aso \
  data.fresh_index=/path/to/fresh-index.json \
  data.recorded_root=/path/to/recorded-data \
  data.proposal_root=/path/to/proposal-data
```

Use `uv run python train.py --cfg job` to inspect the selected settings. Data
files are checked against the manifests in the package.

## Files

| File or folder | Purpose |
| --- | --- |
| `train_notegrab.py` | Launches the NoteGrab training recipe. |
| `validate_notegrab.py` | Checks the saved NoteGrab recipe and runtime components. |
| `stage_metadata.py` | Installs packed metadata while reading audio shards in place. |
| `download_initial_checkpoint.py` | Downloads and verifies the NoteGrab starting checkpoint. |
| `train_aso.py` | Validates ASO inputs and launches its training recipe. |
| `run_aso_training.py` | Applies and records the ASO numerical runtime settings. |
| `aso/` | ASO model, data, and training implementation. |
| `notegrab/` | NoteGrab recipe, models, data, and saved training settings. |
