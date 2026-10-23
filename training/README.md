# Training

NoteSep has two paper recipes: NoteGrab and ASO. The data is separate from the
code repository. Install the training dependencies before downloading data or
running a recipe:

```sh
uv sync --extra train
```

## Prepare the data

The training dataset has not been published yet. The commands below require
the packed training data and its metadata on disk.

If you already have the data package, check it and generate its local path map:

```sh
uv run notesep download-data --output /path/to/notegrab-data \
  --manifest /path/to/notegrab-data/manifest.json
```

The path map lets the training scripts find the data wherever you store it.

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

ASO trains from scratch. Its inputs are an index of synthesized examples,
recorded audio, and precomputed note proposals:

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
| `notegrab/` | NoteGrab data readers, losses, and training settings. Shared model layers are in `src/notesep/models/`. |
