# Training

Start training with `../train.py` and choose a recipe in `../configs/stage/`.
The recipe launchers validate data and checkpoint inputs before calling the
paper training implementation. Large datasets and initialization checkpoints
are supplied separately.

Download a dataset snapshot from its Hugging Face dataset repository at an
immutable commit, then use the generated path map in the training command:

```sh
notesep download-data --output /path/to/notegrab-data \
  --repo-id ORGANIZATION/DATASET --revision DATASET_COMMIT
```

The command checks every file against the dataset manifest. If you already
have the files, use `--manifest /path/to/manifest.json` instead.

Download the exact NoteGrab step-250 initialization checkpoint:

```sh
python training/download_initial_checkpoint.py --output /path/to/checkpoints
```

Pass `/path/to/checkpoints/0441537c.pt` as `data.initialization` in the
NoteGrab command below.

| File or folder | Purpose |
| --- | --- |
| `train_notegrab.py` | Launches the NoteGrab training recipe. |
| `validate_notegrab.py` | Checks NoteGrab source files and saved model, loss, and sampler settings. |
| `stage_metadata.py` | Installs packed metadata while reading the audio shards in place. |
| `download_initial_checkpoint.py` | Downloads and verifies the NoteGrab starting checkpoint. |
| `train_aso.py` | Validates ASO inputs and launches its training recipe. |
| `run_aso_training.py` | Applies and records deterministic runtime settings for ASO. |
| `aso/` | ASO model, data, and training implementation. |
| `notegrab/` | NoteGrab recipe, models, data, and saved training settings. |

## Run NoteGrab

Install the training dependencies with `python -m pip install -e ".[train]"`,
then point the recipe to the data package and its initialization checkpoint:

```sh
python train.py stage=notegrab run.dir=outputs/notegrab \
  data.root=/path/to/notegrab-data \
  data.initialization=/path/to/initialization.pt
```

## Run ASO

Supply the prepared fresh index, recorded data, proposal data, and the validated
proposal hash:

```sh
python train.py stage=aso run.dir=outputs/aso \
  data.fresh_index=/path/to/fresh-index.json \
  data.recorded_root=/path/to/recorded-data \
  data.proposal_root=/path/to/proposal-data \
  training.proposal_validation_sha256=<validated-sha256>
```

Use `python train.py --cfg job` to inspect the selected settings without
starting training. Each data root must match the expected manifest and hashes.
