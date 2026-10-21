# Training

Start training with `../train.py` and choose a recipe in `../configs/stage/`.
The recipe launchers validate data and checkpoint inputs before calling the
paper training implementation. Large datasets and initialization checkpoints
are supplied separately.

| File or folder | Purpose |
| --- | --- |
| `train_notegrab.py` | Launches the NoteGrab training recipe. |
| `validate_notegrab.py` | Checks NoteGrab source files and saved model, loss, and sampler settings. |
| `stage_metadata.py` | Installs packed metadata while reading the audio shards in place. |
| `download_initial_checkpoint.py` | Downloads and verifies the NoteGrab starting checkpoint. |
| `train_aso.py` | Validates ASO inputs and launches its training recipe. |
| `run_aso_deterministic.py` | Applies and records deterministic runtime settings for ASO. |
| `scan_aso_sampler.py` | Defines the fixed-order sampler operation used by deterministic ASO runs. |
| `aso/` | ASO model, data, and training implementation. |
| `notegrab/` | NoteGrab recipe, models, data, and saved training settings. |
