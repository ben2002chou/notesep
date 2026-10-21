# NoteGrab training

`training_contract.json` records the selected model, loss, sampler, and training settings checked by the validator. The stage configuration in `configs/stage/notegrab.yaml` is the user-facing launch configuration. Data paths use portable roots and are resolved through the supplied path map. `environment.json` records the model settings used by this recipe.

The dataset package's `path-map.json` maps `/note-sep-data/synthetic` to the
synthetic corpus, `/note-sep-data/recorded` to the recorded corpus, and
`/note-sep-checkpoints` to the initialization checkpoint files. These names are
stable logical roots; the paths on a user's machine are supplied by the map.

| File or folder | Purpose |
| --- | --- |
| `training_contract.json` | Saved NoteGrab training and model contract checked by the validator. |
| `environment.json` | Model and loss settings needed to match the selected recipe. |
| `models/` | Separator and gating model code used by training. |
| `data/` | Dataset, crop, and sampling logic used by training. |
