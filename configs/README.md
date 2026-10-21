# Training configurations

`train.yaml` is the shared Hydra entry configuration. Stage-specific paths and settings are in `stage/`. The exact training inputs are supplied by the user; these files do not contain machine-specific data locations.

| File or folder | Purpose |
| --- | --- |
| `train.yaml` | Selects the default stage and output directory settings. |
| `stage/` | One configuration per supported training stage. |
