# Training configuration

`train.yaml` selects a stage and sets the output directory. `stage/notegrab.yaml`
and `stage/aso.yaml` list the inputs for each recipe. Values marked `???` must be
provided on the command line.

Inspect either recipe without starting training:

```sh
uv run python train.py --cfg job stage=notegrab
uv run python train.py --cfg job stage=aso
```

Both stages use `run.dir` for checkpoints and logs. Hydra saves its resolved
configuration beside that directory, at `<run.dir>.hydra/`.

- **NoteGrab:** `data.root` is the prepared data package; `data.initialization`
  is the step-250 checkpoint. Model, loss, and sampler settings come from
  `training/notegrab/training_contract.json` and `environment.json`.
- **ASO:** `data.fresh_index`, `data.recorded_root`, and `data.proposal_root`
  identify the synthesized audio, recorded audio, and cached note estimates.
  `training.seed` controls random initialization and sampling. If the data
  package contains a relocation map, pass it as `data.path_map`.

`run.dry_run=true` checks the launcher inputs without updating model weights.
NoteGrab also checks the initialization checkpoint and data bindings. ASO's dry
run only checks the required files exist; its full input checks run at startup.
