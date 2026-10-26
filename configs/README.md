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

- **NoteGrab:** `data.root` points to the prepared data package. The default
  `training.recipe=full` runs the parent, adaptation and continuation in order.
  Their model, loss and sampler settings are in `training/notegrab/`.
- **ASO:** `data.root` points to its data package, `data.proposal_root` to the
  generated NoteGrab estimates, and `data.separator` to the checkpoint that
  produced those estimates. `training.seed` controls initialization and sampling.

For working on a single NoteGrab stage, set `training.recipe` to `parent`,
`adaptation`, or `continuation`. The latter two also need `data.initialization`.
`run.dry_run=true` checks a single stage without updating weights; ASO's dry run
checks input files, while its full input checks run at training startup.

For an interrupted NoteGrab run, repeat its command with `run.resume=true`.
This restores the saved model, optimizer, random states and step count.
