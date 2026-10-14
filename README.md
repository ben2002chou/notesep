# NoteSep inference and editing code

This repository contains the first-party NoteSep Python inference, audio/score editing, and project export code, licensed under AGPL-3.0-only. Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Model weights

No trained checkpoint, research recording, or evaluation dataset is included. The selected checkpoints remain unavailable for public redistribution while their complete training-source admission and asset-rights review is unresolved. The older Studio preview weights have the same unresolved lineage issue. Do not treat the code license as a license for model weights. Neural inference requires weights supplied under separate authorization; `notesep` verifies their pinned hashes before loading.

## Install

Use Python 3.10 or newer in a virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[studio,baseline,test]"
python -m pip check
```

The self-contained NMF example runs without neural weights:

```sh
notesep baseline src/notesep/studio_static/examples/mixture.wav \
  --score src/notesep/studio_static/examples/score.mid \
  --output example-notes --iterations 1000
notesep render example-notes/project.json --output edited.wav
```

To use the neural Studio, configure authorized weights at `NOTESEP_WEIGHTS`, then run `notesep verify` and `notesep studio`. This repository does not grant access to those weights.

## Scope

This is an inference/runtime code release, not a complete paper-reproduction release. Training/evaluation pipelines, private authorities, benchmark datasets, scores, and checkpoints are kept out of this minimal branch. The included original synthesized example is only an editing smoke test; it is not a transcription-accuracy or listening-quality result.
