# NoteGrab training components

- `batching.py` assembles examples into batches.
- `corpus.py` checks the training corpus and its splits.
- `dna_residual_corpus.py` and `dna_residual_store.py` read paired note data.
- `dna_residual_losses.py` defines the training losses.
- `occurrence_audio_augmentation.py` augments audio and note timings together.
- `occurrence_dual_hpss_separator.py` defines the two-branch separator.
- `occurrence_dual_hpss_symmetric_alignment.py` connects its branches.
- `occurrence_full_score_data.py` adds score context to each example.
- `residual_source_data.py` reads recorded notes and duration data, selects
  synthetic examples, and mixes both sources for training.
- `samplers.py` selects batches and preserves sampler state on resume.
- `streaming_dataset.py` reads score-conditioned examples from disk.
- `training_contract.py` defines model and loss settings.
- `__init__.py` marks this folder as a Python package.
