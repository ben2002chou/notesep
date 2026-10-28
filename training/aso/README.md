# ASO training

ASO learns to refine a group of NoteGrab estimates using the mixture and score.
Start training through `train.py stage=aso` at the repository root.

| File | Purpose |
| --- | --- |
| `train.py` | Build the model, run AdamW updates, save checkpoints, and evaluate on the development split. |
| `dataset.py` | Join cached NoteGrab estimates with audio targets and construct the sampling schedule. |
| `engine.py` | Training loss, development metrics, and checkpoint writing. |
| `datasets/synthetic.py` | Read synthesized mixtures and their isolated notes. |
| `datasets/recorded.py` | Read recorded score scenes and targets. |
| `datasets/score.py` | Attach complete score events to synthesized audio. |
| `datasets/streaming.py` | Load note crops from packed audio shards. |
| `datasets/shards.py` | Read and write indexed binary audio shards. |
| `datasets/corpus.py` | Resolve corpus manifests and train/validation splits. |
| `datasets/__init__.py` | Dataset package. |

The network lives in `src/notesep/models/`, shared with inference. `train.py`
passes the forward function and waveform objective to `engine.py` explicitly;
the evaluation code can use the normal score or a shuffled score without
changing module globals.

A training record contains complex spectrograms for the mixture `[F, T]`, raw
note estimates `[N, F, T]`, and targets `[N, F, T]`. Here `N` is the number of
notes, `F` frequency bins, and `T` time frames. Waveform targets and acoustic note
boundaries are included for the waveform loss. The STFT uses 16 kHz audio, a
2,048-sample window, and a 512-sample hop.

The loss combines waveform distortion, ownership cross-entropy, and temporal
smoothness. Checkpoints contain model weights, optimizer state, step count,
and the data and recipe settings needed to resume.

Training logs include the latest mixture's `loss`, the `mean_loss` over the
logged window, and `loss_by_source` for synthetic and recorded mixtures.
`loss_window_steps` gives the number of updates in that average.
