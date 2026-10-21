# ASO training

This folder contains the paper's acoustic source ownership recipe. `train_aso.py` is the command-line wrapper; `aso_engine.py` and `aso_data.py` provide the portable training and input setup.

| File or folder | Purpose |
| --- | --- |
| `aso_data.py` | Resolves and validates the ASO input datasets. |
| `aso_engine.py` | Runs the configured training stage. |
| `source_manifest.json` | Hashes for the selected ASO code, protocol, and path helpers. |
| `docs/` | Training protocol and its amendments. |
| `scripts/` | The selected paper training implementation. |
| `src/` | Model and dataset modules imported by the trainer. |
