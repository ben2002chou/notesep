# ASO model and data modules

| File | Purpose |
| --- | --- |
| `__init__.py` | Declares the model module package. |
| `acoustic_multiplicative_temporal_ownership.py` | Acoustic ownership model used by the selected recipe. |
| `corpus.py` | Validates the frozen synthetic corpus and split indexes. |
| `streaming_dataset.py` | Reads score-addressed examples from the frozen corpus. |
| `fresh_source_complete_data.py` | Loads the fresh-source training examples. |
| `occurrence_full_score_data.py` | Builds score-conditioned training examples. |
| `packed_blob_shards.py` | Reads audio records from packed blob shards. |
| `recorded_natural_score_scene_data.py` | Loads recorded natural-score scenes. |
| `simplex.py` | Implements constrained mixture weights. |
| `temporal_hybrid_set_ownership.py` | Provides the temporal hybrid ownership model. |
