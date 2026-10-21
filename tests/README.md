# Tests

Run the suite from the repository root with `PYTHONPATH=.:src pytest -q`. The tests cover the public inference and editing behavior as well as training path and data-root helpers.

| File | Purpose |
| --- | --- |
| `test_api.py` | Python API separation, rendering, and input validation. |
| `test_cli.py` | Command-line behavior and argument validation. |
| `test_audio_windows.py` | Audio window selection and reconstruction. |
| `test_notegrab_data_root.py` | NoteGrab data package and path-map setup. |
| `test_packed_metadata_relocation.py` | Safe staging of archived training metadata. |
| `test_project_archive.py` | Project archive serialization. |
| `test_project_reopen.py` | Project archive export and reopening. |
| `test_public_transcription.py` | Hosted transcription request and response handling. |
| `test_quality_checks.py` | Output quality and finite-value checks. |
| `test_saved_project_controls.py` | Persistence of note edits in saved projects. |
| `test_score_editor.mjs` | Browser score-editor interactions. |
| `test_score_io.py` | MIDI and score JSON parsing and writing. |
| `test_setup_data.py` | Dataset download and checksum verification. |
| `test_training_paths.py` | Portable path mapping for training inputs. |
