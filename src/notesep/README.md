# NoteSep package

| File or folder | Purpose |
| --- | --- |
| `__init__.py` | Exports the Python API. |
| `api.py` | User-facing NoteSep separation and editing API. |
| `aso_onset_support.py` | Onset features used by the ASO model. |
| `assets.py` | Downloads and verifies model checkpoints. |
| `audio_windows.py` | Selects and restores audio windows. |
| `baselines.py` | Score-informed NMF baseline. |
| `cli.py` | Command-line entry points. |
| `inference.py` | Loads checkpoints and runs note separation. |
| `local_transcription.py` | Optional local transcription integration. |
| `public_transcription.py` | Hosted transcription client. |
| `project_archive.py` | Saves and reopens editable projects. |
| `quality_checks.py` | Validates separation outputs. |
| `runtime_limits.py` | Defines inference resource limits. |
| `score_audio_checks.py` | Checks score and audio compatibility. |
| `score_io.py` | Reads and writes MIDI and score JSON. |
| `source_archive.py` | Exports stems and project assets. |
| `studio.py` | Serves the local editor. |
| `training_data.py` | Verifies and prepares training dataset downloads. |
| `training_paths.py` | Maps portable training paths to local roots. |
| `training_resume.py` | Checks scientific identity when relocating training resumes. |
| `training_window.py` | Computes training audio crop geometry. |
| `transcription.py` | Transcription interface and result normalization. |
| `transcription_checks.py` | Validates transcription output. |
| `legal/` | Third-party licensing materials. |
| `studio_static/` | Browser editor and its bundled assets. |
