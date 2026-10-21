# NoteGrab model code

The source manifest checks the exact selected implementation files before they are loaded. `separator/` contains the selected separator and its training code; `gating/` contains the gate implementation used by that model.

| File or folder | Purpose |
| --- | --- |
| `source_manifest.json` | SHA-256 checksums for the selected model source files. |
| `separator/` | Selected separator, its trainer, data readers, and model helpers. |
| `gating/` | Selective-gate model implementation. |
