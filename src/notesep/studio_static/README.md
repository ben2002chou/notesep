# Studio browser files

`index.html` and `note-editor.html` are the editor pages. JavaScript modules implement audio editing, import/export, score editing, and playback; CSS files control the editor layout and interaction states.

| File or folder | Purpose |
| --- | --- |
| `index.html` | Main Studio interface. |
| `note-editor.html` | Note-level editor page. |
| `note-editor.js` | Note editor interactions. |
| `score-edit.mjs` | Score and piano-roll editing. |
| `audio-edit-values.mjs` | Applies note edit values to audio. |
| `audio-export.mjs` | Exports edited audio. |
| `wav-decode.mjs` | Decodes WAV data in the browser. |
| `space.js` | Space-level editor behavior. |
| `note-editor.css`, `note-editor-layout.css` | Note editor appearance and layout. |
| `space.css`, `space-layout.css`, `space-editor.css` | Space editor appearance and layout. |
| `draw-mode.css` | Note-drawing mode styles. |
| `upload-feedback.css` | Upload progress and error styles. |
| `assets/` | Asset integrity manifest. |
| `examples/` | Small audio and MIDI example for local testing. |
| `vendor/` | Required third-party browser libraries and notices. |
