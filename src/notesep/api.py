"""Separate an aligned score and edit the resulting note waveforms."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import platform

import numpy as np
import soundfile as sf

from .assets import checkpoint_hashes, model_profile, sha256


@dataclass
class NoteBank:
    """A set of aligned note waveforms and their score events.

    Attributes:
        notes: One dictionary per note, with id, pitch, start, and end.
        stems: Float32 waveforms shaped [notes, samples].
        sample_rate: Samples per second; inference returns 16,000 Hz.
        raw_stems: Optional note estimates before ASO, with the same shape.
        provenance: Input, checkpoint, and window information saved on export.
        saved_edits: Edit controls loaded from a project, keyed by note ID.
    """

    notes: list[dict]
    stems: np.ndarray
    sample_rate: int = 16000
    raw_stems: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    saved_edits: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.saved_edits, dict):
            raise ValueError("Saved edit controls must be a note-ID object")
        self.stems = np.asarray(self.stems, dtype=np.float32)
        if (
            self.stems.ndim != 2
            or len(self.stems) != len(self.notes)
            or not np.isfinite(self.stems).all()
            or self.stems.shape[1] == 0
        ):
            raise ValueError("Expected one finite, aligned waveform per note")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("Expected a positive integer sample rate")
        if len({str(n["id"]) for n in self.notes}) != len(self.notes):
            raise ValueError("Note IDs must be unique")
        if self.raw_stems is not None:
            self.raw_stems = np.asarray(self.raw_stems, dtype=np.float32)
            if (
                self.raw_stems.shape != self.stems.shape
                or not np.isfinite(self.raw_stems).all()
            ):
                raise ValueError(
                    "Raw stems must be finite and aligned with separated stems"
                )

    def render(self, edits: dict[str, dict] | None = None) -> np.ndarray:
        """Direct stem sum with optional pitchShift, delta, stretch, gainDb or mute.

        delta is seconds; stretch > 1 lengthens a note at fixed pitch. Unedited
        stems retain every sample and tail. Transformed notes use the Studio's
        100 ms pre-onset / 450 ms post-offset crop, rendered with librosa rather
        than the browser's SoundTouch. Output expands for delayed/stretched tails.
        """
        edits = edits or {}
        unknown = set(edits) - {str(n["id"]) for n in self.notes}
        if unknown:
            raise ValueError(f"Unknown edit note IDs: {sorted(unknown)}")
        rendered = []
        output_samples = self.stems.shape[1]
        for note, stem in zip(self.notes, self.stems):
            edit = edits.get(str(note["id"]), {})
            allowed = {"pitchShift", "delta", "stretch", "gainDb", "mute"}
            if not isinstance(edit, dict) or set(edit) - allowed:
                raise ValueError(
                    "Edit fields: pitchShift, delta, stretch, gainDb, mute"
                )
            pitch = edit.get("pitchShift", 0)
            delta = edit.get("delta", 0)
            stretch = edit.get("stretch", 1)
            gain = edit.get("gainDb", 0)
            if any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                for x in (pitch, delta, stretch, gain)
            ):
                raise ValueError("Audio edits must be finite numbers")
            if (
                not -12 <= pitch <= 12
                or not 0.5 <= stretch <= 2
                or not -60 <= gain <= 24
            ):
                raise ValueError(
                    "Pitch range ±12 semitones; stretch .5–2; gain −60–24 dB"
                )
            mute = edit.get("mute", False)
            if not isinstance(mute, bool):
                raise ValueError("mute must be boolean")
            if mute:
                continue
            if pitch == 0 and delta == 0 and stretch == 1:
                offset, wave = 0, stem
            else:
                left = max(0, round((note["start"] - 0.10) * self.sample_rate))
                right = min(len(stem), round((note["end"] + 0.45) * self.sample_rate))
                wave = stem[left:right].copy()
                if pitch or stretch != 1:
                    import librosa

                    if pitch:
                        wave = librosa.effects.pitch_shift(
                            wave, sr=self.sample_rate, n_steps=pitch
                        )
                    if stretch != 1:
                        wave = librosa.effects.time_stretch(wave, rate=1 / stretch)
                offset = left + round(delta * self.sample_rate)
                if offset < 0:
                    wave = wave[min(-offset, len(wave)) :]
                    offset = 0
            wave = wave * np.float32(10 ** (gain / 20))
            if not np.isfinite(wave).all():
                raise ValueError("Audio edit produced non-finite samples")
            output_samples = max(output_samples, offset + len(wave))
            rendered.append((offset, wave))
        # Prevent malformed project files from allocating unbounded timelines.
        if output_samples > 600 * self.sample_rate:
            raise ValueError("Edited timeline exceeds ten minutes")
        output = np.zeros(output_samples, dtype=np.float32)
        for offset, wave in rendered:
            output[offset : offset + len(wave)] += wave
        return output

    def export(self, directory: str | Path, *, edits: dict | None = None) -> Path:
        """Write a project to an empty directory and return its project.json path.

        Supplying edits also writes edited.wav. Original note stems are saved
        separately so subsequent edits start from the same audio.
        """
        root = Path(directory)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Output directory must be empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        files = []
        for index, (note, stem) in enumerate(zip(self.notes, self.stems)):
            name = f"note-{index:04d}-midi-{note['pitch']}.wav"
            sf.write(root / name, stem, self.sample_rate, subtype="FLOAT")
            record = {"id": str(note["id"]), "audio": name}
            if self.raw_stems is not None:
                raw_name = f"raw-{index:04d}-midi-{note['pitch']}.wav"
                sf.write(
                    root / raw_name,
                    self.raw_stems[index],
                    self.sample_rate,
                    subtype="FLOAT",
                )
                record["raw_audio"] = raw_name
            files.append(record)
        sf.write(root / "sum.wav", self.render(), self.sample_rate, subtype="FLOAT")
        if edits is not None:
            sf.write(
                root / "edited.wav",
                self.render(edits),
                self.sample_rate,
                subtype="FLOAT",
            )
        (root / "score.json").write_text(json.dumps(self.notes, indent=2) + "\n")
        manifest = {
            "schema": "notesep_project_v1",
            "sample_rate": self.sample_rate,
            "samples": self.stems.shape[1],
            "notes": self.notes,
            "files": files,
            "edits": self.saved_edits if edits is None else edits,
            "provenance": self.provenance,
            "playback": "direct_sum_of_note_estimates",
            "files_sha256": {p.name: sha256(p) for p in root.glob("*.wav")},
        }
        (root / "project.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return root / "project.json"

    @classmethod
    def load(cls, project: str | Path) -> NoteBank:
        """Restore a project, checking that its WAV files match the saved metadata."""
        path = Path(project)
        data = json.loads(path.read_text())
        if data.get("schema") != "notesep_project_v1":
            raise ValueError("Unknown project schema")
        arrays, raw_arrays = [], []
        root = path.parent.resolve()
        for item in data["files"]:
            for kind, collection in [("audio", arrays), ("raw_audio", raw_arrays)]:
                if kind not in item:
                    continue
                audio_path = (root / item[kind]).resolve()
                if audio_path.parent != root:
                    raise ValueError(
                        "Project audio must stay inside the project directory"
                    )
                expected = data["files_sha256"].get(item[kind])
                if not expected or sha256(audio_path) != expected:
                    raise ValueError(f"Project audio checksum mismatch: {item[kind]}")
                wave, rate = sf.read(audio_path, dtype="float32")
                if (
                    rate != data["sample_rate"]
                    or wave.ndim != 1
                    or len(wave) != data["samples"]
                ):
                    raise ValueError(
                        "Project stems must share the declared rate and length"
                    )
                collection.append(wave)
        return cls(
            data["notes"],
            np.stack(arrays),
            data["sample_rate"],
            np.stack(raw_arrays) if raw_arrays else None,
            data.get("provenance", {}),
            data.get("edits", {}),
        )


class NoteSep:
    """Run the released NoteGrab and ASO pipeline on an aligned score.

    Checkpoints are loaded on the first separation call and cached thereafter.
    The auto device preference is CUDA, then Apple MPS, then CPU.
    """

    def __init__(self, device: str = "auto", *, profile: str | None = None):
        if device == "auto":
            import torch

            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("Device must be auto, cpu, cuda, or mps")
        self.device = device
        self.profile = model_profile(profile)

    def separate(
        self,
        audio_path: str | Path,
        notes: str | Path | list[dict],
        *,
        apply_aso: bool = True,
        window_start: float | None = None,
        window_seconds: float | None = None,
        score_timeline: str = "recording",
    ) -> NoteBank:
        """Estimate one waveform per score note.

        Args:
            audio_path: Audio recording to read, mix to mono, and resample.
            notes: MIDI/JSON path or note dictionaries with pitch/start/end.
            apply_aso: Refine the note estimates jointly. A single note uses
                NoteGrab alone because there are no competing notes to allocate.
            window_start: Passage start in seconds, or None for an audio excerpt.
            window_seconds: Passage duration; defaults to 12 when selecting one.
            score_timeline: "recording" for full-recording score times, or
                "window" for times already relative to the selected passage.

        Returns:
            A NoteBank with aligned 16 kHz waveforms and window-relative notes.
        """
        from .inference import RATE, decode_audio, separate, validate_notes
        from .score_io import midi_notes, midi_notes_in_window, score_notes_in_window
        from .audio_windows import decode_audio_window

        path = Path(audio_path)
        if score_timeline not in ("recording", "window"):
            raise ValueError("Score timeline must be recording or window")
        selected_window = window_start is not None or window_seconds is not None
        if selected_window:
            audio, actual_start, source_duration = decode_audio_window(
                path,
                0 if window_start is None else window_start,
                12 if window_seconds is None else window_seconds,
            )
        else:
            audio = decode_audio(path)
            actual_start, source_duration = 0.0, sf.info(path).duration
        if isinstance(notes, (str, Path)):
            score_path = Path(notes)
            if score_path.suffix.lower() in (".mid", ".midi"):
                if selected_window and score_timeline == "recording":
                    notes = midi_notes_in_window(
                        score_path.read_bytes(),
                        source_duration,
                        actual_start,
                        len(audio) / RATE,
                    )
                    score_timeline = "window"
                else:
                    notes = midi_notes(score_path.read_bytes(), len(audio) / RATE)
            else:
                notes = json.loads(score_path.read_text())
        if selected_window and score_timeline == "recording":
            notes = score_notes_in_window(
                notes, source_duration, actual_start, len(audio) / RATE
            )
        notes = validate_notes(notes, len(audio) / RATE)
        if len({n["id"] for n in notes}) != len(notes):
            raise ValueError("Note IDs must be unique")
        # A singleton query extracts a note; allocating a whole mixture among
        # one proposal would obscure that distinction. Match Studio's path.
        apply_aso = bool(apply_aso and len(notes) > 1)
        raw, stems = separate(
            audio, notes, self.device, apply_aso=apply_aso, profile=self.profile
        )
        return NoteBank(
            notes,
            stems,
            RATE,
            raw,
            {
                "input_sha256": sha256(path),
                "checkpoint_sha256": checkpoint_hashes(self.profile),
                "model_profile": self.profile,
                "device": self.device,
                "python": platform.python_version(),
                "window_start_seconds": actual_start,
                "source_duration_seconds": source_duration,
                "window_duration_seconds": len(audio) / RATE,
                "score_output_timeline": "window",
                "boundary_note_policy": "clip_to_window_as_in_studio",
                "apply_aso": apply_aso,
                "acoustic_end_policy": "symbolic_end_plus_300ms_clipped",
                "inference_contract": "target_training_crop_symbolic_end300ms_onset_support100ms_v1",
            },
        )
