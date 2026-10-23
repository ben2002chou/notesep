"""Downloadable projects containing aligned float note stems and provenance."""

from pathlib import Path
import io
import json
import math
import shutil
import stat
import zipfile
import soundfile as sf
import numpy as np
from .api import NoteBank
from .assets import sha256

MAX_PROJECT_BYTES = 180_000_000


def read_project_archive(payload, destination):
    """Read a bounded flat project ZIP, preserving its original aligned stems."""
    if not payload or len(payload) > MAX_PROJECT_BYTES:
        raise ValueError("Project ZIP is empty or too large")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if not 1 <= len(entries) <= 256 or len(set(names)) != len(names):
            raise ValueError("Invalid project member count or duplicate names")
        if sum(item.file_size for item in entries) > MAX_PROJECT_BYTES:
            raise ValueError("Unpacked project is too large")
        for item in entries:
            if (
                Path(item.filename).name != item.filename
                or "\\" in item.filename
                or item.filename in (".", "..")
                or item.is_dir()
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError(
                    "Project ZIP must contain ordinary files in one flat directory"
                )
        if "project.json" not in names:
            raise ValueError("Project ZIP has no project.json")
        if archive.getinfo("project.json").file_size > 1_000_000:
            raise ValueError("Project manifest is too large")
        manifest = json.loads(archive.read("project.json"))
        notes = manifest.get("notes", [])
        files = manifest.get("files", [])
        if not 1 <= len(notes) <= 96 or len(files) != len(notes):
            raise ValueError("Keep 1–96 aligned project notes")
        if (
            manifest.get("sample_rate") != 16000
            or not isinstance(manifest.get("samples"), int)
            or not 17600 <= manifest["samples"] <= 208000
        ):
            raise ValueError("Project must contain a 1.1–13 s mono 16 kHz edit window")
        if [str(row["id"]) for row in files] != [str(note["id"]) for note in notes]:
            raise ValueError("Project note/stem identity order changed")
        wave_names = {
            value
            for row in files
            for key, value in row.items()
            if key in ("audio", "raw_audio")
        }
        if any(
            not isinstance(name, str) or not name.endswith(".wav")
            for name in wave_names
        ):
            raise ValueError("Project stems must be WAV files")
        allowed = wave_names | {
            "project.json",
            "score.json",
            "IMPORT.txt",
            "input.wav",
            "sum.wav",
            "edited.wav",
        }
        if set(names) - allowed:
            raise ValueError("Unexpected project file")
        for item in entries:
            with (
                archive.open(item) as source,
                (root / item.filename).open("xb") as target,
            ):
                shutil.copyfileobj(source, target)
    bank = NoteBank.load(root / "project.json")
    # validate_notes is deliberately imported here: archive reading needs no
    # model initialization, and preserves the bank's original extra metadata.
    from .inference import validate_notes

    validate_notes(bank.notes, bank.stems.shape[1] / bank.sample_rate)
    original = root / "input.wav"
    input_available = original.is_file()
    if input_available:
        expected = manifest.get("files_sha256", {}).get(
            "input.wav"
        ) or bank.provenance.get("input_sha256")
        if not expected or sha256(original) != expected:
            raise ValueError("Original edit-window checksum mismatch")
        info = sf.info(original)
        if (
            info.samplerate != 16000
            or info.channels != 1
            or info.frames != bank.stems.shape[1]
        ):
            raise ValueError("Original edit-window geometry changed")
        if not np.isfinite(sf.read(original, dtype="float32")[0]).all():
            raise ValueError("Original edit window has non-finite samples")
    start = bank.provenance.get("window_start_seconds", 0)
    duration = bank.stems.shape[1] / 16000
    source_duration = bank.provenance.get("source_duration_seconds", start + duration)
    if (
        any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in (start, source_duration)
        )
        or start < 0
        or source_duration < start + duration - 1e-6
    ):
        raise ValueError("Invalid source-window provenance")
    return bank, manifest.get("edits", {}), input_available


def write_project_archive(folder, version, bank):
    folder = Path(folder)
    project = folder / ("project-" + version)
    bank.export(project)
    if (folder / "input.wav").is_file():
        shutil.copy2(folder / "input.wav", project / "input.wav")
        manifest_path = project / "project.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["input.wav"] = sha256(project / "input.wav")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    start = float(
        bank.provenance.get(
            "daw_window_start_seconds", bank.provenance.get("window_start_seconds", 0)
        )
    )
    if "window_start_seconds" in bank.provenance:
        (project / "IMPORT.txt").write_text(
            f"Place the aligned note WAVs at {start:.6f} seconds in your DAW.\n"
            "Each WAV begins at the selected edit window's start, including leading silence.\n"
            "Keep the WAVs aligned with one another. Score times are relative to this window.\n"
            "The edited WAV contains this window; it does not replace the full recording.\n"
        )
    destination = folder / ("bank-" + version + ".zip")
    try:
        with zipfile.ZipFile(
            destination, "x", compression=zipfile.ZIP_STORED
        ) as archive:
            for path in sorted(project.iterdir()):
                archive.write(path, path.name)
    finally:
        shutil.rmtree(project)
    return destination
