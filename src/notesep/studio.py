"""NoteSep Studio: score-query inference and direct-sum note editing."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import spaces
from gradio import Server
import numpy as np
import soundfile as sf

from notesep.audio_windows import MAX_WINDOW_SECONDS, decode_audio_window
from notesep.inference import RATE, decode_audio, load_models, separate, validate_notes
from notesep.public_transcription import (
    transcribe_with_public_muscriptor,
    transcribe_with_yourmt3,
)
from notesep.quality_checks import separated_coverage
from notesep.runtime_limits import separation_gpu_duration
from notesep.score_audio_checks import audit_score_audio, audit_separated_notes
from notesep.score_io import (
    MAX_MIDI_BYTES,
    midi_note_event_count,
    midi_notes,
    midi_notes_in_window,
)
from notesep.transcription_checks import compare_transcriptions
from notesep.api import NoteBank
from notesep.assets import checkpoint_hashes, model_profile, sha256
from notesep.project_archive import (
    write_project_archive,
    read_project_archive,
    MAX_PROJECT_BYTES,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "studio_static"
RUNTIME = Path(os.environ.get("NOTESEP_RUNTIME", Path("/tmp") / "notesep-studio"))
RUNTIME.mkdir(exist_ok=True)
SESSION_RE = re.compile(r"[a-f0-9]{32}\Z")
MAX_AUDIO_BYTES = 128_000_000
LIFETIME = 3600
MAX_OUTPUT_GENERATIONS = 3
DEVICE = os.environ.get("NOTESEP_DEVICE", "cuda")
PROFILE = model_profile()

# ZeroGPU stages this CUDA placement during startup; it is not a lazy transfer
# inside the GPU function. Locally, NOTESEP_DEVICE=mps or cpu also works.
load_models(DEVICE, PROFILE)

app = Server()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def cleanup():
    now = time.time()
    for folder in RUNTIME.iterdir():
        if (
            folder.is_dir()
            and SESSION_RE.fullmatch(folder.name)
            and now - folder.stat().st_mtime > LIFETIME
        ):
            shutil.rmtree(folder)


def cleanup_loop():
    while True:
        time.sleep(60)
        try:
            cleanup()
        except OSError:
            logger.warning("Could not clean up expired Studio sessions", exc_info=True)


threading.Thread(
    target=cleanup_loop, name="notesep-session-cleanup", daemon=True
).start()


def session_folder(session: str) -> Path:
    if not isinstance(session, str) or not SESSION_RE.fullmatch(session):
        raise ValueError("Invalid session ID")
    folder = RUNTIME / session
    if not folder.is_dir():
        raise ValueError("This upload has expired. Please upload it again.")
    folder.touch()
    return folder


def prune_old_outputs(folder: Path, keep: int = MAX_OUTPUT_GENERATIONS) -> None:
    """Bound per-session storage while preserving a few immutable edit runs."""
    generations: dict[str, list[Path]] = {}
    for path in folder.iterdir():
        match = re.fullmatch(
            r"(?:(?:note|raw)-([a-f0-9]{12})-\d{1,2}\.wav|bank-([a-f0-9]{12})\.zip)",
            path.name,
        )
        if match and path.is_file():
            generations.setdefault(match.group(1) or match.group(2), []).append(path)
    newest = sorted(
        generations.values(),
        key=lambda paths: max(path.stat().st_mtime_ns for path in paths),
        reverse=True,
    )
    for paths in newest[keep:]:
        for path in paths:
            path.unlink(missing_ok=True)


def decode_base64(data: str, limit: int) -> bytes:
    if not isinstance(data, str) or len(data) > limit * 4 // 3 + 256:
        raise ValueError("File is too large")
    payload = data.partition(",")[2] if data.startswith("data:") else data
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid file data") from error
    if not decoded or len(decoded) > limit:
        raise ValueError("File is empty or too large")
    return decoded


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HTMLResponse(
        (STATIC / "index.html").read_text(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/notesep-source.zip")
async def serving_source():
    from notesep.source_archive import build_source_archive

    return Response(
        build_source_archive(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="notesep-source.zip"'},
    )


@app.get("/notesep-config")
async def public_config():
    """Expose only non-secret transcription settings needed by the UI."""
    providers = [
        {"id": "yourmt3", "label": "YourMT3+ · public"},
        {"id": "muscriptor_public", "label": "MuScriptor · public"},
    ]
    return JSONResponse(
        {
            "provider": "YourMT3+",
            "defaultProvider": "yourmt3",
            "providers": providers,
            "instrumentHint": True,
            "ready": True,
            "fileLimits": {
                "audioBytes": MAX_AUDIO_BYTES,
                "midiBytes": MAX_MIDI_BYTES,
                "projectBytes": MAX_PROJECT_BYTES,
            },
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/result/{session}/{filename}")
async def result_file(session: str, filename: str):
    try:
        folder = session_folder(session)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if not re.fullmatch(
        r"(?:(?:input|(?:note|raw)-[a-f0-9]{12}-\d{1,2})\.wav|(?:bank|project)-[a-f0-9]{12}\.zip)",
        filename,
    ):
        raise HTTPException(status_code=404)
    path = folder / filename
    if not path.is_file():
        raise HTTPException(status_code=404)
    if filename.endswith(".zip"):
        return FileResponse(
            path,
            media_type="application/zip",
            filename="notesep-notes.zip",
            headers={"Cache-Control": "private, no-store"},
        )
    return FileResponse(
        path, media_type="audio/wav", headers={"Cache-Control": "private, no-store"}
    )


@app.api(name="prepare", concurrency_limit=1)
def prepare(
    audio_base64: str,
    filename: str,
    midi_base64: str = "",
    window_start: float = 0.0,
    window_seconds: float = MAX_WINDOW_SECONDS,
) -> dict:
    """Load a bounded edit window from a full source recording and optional score."""
    cleanup()
    suffix = Path(filename).suffix.lower()
    if suffix not in (".wav", ".flac", ".mp3", ".aif", ".aiff"):
        raise ValueError("Please upload WAV, FLAC, MP3 or AIFF audio")
    payload = decode_base64(audio_base64, MAX_AUDIO_BYTES)
    session = uuid4().hex
    folder = RUNTIME / session
    folder.mkdir()
    try:
        source = folder / ("upload" + suffix)
        source.write_bytes(payload)
        audio, actual_window_start, source_duration = decode_audio_window(
            source, window_start, window_seconds
        )
        duration = len(audio) / RATE
        sf.write(folder / "input.wav", audio, RATE, subtype="FLOAT")
        context = {
            "source_recording_sha256": hashlib.sha256(payload).hexdigest(),
            "window_start_seconds": actual_window_start,
            "source_duration_seconds": source_duration,
            "window_duration_seconds": duration,
            "score_timeline": "window",
        }
        (folder / "window.json").write_text(json.dumps(context) + "\n")
        source.unlink()
        if midi_base64:
            score = midi_notes_in_window(
                decode_base64(midi_base64, MAX_MIDI_BYTES),
                source_duration,
                actual_window_start,
                duration,
            )
        else:
            score = []
        audio_check = audit_score_audio(audio, score, RATE) if score else None
        return {
            "session": session,
            "duration": duration,
            "sourceDuration": source_duration,
            "windowStart": actual_window_start,
            "notes": score,
            "audioCheck": audio_check,
            "audioUrl": f"/result/{session}/input.wav",
        }
    except Exception:
        shutil.rmtree(folder)
        raise


@app.api(name="restore_project")
def restore_project(project_base64: str, filename: str) -> dict:
    """Reopen saved estimates without inference or external transcription."""
    if Path(filename).suffix.lower() != ".zip":
        raise ValueError("Choose a NoteSep project ZIP")
    payload = decode_base64(project_base64, MAX_PROJECT_BYTES)
    cleanup()
    session = uuid4().hex
    folder = RUNTIME / session
    folder.mkdir()
    try:
        bank, edits, has_input = read_project_archive(payload, folder / "imported")
        version = uuid4().hex[:12]
        output = []
        for index, note in enumerate(bank.notes):
            final = f"note-{version}-{index}.wav"
            sf.write(folder / final, bank.stems[index], RATE, subtype="FLOAT")
            raw = f"raw-{version}-{index}.wav"
            sf.write(
                folder / raw,
                bank.raw_stems[index]
                if bank.raw_stems is not None
                else bank.stems[index],
                RATE,
                subtype="FLOAT",
            )
            output.append(
                {
                    **note,
                    "event": index,
                    "qualityFlags": [],
                    "outputs": {
                        "aso": {"audio": f"/result/{session}/{final}"},
                        "notegrab": {"audio": f"/result/{session}/{raw}"},
                    },
                }
            )
        if has_input:
            shutil.copy2(folder / "imported/input.wav", folder / "input.wav")
        archive = folder / f"bank-{version}.zip"
        archive.write_bytes(payload)
        context = {
            key: value
            for key, value in bank.provenance.items()
            if key
            in (
                "source_recording_sha256",
                "window_start_seconds",
                "source_duration_seconds",
                "window_duration_seconds",
                "score_timeline",
                "daw_recording_start_seconds",
                "daw_window_start_seconds",
            )
        }
        (folder / "window.json").write_text(json.dumps(context) + "\n")
        shutil.rmtree(folder / "imported")
        duration = bank.stems.shape[1] / RATE
        start = float(bank.provenance.get("window_start_seconds", 0))
        return {
            "id": "restored-" + session,
            "title": "Saved NoteSep project",
            "session": session,
            "duration": duration,
            "windowStart": start,
            "sourceDuration": float(
                bank.provenance.get("source_duration_seconds", start + duration)
            ),
            "notes": output,
            "scoreOnly": False,
            "method": "NoteSep"
            if bank.provenance.get("apply_aso", True)
            else "NoteGrab",
            "inputAudio": f"/result/{session}/input.wav" if has_input else None,
            "projectArchive": f"/result/{session}/{archive.name}",
            "edits": edits,
            "provenance": bank.provenance,
            "canReseparate": has_input,
            "forceNoteGrab": len(bank.notes) > 1
            and bank.provenance.get("apply_aso") is False,
        }
    except Exception:
        shutil.rmtree(folder)
        raise


@app.api(name="save_project")
def save_project(
    session: str, filename: str, edits_json: str, recording_start: float | None = None
) -> dict:
    """Save browser edit controls and DAW origin while retaining base samples."""
    import math

    if recording_start is not None and (
        not math.isfinite(recording_start) or recording_start < 0
    ):
        raise ValueError("Use a nonnegative DAW recording position")
    if len(edits_json) > 1_000_000:
        raise ValueError("Edit settings are too large")
    if not re.fullmatch(r"(?:bank|project)-[a-f0-9]{12}\.zip", filename):
        raise ValueError("Invalid project file")
    folder = session_folder(session)
    source = folder / filename
    if not source.is_file():
        raise ValueError("The saved note bank has expired")
    temporary = folder / ("position-" + uuid4().hex)
    try:
        bank, _, _ = read_project_archive(source.read_bytes(), temporary)
        edits = json.loads(edits_json)
        if not isinstance(edits, dict) or set(edits) - {
            str(note["id"]) for note in bank.notes
        }:
            raise ValueError("Invalid saved edit note IDs")
        for values in edits.values():
            if not isinstance(values, dict) or set(values) - {
                "pitchShift",
                "delta",
                "stretch",
                "gainDb",
                "mute",
            }:
                raise ValueError("Invalid saved edit fields")
            pitch, delta, stretch, gain = (
                values.get("pitchShift", 0),
                values.get("delta", 0),
                values.get("stretch", 1),
                values.get("gainDb", 0),
            )
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in (pitch, delta, stretch, gain)
            ):
                raise ValueError("Saved edits must be finite numbers")
            if (
                not -12 <= pitch <= 12
                or abs(delta) > 600
                or not 0.5 <= stretch <= 2
                or not -60 <= gain <= 24
                or not isinstance(values.get("mute", False), bool)
            ):
                raise ValueError("Saved edit controls exceed supported bounds")
        bank.saved_edits = edits
        if recording_start is not None:
            bank.provenance.update(
                daw_recording_start_seconds=recording_start,
                daw_window_start_seconds=recording_start
                + float(bank.provenance.get("window_start_seconds", 0)),
            )
        archive = write_project_archive(folder, uuid4().hex[:12], bank)
        destination = archive.with_name(archive.name.replace("bank-", "project-", 1))
        archive.rename(destination)
        # Metadata generations are pruned independently of active note WAVs.
        generations = sorted(
            folder.glob("project-*.zip"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in generations[3:]:
            path.unlink()
        return {
            "projectArchive": f"/result/{session}/{destination.name}",
            "dawStartSeconds": bank.provenance.get(
                "daw_window_start_seconds",
                bank.provenance.get("window_start_seconds", 0),
            ),
        }
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)


def _transcribe_yourmt3(folder: Path, duration: float) -> dict:
    """Run the public YourMT3+ service and normalize its MIDI response."""
    midi_data = transcribe_with_yourmt3(folder / "input.wav")
    return {
        "notes": midi_notes(midi_data, duration),
        "rawMidiNoteEvents": midi_note_event_count(midi_data),
    }


def _draft_response(
    session: str,
    folder: Path,
    duration: float,
    notes: list[dict],
    provider: str,
    instrument: str = "",
    raw_midi_note_events: int | None = None,
) -> dict:
    audio = decode_audio(folder / "input.wav")
    return {
        "session": session,
        "notes": notes,
        "duration": duration,
        "provider": provider,
        "instrument": instrument,
        "rawMidiNoteEvents": raw_midi_note_events,
        "audioCheck": audit_score_audio(audio, notes, RATE),
    }


@app.api(name="transcribe", concurrency_limit=1)
def transcribe(session: str, instrument: str = "", provider: str = "yourmt3") -> dict:
    """Create an editable first-pass score with the configured AMT model."""
    folder = session_folder(session)
    duration = sf.info(folder / "input.wav").duration
    instrument = instrument.strip()
    provider = provider.strip().lower()
    if provider == "muscriptor_public":
        data = transcribe_with_public_muscriptor(folder / "input.wav", instrument)
        notes = midi_notes(data, duration)
        raw_note_events = midi_note_event_count(data)
        return _draft_response(
            session,
            folder,
            duration,
            notes,
            "MuScriptor · public",
            instrument or "",
            raw_note_events,
        )

    if provider != "yourmt3":
        raise ValueError("Choose YourMT3+ or MuScriptor for transcription.")
    if instrument:
        raise ValueError(
            "Instrument hints are available with MuScriptor; clear the hint to use YourMT3+."
        )
    draft = _transcribe_yourmt3(folder, duration)
    return _draft_response(
        session,
        folder,
        duration,
        draft["notes"],
        "YourMT3+",
        "",
        draft["rawMidiNoteEvents"],
    )


@app.api(name="cross_check", concurrency_limit=1)
def cross_check(
    session: str, notes_json: str, provider: str, instrument: str = ""
) -> dict:
    """Compare a score with an independent public AMT draft without replacing it."""
    folder = session_folder(session)
    duration = sf.info(folder / "input.wav").duration
    try:
        current = json.loads(notes_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("The current score could not be read") from error
    if not isinstance(current, list) or len(current) > 96:
        raise ValueError("Cross-check a score with at most 96 notes")
    current = validate_notes(current, duration) if current else []
    provider = provider.strip().lower()
    instrument = instrument.strip()
    if provider == "muscriptor_public":
        data = transcribe_with_public_muscriptor(folder / "input.wav", instrument)
        independent = midi_notes(data, duration)
        independent_label = "MuScriptor · public"
    elif provider == "yourmt3":
        if instrument:
            raise ValueError("YourMT3+ cannot apply a single-instrument hint")
        draft = _transcribe_yourmt3(folder, duration)
        independent = draft["notes"]
        independent_label = "YourMT3+"
    else:
        raise ValueError("Choose an available MuScriptor or YourMT3+ draft")
    return compare_transcriptions(
        current, independent, "Current score", independent_label
    )


@app.api(name="separate", concurrency_limit=1)
@spaces.GPU(duration=separation_gpu_duration)
def run_separation(session: str, notes_json: str, apply_aso: bool = True) -> dict:
    """Rerun frozen NoteGrab and ASO on the corrected score, never subtracting."""
    folder = session_folder(session)
    audio = decode_audio(folder / "input.wav")
    notes = validate_notes(json.loads(notes_json), len(audio) / RATE)
    use_aso = bool(apply_aso) and len(notes) > 1
    raw, separated = separate(audio, notes, DEVICE, apply_aso=use_aso, profile=PROFILE)
    audio_check = audit_score_audio(audio, notes, RATE)
    separated_check = audit_separated_notes(separated, notes, RATE)
    reconstruction = None if not use_aso else separated_coverage(audio, separated, RATE)
    input_flags = {item["id"]: item["flags"] for item in audio_check["notes"]}
    output_flags = {item["id"]: item["flags"] for item in separated_check["notes"]}
    version = uuid4().hex[:12]
    output = []
    for index, note in enumerate(notes):
        raw_file, final_file = (
            f"raw-{version}-{index}.wav",
            f"note-{version}-{index}.wav",
        )
        sf.write(folder / raw_file, raw[index], RATE, subtype="FLOAT")
        sf.write(folder / final_file, separated[index], RATE, subtype="FLOAT")
        note_id = str(note["id"])
        flags = list(
            dict.fromkeys(input_flags.get(note_id, []) + output_flags.get(note_id, []))
        )
        output.append(
            {
                "id": note["id"],
                "event": index,
                "pitch": note["pitch"],
                "start": note["start"],
                "end": note["end"],
                "qualityFlags": flags,
                "outputs": {
                    "aso": {"audio": f"/result/{session}/{final_file}"},
                    "notegrab": {"audio": f"/result/{session}/{raw_file}"},
                },
            }
        )
    context_path = folder / "window.json"
    window_context = (
        json.loads(context_path.read_text()) if context_path.is_file() else {}
    )
    bank = NoteBank(
        notes,
        separated,
        RATE,
        raw,
        {
            **window_context,
            "input_sha256": sha256(folder / "input.wav"),
            "checkpoint_sha256": checkpoint_hashes(PROFILE),
            "model_profile": PROFILE,
            "apply_aso": use_aso,
            "acoustic_end_policy": "symbolic_end_plus_300ms_clipped",
        },
    )
    archive = write_project_archive(folder, version, bank)
    prune_old_outputs(folder)
    method = "NoteSep" if use_aso else "NoteGrab"
    return {
        "id": "live-" + session,
        "title": f"Your recording · {method}",
        "method": method,
        "duration": len(audio) / RATE,
        "notes": output,
        "scoreOnly": False,
        "inputAudio": f"/result/{session}/input.wav",
        "projectArchive": f"/result/{session}/{archive.name}",
        "audioCheck": audio_check,
        "separatedNoteCheck": separated_check,
        "reconstructionCheck": reconstruction,
    }


def launch(host="127.0.0.1", port=7860):
    app.launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    launch()
