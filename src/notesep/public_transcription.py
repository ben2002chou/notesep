"""Small client for the public MuScriptor MIDI endpoint."""
from __future__ import annotations

from pathlib import Path
import base64
import re
import time

import requests


MUSCRIPTOR_PUBLIC_MIDI_URL = "https://muscriptor.kyutai.org/transcribe/midi"
MUSCRIPTOR_PUBLIC_TIMEOUT_SECONDS = 120
MUSCRIPTOR_BUSY_RETRIES = 3
MIDI_HTML_RE = re.compile(r"data:audio/(?:midi|mid);base64,([A-Za-z0-9+/=]+)")


def transcribe_with_yourmt3(audio_path: Path) -> bytes:
    """Request one bounded window and return validated MIDI bytes."""
    from gradio_client import Client, handle_file
    from .score_io import MAX_MIDI_BYTES
    client = Client('mimbres/YourMT3', max_workers=1, verbose=False,
                    httpx_kwargs={'timeout': 100})
    try:
        job = client.submit(handle_file(str(audio_path)), api_name='/process_audio')
        try:
            result = job.result(timeout=90)
        except TimeoutError as error:
            job.cancel()
            raise ValueError('YourMT3+ did not finish within 90 seconds. Retry, use a shorter window, or import aligned MIDI.') from error
    finally:
        client.close()
    match = MIDI_HTML_RE.search(result) if isinstance(result, str) else None
    if not match or len(match.group(1)) > 4 * ((MAX_MIDI_BYTES + 2) // 3):
        raise ValueError('YourMT3+ did not return bounded MIDI; import aligned MIDI instead')
    try:
        data = base64.b64decode(match.group(1), validate=True)
    except ValueError as error:
        raise ValueError('YourMT3+ returned invalid MIDI encoding') from error
    if not data.startswith(b'MThd') or len(data) > MAX_MIDI_BYTES:
        raise ValueError('YourMT3+ did not return valid MIDI; import aligned MIDI instead')
    return data


def _retry_delay(response: requests.Response, attempt: int) -> float:
    """Honor a short server retry hint, otherwise use bounded exponential backoff."""
    try:
        return min(8.0, max(0.0, float(response.headers.get("retry-after", ""))))
    except (TypeError, ValueError):
        return min(8.0, 1.0 * (2 ** attempt))


def transcribe_with_public_muscriptor(audio_path: Path, instrument: str = "") -> bytes:
    """Send one selected audio window and return its MIDI response.

    The Space caller is responsible for bounding the audio window. This helper
    intentionally sends no user-controlled URL or request headers.
    """
    form = {"detect_tempo": "false"}
    if instrument:
        form["instruments"] = instrument
    try:
        for attempt in range(MUSCRIPTOR_BUSY_RETRIES + 1):
            with audio_path.open("rb") as audio_file:
                response = requests.post(
                    MUSCRIPTOR_PUBLIC_MIDI_URL,
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                    data=form,
                    timeout=MUSCRIPTOR_PUBLIC_TIMEOUT_SECONDS,
                )
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                busy = (response.status_code == 503 and
                        "server busy" in response.text.lower())
                if not busy or attempt >= MUSCRIPTOR_BUSY_RETRIES:
                    raise
                time.sleep(_retry_delay(response, attempt))
                continue
            break
    except requests.RequestException as error:
        raise ValueError(
            "MuScriptor's public transcription service is temporarily unavailable. "
            "Try YourMT3+, import aligned MIDI, or add notes manually."
        ) from error

    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "audio/midi" or not response.content:
        raise ValueError(
            "MuScriptor's public service did not return a valid MIDI draft. "
            "Try YourMT3+, import aligned MIDI, or add notes manually."
        )
    return response.content
