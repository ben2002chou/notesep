"""Read one bounded edit window from a source recording."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


RATE = 16_000
MIN_WINDOW_SECONDS = 1.1
MAX_WINDOW_SECONDS = 12.0
MAX_SOURCE_SECONDS = 300.0


def decode_audio_window(
    path: Path, start_seconds: float, window_seconds: float
) -> tuple[np.ndarray, float, float]:
    """Decode/resample only the requested region; return audio, actual start, source duration."""
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (start_seconds, window_seconds)
    ):
        raise ValueError("Choose a valid edit-window start and length")
    start_seconds, window_seconds = float(start_seconds), float(window_seconds)
    if (
        start_seconds < 0
        or window_seconds < MIN_WINDOW_SECONDS
        or window_seconds > MAX_WINDOW_SECONDS
    ):
        raise ValueError(
            "Choose a window from 1.1 to 12 seconds, starting at or after 0 seconds"
        )

    info = sf.info(path)
    source_duration = info.duration
    if source_duration < MIN_WINDOW_SECONDS or source_duration > MAX_SOURCE_SECONDS:
        raise ValueError("Use a recording between 1.1 seconds and 5 minutes long")
    if start_seconds >= source_duration:
        raise ValueError("The edit-window start is beyond the end of the recording")

    first_frame = min(info.frames, round(start_seconds * info.samplerate))
    requested_frames = round(window_seconds * info.samplerate)
    frame_count = min(requested_frames, info.frames - first_frame)
    actual_duration = frame_count / info.samplerate
    if actual_duration < MIN_WINDOW_SECONDS:
        raise ValueError(
            "Move the edit window earlier so it contains at least 1.1 seconds"
        )

    with sf.SoundFile(path) as source:
        # MP3 frames share a bit reservoir and decoder filter state. Seeking
        # directly to the edit boundary can produce an audible onset transient.
        # Decode at most one extra second, then discard it before resampling so
        # the window retains the same sample origin as a continuous decode.
        preroll = min(first_frame, info.samplerate) if info.format == "MP3" else 0
        source.seek(first_frame - preroll)
        audio = source.read(
            frames=frame_count + preroll, dtype="float32", always_2d=True
        )[preroll:]
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    gcd = math.gcd(info.samplerate, RATE)
    result = resample_poly(audio, RATE // gcd, info.samplerate // gcd).astype(
        np.float32
    )
    return result, first_frame / info.samplerate, source_duration
