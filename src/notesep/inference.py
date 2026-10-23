"""Frozen NoteGrab + ASO inference for short, aligned score queries.

Uses the same 16 kHz training crop, HPSS, selective gate, and 100 ms pre-ASO
onset support as the released qualitative demo. No waveform subtraction.
"""
from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from .assets import model_profile, weights_directory, checkpoint_hashes
from .training_window import TrainingWindow, place_prediction
from .aso_onset_support import onset_supported_forward
from .models.occurrence_dual_hpss_separator import (
    occurrence_dual_hpss_separator_from_contract,
)
from .models.hpss_target import project_hpss_batch
from .models.acoustic_multiplicative_temporal_ownership import (
    AcousticMultiplicativeTemporalOwnership,
)
from .models.temporal_hybrid_set_ownership import TemporalHybridOwnershipConfig

RATE = 16000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(device_name: str, profile=None):
    selected = model_profile(profile)
    return _load_models(device_name, selected, str(weights_directory(selected).resolve()))


@lru_cache(maxsize=4)
def _load_models(device_name: str, profile: str, weights_path: str):
    device = torch.device(device_name)
    separator_path = Path(weights_path) / "separator.pt"
    aso_path = Path(weights_path) / "aso.pt"
    expected = checkpoint_hashes(profile)
    if file_sha256(separator_path) != expected['separator.pt'] or file_sha256(aso_path) != expected['aso.pt']:
        raise RuntimeError("Frozen checkpoint checksum mismatch")
    checkpoint = torch.load(separator_path, map_location="cpu", weights_only=False)
    geometry = TrainingWindow.from_checkpoint(checkpoint)
    if (geometry.lead_samples, geometry.release_samples,
        geometry.minimum_window_samples, geometry.window_quantum) != (4000, 2560, 16384, 4096):
        raise RuntimeError("Unexpected separator training-window contract")
    separator = occurrence_dual_hpss_separator_from_contract(checkpoint["model_contract"])
    separator.load_state_dict(checkpoint["model"], strict=True)
    separator.to(device).eval()
    del checkpoint
    checkpoint = torch.load(aso_path, map_location="cpu", weights_only=False)
    valid_names = {field.name for field in fields(TemporalHybridOwnershipConfig)}
    config = TemporalHybridOwnershipConfig(**{
        key: value for key, value in checkpoint["contract"]["model_contract"].items()
        if key in valid_names
    })
    aso = AcousticMultiplicativeTemporalOwnership(config).eval()
    aso.load_state_dict(checkpoint["model"], strict=True)
    return separator, aso, geometry


def decode_audio(path: Path) -> np.ndarray:
    info = sf.info(path)
    if info.duration < 1.1 or info.duration > 13.0:
        raise ValueError("Upload a 1.1–12 second excerpt (13 seconds maximum including padding).")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    gcd = math.gcd(sample_rate, RATE)
    result = resample_poly(audio, RATE // gcd, sample_rate // gcd).astype(np.float32)
    if len(result) > 13 * RATE:
        raise ValueError("Excerpt is too long")
    return result


def validate_notes(notes: list[dict], duration: float) -> list[dict]:
    if not isinstance(notes, list) or not 1 <= len(notes) <= 96:
        raise ValueError("Provide between 1 and 96 score notes")
    result = []
    for index, item in enumerate(notes):
        if not isinstance(item, dict):
            raise ValueError("Invalid score note")
        pitch, start, end = item.get("pitch"), item.get("start"), item.get("end")
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 21 <= pitch <= 108:
            raise ValueError(f"Note {index + 1} has an invalid MIDI pitch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (start, end)):
            raise ValueError(f"Note {index + 1} has invalid timing")
        start, end = float(start), float(end)
        if not 0 <= start < end <= duration + .001:
            raise ValueError(f"Note {index + 1} must have positive duration within the excerpt")
        if round(min(end, duration) * RATE) <= round(start * RATE):
            raise ValueError(f"Note {index + 1} must span at least one audio sample within the excerpt")
        result.append({"id": str(item.get("id", index)), "pitch": pitch,
                       "start": start, "end": min(end, duration)})
    return result


@torch.inference_mode()
def separate(audio: np.ndarray, notes: list[dict], device_name: str, *, apply_aso: bool = True, profile=None) -> tuple[np.ndarray, np.ndarray]:
    if audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono float32 audio")
    total = len(audio)
    notes = validate_notes(notes, total / RATE)
    separator, aso, geometry = load_models(device_name, profile)
    device = torch.device(device_name)
    bounds = [(note["pitch"], round(note["start"] * RATE),
               min(total, round((note["end"] + .30) * RATE))) for note in notes]
    raw = []
    for note, (pitch, onset, acoustic) in zip(notes, bounds):
        offset = round(note["end"] * RATE)
        crop_start, crop_stop = geometry.bounds(onset, offset, acoustic, total)
        valid_count = crop_stop - crop_start
        padded = max(16384, math.ceil(valid_count / 1024) * 1024)
        mixture = torch.zeros((1, padded), device=device)
        mixture[0, :valid_count] = torch.from_numpy(audio[crop_start:crop_stop]).to(device)
        valid = torch.tensor([valid_count], device=device)
        harmonic, percussive = project_hpss_batch(
            mixture, valid, sample_rate=RATE, n_fft=separator.config.n_fft,
            hop_length=separator.config.hop_length, kernel_size=31, power=2., margin=1.)
        gate = any(pitch + 12 == other_pitch and
                   min(acoustic, other_end) - max(onset, other_start) >=
                   max(1280, .25 * min(acoustic - onset, other_end - other_start))
                   for other_pitch, other_start, other_end in bounds)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, _ = separator(
                harmonic, percussive, torch.tensor([pitch], device=device), valid,
                torch.tensor([onset - crop_start], device=device),
                torch.tensor([acoustic - crop_start], device=device),
                input_gate_strength=torch.tensor([float(gate)], device=device))
        raw.append(place_prediction(prediction[0, :valid_count].float().cpu().numpy(),
                                    crop_start, 0, total))
    raw_bank = np.stack(raw).astype(np.float32)
    if not apply_aso:
        return raw_bank, raw_bank.copy()
    hann = torch.hann_window(2048)
    def stft(signal):
        return torch.stft(torch.from_numpy(signal), 2048, 512, window=hann,
                          center=True, pad_mode="constant", return_complex=True)
    onsets = torch.tensor([[round(note["start"] * RATE) for note in notes]])
    acoustic_offsets = torch.tensor([[end for _, _, end in bounds]])
    separated, _ = onset_supported_forward(
        aso, stft(audio)[None], stft(raw_bank)[None],
        torch.tensor([[note["pitch"] for note in notes]]),
        onsets=onsets, acoustic_offsets=acoustic_offsets)
    result = torch.istft(separated[0], 2048, 512, window=hann,
                         center=True, length=total).cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite model output")
    return raw_bank, result
