"""Portable adapter for the paper's unchanged float64 score-informed NMF function."""
from __future__ import annotations
from typing import Any
from types import SimpleNamespace
import numpy as np
implementation = SimpleNamespace(MODEL_RATE=16000)

def infer_nmf_piece_float64(
    frozen: Any,
    mixture48: np.ndarray,
    piece: dict[str, Any],
    prepared: list[dict[str, Any]],
    max_iter: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Run frozen Ewert--Mueller configuration A in float64 solver precision."""
    import librosa
    from sklearn.decomposition import NMF

    mixture16 = frozen.to_16k(mixture48, frozen.SOURCE_RATE)
    n_fft, hop = 2048, 512
    spectrum = librosa.stft(mixture16, n_fft=n_fft, hop_length=hop)
    magnitude, _phase = librosa.magphase(spectrum)
    # Upstream W/H are created by bare numpy.zeros and therefore float64.  A
    # modern sklearn custom initialization requires X, W, and H to share a
    # dtype, so promote X instead of demoting the upstream initializers.
    magnitude = np.asarray(magnitude, dtype=np.float64)
    pitches = sorted({int(event["pitch"]) for event in piece["events"]})
    pitch_index = {pitch: index for index, pitch in enumerate(pitches)}
    frequencies = librosa.fft_frequencies(sr=implementation.MODEL_RATE, n_fft=n_fft)
    nmf_dtype = magnitude.dtype
    w_init = np.zeros((len(frequencies), len(pitches)), dtype=nmf_dtype)
    for column, pitch in enumerate(pitches):
        partial = 1
        while True:
            low = float(librosa.midi_to_hz(pitch - 1.0)) * partial
            if low > frequencies[-1]:
                break
            high = min(
                float(frequencies[-1]),
                float(librosa.midi_to_hz(pitch + 1.0)) * partial,
            )
            first = max(0, int(np.searchsorted(frequencies, low, side="right") - 1))
            last = int(np.searchsorted(frequencies, high, side="left"))
            w_init[first : last + 1, column] = 1.0 / (partial * partial)
            partial += 1
    frames = magnitude.shape[1]
    h_init = np.zeros((len(pitches), frames), dtype=nmf_dtype)
    event_masks: list[np.ndarray] = []
    for event in piece["events"]:
        start, stop = librosa.time_to_frames(
            [float(event["onset_seconds"]) - 0.2, float(event["offset_seconds"]) + 1.0],
            sr=implementation.MODEL_RATE,
            hop_length=hop,
        )
        start = max(0, int(start))
        stop = min(frames, max(start + 1, int(stop)))
        mask = np.zeros(frames, dtype=nmf_dtype)
        mask[start:stop] = 1.0
        event_masks.append(mask)
        h_init[pitch_index[int(event["pitch"])], start:stop] = 1.0

    transformer = NMF(
        n_components=len(pitches),
        solver="mu",
        init="custom",
        max_iter=int(max_iter),
        random_state=0,
    )
    h_fit_t = transformer.fit_transform(magnitude.T, W=h_init.T, H=w_init.T)
    h_fit = h_fit_t.T
    w_fit = transformer.components_.T
    denominator = np.maximum(w_fit @ h_fit, np.finfo(w_fit.dtype).eps)
    estimates: list[np.ndarray] = []
    for item, event_mask in zip(prepared, event_masks, strict=True):
        row = pitch_index[int(item["event"]["pitch"])]
        numerator = np.outer(w_fit[:, row], h_fit[row] * event_mask)
        note_spectrum = (numerator / denominator) * spectrum
        full_note = librosa.istft(
            note_spectrum,
            hop_length=hop,
            length=len(mixture16),
        )
        start = int(item["crop_start16"])
        estimates.append(
            frozen.fit(
                full_note[start : start + len(item["mixture"])],
                len(item["mixture"]),
            )
        )
    return estimates, {
        "n_fft": n_fft,
        "hop_length": hop,
        "phi_semitones": 1.0,
        "tolerance_on_seconds": 0.2,
        "tolerance_off_seconds": 1.0,
        "mask_resynthesis": True,
        "solver": "sklearn_NMF_mu_frobenius",
        "solver_dtype": str(magnitude.dtype),
        "max_iter": int(max_iter),
        "actual_iterations": int(transformer.n_iter_),
        "pitch_components": len(pitches),
    }


def separate_baseline(audio_path, notes, *, max_iter=1000):
    from .api import NoteBank
    from .inference import RATE, decode_audio, validate_notes
    from .score_io import midi_notes
    from .assets import sha256
    from pathlib import Path
    import json
    audio = decode_audio(Path(audio_path))
    if isinstance(notes, (str, Path)):
        path = Path(notes)
        notes = (midi_notes(path.read_bytes(), len(audio)/RATE)
                 if path.suffix.lower() in ('.mid', '.midi') else json.loads(path.read_text()))
    notes = validate_notes(notes, len(audio)/RATE)
    events = [{'pitch':n['pitch'], 'onset_seconds':n['start'], 'offset_seconds':n['end']}
              for n in notes]
    def fit(wave, count):
        result = np.zeros(count, dtype=np.float32)
        result[:min(count,len(wave))] = wave[:count]
        return result
    adapter = SimpleNamespace(SOURCE_RATE=RATE, to_16k=lambda x, rate:x, fit=fit)
    prepared = [{'event':e, 'mixture':audio, 'crop_start16':0} for e in events]
    estimates, contract = infer_nmf_piece_float64(adapter, audio, {'events':events}, prepared, max_iter)
    return NoteBank(notes, np.stack(estimates), RATE, provenance={
        'method':'score_informed_nmf', 'contract':contract, 'input_sha256':sha256(audio_path),
        'original_runner_sha256':'61433085b089773116b2c12f2c56302db8cc4d8c0afc0608781b63936e57ff92',
        'endpoint_policy':'provided_score_end', 'role':'portable user tool; frozen paper evaluator is separate'})
