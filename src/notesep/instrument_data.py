"""Read PHENICX and Bach10 and reproduce the paper's crop geometry."""
from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Any
import librosa
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
import soundfile as sf
import museval
from museval.metrics import Framing

MODEL_RATE = 16_000
SOURCE_RATE = 44_100
GEOMETRY_RATE = 24_000
LEAD_24K = 4_000
RELEASE_24K = 2_560
MINIMUM_24K = 16_384
QUANTUM_24K = 4_096
RATE = MODEL_RATE
CLASS_TO_STEM = {"Violin": "violin", "Viola": "viola", "Cello": "cello", "Bass": "doublebass", "Flute": "flute", "Clarinet": "clarinet", "Oboe": "oboe", "Bassoon": "bassoon", "Horn": "horn", "Trumpet": "trumpet"}
CLASSES = tuple(CLASS_TO_STEM)
BACH_PARTS = ("violin", "clarinet", "saxphone", "bassoon")

def mono(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2:
        values = values.mean(axis=1)
    return np.ascontiguousarray(values.reshape(-1))

def fit(values: np.ndarray, length: int) -> np.ndarray:
    values = mono(values)
    if len(values) < length:
        values = np.pad(values, (0, length - len(values)))
    return np.ascontiguousarray(values[:length])

def to_16k(values: np.ndarray) -> np.ndarray:
    divisor = math.gcd(SOURCE_RATE, MODEL_RATE)
    output = resample_poly(mono(values), MODEL_RATE // divisor, SOURCE_RATE // divisor)
    return np.ascontiguousarray(output, dtype=np.float32)

def load_piece_audio(dataset_root: Path, piece: str) -> tuple[np.ndarray, np.ndarray]:
    audio_root = dataset_root / "audio" / piece
    wav_paths = sorted(audio_root.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(audio_root)
    loaded: dict[Path, np.ndarray] = {}
    maximum = 0
    for path in wav_paths:
        values, rate = sf.read(path, dtype="float64", always_2d=True)
        if int(rate) != SOURCE_RATE or values.shape[1] != 1 or not np.isfinite(values).all():
            raise ValueError(f"invalid source WAV: {path}")
        loaded[path] = mono(values)
        maximum = max(maximum, len(values))

    targets44 = np.zeros((len(CLASSES), maximum), dtype=np.float64)
    covered: set[Path] = set()
    for class_index, class_name in enumerate(CLASSES):
        stem = CLASS_TO_STEM[class_name]
        for path in sorted(audio_root.glob(f"{stem}*.wav")):
            targets44[class_index] += fit(loaded[path], maximum)
            covered.add(path)
    uncovered = sorted(path.name for path in set(wav_paths) - covered)
    if uncovered:
        raise ValueError(f"unmapped source WAVs in {piece}: {uncovered}")

    mixture44 = targets44.sum(axis=0)
    peak = float(np.max(np.abs(mixture44)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"invalid mixture peak for {piece}: {peak}")
    scale = peak / 0.75
    mixture44 /= scale
    targets44 /= scale
    mixture44[np.abs(mixture44) < 1e-4] = 0.0
    targets44[np.abs(targets44) < 1e-4] = 0.0

    mixture = to_16k(mixture44)
    targets = np.stack([fit(to_16k(row), len(mixture)) for row in targets44])
    return mixture, np.ascontiguousarray(targets, dtype=np.float32)

def load_events(dataset_root: Path, piece: str) -> list[dict[str, Any]]:
    annotation_root = dataset_root / "annotations" / piece
    events: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASSES):
        path = annotation_root / f"{CLASS_TO_STEM[class_name]}.txt"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for note_index, row in enumerate(csv.reader(handle)):
                if not row:
                    continue
                onset, offset = float(row[0]), float(row[1])
                pitch = int(librosa.note_to_midi(row[2].strip(), round_midi=True))
                if onset < 0 or offset <= onset or not 0 <= pitch <= 127:
                    raise ValueError(f"invalid annotation in {path}: {row}")
                events.append({
                    "class": class_name,
                    "class_index": class_index,
                    "note_index": note_index,
                    "pitch": pitch,
                    "onset_seconds": onset,
                    "offset_seconds": offset,
                })
    events.sort(key=lambda row: (
        row["onset_seconds"], row["class_index"], row["note_index"], row["pitch"]
    ))
    for event_index, event in enumerate(events):
        event["event_index"] = event_index
    return events

def crop_geometry(piece_samples: int, event: dict[str, Any]) -> tuple[int, int]:
    total24 = int(math.ceil(piece_samples * GEOMETRY_RATE / MODEL_RATE))
    onset24 = int(round(float(event["onset_seconds"]) * GEOMETRY_RATE))
    acoustic24 = int(math.ceil(
        (float(event["offset_seconds"]) - float(event["onset_seconds"])) * GEOMETRY_RATE
    ))
    left = max(0, onset24 - LEAD_24K)
    right = min(total24, onset24 + acoustic24 + RELEASE_24K)
    desired = max(MINIMUM_24K, right - left)
    desired = min(total24, int(math.ceil(desired / QUANTUM_24K) * QUANTUM_24K))
    start_min = max(0, right - desired)
    start_max = min(left, total24 - desired)
    start24 = max(0, min((start_min + start_max) // 2, total24 - desired))
    stop24 = start24 + desired
    start16 = int(round(start24 * MODEL_RATE / GEOMETRY_RATE))
    samples16 = int(round((stop24 - start24) * MODEL_RATE / GEOMETRY_RATE))
    start16 = max(0, min(start16, piece_samples))
    return start16, min(piece_samples, start16 + samples16)

def waveform_metrics(target: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    target64 = np.asarray(target, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    target_energy = max(float(np.dot(target64, target64)), 1e-20)
    error = estimate64 - target64
    target_zero = target64 - target64.mean()
    estimate_zero = estimate64 - estimate64.mean()
    zero_energy = max(float(np.dot(target_zero, target_zero)), 1e-20)
    gain = float(np.dot(estimate_zero, target_zero)) / zero_energy
    projection = gain * target_zero
    residual = estimate_zero - projection
    return {
        "sdr_db": float(10.0 * math.log10(
            target_energy / max(float(np.dot(error, error)), 1e-20)
        )),
        "si_sdr_db": float(10.0 * math.log10(
            max(float(np.dot(projection, projection)), 1e-20)
            / max(float(np.dot(residual, residual)), 1e-20)
        )),
        "waveform_nmse": float(np.dot(error, error) / target_energy),
    }

def bach_events(piece_root: Path) -> list[dict[str, Any]]:
    cells = loadmat(piece_root / f"{piece_root.name}-GTNotes.mat", squeeze_me=True)["GTNotes"]
    events: list[dict[str, Any]] = []
    for part_index, (part, notes) in enumerate(zip(BACH_PARTS, cells, strict=True)):
        for part_note_index, note in enumerate(notes):
            note = np.asarray(note, dtype=np.float64)
            onset_frame, offset_frame = int(note[0, 0]), int(note[0, -1])
            events.append({
                "part": part, "part_index": part_index, "part_note_index": part_note_index,
                "pitch": int(np.round(np.mean(note[1]))),
                "onset_seconds": (onset_frame - 1) * 0.010,
                "offset_seconds": offset_frame * 0.010,
                "identity": [int(np.round(np.mean(note[1]))), onset_frame, offset_frame],
            })
    events.sort(key=lambda row: (
        row["onset_seconds"], row["part_index"], row["part_note_index"], row["pitch"]
    ))
    for index, row in enumerate(events):
        row["event_index"] = index
    return events

def read_44k(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if int(rate) != SOURCE_RATE or values.shape[1] != 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid official audio: {path}")
    return np.ascontiguousarray(values[:, 0])


def load_phenicx(dataset_root: Path, piece: str):
    mixture, targets = load_piece_audio(dataset_root, piece)
    events = load_events(dataset_root, piece)
    for event in events:
        event['part'] = event['class']
        event['part_index'] = event['class_index']
    return mixture, targets, events, CLASSES


def load_bach10(dataset_root: Path, piece: str):
    root = dataset_root / piece
    mixture = to_16k(read_44k(root / f'{piece}.wav'))
    targets = np.stack([
        fit(to_16k(read_44k(root / f'{piece}-{part}.wav')), len(mixture))
        for part in BACH_PARTS
    ]).astype(np.float32)
    return mixture, targets, bach_events(root), BACH_PARTS


def load_phenicx_references(dataset_root: Path, piece: str):
    """Read the original 44.1-kHz mixture and instrument references."""
    paths = sorted((dataset_root / 'audio' / piece).glob('*.wav'))
    if not paths:
        raise FileNotFoundError(dataset_root / 'audio' / piece)
    sources = {path: read_44k(path) for path in paths}
    length = max(map(len, sources.values()))
    targets = np.zeros((len(CLASSES), length), dtype=np.float64)
    used = set()
    for index, label in enumerate(CLASSES):
        for path, source in sources.items():
            if path.name.startswith(CLASS_TO_STEM[label]):
                targets[index] += fit(source, length)
                used.add(path)
    if used != set(paths):
        raise ValueError('Unrecognized source audio')
    mixture = targets.sum(axis=0)
    scale = float(np.max(np.abs(mixture))) / 0.75
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid mixture level')
    mixture /= scale
    targets /= scale
    mixture[np.abs(mixture) < 1e-4] = 0
    targets[np.abs(targets) < 1e-4] = 0
    return mixture, targets


def load_bach10_references(dataset_root: Path, piece: str):
    root = dataset_root / piece
    mixture = read_44k(root / f'{piece}.wav')
    targets = np.stack([read_44k(root / f'{piece}-{part}.wav') for part in BACH_PARTS])
    return mixture, targets


METRIC_RATE = 44_100

def si_sdr(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    energy = max(float(reference @ reference), 1e-20)
    projection = reference * (float(estimate @ reference) / energy)
    residual = estimate - projection
    return float(10 * math.log10(
        max(float(projection @ projection), 1e-20)
        / max(float(residual @ residual), 1e-20)
    ))

def resample_estimate(wave, length):
    wave = resample_poly(np.asarray(wave, dtype=np.float64).reshape(-1), 441, 160)
    if len(wave) < length:
        wave = np.pad(wave, (0, length - len(wave)))
    return wave[:length, None]

def score_parts(mixture, references, estimates, labels):
    """Score one piece; references and mixture are the original 44.1-kHz audio."""
    mixture = np.asarray(mixture, dtype=np.float64).reshape(-1, 1)
    references = np.asarray(references, dtype=np.float64).reshape(len(labels), -1, 1)
    if references.shape[1] != len(mixture):
        raise ValueError('Reference and mixture lengths differ')
    active = [i for i, reference in enumerate(references) if float(reference.reshape(-1) @ reference.reshape(-1)) > 1e-12]
    selected = references[active].copy()
    selected_estimates = np.stack([resample_estimate(estimates[i], len(mixture)) for i in active])
    for source in selected:
        for window in Framing(METRIC_RATE, METRIC_RATE, len(mixture)):
            if np.allclose(source[window], 0, atol=1e-2):
                source[window] = 0
    sdr, _, _, _ = museval.evaluate(selected, selected_estimates, win=METRIC_RATE, hop=METRIC_RATE, mode='v4')
    results = []
    for row, index in enumerate(active):
        values = np.asarray(sdr[row], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f'No valid SDR frames for {labels[index]}')
        estimate_score = si_sdr(references[index], selected_estimates[row])
        mixture_score = si_sdr(references[index], mixture)
        results.append({
            'part': labels[index],
            'sdr': float(np.median(finite)),
            'si_sdr': estimate_score,
            'si_sdri': estimate_score - mixture_score,
        })
    return results

def summarize(dataset, piece_scores):
    """Combine part scores using the paper's dataset-specific aggregation."""
    expected_pieces, expected_rows = (4, 38) if dataset == 'phenicx' else (10, 40)
    if len(piece_scores) != expected_pieces:
        raise ValueError(f'{dataset} requires {expected_pieces} pieces')
    summary = {}
    for method in ('notegrab', 'notesep'):
        rows = [row for scores in piece_scores for row in scores[method]]
        if len(rows) != expected_rows:
            raise ValueError(f'{dataset} requires {expected_rows} part scores')
        if dataset == 'phenicx':
            parts = sorted({row['part'] for row in rows})
            sdr = float(np.mean([
                np.median([row['sdr'] for row in rows if row['part'] == part])
                for part in parts
            ]))
        else:
            sdr = float(np.median([row['sdr'] for row in rows]))
        summary[method] = {
            'sdr': sdr,
            'si_sdr': float(np.median([row['si_sdr'] for row in rows])),
            'si_sdri': float(np.median([row['si_sdri'] for row in rows])),
            'parts': len(rows),
        }
    return summary
