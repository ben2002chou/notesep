"""Evaluate score-informed note editing against isolated references."""
import argparse
from collections import defaultdict
import io
import json
import math
import re
from statistics import median
from pathlib import Path
import zipfile

import numpy as np
import torch

from notesep.evaluate import load_models, read_audio, to_model_rate, source_context, separate_crop

RATE = 16000


def two_f_from_peaq(stdout: str) -> float:
    """Read PEAQ basic-model output and calculate the 2f score (0–100)."""
    modulation = re.findall(r"AvgModDiff1b:\s*([-+0-9.eE]+)", stdout)
    distortion = re.findall(r"ADBb:\s*([-+0-9.eE]+)", stdout)
    if not modulation or not distortion:
        raise ValueError("PEAQ output is missing the measurements needed for 2f")
    modulation = float(modulation[-1])
    distortion = float(distortion[-1])
    score = 56.1345 / (1.0 + (-0.0282 * modulation - 0.8628) ** 2)
    return float(np.clip(score - 27.1451 * distortion + 86.3515, 0.0, 100.0))


def place(values, start, length):
    """Place a note at a sample offset on a fixed-length timeline."""
    result = np.zeros(length, dtype=np.float32)
    source = max(0, -start)
    destination = max(0, start)
    count = min(len(values) - source, length - destination)
    if count > 0:
        result[destination : destination + count] = values[source : source + count]
    return result


def edit_segment(original, transformed, start, stop, edit_type, value, fade_samples):
    """Blend an edited note segment into its original timeline."""
    length = stop - start
    if length <= 0:
        raise ValueError("Empty edit support")
    fade = min(int(fade_samples), length // 2)
    mask = np.ones(length, np.float32)
    if fade:
        phase = np.linspace(0.0, np.pi / 2.0, fade, endpoint=False, dtype=np.float64)
        ramp = np.square(np.sin(phase)).astype(np.float32)
        mask[:fade] = ramp
        mask[-fade:] = ramp[::-1]
    kept = original * (np.float32(1.0) - place(mask, start, len(original)))
    shift = int(round(value * 16000 / 1000.0)) if edit_type == "retiming" else 0
    inserted = place(transformed * mask, start + shift, len(original))
    return np.ascontiguousarray((kept + inserted).astype(np.float32))


def other_note_sums(bank):
    """Compute sums excluding each note without subtracting an estimate."""
    count, samples = bank.shape
    prefix = np.zeros((count + 1, samples), np.float32)
    suffix = np.zeros((count + 1, samples), np.float32)
    for index in range(count):
        prefix[index + 1] = prefix[index] + bank[index]
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] + bank[index]
    return prefix, suffix


def pitch_shift(notes, semitones):
    """Apply the paper's pitch transform with per-note level preservation."""
    import torch
    import torchaudio.functional as audio_functional

    if not notes:
        return []
    lengths = [len(note) for note in notes]
    maximum = max(lengths)
    batch = np.zeros((len(notes), maximum), dtype=np.float32)
    for index, note in enumerate(notes):
        batch[index, : len(note)] = note
    samples = torch.from_numpy(batch)
    original_rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
    shifted = audio_functional.pitch_shift(
        samples, sample_rate=16000, n_steps=float(semitones),
        n_fft=512, win_length=512, hop_length=128,
    ).float()
    if shifted.shape[-1] < maximum:
        shifted = torch.nn.functional.pad(shifted, (0, maximum - shifted.shape[-1]))
    shifted = shifted[..., :maximum]
    shifted_rms = shifted.square().mean(dim=-1, keepdim=True).sqrt()
    scale = torch.where(original_rms > 1e-7, original_rms / shifted_rms.clamp_min(1e-7),
                        torch.zeros_like(original_rms)).clamp(max=4.0)
    shifted = shifted * scale
    result = [shifted[index, :length].numpy().astype(np.float32, copy=True)
              for index, length in enumerate(lengths)]
    if any(not np.isfinite(note).all() for note in result):
        raise FloatingPointError("Pitch edit produced non-finite audio")
    return result


def transform(notes, edit_type, values):
    """Apply one benchmark edit to each note."""
    if edit_type == "rebalancing":
        return [(note * np.float32(10.0 ** (value / 20.0))).astype(np.float32)
                for note, value in zip(notes, values, strict=True)]
    if edit_type == "retiming":
        return [note.copy() for note in notes]
    if edit_type in {"repitching", "retuning"}:
        steps = values if edit_type == "repitching" else [value / 100.0 for value in values]
        result = [None] * len(notes)
        for step in sorted(set(steps)):
            indices = [index for index, value in enumerate(steps) if value == step]
            for index, note in zip(indices, pitch_shift([notes[i] for i in indices], step), strict=True):
                result[index] = note
        return result
    raise ValueError(f"Unknown edit type: {edit_type}")


def peaq_score(reference, estimate, executable):
    """Run PEAQ Basic on 16 kHz audio and return the paper's 2f score."""
    import subprocess
    import tempfile
    from pathlib import Path
    from scipy.signal import resample_poly
    import soundfile as sf

    reference_48k = resample_poly(reference, 3, 1).astype(np.float64)
    estimate_48k = resample_poly(estimate, 3, 1).astype(np.float64)
    with tempfile.TemporaryDirectory() as folder:
        reference_path = Path(folder) / "reference.wav"
        estimate_path = Path(folder) / "estimate.wav"
        sf.write(reference_path, reference_48k, 48000, subtype="PCM_16")
        sf.write(estimate_path, estimate_48k, 48000, subtype="PCM_16")
        result = subprocess.run(
            [str(executable), "-r", str(reference_path), "-t", str(estimate_path)],
            check=True, capture_output=True, text=True,
        )
    return two_f_from_peaq(result.stdout)



def evaluate_piece(root, piece, rows, edits, separator, aso, device, peaq, max_requests=0):
    """Evaluate all requested note edits in a recording."""
    folder = root / piece['root']
    mixture, rate = read_audio(folder / 'mixture.wav')
    if rate != 24000:
        raise ValueError('Expected 24 kHz benchmark audio')
    metadata = json.loads((folder / 'piece.json').read_text())
    events = {int(event['event_index']): event for event in metadata['events']}
    references = {}
    with zipfile.ZipFile(folder / 'occurrences.zip') as archive:
        for row in rows:
            audio, note_rate = read_audio(io.BytesIO(archive.read(row['occurrence_member'])))
            if note_rate != 24000:
                raise ValueError('Expected 24 kHz note references')
            references[int(row['event_index'])] = audio
    groups = defaultdict(list)
    selected = rows[:max_requests] if max_requests else rows
    for row in selected:
        groups[(int(row['crop_start_24k']), int(row['crop_stop_24k']))].append(row)
    boundary = edits['boundary']
    lead = round(boundary['lead_ms'] * RATE / 1000)
    release = round(boundary['release_ms'] * RATE / 1000)
    fade = round(boundary['crossfade_ms'] * RATE / 1000)
    results = []
    for (start, stop), requests in groups.items():
        audio = to_model_rate(mixture[start:stop], rate)
        context = source_context(metadata, start, len(audio))
        positions = {item['event_index']: index for index, item in enumerate(context)}
        targets, supports = [], []
        crop_start = round(start * RATE / 24000)
        for item in context:
            event = events[item['event_index']]
            note = references[item['event_index']]
            onset_24k = int(event['onset_sample'])
            target_24k = place(note, onset_24k - start, stop - start)
            targets.append(to_model_rate(target_24k, 24000))
            onset = round(onset_24k * RATE / 24000) - crop_start
            left = max(0, onset - lead)
            right = min(len(audio), max(onset + math.ceil(len(note) * RATE / 24000),
                                        int(item['offset']) + release))
            if right <= left:
                raise ValueError('Empty note edit support')
            supports.append((left, right))
        targets = np.stack(targets)
        target_prefix, target_suffix = other_note_sums(targets)
        raw, separated = separate_crop(audio, context, separator, aso, device)
        estimates = {'raw': raw, 'aso': separated}
        other_scores = {}
        positions_for_rows = [positions[int(row['event_index'])] for row in requests]
        for edit_type in edits['conditions']:
            values = [float(edits['assignments'][row['request_id']][edit_type]) for row in requests]
            target_segments = [targets[position, supports[position][0]:supports[position][1]]
                               for position in positions_for_rows]
            target_transforms = transform(target_segments, edit_type, values)
            for method, bank in estimates.items():
                prefix, suffix = other_note_sums(bank)
                segments = [bank[position, supports[position][0]:supports[position][1]]
                            for position in positions_for_rows]
                changed = transform(segments, edit_type, values)
                for row, position, value, target_change, estimate_change in zip(
                    requests, positions_for_rows, values, target_transforms, changed, strict=True
                ):
                    left, right = supports[position]
                    target_edit = edit_segment(targets[position], target_change, left, right, edit_type, value, fade)
                    estimate_edit = edit_segment(bank[position], estimate_change, left, right, edit_type, value, fade)
                    target_other = target_prefix[position] + target_suffix[position + 1]
                    estimate_other = prefix[position] + suffix[position + 1]
                    other_key = (method, position)
                    if other_key not in other_scores:
                        other_scores[other_key] = peaq_score(target_other, estimate_other, peaq)
                    results.append({
                        'request_id': row['request_id'], 'edit': edit_type, 'method': method,
                        'target_2f': peaq_score(target_edit, estimate_edit, peaq),
                        'other_2f': other_scores[other_key],
                        'mix_2f': peaq_score(target_other + target_edit, estimate_other + estimate_edit, peaq),
                    })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--peaq', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.data / 'evaluation.json').read_text())
    edits = json.loads((args.data / 'editing.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator, aso, _ = load_models(str(device), weights_dir=args.weights)
    aso.to(device)
    grouped = defaultdict(list)
    for row in evaluation['rows']:
        grouped[row['piece_id']].append(row)
    results = []
    pieces = sorted(evaluation['pieces'], key=lambda piece: piece['piece_id'])
    for piece in pieces:
        results.extend(evaluate_piece(args.data, piece, grouped[piece['piece_id']], edits,
                                      separator, aso, device, args.peaq))
        print(piece['piece_id'], len(results), 'edit results', flush=True)
    summary = {}
    for edit in edits['conditions']:
        summary[edit] = {}
        for method in ('raw', 'aso'):
            selected = [row for row in results if row['edit'] == edit and row['method'] == method]
            summary[edit][method] = {key: {'mean': float(np.mean([row[key] for row in selected])),
                                          'median': median(row[key] for row in selected)}
                                     for key in ('target_2f', 'other_2f', 'mix_2f')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'rows': results}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
