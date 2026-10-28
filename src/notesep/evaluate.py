"""Evaluate NoteSep checkpoints on the SCNS separation benchmark."""

import argparse
import io
import json
import zipfile
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .inference import load_models
from .models.hpss import project_hpss_batch

RATE = 16000


def score(target, estimate):
    """Score one aligned estimate against its isolated reference."""
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if target.ndim != 1 or estimate.ndim != 1:
        raise ValueError("Expected mono waveforms")
    if len(estimate) < len(target):
        estimate = np.pad(estimate, (0, len(target) - len(estimate)))
    estimate = estimate[: len(target)]
    energy = max(float(np.dot(target, target)), 1e-20)
    gain = float(np.dot(estimate, target)) / energy
    residual = estimate - gain * target
    off_target = float(np.dot(residual, residual) / energy)

    return {
        "target_projection_gain": gain,
        "off_target_energy_ratio": off_target,
        "si_sdr_db": 10.0 * np.log10(max(gain * gain, 1e-30) / max(off_target, 1e-30)),
    }


def octave_confusion(estimate, target, candidates):
    """Measure whether an octave competitor matches the estimate better."""

    def correlation(reference):
        reference = np.asarray(reference, dtype=np.float64)
        prediction = np.asarray(estimate, dtype=np.float64)[: len(reference)]
        if len(prediction) < len(reference):
            prediction = np.pad(prediction, (0, len(reference) - len(prediction)))
        denominator = float(
            np.dot(prediction, prediction) * np.dot(reference, reference)
        )
        return (
            float(np.dot(prediction, reference) ** 2 / denominator)
            if denominator
            else 0.0
        )

    target_score = correlation(target)
    scores = {direction: [] for direction in ("lower", "upper")}
    for direction, reference in candidates:
        scores[direction].append(correlation(reference))
    result = {
        direction: max(values) if values else None
        for direction, values in scores.items()
    }
    return {
        "target_rho2": target_score,
        "lower_max_rho2": result["lower"],
        "upper_max_rho2": result["upper"],
        "confusion": any(
            value is not None and value > target_score for value in result.values()
        ),
    }


def source_context(metadata, crop_start_24k, samples):
    """Find all notes that can contribute to a benchmark crop."""
    crop_start = round(crop_start_24k * RATE / 24000)
    crop_stop = crop_start + samples
    context = []
    for event in metadata["events"]:
        onset = round(float(event["onset_seconds"]) * RATE)
        acoustic_end = onset + (int(event["acoustic_samples"]) * RATE + 23999) // 24000
        offset = round(int(event["conditioning_offset_sample"]) * RATE / 24000)
        if max(acoustic_end, offset) <= crop_start or onset >= crop_stop:
            continue
        context.append(
            {
                "event_index": int(event["event_index"]),
                "pitch": int(event["pitch"]),
                "onset": max(0, min(samples, onset - crop_start)),
                "offset": max(1, min(samples, offset - crop_start)),
                "gate_onset": onset - crop_start,
                "gate_offset": offset - crop_start,
            }
        )
    if not context or len(context) > 64:
        raise ValueError(f"Invalid note count in crop: {len(context)}")
    return context


def octave_gate(context, device):
    """Gate lower notes when their octave partner overlaps."""
    pitch = torch.tensor([event["pitch"] for event in context], device=device)
    onset = torch.tensor([event["gate_onset"] for event in context], device=device)
    offset = torch.tensor([event["gate_offset"] for event in context], device=device)
    duration = (offset - onset).clamp_min(1)
    overlap = (
        torch.minimum(offset[:, None], offset[None, :])
        - torch.maximum(onset[:, None], onset[None, :])
    ).clamp_min(0)
    active = (
        (pitch[:, None] + 12 == pitch[None, :])
        & (overlap >= 1280)
        & (overlap >= 0.25 * torch.minimum(duration[:, None], duration[None, :]))
    )
    return active.any(dim=1).float()


@torch.inference_mode()
def separate_crop(audio, context, separator, aso, device, batch_size=4):
    """Return the raw and ASO estimates for every note in one frozen crop."""
    mixture = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(device)
    samples = len(mixture)
    padded_samples = max(16384, ((samples + 1023) // 1024) * 1024)
    padded = torch.zeros((1, padded_samples), device=device)
    padded[0, :samples] = mixture
    valid = torch.tensor([samples], device=device)
    harmonic, percussive = project_hpss_batch(
        padded,
        valid,
        sample_rate=int(separator.config.sample_rate),
        n_fft=int(separator.config.n_fft),
        hop_length=int(separator.config.hop_length),
        kernel_size=31,
        power=2.0,
        margin=1.0,
    )
    pitch = torch.tensor([event["pitch"] for event in context], device=device)
    onset = torch.tensor([event["onset"] for event in context], device=device)
    offset = torch.tensor([event["offset"] for event in context], device=device)
    gate = octave_gate(context, device)
    raw_parts = []
    for start in range(0, len(context), batch_size):
        stop = min(start + batch_size, len(context))
        count = stop - start
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            result, _ = separator(
                harmonic.expand(count, -1),
                percussive.expand(count, -1),
                pitch[start:stop],
                valid.expand(count),
                onset[start:stop],
                offset[start:stop],
                input_gate_strength=gate[start:stop],
            )
        raw_parts.append(result.float())
    raw = torch.cat(raw_parts)[:, :samples]
    window = torch.hann_window(2048, device=device)

    def stft(audio):
        return torch.stft(
            audio,
            2048,
            512,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )

    def istft(spectrum):
        return torch.istft(
            spectrum, 2048, 512, window=window, center=True, length=samples
        ).float()

    # ASO takes a mixture-consistent partition of all raw note estimates.
    mixture_spec, raw_spec = stft(mixture), stft(raw)
    total = raw_spec.abs().sum(0, keepdim=True)
    shares = torch.where(
        total > 1e-20,
        raw_spec.abs() / total.clamp_min(1e-20),
        torch.zeros_like(raw_spec.abs()),
    )
    partition = shares * mixture_spec[None]
    if len(context) == 1:
        refined = istft(partition)
    else:
        result, _ = aso(
            mixture_spec[None],
            partition[None],
            raw_spec[None],
            pitch[None],
            onsets=onset[None],
            acoustic_offsets=offset[None],
        )
        refined = istft(result[0])
    return raw.cpu().numpy(), refined.cpu().numpy()


def read_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def to_model_rate(audio, rate):
    if rate == 16000:
        return audio
    divisor = gcd(rate, 16000)
    return resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)


def place(reference, start, length):
    target = np.zeros(length, dtype=np.float32)
    left = max(0, start)
    source = max(0, -start)
    count = min(len(reference) - source, length - left)
    if count > 0:
        target[left : left + count] = reference[source : source + count]
    return target


def evaluate_piece(root, piece, rows, separator, aso, device):
    """Score each requested note using its benchmark crop and reference."""
    folder = root / piece["root"]
    mixture, rate = read_audio(folder / "mixture.wav")
    if rate != 24000:
        raise ValueError(f"Expected 24 kHz mixture in {folder}")
    metadata = json.loads((folder / "piece.json").read_text())
    references = {}
    with zipfile.ZipFile(folder / "occurrences.zip") as archive:
        for row in rows:
            audio, note_rate = read_audio(
                io.BytesIO(archive.read(row["occurrence_member"]))
            )
            references[int(row["event_index"])] = to_model_rate(audio, note_rate)
    crops = defaultdict(list)
    for row in rows:
        crops[(int(row["crop_start_24k"]), int(row["crop_stop_24k"]))].append(row)
    result = []
    for (start, stop), requests in crops.items():
        audio = to_model_rate(mixture[start:stop], rate)
        # Include every sounding note, even when only one is being scored.
        context = source_context(metadata, start, len(audio))
        index = {item["event_index"]: i for i, item in enumerate(context)}
        raw, refined = separate_crop(audio, context, separator, aso, device)
        for row in requests:
            position = index[int(row["event_index"])]
            onset = round(int(row["onset_sample_24k"]) * 16000 / 24000) - round(
                start * 16000 / 24000
            )
            target = place(
                references[int(row["event_index"])], onset, int(row["samples_16k"])
            )
            candidates = []
            for competitor in row["octave_competitors"]:
                other = next(
                    item
                    for item in rows
                    if int(item["event_index"]) == int(competitor["event_index"])
                )
                other_onset = round(
                    int(other["onset_sample_24k"]) * 16000 / 24000
                ) - round(start * 16000 / 24000)
                other_target = place(
                    references[int(other["event_index"])], other_onset, len(target)
                )
                candidates.append((competitor["direction"], other_target))
            metrics = {}
            for method, estimate in (
                ("raw", raw[position]),
                ("aso", refined[position]),
            ):
                metrics[method] = score(target, estimate)
                if candidates:
                    metrics[method]["octave"] = octave_confusion(
                        estimate, target, candidates
                    )
            result.append(
                {
                    "request_id": row["request_id"],
                    "piece_id": row["piece_id"],
                    "instrument": row["instrument"],
                    **metrics,
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", type=Path, help="Directory containing evaluation.json and pieces/"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, help="Directory containing separator.pt and aso.pt"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    data = json.loads((args.data / "evaluation.json").read_text())
    separator, aso, _ = load_models(args.device, weights_dir=args.weights)
    aso.to(args.device)
    grouped = defaultdict(list)
    for row in data["rows"]:
        grouped[row["piece_id"]].append(row)
    rows = []
    for piece in data["pieces"]:
        rows.extend(
            evaluate_piece(
                args.data,
                piece,
                grouped[piece["piece_id"]],
                separator,
                aso,
                torch.device(args.device),
            )
        )
        print(f"{len(rows)}/{len(data['rows'])} notes", flush=True)
    summary = {}
    for method in ("raw", "aso"):
        values = [row[method] for row in rows]
        eligible = [value["octave"] for value in values if "octave" in value]
        summary[method] = {
            "si_sdr_median_db": float(np.median([v["si_sdr_db"] for v in values])),
            "si_sdr_mean_db": float(np.mean([v["si_sdr_db"] for v in values])),
            "off_target_energy_median": float(
                np.median([v["off_target_energy_ratio"] for v in values])
            ),
            "target_gain_median": float(
                np.median([v["target_projection_gain"] for v in values])
            ),
            "octave_confusion_percent": (
                100.0 * sum(v["confusion"] for v in eligible) / len(eligible)
            )
            if eligible
            else None,
            "octave_eligible_notes": len(eligible),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
