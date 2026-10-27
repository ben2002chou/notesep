"""Reproduce the paper's instrument-separation evaluation."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .instrument_data import (
    RATE,
    crop_geometry,
    load_bach10,
    load_phenicx,
    load_bach10_references,
    load_phenicx_references,
    score_parts,
    summarize,
)
from .models.aso import (
    ThresholdPrunedTemporalOwnership,
    raw_energy_budget_threshold,
    threshold_credibility,
)
from .models.hpss import project_hpss_batch
from .models.hpss_separator import occurrence_dual_hpss_separator_from_contract
from .models.ownership import TemporalHybridOwnershipConfig

NFFT = 2048
HOP = 512


def stft(y):
    return torch.stft(
        y,
        NFFT,
        HOP,
        window=torch.hann_window(NFFT, device=y.device),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )


def istft(y, length):
    return torch.istft(
        y,
        NFFT,
        HOP,
        window=torch.hann_window(NFFT, device=y.device),
        center=True,
        length=length,
    )


@torch.inference_mode()
def infer_notes(model, crop, events, device, batch_size=16):
    """Separate every score event overlapping the same audio crop."""
    length = len(crop)
    padded = max(16384, math.ceil(length / 1024) * 1024)
    strengths = selective_strengths(events)
    outputs = []
    for first in range(0, len(events), batch_size):
        selected = events[first : first + batch_size]
        mixture = torch.zeros((len(selected), padded), device=device)
        mixture[:, :length] = torch.as_tensor(crop, device=device)
        valid = torch.full((len(selected),), length, device=device, dtype=torch.long)
        harmonic, percussive = project_hpss_batch(
            mixture,
            valid,
            sample_rate=int(model.config.sample_rate),
            n_fft=int(model.config.n_fft),
            hop_length=int(model.config.hop_length),
            kernel_size=31,
            power=2.0,
            margin=1.0,
        )
        pitch = torch.tensor([e["pitch"] for e in selected], device=device)
        onset = torch.tensor([e["local_onset"] for e in selected], device=device)
        offset = torch.tensor([e["local_offset"] for e in selected], device=device)
        gates = torch.tensor(
            strengths[first : first + batch_size], device=device, dtype=harmonic.dtype
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            estimate, _ = model(
                harmonic,
                percussive,
                pitch,
                valid,
                onset,
                offset,
                input_gate_strength=gates,
            )
        outputs.extend(estimate[:, :length].float().cpu().numpy())
    return outputs


def streamed_aso_forward(
    model: ThresholdPrunedTemporalOwnership,
    mixture: torch.Tensor,
    raw: torch.Tensor,
    pitch: torch.Tensor,
    onset: torch.Tensor,
    offset: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Exact two-pass inference with bounded per-note hidden-state memory.

    The first pass accumulates the unchanged Deep-Sets mean/max statistics.
    The second recomputes each independent note trunk with that global context.
    Allocation is then normalized jointly across the complete note bank.
    """
    if chunk_size < 1 or raw.shape[0] != 1 or raw.shape[1] < 2:
        raise ValueError("invalid streamed ASO geometry")
    if model.training or model.config.phase_mode != "mixture":
        raise ValueError("streamed path is inference-only mixture-phase ASO")
    if model._pruning_context is not None or model._acoustic_coordinates is not None:
        raise RuntimeError("reentrant streamed ASO is unsupported")
    frames = mixture.shape[-1]
    centers = (
        torch.arange(frames, device=mixture.device, dtype=mixture.real.dtype)
        * model.hop_length
    )
    onset_distance = torch.tanh(
        (centers[None, None] - onset[..., None].to(centers.dtype))
        / model.boundary_scale_samples
    )
    offset_distance = torch.tanh(
        (offset[..., None].to(centers.dtype) - centers[None, None])
        / model.boundary_scale_samples
    )
    model._acoustic_coordinates = torch.stack((onset_distance, offset_distance), dim=2)
    magnitude = raw.abs()
    raw_threshold = raw_energy_budget_threshold(magnitude, model.energy_budget)
    threshold = (
        raw_threshold.detach().to(magnitude) * model.log_threshold_multiplier.exp()
    )
    gate = threshold_credibility(magnitude, threshold)
    model._pruning_context = {
        "raw_magnitude": magnitude,
        "mixture": mixture,
        "gate": gate,
        "active": (magnitude * gate).sum(1, keepdim=True) > 1e-20,
    }
    try:
        features, base = model._features(mixture, raw, pitch)
        note_count = features.shape[1]

        def encode(value: torch.Tensor) -> torch.Tensor:
            local = F.silu(model.input(value))
            return model.temporal(model.local(local))

        local_sum = None
        local_max = None
        for start in range(0, note_count, chunk_size):
            local = encode(features[:, start : start + chunk_size].flatten(0, 1))
            current_sum = local.sum(0, keepdim=True)
            current_max = local.amax(0, keepdim=True)
            local_sum = current_sum if local_sum is None else local_sum + current_sum
            local_max = (
                current_max
                if local_max is None
                else torch.maximum(local_max, current_max)
            )
        assert local_sum is not None and local_max is not None
        local_mean = local_sum / note_count

        deltas = []
        for start in range(0, note_count, chunk_size):
            count = min(chunk_size, note_count - start)
            local = encode(features[:, start : start + count].flatten(0, 1))
            memory = model.set_fuse(
                torch.cat(
                    (
                        local,
                        local_mean.expand(count, -1, -1, -1),
                        local_max.expand(count, -1, -1, -1),
                    ),
                    dim=1,
                )
            )
            deltas.append(model._predict_magnitude_delta(memory))
        delta = torch.cat(deltas, dim=0)[None]
        delta = model._refine_magnitude_delta(features, delta)
        model._allocate_ownership(base, delta)
        context = model._pruning_context
        mix = mixture[:, None]
        phase = torch.polar(torch.ones_like(mix.real), torch.angle(mix))
        output = torch.empty_like(raw)
        for start in range(0, note_count, chunk_size):
            stop = min(start + chunk_size, note_count)
            output[:, start:stop] = torch.where(
                context["active"],
                context["ownership"][:, start:stop] * mix,
                context["raw_magnitude"][:, start:stop] * phase,
            )
        return output
    finally:
        model._pruning_context = None
        model._acoustic_coordinates = None


def selective_strengths(events: list[dict[str, Any]]) -> list[float]:
    pitch = np.asarray([int(row["pitch"]) for row in events])
    onset = np.asarray([int(row["global_onset"]) for row in events])
    offset = np.asarray([int(row["global_offset"]) for row in events])
    duration = np.maximum(offset - onset, 1)
    overlap = np.maximum(
        np.minimum(offset[:, None], offset[None, :])
        - np.maximum(onset[:, None], onset[None, :]),
        0,
    )
    lower = pitch[:, None] + 12 == pitch[None, :]
    active = (
        lower
        & (overlap >= 1_280)
        & (overlap >= 0.25 * np.minimum(duration[:, None], duration[None, :]))
    )
    return active.any(axis=1).astype(np.float32).tolist()


@torch.inference_mode()
def apply_aso(
    mixture: np.ndarray,
    raw: list[np.ndarray],
    events: list[dict[str, Any]],
    aso: torch.nn.Module,
    device: torch.device,
    *,
    chunk_size: int,
) -> np.ndarray:
    samples = len(mixture)
    raw_wave = torch.from_numpy(np.stack(raw).astype(np.float32)).to(device)
    if len(events) == 1:
        return raw_wave.cpu().numpy()
    mixture_wave = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).to(device)
    mixture_spec = stft(mixture_wave)
    raw_spec = stft(raw_wave)
    pitch = torch.tensor([int(row["pitch"]) for row in events], device=device)[None]
    onset = torch.tensor([int(row["local_onset"]) for row in events], device=device)[
        None
    ]
    offset = torch.tensor([int(row["local_offset"]) for row in events], device=device)[
        None
    ]
    output = streamed_aso_forward(
        aso,
        mixture_spec[None],
        raw_spec[None],
        pitch,
        onset,
        offset,
        chunk_size,
    )
    wave = istft(output[0], samples).float().cpu().numpy()
    if not np.isfinite(wave).all():
        raise ValueError("non-finite ASO bank")
    return wave


def load_models(separator_path, aso_path, device):
    checkpoint = torch.load(separator_path, map_location="cpu", weights_only=False)
    separator = occurrence_dual_hpss_separator_from_contract(
        checkpoint["model_contract"]
    )
    separator.load_state_dict(checkpoint["model"], strict=True)
    checkpoint = torch.load(aso_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["contract"]["model"]
    config = TemporalHybridOwnershipConfig(
        **{
            field.name: model_config[field.name]
            for field in fields(TemporalHybridOwnershipConfig)
            if field.name in model_config
        }
    )
    aso = ThresholdPrunedTemporalOwnership(config)
    aso.load_state_dict(checkpoint["model"], strict=True)
    return separator.to(device).eval(), aso.to(device).eval()


def separate_piece(
    mixture, events, parts, separator, aso, device, batch_size, note_chunk_size
):
    """Return one summed track per instrument, before and after ASO."""
    groups = OrderedDict()
    for event in events:
        event["global_onset"] = round(event["onset_seconds"] * RATE)
        event["global_offset"] = round(event["offset_seconds"] * RATE)
        event["crop"] = crop_geometry(len(mixture), event)
        groups.setdefault(event["crop"], []).append(event)
    estimates = {
        name: np.zeros((len(parts), len(mixture)), dtype=np.float32)
        for name in ("notegrab", "notesep")
    }
    for index, ((start, stop), addressed) in enumerate(groups.items(), 1):
        crop = np.ascontiguousarray(mixture[start:stop], dtype=np.float32)
        context = [
            e for e in events if e["global_offset"] > start and e["global_onset"] < stop
        ]
        for event in context:
            event["local_onset"] = max(
                0, min(len(crop) - 1, event["global_onset"] - start)
            )
            event["local_offset"] = max(
                event["local_onset"] + 1, min(len(crop), event["global_offset"] - start)
            )
        positions = {event["event_index"]: i for i, event in enumerate(context)}
        raw = infer_notes(separator, crop, context, device, batch_size=batch_size)
        refined = apply_aso(crop, raw, context, aso, device, chunk_size=note_chunk_size)
        for event in addressed:
            onset = max(0, min(len(mixture), event["global_onset"]))
            end = max(onset, min(len(mixture), event["global_offset"]))
            local_start = onset - start
            count = min(end - onset, len(crop) - local_start)
            if local_start < 0 or count <= 0:
                raise ValueError("Invalid note crop")
            row = positions[event["event_index"]]
            for name, bank in (("notegrab", raw), ("notesep", refined)):
                estimates[name][event["part_index"], onset : onset + count] += bank[
                    row
                ][local_start : local_start + count]
        if index % 100 == 0:
            print(f"{index}/{len(groups)} crops", flush=True)
    return estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("phenicx", "bach10"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--piece")
    parser.add_argument("--separator", type=Path)
    parser.add_argument("--aso", type=Path)
    parser.add_argument(
        "--summarize", type=Path, help="Summarize scores from all pieces"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--note-chunk-size", type=int, default=4)
    args = parser.parse_args()
    if args.summarize is not None:
        files = sorted(args.summarize.glob("*/scores.json"))
        scores = [json.loads(path.read_text()) for path in files]
        args.output.write_text(
            json.dumps(summarize(args.dataset, scores), indent=2) + "\n"
        )
        return
    if any(
        value is None
        for value in (args.data_root, args.piece, args.separator, args.aso)
    ):
        parser.error(
            "A piece evaluation needs --data-root, --piece, --separator and --aso"
        )
    device = torch.device(args.device)
    loader = load_phenicx if args.dataset == "phenicx" else load_bach10
    mixture, targets, events, parts = loader(args.data_root, args.piece)
    separator, aso = load_models(args.separator, args.aso, device)
    estimates = separate_piece(
        mixture,
        events,
        parts,
        separator,
        aso,
        device,
        args.batch_size,
        args.note_chunk_size,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    reference_loader = (
        load_phenicx_references if args.dataset == "phenicx" else load_bach10_references
    )
    scoring_mixture, scoring_targets = reference_loader(args.data_root, args.piece)
    scores = {
        name: score_parts(scoring_mixture, scoring_targets, bank, parts)
        for name, bank in estimates.items()
    }
    (args.output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    sf.write(args.output / "mixture.wav", mixture, RATE, subtype="FLOAT")
    for index, part in enumerate(parts):
        sf.write(
            args.output / f"{part}-reference.wav", targets[index], RATE, subtype="FLOAT"
        )
        for name, bank in estimates.items():
            sf.write(
                args.output / f"{part}-{name}.wav", bank[index], RATE, subtype="FLOAT"
            )


if __name__ == "__main__":
    main()
