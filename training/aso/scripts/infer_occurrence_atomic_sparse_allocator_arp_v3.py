#!/usr/bin/env python3
"""Target-blind joint sparse-allocator inference for ARP-Test score cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model_exp.occurrence_atomic_merger import (
    OccurrenceAtomicMerger,
    OccurrenceAtomicMergerConfig,
)
from scripts.audit_occurrence_atomic_merge_feasibility import istft, stft
from scripts.evaluate_occurrence_atomic_sparse_allocation import sparsemax
from scripts.train_occurrence_atomic_merger_canary import (
    INPUT_FEATURES,
    SAMPLE_RATE,
    padded_pool,
    teacher_evidence,
)
from scripts.train_occurrence_score_remix_ssl import load_dual_hpss_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(
    run: Path, device: torch.device
) -> tuple[torch.nn.Module, OccurrenceAtomicMerger, argparse.Namespace, dict[str, Any]]:
    run = run.expanduser().resolve()
    contract_path = run / "contract.json"
    result_path = run / "result.json"
    checkpoint_path = run / "checkpoints" / "best.pt"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "occurrence_atomic_sparse_allocator_canary_v1":
        raise ValueError("atomic run has the wrong contract schema")
    if result.get("status") != "complete":
        raise ValueError("atomic run is not complete")
    arguments = argparse.Namespace(**contract["arguments"])
    if int(arguments.atom_frequency_bins) != 1 or int(arguments.atom_time_frames) != 1:
        raise ValueError("ARP atomic inference requires the saved one-bin router")
    backbone, backbone_info = load_dual_hpss_checkpoint(
        Path(arguments.initializer), device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("schema") != "occurrence_atomic_sparse_allocator_canary_v1":
        raise ValueError("atomic router checkpoint has the wrong schema")
    router = OccurrenceAtomicMerger(
        OccurrenceAtomicMergerConfig(**state["config"])
    ).to(device)
    router.load_state_dict(state["model"], strict=True)
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "kind": "occurrence_atomic_sparse_allocator_arp_v3_v1",
        "run": str(run),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(state["step"]),
        "global_step": int(state["step"]),
        "source_run": str(contract["source_run"]),
        "selection": str(contract["selection"]),
        "backbone": backbone_info,
        "inference_inputs": [
            "mixture_waveform",
            "all_score_event_pitches",
            "all_score_event_onsets",
            "all_score_event_offsets",
        ],
        "isolated_target_input": False,
        "assignment": "sparsemax_temperature_1_over_all_active_events_plus_background",
    }
    return backbone, router, arguments, provenance


@torch.inference_mode()
def build_features(
    backbone: torch.nn.Module,
    inference: dict[str, Any],
    score: dict[str, Any],
    device: torch.device,
    arguments: argparse.Namespace,
    *,
    cached_evidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    events = list(score["events"])
    if not events or len(events) > int(arguments.maximum_sources):
        raise ValueError(f"ARP score event count outside router authority: {len(events)}")
    target_positions = [index for index, event in enumerate(events) if event["is_target"]]
    if len(target_positions) != 1:
        raise ValueError("ARP score must contain exactly one designated target")
    mixture_waveform = torch.as_tensor(
        np.asarray(inference["mixture"], dtype=np.float32), device=device
    )
    valid_samples = min(int(inference["valid_samples"]), int(mixture_waveform.numel()))
    pitches = torch.tensor([int(event["pitch"]) for event in events], device=device)
    onsets = torch.tensor([int(event["onset_sample"]) for event in events], device=device)
    offsets = torch.tensor([int(event["offset_sample"]) for event in events], device=device)
    bank = {
        "mixture": mixture_waveform[None],
        "valid": torch.tensor([valid_samples], device=device),
        "assignment": torch.zeros(len(events), dtype=torch.long, device=device),
        "pitch": pitches,
        "onset": onsets,
        "offset": offsets,
    }
    if cached_evidence is None:
        teacher_waveform, harmonic_logits, percussive_logits = teacher_evidence(
            backbone, bank, query_microbatch=int(arguments.query_microbatch)
        )
    else:
        teacher_waveform, harmonic_logits, percussive_logits = cached_evidence
        if any(value.shape[0] != len(events) or value.device != device
               for value in cached_evidence):
            raise ValueError("cached evidence note count/device mismatch")
    mixture = stft(mixture_waveform, int(arguments.n_fft))
    anchors = stft(teacher_waveform, int(arguments.n_fft))
    anchor_background = mixture - anchors.sum(dim=0)
    anchor_bank = torch.cat((anchors, anchor_background[None]), dim=0)
    sources, frequency, frames = anchor_bank.shape

    harmonic_logits = F.interpolate(
        harmonic_logits[:, None], size=(frequency, frames), mode="bilinear", align_corners=False
    )[:, 0]
    percussive_logits = F.interpolate(
        percussive_logits[:, None], size=(frequency, frames), mode="bilinear", align_corners=False
    )[:, 0]
    harmonic_logits = torch.cat(
        (harmonic_logits, -harmonic_logits.amax(dim=0, keepdim=True)), dim=0
    )
    percussive_logits = torch.cat(
        (percussive_logits, -percussive_logits.amax(dim=0, keepdim=True)), dim=0
    )

    mixture_power = mixture.abs().square()
    mixture_floor = 0.01 * mixture_power.mean().clamp_min(1e-10)
    phase_ratio = anchor_bank * mixture.conj()[None] / (mixture_power + mixture_floor)[None]
    log_anchor = torch.log1p(100.0 * anchor_bank.abs()) / 5.0
    log_mixture = torch.log1p(100.0 * mixture.abs()) / 5.0
    log_flux = F.pad((log_mixture[:, 1:] - log_mixture[:, :-1]).abs(), (1, 0))
    anchor_share = anchor_bank.abs() / anchor_bank.abs().sum(dim=0, keepdim=True).clamp_min(1e-8)

    frequency_hz = torch.arange(frequency, device=device) * SAMPLE_RATE / int(arguments.n_fft)
    f0 = 440.0 * torch.pow(
        torch.tensor(2.0, device=device), (pitches.float() - 69.0) / 12.0
    )
    relative = 12.0 * torch.log2(
        frequency_hz.clamp_min(SAMPLE_RATE / int(arguments.n_fft))[None] / f0[:, None]
    )
    zero_frequency = torch.zeros((1, frequency), device=device)
    relative_clipped = torch.cat(((relative / 48.0).clamp(-1.0, 1.0), zero_frequency), dim=0)
    relative_sine = torch.cat((torch.sin(torch.pi * relative / 6.0), zero_frequency), dim=0)
    relative_cosine = torch.cat((torch.cos(torch.pi * relative / 6.0), zero_frequency), dim=0)

    frame_sample = torch.arange(frames, device=device) * (int(arguments.n_fft) // 4)
    active = (frame_sample[None] >= onsets[:, None]) & (frame_sample[None] < offsets[:, None])
    onset_state = active & (frame_sample[None] < onsets[:, None] + SAMPLE_RATE // 25)
    held_state = active & ~onset_state
    off_state = ~active
    since = ((frame_sample[None] - onsets[:, None]).float() / (2 * SAMPLE_RATE)).clamp(0, 1)
    until = ((offsets[:, None] - frame_sample[None]).float() / (2 * SAMPLE_RATE)).clamp(0, 1)
    off_state = torch.cat((off_state, torch.ones((1, frames), dtype=torch.bool, device=device)))
    held_state = torch.cat((held_state, torch.zeros((1, frames), dtype=torch.bool, device=device)))
    onset_state = torch.cat((onset_state, torch.zeros((1, frames), dtype=torch.bool, device=device)))
    since = torch.cat((since, torch.zeros((1, frames), device=device)))
    until = torch.cat((until, torch.zeros((1, frames), device=device)))
    background = torch.zeros((sources, 1, 1), device=device)
    background[-1] = 1.0
    frequency_coordinate = torch.linspace(0, 1, frequency, device=device)[None, :, None]
    time_coordinate = torch.linspace(0, 1, frames, device=device)[None, None, :]
    expanded_log_mixture = log_mixture[None].expand(sources, -1, -1)
    expanded_flux = log_flux[None].expand(sources, -1, -1)
    features = torch.stack(
        (
            (phase_ratio.real / 4.0).clamp(-1, 1),
            (phase_ratio.imag / 4.0).clamp(-1, 1),
            log_anchor,
            expanded_log_mixture,
            (harmonic_logits / 8.0).clamp(-1, 1),
            (percussive_logits / 8.0).clamp(-1, 1),
            relative_clipped[:, :, None].expand(-1, -1, frames),
            relative_sine[:, :, None].expand(-1, -1, frames),
            relative_cosine[:, :, None].expand(-1, -1, frames),
            off_state[:, None].expand(-1, frequency, -1).float(),
            held_state[:, None].expand(-1, frequency, -1).float(),
            onset_state[:, None].expand(-1, frequency, -1).float(),
            since[:, None].expand(-1, frequency, -1),
            until[:, None].expand(-1, frequency, -1),
            background.expand(-1, frequency, frames),
            anchor_share,
            expanded_flux,
            frequency_coordinate.expand(sources, -1, frames),
            time_coordinate.expand(sources, frequency, -1),
        ),
        dim=1,
    )
    if features.shape[1] != INPUT_FEATURES:
        raise RuntimeError("atomic ARP feature contract changed")
    tile = (int(arguments.atom_frequency_bins), int(arguments.atom_time_frames))
    return padded_pool(features, tile), mixture, teacher_waveform, target_positions[0]


@torch.inference_mode()
def infer(
    backbone: torch.nn.Module,
    router: OccurrenceAtomicMerger,
    arguments: argparse.Namespace,
    inference_rows: list[dict[str, Any]],
    full_score_rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    teacher_outputs: list[np.ndarray] = []
    sparse_outputs: list[np.ndarray] = []
    started = time.time()
    for index, (inference_row, score_row) in enumerate(
        zip(inference_rows, full_score_rows, strict=True), start=1
    ):
        features, mixture, teacher_waveform, target_position = build_features(
            backbone, inference_row, score_row, device, arguments
        )
        allocation = sparsemax(router(features.float()), 1.0)
        estimated_bank = allocation * mixture[None]
        length = len(inference_row["mixture"])
        sparse_waveform = istft(
            estimated_bank[target_position], int(arguments.n_fft), length
        )
        teacher_outputs.append(
            teacher_waveform[target_position, :length].float().cpu().numpy()
        )
        sparse_outputs.append(sparse_waveform.float().cpu().numpy())
        if index == 1 or index % 8 == 0 or index == len(inference_rows):
            print(
                json.dumps(
                    {
                        "atomic_sparse_progress": index,
                        "cases": len(inference_rows),
                        "events": len(score_row["events"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    elapsed = time.time() - started
    return teacher_outputs, sparse_outputs, {
        "cases": float(len(inference_rows)),
        "elapsed_seconds": elapsed,
        "audio_seconds": sum(len(row["mixture"]) for row in inference_rows) / SAMPLE_RATE,
    }
