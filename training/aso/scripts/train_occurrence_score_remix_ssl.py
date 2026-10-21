#!/usr/bin/env python3
"""Adapt an RT2 occurrence separator on real audio by score-conditioned remixing."""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Iterator

import numpy as np
import pretty_midi
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.hpss_target import project_hpss_batch  # noqa: E402
from model_exp.dna_residual_v13_streaming import (  # noqa: E402
    FrozenV13Contract,
    ResumeSafeBatchSampler,
    V13StreamingResidualDataset,
)
from model_exp.occurrence_dual_hpss_separator import (  # noqa: E402
    occurrence_dual_hpss_separator_from_contract,
)
from model_exp.occurrence_score_remix_ssl import (  # noqa: E402
    batch_derangement,
    collision_free_partner_indices,
    collision_free_triplet_indices,
    confidence_filter,
    masked_rms,
    note_crop_bounds,
    note_fits_timeline,
    per_example_masked_waveform_l1,
    per_example_pre_onset_rms,
    per_example_projection_gain,
    real_mixture_invariance_remix,
    relative_rms_interferer_scale,
    restore_teacher_state,
    rt2_absent_bounds,
    score_absent_pitch,
    score_complete_mixture_closure,
    score_event_groups_temporally_collide,
    select_overlapping_competing_score_events,
    select_score_group_pair_positions,
    select_collision_free_triplet_anchors,
    self_remix,
    symmetric_real_mixture_remix,
    threeway_real_mixture_invariance_remix,
    trainable_parameter_names,
    weighted_mean,
)
from model_exp.occurrence_nonlinear_property_view import (  # noqa: E402
    apply_nonlinear_property_view,
)
from model_exp.real_score_property_metrics import (  # noqa: E402
    competing_harmonic_leakage_ratio,
)
import scripts.train_dna_residual_v13_streaming as synthetic_corpus  # noqa: E402


SCHEMA = "occurrence_score_conditioned_remix_ssl_v11"
RESUME_SCHEMAS = {
    SCHEMA,
    "occurrence_score_conditioned_remix_ssl_v10",
    "occurrence_score_conditioned_remix_ssl_v9",
    "occurrence_score_conditioned_remix_ssl_v8",
    "occurrence_score_conditioned_remix_ssl_v7",
    "occurrence_score_conditioned_remix_ssl_v6",
    "occurrence_score_conditioned_remix_ssl_v5",
    "occurrence_score_conditioned_remix_ssl_v4",
    "occurrence_score_conditioned_remix_ssl_v3",
    "occurrence_score_conditioned_remix_ssl_v2",
}
SAMPLE_RATE = 16_000


def gradient_pair_geometry(
    left_gradients: tuple[torch.Tensor | None, ...],
    right_gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, float | bool]:
    """Measure objective geometry without composing or mutating gradients."""

    if len(left_gradients) != len(right_gradients):
        raise ValueError("gradient tuples must align")
    reference = next(
        (gradient for gradient in (*left_gradients, *right_gradients) if gradient is not None),
        None,
    )
    if reference is None:
        raise ValueError("at least one gradient must be present")
    zero = torch.zeros((), device=reference.device, dtype=torch.float32)
    left_sq = zero.clone()
    right_sq = zero.clone()
    dot = zero.clone()
    for left, right in zip(left_gradients, right_gradients, strict=True):
        if left is not None:
            left_sq = left_sq + left.float().square().sum()
        if right is not None:
            right_sq = right_sq + right.float().square().sum()
        if left is not None and right is not None:
            dot = dot + (left.float() * right.float()).sum()
    left_norm = left_sq.sqrt()
    right_norm = right_sq.sqrt()
    cosine = dot / (left_norm * right_norm).clamp_min(1e-12)
    conflicting = bool(dot < 0.0)
    # If PCGrad protects ``right``, this is the norm fraction of a standalone
    # ``left`` objective that survives removal of its opposing right-parallel
    # component.  It diagnoses whether rehearsal structurally suppresses the
    # property signal rather than merely reporting that the aggregate conflicts.
    survival = torch.ones_like(cosine)
    if conflicting and left_norm > 0.0 and right_norm > 0.0:
        survival = (1.0 - cosine.square()).clamp_min(0.0).sqrt()
    return {
        "left_gradient_norm": float(left_norm),
        "right_gradient_norm": float(right_norm),
        "gradient_cosine": float(cosine),
        "gradient_conflicting": conflicting,
        "left_pcgrad_survival_fraction": float(survival),
    }


def compose_real_protected_replay_gradients(
    real_gradients: tuple[torch.Tensor | None, ...],
    replay_gradients: tuple[torch.Tensor | None, ...],
    *,
    replay_weight: float,
    mode: str,
    maximum_replay_to_real_norm_ratio: float,
) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float | bool]]:
    """Compose real and rehearsal gradients with an explicit protected objective."""

    if mode not in {
        "sum",
        "protect_real_pcgrad",
        "protect_real_gate",
        "protect_replay_pcgrad",
        "protect_replay_gate",
    }:
        raise ValueError(f"unknown replay gradient mode: {mode}")
    if replay_weight < 0.0 or maximum_replay_to_real_norm_ratio < 0.0:
        raise ValueError("replay gradient weights and caps must be nonnegative")
    if len(real_gradients) != len(replay_gradients):
        raise ValueError("real and replay gradient tuples must align")
    reference = next(
        (gradient for gradient in (*real_gradients, *replay_gradients) if gradient is not None),
        None,
    )
    if reference is None:
        raise ValueError("at least one real or replay gradient must be present")
    zero = torch.zeros((), device=reference.device, dtype=torch.float32)
    real_sq = zero.clone()
    replay_sq = zero.clone()
    dot = zero.clone()
    for real_gradient, replay_gradient in zip(
        real_gradients, replay_gradients, strict=True
    ):
        if real_gradient is not None:
            real_sq = real_sq + real_gradient.float().square().sum()
        if replay_gradient is not None:
            replay_sq = replay_sq + replay_gradient.float().square().sum()
        if real_gradient is not None and replay_gradient is not None:
            dot = dot + (real_gradient.float() * replay_gradient.float()).sum()
    real_norm = real_sq.sqrt()
    replay_norm = replay_sq.sqrt()
    cosine = dot / (real_norm * replay_norm).clamp_min(1e-12)
    conflicting = bool(dot < 0.0)

    real_components: list[torch.Tensor | None] = list(real_gradients)
    replay_components: list[torch.Tensor | None] = list(replay_gradients)
    projected = False
    if mode == "protect_real_gate" and conflicting:
        replay_components = [None for _ in replay_gradients]
    elif mode == "protect_real_pcgrad" and conflicting:
        coefficient = dot / real_sq.clamp_min(1e-12)
        replay_components = [
            (
                replay_gradient - coefficient.to(replay_gradient.dtype) * real_gradient
                if real_gradient is not None and replay_gradient is not None
                else replay_gradient
            )
            for real_gradient, replay_gradient in zip(
                real_gradients, replay_gradients, strict=True
            )
        ]
        projected = True
    elif mode == "protect_replay_gate" and conflicting:
        real_components = [None for _ in real_gradients]
    elif mode == "protect_replay_pcgrad" and conflicting:
        coefficient = dot / replay_sq.clamp_min(1e-12)
        real_components = [
            (
                real_gradient - coefficient.to(real_gradient.dtype) * replay_gradient
                if real_gradient is not None and replay_gradient is not None
                else real_gradient
            )
            for real_gradient, replay_gradient in zip(
                real_gradients, replay_gradients, strict=True
            )
        ]
        projected = True

    replay_component_sq = zero.clone()
    for replay_gradient in replay_components:
        if replay_gradient is not None:
            replay_component_sq = (
                replay_component_sq + replay_gradient.float().square().sum()
            )
    replay_component_norm = replay_component_sq.sqrt()
    applied_weight = replay_weight
    if mode == "protect_real_gate" and conflicting:
        applied_weight = 0.0
    if maximum_replay_to_real_norm_ratio and replay_component_norm > 0.0:
        cap = (
            maximum_replay_to_real_norm_ratio
            * real_norm
            / (replay_weight * replay_component_norm).clamp_min(1e-12)
        )
        applied_weight *= min(1.0, float(cap))

    combined: list[torch.Tensor | None] = []
    for real_gradient, replay_gradient in zip(
        real_components, replay_components, strict=True
    ):
        if real_gradient is None:
            value = (
                None
                if replay_gradient is None
                else applied_weight * replay_gradient
            )
        elif replay_gradient is None:
            value = real_gradient
        else:
            value = real_gradient + applied_weight * replay_gradient
        combined.append(value)
    raw_ratio = replay_weight * replay_norm / real_norm.clamp_min(1e-12)
    applied_ratio = (
        applied_weight * replay_component_norm / real_norm.clamp_min(1e-12)
    )
    return tuple(combined), {
        "real_gradient_norm": float(real_norm),
        "replay_gradient_norm": float(replay_norm),
        "gradient_conflict_cosine": float(cosine),
        "gradient_conflicting": conflicting,
        "replay_to_real_gradient_norm_ratio": float(raw_ratio),
        "applied_replay_to_real_gradient_norm_ratio": float(applied_ratio),
        "replay_gradient_applied_weight": float(applied_weight),
        "replay_gradient_projected": projected,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dual_hpss_checkpoint(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the exact ordinary margin-1 HPSS occurrence authority."""

    resolved = path.expanduser().resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != (
        "dna_residual_v13_streaming_checkpoint_v1"
    ):
        raise ValueError("initializer has the wrong dual HPSS schema")
    model = occurrence_dual_hpss_separator_from_contract(
        checkpoint.get("model_contract")
    )
    model.load_state_dict(checkpoint.get("model"), strict=True)
    loss_contract = checkpoint.get("loss_contract") or {}
    decomposition = loss_contract.get("decomposition") or {}
    if decomposition.get("type") != "ordinary_hpss_harmonic_and_percussive":
        raise ValueError("initializer is not the ordinary HPSS experiment")
    if float(decomposition.get("margin", -1.0)) != 1.0:
        raise ValueError("initializer HPSS margin is not one")
    provenance = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "schema": checkpoint["schema"],
        "global_step": int(checkpoint["global_step"]),
        "model_contract": checkpoint["model_contract"],
        "loss_contract": loss_contract,
        "corpus_binding": checkpoint.get("corpus_binding"),
        "run_contract": checkpoint.get("run_contract"),
    }
    return model.to(device).eval(), provenance


@dataclass(frozen=True)
class Track:
    source_id: str
    audio_path: Path
    duration_sec: float
    notes: tuple[tuple[float, float, int], ...]
    corpus: str = "maestro"
    score_complete: bool = True
    active_pitch_set_complete: bool = True


@dataclass(frozen=True)
class Query:
    track: int
    note: int
    pitch: int
    onset_polyphony: int
    active_pitch_bits: int
    corpus: str = "maestro"
    score_complete: bool = True
    active_pitch_set_complete: bool = True


def _pretty_midi_notes(path: Path) -> tuple[tuple[float, float, int], ...]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return tuple(
        sorted(
            (
                (float(note.start), float(note.end), int(note.pitch))
                for instrument in midi.instruments
                if not instrument.is_drum
                for note in instrument.notes
                if note.end > note.start
            ),
            key=lambda value: (value[0], value[1], value[2]),
        )
    )


def _load_aligned_manifest_tracks(
    entry: dict[str, Any], *, corpus: str, root: Path, split: str, maximum: int
) -> tuple[list[Track], dict[str, Any]]:
    """Load a dataset-specific conversion through one canonical track manifest.

    Dataset converters own their native annotation quirks.  The trainer only
    consumes frozen JSONL rows containing an audio path and normalized
    ``[onset_sec, offset_sec, midi_pitch]`` events.  Partial melody/F0 labels
    must be declared explicitly so full-mixture closure cannot silently treat
    them as complete scores.
    """

    manifest = Path(entry["manifest"])
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.expanduser().resolve()
    allow_partial = bool(entry.get("allow_partial_score", False))
    local: list[Track] = []
    source_ids: set[str] = set()
    scope_counts: dict[str, int] = {}
    active_pitch_scope_counts: dict[str, int] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != split:
                continue
            source_id = str(row["source_id"])
            if source_id in source_ids:
                raise ValueError(f"duplicate source_id in {manifest}:{line_number}")
            source_ids.add(source_id)
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_path = audio_path.expanduser().resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            duration_sec = float(row["duration_sec"])
            score_scope = str(row.get("score_scope", "complete"))
            if score_scope not in {"complete", "partial"}:
                raise ValueError(f"invalid score_scope in {manifest}:{line_number}")
            if score_scope == "partial" and not allow_partial:
                raise ValueError(
                    f"partial score requires allow_partial_score in {corpus}"
                )
            active_pitch_set_complete = bool(
                row.get("active_pitch_set_complete", score_scope == "complete")
            )
            notes = tuple(
                sorted(
                    (
                        (float(note[0]), float(note[1]), int(note[2]))
                        for note in row["notes"]
                    ),
                    key=lambda value: (value[0], value[1], value[2]),
                )
            )
            if duration_sec <= 0.0 or not notes:
                raise ValueError(f"empty or invalid track in {manifest}:{line_number}")
            if any(
                onset < 0.0
                or offset <= onset
                or offset > duration_sec + 0.1
                or not 0 <= pitch < 128
                for onset, offset, pitch in notes
            ):
                raise ValueError(f"invalid note event in {manifest}:{line_number}")
            local.append(
                Track(
                    source_id=f"{corpus}:{source_id}",
                    audio_path=audio_path,
                    duration_sec=duration_sec,
                    notes=notes,
                    corpus=corpus,
                    score_complete=score_scope == "complete",
                    active_pitch_set_complete=active_pitch_set_complete,
                )
            )
            scope_counts[score_scope] = scope_counts.get(score_scope, 0) + 1
            active_pitch_scope = (
                "complete" if active_pitch_set_complete else "partial"
            )
            active_pitch_scope_counts[active_pitch_scope] = (
                active_pitch_scope_counts.get(active_pitch_scope, 0) + 1
            )
            if maximum and len(local) >= maximum:
                break
    return local, {
        "path": str(manifest),
        "sha256": file_sha256(manifest),
        "score_scope_track_counts": scope_counts,
        "active_pitch_scope_track_counts": active_pitch_scope_counts,
    }


def _load_configured_real_tracks(
    config_path: Path,
) -> tuple[list[Track], dict[str, float], dict[str, Any]]:
    """Load QC-approved aligned real recordings from a frozen mixture config."""

    resolved = config_path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "occurrence_real_ssl_corpus_mixture_v1":
        raise ValueError("real corpus config has the wrong schema")
    entries = payload.get("corpora")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("real corpus config must contain at least two corpora")
    tracks: list[Track] = []
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    canonical_manifests: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        corpus = str(entry["id"])
        kind = str(entry["kind"])
        root = Path(entry["root"]).expanduser().resolve()
        split = str(entry.get("split", "train"))
        maximum = int(entry.get("max_tracks", 0))
        weight = float(entry["weight"])
        if split != "train" or weight <= 0.0:
            raise ValueError(f"invalid real corpus split or weight for {corpus}")
        local: list[Track] = []
        if kind == "maestro_v3":
            csv_path = root / "maestro-v3.0.0.csv"
            with csv_path.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] != split:
                        continue
                    notes = _pretty_midi_notes(root / row["midi_filename"])
                    if notes:
                        local.append(
                            Track(
                                source_id=f"{corpus}:{Path(row['audio_filename']).stem}",
                                audio_path=(root / row["audio_filename"]).resolve(),
                                duration_sec=float(row["duration"]),
                                notes=notes,
                                corpus=corpus,
                            )
                        )
                    if maximum and len(local) >= maximum:
                        break
        elif kind == "urmp":
            validation_pieces = {1, 2, 12, 13, 24, 25, 31, 38, 39}
            for piece_dir in sorted(root.iterdir()):
                if not piece_dir.is_dir() or not piece_dir.name[:1].isdigit():
                    continue
                piece_number = int(piece_dir.name.split("_", 1)[0])
                if piece_number in validation_pieces:
                    continue
                audio_path = piece_dir / f"AuMix_{piece_dir.name}.wav"
                midi_path = piece_dir / f"Sco_{piece_dir.name}.mid"
                if not audio_path.is_file() or not midi_path.is_file():
                    continue
                notes = _pretty_midi_notes(midi_path)
                if notes:
                    info = sf.info(audio_path)
                    local.append(
                        Track(
                            source_id=f"{corpus}:{piece_dir.name}",
                            audio_path=audio_path.resolve(),
                            duration_sec=float(info.frames / info.samplerate),
                            notes=notes,
                            corpus=corpus,
                        )
                    )
                if maximum and len(local) >= maximum:
                    break
        elif kind == "guitarset":
            train_players = set(map(str, entry.get("train_players", ["00", "01", "02", "03"])))
            annotation_root = root / "annotation"
            audio_root = root / "audio" / "mono_mic"
            for annotation_path in sorted(annotation_root.glob("*.jams")):
                if annotation_path.name.split("_", 1)[0] not in train_players:
                    continue
                audio_path = audio_root / annotation_path.name.replace(".jams", "_mic.wav")
                if not audio_path.is_file():
                    audio_path = audio_root / annotation_path.name.replace(".jams", "_mix.wav")
                if not audio_path.is_file():
                    continue
                jams = json.loads(annotation_path.read_text(encoding="utf-8"))
                notes = tuple(
                    sorted(
                        (
                            (float(row["time"]), float(row["time"]) + float(row["duration"]), int(round(row["value"])))
                            for annotation in jams.get("annotations", [])
                            if annotation.get("namespace") == "note_midi"
                            for row in annotation.get("data", [])
                            if float(row["duration"]) > 0.0
                        ),
                        key=lambda value: (value[0], value[1], value[2]),
                    )
                )
                if notes:
                    info = sf.info(audio_path)
                    local.append(
                        Track(
                            source_id=f"{corpus}:{annotation_path.stem}",
                            audio_path=audio_path.resolve(),
                            duration_sec=float(info.frames / info.samplerate),
                            notes=notes,
                            corpus=corpus,
                        )
                    )
                if maximum and len(local) >= maximum:
                    break
        elif kind == "aligned_manifest_v1":
            local, canonical_manifests[corpus] = _load_aligned_manifest_tracks(
                entry,
                corpus=corpus,
                root=root,
                split=split,
                maximum=maximum,
            )
        else:
            raise ValueError(f"unsupported real corpus kind: {kind}")
        if not local:
            raise ValueError(f"real corpus {corpus} loaded no tracks")
        tracks.extend(local)
        weights[corpus] = weight
        counts[corpus] = len(local)
    total_weight = sum(weights.values())
    if len(weights) < 2 or not math.isclose(total_weight, 1.0, abs_tol=1e-8):
        raise ValueError("enabled real corpus weights must sum to one")
    return tracks, weights, {
        "schema": payload["schema"],
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "corpus_track_counts": counts,
        "corpus_weights": weights,
        "canonical_manifests": canonical_manifests,
        "split": "train_only",
        "synthetic_counted_as_real": False,
    }


class MaestroScoreCropDataset(Dataset[dict[str, Any]]):
    """Real MAESTRO crops with score queries and no source targets."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        crop_samples: int,
        lead_samples: int,
        release_samples: int,
        max_tracks: int,
        minimum_note_sec: float,
        maximum_anchor_polyphony: int,
        seed: int,
        tracks_override: list[Track] | None = None,
    ) -> None:
        if split != "train":
            raise ValueError("SSL adaptation is restricted to MAESTRO train")
        self.root = root.expanduser().resolve()
        self.crop_samples = int(crop_samples)
        self.lead_samples = int(lead_samples)
        self.release_samples = int(release_samples)
        self.seed = int(seed)
        tracks: list[Track] = list(tracks_override or [])
        if tracks_override is None:
            csv_path = self.root / "maestro-v3.0.0.csv"
            with csv_path.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] != split:
                        continue
                    notes = _pretty_midi_notes(self.root / row["midi_filename"])
                    if notes:
                        tracks.append(
                            Track(
                                source_id=Path(row["audio_filename"]).stem,
                                audio_path=(self.root / row["audio_filename"]).resolve(),
                                duration_sec=float(row["duration"]),
                                notes=notes,
                            )
                        )
                    if max_tracks and len(tracks) >= max_tracks:
                        break
        if not tracks:
            raise ValueError("no MAESTRO train tracks were loaded")
        self.tracks = tuple(tracks)
        maximum_note_sec = (
            self.crop_samples - self.lead_samples - self.release_samples
        ) / SAMPLE_RATE
        queries: list[Query] = []
        for track_index, track in enumerate(self.tracks):
            starts = [note[0] for note in track.notes]
            ends = sorted(note[1] for note in track.notes)
            intervals_by_pitch: dict[int, list[tuple[float, float]]] = {}
            for note_onset, note_offset, note_pitch in track.notes:
                intervals_by_pitch.setdefault(note_pitch, []).append(
                    (note_onset, note_offset)
                )
            pitch_intervals: dict[int, tuple[list[float], list[float]]] = {}
            for candidate_pitch, intervals in intervals_by_pitch.items():
                pitch_starts = [interval[0] for interval in intervals]
                prefix_max_ends: list[float] = []
                maximum_end = -math.inf
                for _, interval_end in intervals:
                    maximum_end = max(maximum_end, interval_end)
                    prefix_max_ends.append(maximum_end)
                pitch_intervals[candidate_pitch] = (
                    pitch_starts,
                    prefix_max_ends,
                )
            for note_index, (onset, offset, pitch) in enumerate(track.notes):
                onset_polyphony = bisect.bisect_right(
                    starts, onset
                ) - bisect.bisect_right(ends, onset)
                if (
                    minimum_note_sec <= offset - onset <= maximum_note_sec
                    and (
                        maximum_anchor_polyphony <= 0
                        or onset_polyphony <= maximum_anchor_polyphony
                    )
                    and note_fits_timeline(
                        onset_sec=onset,
                        offset_sec=offset,
                        duration_sec=track.duration_sec,
                        crop_samples=self.crop_samples,
                        sample_rate=SAMPLE_RATE,
                    )
                ):
                    total = max(
                        self.crop_samples,
                        int(round(track.duration_sec * SAMPLE_RATE)),
                    )
                    identity = f"{track.source_id}:{note_index}:{self.seed}"
                    crop_start, _, _ = note_crop_bounds(
                        onset_sample=int(round(onset * SAMPLE_RATE)),
                        offset_sample=int(round(offset * SAMPLE_RATE)),
                        total_samples=total,
                        crop_samples=self.crop_samples,
                        lead_samples=self.lead_samples,
                        identity=identity,
                    )
                    crop_start_sec = crop_start / SAMPLE_RATE
                    crop_stop_sec = min(
                        track.duration_sec,
                        (crop_start + self.crop_samples) / SAMPLE_RATE,
                    )
                    active_pitch_bits = 0
                    for candidate_pitch, (
                        pitch_starts,
                        prefix_max_ends,
                    ) in pitch_intervals.items():
                        last_started = (
                            bisect.bisect_left(pitch_starts, crop_stop_sec) - 1
                        )
                        if (
                            last_started >= 0
                            and prefix_max_ends[last_started] > crop_start_sec
                        ):
                            active_pitch_bits |= 1 << candidate_pitch
                    queries.append(
                        Query(
                            track=track_index,
                            note=note_index,
                            pitch=int(pitch),
                            onset_polyphony=int(onset_polyphony),
                            active_pitch_bits=int(active_pitch_bits),
                            corpus=track.corpus,
                            score_complete=track.score_complete,
                            active_pitch_set_complete=(
                                track.active_pitch_set_complete
                            ),
                        )
                    )
        self.queries = tuple(queries)
        self.query_indices_by_corpus = {
            corpus: tuple(
                index for index, query in enumerate(self.queries) if query.corpus == corpus
            )
            for corpus in sorted({query.corpus for query in self.queries})
        }
        if len(self.queries) < 2:
            raise ValueError("MAESTRO query filter retained fewer than two notes")

    def __len__(self) -> int:
        return len(self.queries)

    @staticmethod
    def _read_resampled(path: Path, start_sec: float, samples: int) -> tuple[np.ndarray, int]:
        with sf.SoundFile(path) as handle:
            source_rate = int(handle.samplerate)
            source_start = max(0, int(math.floor(start_sec * source_rate)))
            source_count = int(math.ceil(samples * source_rate / SAMPLE_RATE)) + 4
            handle.seek(source_start)
            audio = handle.read(source_count, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1, dtype=np.float32)
        if source_rate != SAMPLE_RATE:
            divisor = math.gcd(source_rate, SAMPLE_RATE)
            mono = resample_poly(
                mono,
                SAMPLE_RATE // divisor,
                source_rate // divisor,
            ).astype(np.float32, copy=False)
        output = np.zeros(samples, dtype=np.float32)
        valid = min(samples, len(mono))
        output[:valid] = mono[:valid]
        return output, valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = self.queries[int(index)]
        track = self.tracks[query.track]
        onset_sec, offset_sec, pitch = track.notes[query.note]
        if pitch != query.pitch:
            raise AssertionError("query pitch metadata changed")
        total = max(self.crop_samples, int(round(track.duration_sec * SAMPLE_RATE)))
        onset = int(round(onset_sec * SAMPLE_RATE))
        offset = int(round(offset_sec * SAMPLE_RATE))
        identity = f"{track.source_id}:{query.note}:{self.seed}"
        start, local_onset, local_offset = note_crop_bounds(
            onset_sample=onset,
            offset_sample=offset,
            total_samples=total,
            crop_samples=self.crop_samples,
            lead_samples=self.lead_samples,
            identity=identity,
        )
        audio, valid = self._read_resampled(
            track.audio_path, start / SAMPLE_RATE, self.crop_samples
        )
        crop_stop_sec = (start + valid) / SAMPLE_RATE
        crop_start_sec = start / SAMPLE_RATE
        active_pitches = [
            candidate_pitch
            for candidate_pitch in range(128)
            if query.active_pitch_bits & (1 << candidate_pitch)
        ]
        active_pitch_mask = torch.zeros(128, dtype=torch.bool)
        active_pitch_mask[active_pitches] = True
        absent = score_absent_pitch(active_pitches, target_pitch=pitch, identity=identity)
        # Preserve every transcribed occurrence overlapping the crop, rather
        # than collapsing to the active-pitch set.  Full-mixture closure and
        # certified collision-free remixing additionally require the manifest
        # to declare this track's score complete; partial melody annotations
        # remain usable only by objectives that do not assume exhaustiveness.
        score_events = []
        for event_onset_sec, event_offset_sec, event_pitch in track.notes:
            event_onset = max(
                0, int(round(event_onset_sec * SAMPLE_RATE)) - start
            )
            event_offset = min(
                valid, int(round(event_offset_sec * SAMPLE_RATE)) - start
            )
            if event_onset < event_offset:
                score_events.append((int(event_pitch), event_onset, event_offset))
        return {
            "mixture": torch.from_numpy(audio),
            "corpus": query.corpus,
            "pitch": torch.tensor(pitch, dtype=torch.long),
            "absent_pitch": torch.tensor(absent, dtype=torch.long),
            "valid_samples": torch.tensor(valid, dtype=torch.long),
            "onset": torch.tensor(min(local_onset, valid), dtype=torch.long),
            "offset": torch.tensor(min(local_offset, valid), dtype=torch.long),
            "active_pitch_mask": active_pitch_mask,
            "query_onset_polyphony": torch.tensor(
                query.onset_polyphony, dtype=torch.long
            ),
            "score_events": tuple(score_events),
            "score_complete": query.score_complete,
            "active_pitch_set_complete": query.active_pitch_set_complete,
            "identity": identity,
        }


def collate_score_crops(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Default-collate tensors while retaining ragged complete score events."""

    score_events = [row["score_events"] for row in rows]
    output = default_collate(
        [{key: value for key, value in row.items() if key != "score_events"} for row in rows]
    )
    output["score_events"] = score_events
    return output


class DeterministicStepBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        size: int,
        *,
        batch_size: int,
        start_step: int,
        stop_step: int,
        seed: int,
        collision_free_queries: tuple[Query, ...] | None = None,
        coverage_groups: dict[str, tuple[int, ...]] | None = None,
        coverage_weights: dict[str, float] | None = None,
        batch_strata: dict[str, dict[str, tuple[int, ...]]] | None = None,
        batch_stratum_weights: dict[str, float] | None = None,
        pair_anchor_groups: dict[str, tuple[int, ...]] | None = None,
        pair_partner_groups: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        self.collision_free_queries = collision_free_queries
        self.coverage_groups = coverage_groups
        self.coverage_weights = coverage_weights
        self.batch_strata = batch_strata
        self.batch_stratum_weights = batch_stratum_weights
        self.pair_anchor_groups = pair_anchor_groups
        self.pair_partner_groups = pair_partner_groups
        if self.batch_size < 2 or self.batch_size > self.size:
            raise ValueError("batch size must be in [2, dataset size]")
        if collision_free_queries is not None:
            if len(collision_free_queries) != self.size:
                raise ValueError("collision-free query metadata has wrong length")
            if self.batch_size % 2:
                raise ValueError("collision-free batches require an even batch size")
        if (coverage_groups is None) != (coverage_weights is None):
            raise ValueError("coverage groups and weights must be provided together")
        if coverage_groups is not None and coverage_weights is not None:
            if set(coverage_groups) != set(coverage_weights):
                raise ValueError("coverage group and weight keys differ")
            if any(not values for values in coverage_groups.values()):
                raise ValueError("coverage groups must be nonempty")
            if any(weight <= 0.0 for weight in coverage_weights.values()):
                raise ValueError("coverage weights must be positive")
            if not math.isclose(sum(coverage_weights.values()), 1.0, abs_tol=1e-8):
                raise ValueError("coverage weights must sum to one")

        if (batch_strata is None) != (batch_stratum_weights is None):
            raise ValueError("batch strata and weights must be provided together")
        if batch_strata is not None and batch_stratum_weights is not None:
            if coverage_groups is None or coverage_weights is None:
                raise ValueError("batch strata require coverage groups and weights")
            if set(batch_strata) != set(batch_stratum_weights):
                raise ValueError("batch stratum and weight keys differ")
            if any(weight <= 0.0 for weight in batch_stratum_weights.values()):
                raise ValueError("batch stratum weights must be positive")
            if not math.isclose(
                sum(batch_stratum_weights.values()), 1.0, abs_tol=1e-8
            ):
                raise ValueError("batch stratum weights must sum to one")
            for groups in batch_strata.values():
                if set(groups) != set(coverage_groups):
                    raise ValueError("batch stratum coverage keys differ")
                if any(not indices for indices in groups.values()):
                    raise ValueError("every batch stratum coverage group must be nonempty")
        if (pair_anchor_groups is None) != (pair_partner_groups is None):
            raise ValueError("pair anchor and partner groups must be provided together")
        if pair_anchor_groups is not None and pair_partner_groups is not None:
            if coverage_groups is None or coverage_weights is None:
                raise ValueError("pair groups require coverage groups and weights")
            if batch_strata is not None:
                raise ValueError("pair groups and batch strata are mutually exclusive")
            if (
                set(pair_anchor_groups) != set(coverage_groups)
                or set(pair_partner_groups) != set(coverage_groups)
            ):
                raise ValueError("pair group coverage keys differ")
            if any(not indices for indices in pair_anchor_groups.values()):
                raise ValueError("every pair anchor group must be nonempty")
            if not any(pair_partner_groups.values()):
                raise ValueError("at least one pair partner group must be nonempty")

    def _coverage_index(
        self,
        generator: torch.Generator,
        groups: dict[str, tuple[int, ...]] | None = None,
    ) -> int:
        active_groups = self.coverage_groups if groups is None else groups
        if active_groups is None or self.coverage_weights is None:
            return int(torch.randint(self.size, (), generator=generator).item())
        names = sorted(active_groups)
        weights = torch.tensor(
            [self.coverage_weights[name] for name in names], dtype=torch.float64
        )
        group = names[int(torch.multinomial(weights, 1, generator=generator).item())]
        candidates = active_groups[group]
        position = int(torch.randint(len(candidates), (), generator=generator).item())
        return int(candidates[position])

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            generator = torch.Generator().manual_seed(self.seed + step * 1_000_003)
            active_groups = self.coverage_groups
            if self.batch_strata is not None and self.batch_stratum_weights is not None:
                names = sorted(self.batch_strata)
                weights = torch.tensor(
                    [self.batch_stratum_weights[name] for name in names],
                    dtype=torch.float64,
                )
                stratum = names[
                    int(torch.multinomial(weights, 1, generator=generator).item())
                ]
                active_groups = self.batch_strata[stratum]
            if self.collision_free_queries is None:
                yield torch.randperm(
                    self.size, generator=generator
                )[: self.batch_size].tolist()
                continue
            selected: list[int] = []
            selected_set: set[int] = set()
            for _ in range(self.batch_size // 2):
                pair: tuple[int, int] | None = None
                for _attempt in range(1024):
                    anchor_groups = self.pair_anchor_groups or active_groups
                    partner_groups = self.pair_partner_groups or active_groups
                    if partner_groups is not None:
                        partner_groups = {
                            name: indices
                            for name, indices in partner_groups.items()
                            if indices
                        }
                    anchor = self._coverage_index(generator, anchor_groups)
                    other = self._coverage_index(generator, partner_groups)
                    if (
                        anchor == other
                        or anchor in selected_set
                        or other in selected_set
                    ):
                        continue
                    anchor_query = self.collision_free_queries[anchor]
                    other_query = self.collision_free_queries[other]
                    if (
                        not anchor_query.active_pitch_set_complete
                        or not other_query.active_pitch_set_complete
                    ):
                        continue
                    if (
                        other_query.active_pitch_bits
                        & (1 << anchor_query.pitch)
                        or anchor_query.active_pitch_bits
                        & (1 << other_query.pitch)
                    ):
                        continue
                    pair = (anchor, other)
                    break
                if pair is None:
                    raise RuntimeError(
                        "could not construct a collision-free score pair"
                    )
                selected.extend(pair)
                selected_set.update(pair)
            yield selected


def per_example_multiresolution_logmag_l1(
    estimate: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    losses = []
    for n_fft in (256, 512, 1024):
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        estimate_stft = torch.stft(
            estimate.float(), n_fft=n_fft, hop_length=n_fft // 4, window=window.float(),
            center=True, return_complex=True
        )
        target_stft = torch.stft(
            target.float(), n_fft=n_fft, hop_length=n_fft // 4, window=window.float(),
            center=True, return_complex=True
        )
        losses.append(
            (torch.log1p(estimate_stft.abs()) - torch.log1p(target_stft.abs()))
            .abs()
            .mean(dim=(-2, -1))
        )
    return torch.stack(losses, dim=0).mean(dim=0)
def project_input(model: torch.nn.Module, mixture: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return project_hpss_batch(
        mixture,
        valid,
        sample_rate=int(model.config.sample_rate),
        n_fft=int(model.config.n_fft),
        hop_length=int(model.config.hop_length),
        kernel_size=31,
        power=2.0,
        margin=1.0,
    )


def forward_model(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    pitch: torch.Tensor,
    valid: torch.Tensor,
    onset: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    harmonic, percussive = project_input(model, mixture, valid)
    estimate, _ = model(harmonic, percussive, pitch, valid, onset, offset)
    return estimate


def forward_model_chunked(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    pitch: torch.Tensor,
    valid: torch.Tensor,
    onset: torch.Tensor,
    offset: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Bound activation-recompute memory for many score queries of one crop."""

    if chunk_size < 1:
        raise ValueError("forward chunk size must be positive")
    if mixture.shape[0] <= chunk_size:
        return forward_model(model, mixture, pitch, valid, onset, offset)
    return torch.cat(
        [
            forward_model(
                model,
                mixture[start : start + chunk_size],
                pitch[start : start + chunk_size],
                valid[start : start + chunk_size],
                onset[start : start + chunk_size],
                offset[start : start + chunk_size],
            )
            for start in range(0, mixture.shape[0], chunk_size)
        ],
        dim=0,
    )


def add_parameter_gradients(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    *,
    right_weight: float = 1.0,
) -> tuple[torch.Tensor | None, ...]:
    """Add aligned gradient tuples without constructing another autograd graph."""

    if len(left) != len(right):
        raise ValueError("gradient tuples must align")
    combined: list[torch.Tensor | None] = []
    for left_gradient, right_gradient in zip(left, right, strict=True):
        if left_gradient is None:
            combined.append(
                None
                if right_gradient is None
                else right_gradient.detach().mul(right_weight)
            )
        elif right_gradient is None:
            combined.append(left_gradient.detach())
        else:
            combined.append(
                left_gradient.detach().add(
                    right_gradient.detach(), alpha=right_weight
                )
            )
    return tuple(combined)


def streaming_score_complete_closure_gradients(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    pitch: torch.Tensor,
    valid: torch.Tensor,
    onset: torch.Tensor,
    offset: torch.Tensor,
    note_to_mixture: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    chunk_size: int,
    closure_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Return exact closure loss/gradients while retaining only one query chunk.

    Closure depends on note outputs only through their per-mixture sum.  A
    graph-free pass obtains that sum and its loss cotangent.  Query chunks are
    then recomputed and differentiated against the cotangent, which is the
    exact vector-Jacobian product for the original summed closure objective.
    """

    if chunk_size < 1:
        raise ValueError("forward chunk size must be positive")
    target = mixture if closure_target is None else closure_target
    if target.shape != mixture.shape:
        raise ValueError("closure target and model input must have the same shape")
    indexed_mixture = mixture.index_select(0, note_to_mixture)
    indexed_valid = valid.index_select(0, note_to_mixture)
    with torch.no_grad():
        note_estimates = forward_model_chunked(
            model,
            indexed_mixture,
            pitch,
            indexed_valid,
            onset,
            offset,
            chunk_size=chunk_size,
        )
        summed = torch.zeros_like(target)
        summed.index_add_(0, note_to_mixture, note_estimates)

    summed_proxy = summed.detach().requires_grad_(True)
    proxy_to_mixture = torch.arange(
        mixture.shape[0], device=mixture.device, dtype=torch.long
    )
    _, closure_error = score_complete_mixture_closure(
        summed_proxy,
        proxy_to_mixture,
        target,
        valid,
    )
    closure_loss = closure_error.mean() / masked_rms(
        target, valid
    ).mean().clamp_min(1e-4)
    summed_cotangent = torch.autograd.grad(closure_loss, summed_proxy)[0].detach()

    accumulated: tuple[torch.Tensor | None, ...] = tuple(None for _ in parameters)
    for start in range(0, pitch.shape[0], chunk_size):
        stop = min(start + chunk_size, pitch.shape[0])
        chunk_estimate = forward_model(
            model,
            indexed_mixture[start:stop],
            pitch[start:stop],
            indexed_valid[start:stop],
            onset[start:stop],
            offset[start:stop],
        )
        chunk_cotangent = summed_cotangent.index_select(
            0, note_to_mixture[start:stop]
        )
        surrogate = (chunk_estimate * chunk_cotangent).sum()
        chunk_gradients = torch.autograd.grad(
            surrogate,
            parameters,
            allow_unused=True,
        )
        accumulated = add_parameter_gradients(accumulated, chunk_gradients)
    return closure_loss.detach(), accumulated


def objective_scheduled(step: int, *, every: int, phase: int) -> bool:
    if every < 1 or not 0 <= phase < every:
        raise ValueError("objective cadence is invalid")
    return step % every == phase


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument(
        "--real-corpus-config",
        type=Path,
        help="QC-gated multi-corpus real-audio mixture config; MAESTRO-only when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "remix",
            "real_mix_invariance",
            "symmetric_real_mix",
            "asymmetric_real_mix",
            "threeway_real_mix",
            "distill",
        ),
        default="remix",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("ownership", "heads", "heads_and_film", "full"),
        default="heads",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--threeway-candidate-pool-size", type=int, default=0,
        help="For threeway_real_mix, proposal-pool size; zero uses batch-size.",
    )
    parser.add_argument(
        "--threeway-max-anchors", type=int, default=0,
        help="For threeway_real_mix, cap valid anchors used per gradient; zero uses all.",
    )
    parser.add_argument("--crop-samples", type=int, default=49152)
    parser.add_argument("--lead-samples", type=int, default=4000)
    parser.add_argument("--release-samples", type=int, default=2560)
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument("--minimum-note-sec", type=float, default=0.10)
    parser.add_argument(
        "--maximum-anchor-polyphony",
        type=int,
        default=0,
        help="Keep queries at or below this scored onset polyphony; 0 disables.",
    )
    parser.add_argument(
        "--sparse-batch-fraction",
        type=float,
        default=0.0,
        help=(
            "When positive, draw this fraction of complete batches from onset "
            "polyphony one and the remainder from higher retained polyphony."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--consistency-spectral-weight", type=float, default=0.0)
    parser.add_argument("--consistency-projection-weight", type=float, default=0.0)
    parser.add_argument(
        "--score-complete-closure-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary real-mixture closure on every transcribed occurrence in "
            "a low-cardinality crop.  It has no residual channel; zero disables."
        ),
    )
    parser.add_argument(
        "--score-complete-max-events",
        type=int,
        default=4,
        help=(
            "Only apply score-complete closure to crops with 1 through this "
            "many complete score occurrences."
        ),
    )
    parser.add_argument(
        "--score-complete-max-crops",
        type=int,
        default=1,
        help=(
            "Maximum complete mixtures contributing closure per update.  Each "
            "selected mixture still uses every one of its score events."
        ),
    )
    parser.add_argument(
        "--score-closure-forward-chunk-size",
        type=int,
        default=4,
        help=(
            "Maximum score queries in one model call for complete/group closure. "
            "Chunking preserves the exact loss while bounding checkpoint recompute memory."
        ),
    )
    parser.add_argument("--score-complete-closure-every", type=int, default=1)
    parser.add_argument("--score-complete-closure-phase", type=int, default=0)
    parser.add_argument(
        "--score-group-closure-weight",
        type=float,
        default=0.0,
        help=(
            "On a mixture of two real recordings, sum every score query from "
            "the anchor recording and reconstruct that exact original recording; "
            "zero disables this stem-free score-conditioned MixIT objective."
        ),
    )
    parser.add_argument(
        "--score-group-symmetric-targets",
        action="store_true",
        help=(
            "Reconstruct both original recording groups from their score-query "
            "sums; the historical objective reconstructs only the anchor group."
        ),
    )
    parser.add_argument(
        "--score-group-max-events",
        type=int,
        default=16,
        help="Maximum complete anchor-score events in one group-closure example.",
    )
    parser.add_argument(
        "--score-group-max-crops",
        type=int,
        default=1,
        help="Maximum complete recording groups used per update.",
    )
    parser.add_argument(
        "--score-group-candidate-pool-size",
        type=int,
        default=0,
        help=(
            "Even asymmetric-pair proposal batch used to find complete score "
            "groups while retaining the configured gradient batch; zero disables."
        ),
    )
    parser.add_argument(
        "--output-idempotence-weight",
        type=float,
        default=0.0,
        help=(
            "Require a detached predicted note to reproduce itself when it is "
            "re-extracted with the same score query; zero disables."
        ),
    )
    parser.add_argument(
        "--output-purity-weight",
        type=float,
        default=0.0,
        help=(
            "Penalize recovery of a simultaneous differently pitched scored "
            "event from a detached predicted note; zero disables."
        ),
    )
    parser.add_argument(
        "--direct-competing-harmonic-weight",
        type=float,
        default=0.0,
        help=(
            "Directly penalize first-pass output energy in simultaneous "
            "competitor-exclusive score harmonics; zero disables."
        ),
    )
    parser.add_argument(
        "--direct-competing-gradient-diagnostic-every",
        type=int,
        default=0,
        help="Measure the unweighted direct first-pass gradient norm every N steps.",
    )
    parser.add_argument(
        "--direct-competing-gradient-diagnostic-phase",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--remix-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the pseudo-target remix waveform and spectral objective; "
            "zero provides a synthetic-replay-only causal control."
        ),
    )
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument("--remix-projection-weight", type=float, default=0.0)
    parser.add_argument("--removal-completeness-weight", type=float, default=0.0)
    parser.add_argument("--other-note-preservation-weight", type=float, default=0.0)
    parser.add_argument(
        "--nonlinear-property-view-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of applying the shared-dynamics/saturation attributed "
            "view to a real paired mixture; zero preserves the prior objective."
        ),
    )
    parser.add_argument(
        "--nonlinear-property-view-warmup-steps",
        type=int,
        default=0,
        help=(
            "If positive, linearly ramp the nonlinear-view probability from "
            "zero to its configured value over this many adaptation steps."
        ),
    )
    parser.add_argument(
        "--shared-dynamics-strength",
        type=float,
        default=0.0,
        help="Nonnegative strength for the common mixture-derived compressor envelope.",
    )
    parser.add_argument(
        "--shared-dynamics-window-samples",
        type=int,
        default=1024,
        help="Positive causal envelope window for shared mixture dynamics.",
    )
    parser.add_argument(
        "--saturation-attribution-drive",
        type=float,
        default=0.0,
        help=(
            "Nonnegative mild soft-saturation drive; its interaction is split "
            "between the queried note and complement by a two-player Shapley value."
        ),
    )
    parser.add_argument("--synthetic-replay-weight", type=float, default=0.0)
    parser.add_argument(
        "--synthetic-replay-target",
        choices=("exact_target", "frozen_teacher"),
        default="exact_target",
    )
    parser.add_argument("--synthetic-replay-spectral-weight", type=float, default=0.5)
    parser.add_argument(
        "--synthetic-replay-spectral-margin",
        type=float,
        default=0.0,
        help=(
            "Allowed per-example synthetic spectral replay loss before its "
            "coordinate contributes; 0 preserves the legacy spectral term."
        ),
    )
    parser.add_argument(
        "--synthetic-replay-spectral-guard-weight",
        type=float,
        default=0.0,
        help=(
            "Direct weight on spectral replay excess, applied outside the "
            "aggregate replay margin; 0 preserves the prior formulation."
        ),
    )
    parser.add_argument("--synthetic-replay-projection-weight", type=float, default=2.0)
    parser.add_argument(
        "--synthetic-replay-margin",
        type=float,
        default=0.0,
        help=(
            "Allowed normalized synthetic replay loss before retention activates; "
            "0 preserves the legacy always-on replay loss."
        ),
    )
    parser.add_argument("--synthetic-replay-batch-size", type=int, default=2)
    parser.add_argument("--synthetic-replay-fixed-samples", type=int, default=61_440)
    parser.add_argument("--synthetic-replay-num-workers", type=int, default=2)
    parser.add_argument(
        "--gradient-conflict-every",
        type=int,
        default=0,
        help="Measure real-SSL versus replay gradient cosine every N updates; 0 disables.",
    )
    parser.add_argument(
        "--replay-gradient-mode",
        choices=(
            "sum",
            "protect_real_pcgrad",
            "protect_real_gate",
            "protect_replay_pcgrad",
            "protect_replay_gate",
        ),
        default="sum",
        help=(
            "How gradients are combined. protect_replay modes prevent a conflicting "
            "real-SSL update from increasing the rehearsed original-task loss."
        ),
    )
    parser.add_argument(
        "--maximum-replay-to-real-gradient-norm-ratio",
        type=float,
        default=0.0,
        help="Cap the applied replay gradient norm relative to real SSL; 0 disables.",
    )
    parser.add_argument("--synthetic-replay-frozen-manifest", type=Path)
    parser.add_argument("--synthetic-replay-ordinal-view", type=Path)
    parser.add_argument("--absent-weight", type=float, default=0.02)
    parser.add_argument("--absent-every", type=int, default=4)
    parser.add_argument("--absent-phase", type=int, default=0)
    parser.add_argument(
        "--absent-gradient-diagnostic-every",
        type=int,
        default=0,
        help="Measure absent-query gradient scale and replay conflict every N steps.",
    )
    parser.add_argument("--absent-gradient-diagnostic-phase", type=int, default=0)
    parser.add_argument(
        "--pre-onset-leakage-weight",
        type=float,
        default=0.0,
        help=(
            "Score-grounded penalty for queried-note energy before its onset; "
            "0 preserves the previous objective."
        ),
    )
    parser.add_argument(
        "--pre-onset-leakage-guard-samples",
        type=int,
        default=512,
        help="Exclude this onset-adjacent prefix from the leakage penalty.",
    )
    parser.add_argument(
        "--pre-onset-leakage-min-samples",
        type=int,
        default=1600,
        help="Only score a pre-onset negative region at least this long.",
    )
    parser.add_argument("--residual-scale-min", type=float, default=0.7)
    parser.add_argument("--residual-scale-max", type=float, default=1.3)
    parser.add_argument(
        "--interferer-scale-mode",
        choices=("raw_gain", "relative_rms"),
        default="raw_gain",
        help="Interpret residual scale as raw gain or desired interferer/anchor RMS.",
    )
    parser.add_argument("--confidence-filter", action="store_true")
    parser.add_argument("--view-noise-snr-db", type=float, default=30.0)
    parser.add_argument("--maximum-relative-disagreement", type=float, default=0.35)
    parser.add_argument("--minimum-harmonic-concentration", type=float, default=0.20)
    parser.add_argument("--minimum-target-to-mixture-rms", type=float, default=0.02)
    parser.add_argument("--maximum-target-to-mixture-rms", type=float, default=1.50)
    parser.add_argument("--teacher-ema-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--model-save-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output_dir}")
    if args.steps < 1 or args.batch_size < 2:
        raise ValueError("steps must be positive and batch-size must be at least two")
    if args.threeway_candidate_pool_size < 0 or args.threeway_max_anchors < 0:
        raise ValueError("three-way candidate-pool arguments must be nonnegative")
    if args.save_every < 1 or args.model_save_every < 0:
        raise ValueError("save-every must be positive and model-save-every nonnegative")
    if (
        args.pre_onset_leakage_weight < 0.0
        or args.pre_onset_leakage_guard_samples < 0
        or args.pre_onset_leakage_min_samples < 1
    ):
        raise ValueError("pre-onset leakage arguments are invalid")
    if not 0.0 <= args.teacher_ema_decay < 1.0:
        raise ValueError("teacher-ema-decay must be in [0, 1)")
    if (
        args.consistency_weight < 0.0
        or args.consistency_spectral_weight < 0.0
        or args.consistency_projection_weight < 0.0
        or args.score_complete_closure_weight < 0.0
        or args.score_group_closure_weight < 0.0
        or args.output_idempotence_weight < 0.0
        or args.output_purity_weight < 0.0
        or args.direct_competing_harmonic_weight < 0.0
        or args.remix_weight < 0.0
        or args.remix_projection_weight < 0.0
        or args.removal_completeness_weight < 0.0
        or args.other_note_preservation_weight < 0.0
        or args.nonlinear_property_view_probability < 0.0
        or args.nonlinear_property_view_probability > 1.0
        or args.nonlinear_property_view_warmup_steps < 0
        or args.shared_dynamics_strength < 0.0
        or args.saturation_attribution_drive < 0.0
        or args.synthetic_replay_weight < 0.0
        or args.synthetic_replay_margin < 0.0
        or args.synthetic_replay_spectral_margin < 0.0
        or args.synthetic_replay_spectral_guard_weight < 0.0
        or args.synthetic_replay_spectral_weight < 0.0
        or args.synthetic_replay_projection_weight < 0.0
    ):
        raise ValueError("consistency weights must be nonnegative")
    if args.score_complete_max_events < 1:
        raise ValueError("score-complete-max-events must be positive")
    if (
        args.direct_competing_harmonic_weight
        and args.method != "asymmetric_real_mix"
    ):
        raise ValueError(
            "direct competing-harmonic exclusion currently requires asymmetric_real_mix"
        )
    if args.direct_competing_gradient_diagnostic_every < 0:
        raise ValueError("direct gradient diagnostic cadence must be nonnegative")
    if args.direct_competing_gradient_diagnostic_every and not (
        0
        <= args.direct_competing_gradient_diagnostic_phase
        < args.direct_competing_gradient_diagnostic_every
    ):
        raise ValueError("direct gradient diagnostic phase is invalid")
    if args.score_complete_max_crops < 1:
        raise ValueError("score-complete-max-crops must be positive")
    if args.score_closure_forward_chunk_size < 1:
        raise ValueError("score-closure-forward-chunk-size must be positive")
    if (
        args.score_complete_closure_every < 1
        or not 0
        <= args.score_complete_closure_phase
        < args.score_complete_closure_every
    ):
        raise ValueError("score-complete closure cadence is invalid")
    if args.absent_every < 1 or not 0 <= args.absent_phase < args.absent_every:
        raise ValueError("absent cadence is invalid")
    if args.absent_gradient_diagnostic_every < 0:
        raise ValueError("absent gradient diagnostic cadence must be nonnegative")
    if args.absent_gradient_diagnostic_every and not (
        0
        <= args.absent_gradient_diagnostic_phase
        < args.absent_gradient_diagnostic_every
    ):
        raise ValueError("absent gradient diagnostic phase is invalid")
    if args.score_group_max_events < 1 or args.score_group_max_crops < 1:
        raise ValueError("score-group event/crop limits must be positive")
    if args.score_group_candidate_pool_size < 0:
        raise ValueError("score-group candidate-pool size must be nonnegative")
    if args.score_group_closure_weight and args.method != "asymmetric_real_mix":
        raise ValueError("score-group closure currently requires asymmetric_real_mix")
    if args.score_group_symmetric_targets and not args.score_group_closure_weight:
        raise ValueError("symmetric score-group targets require score-group closure")
    if args.score_group_closure_weight and args.nonlinear_property_view_probability:
        raise ValueError("score-group closure needs an attributed nonlinear group target")
    if args.score_group_candidate_pool_size:
        if not args.score_group_closure_weight:
            raise ValueError("score-group candidate pooling requires its closure loss")
        if args.method != "asymmetric_real_mix":
            raise ValueError("score-group candidate pooling requires asymmetric_real_mix")
        if (
            args.score_group_candidate_pool_size < args.batch_size
            or args.score_group_candidate_pool_size % 2
            or args.batch_size % 2
        ):
            raise ValueError("score-group proposal and gradient batches must be even")
    if (
        args.removal_completeness_weight
        or args.other_note_preservation_weight
    ) and args.method not in {"symmetric_real_mix", "asymmetric_real_mix"}:
        raise ValueError("removal and preservation losses require a paired real mix")
    if (
        args.nonlinear_property_view_probability
        and args.method not in {"symmetric_real_mix", "asymmetric_real_mix"}
    ):
        raise ValueError("nonlinear property views require a paired real mix")
    if (
        args.nonlinear_property_view_probability
        and args.shared_dynamics_strength == 0.0
        and args.saturation_attribution_drive == 0.0
    ):
        raise ValueError("a nonlinear property view needs dynamics or saturation")
    if args.shared_dynamics_window_samples < 1:
        raise ValueError("shared-dynamics-window-samples must be positive")
    if args.minimum_target_to_mixture_rms > args.maximum_target_to_mixture_rms:
        raise ValueError("confidence energy bounds are reversed")
    if args.maximum_anchor_polyphony < 0:
        raise ValueError("maximum-anchor-polyphony must be nonnegative")
    if not 0.0 <= args.sparse_batch_fraction < 1.0:
        raise ValueError("sparse-batch-fraction must be in [0, 1)")
    if args.sparse_batch_fraction and args.maximum_anchor_polyphony <= 1:
        raise ValueError(
            "positive sparse-batch-fraction requires retained polyphony above one"
        )
    if (
        args.interferer_scale_mode == "relative_rms"
        and args.method not in {"symmetric_real_mix", "asymmetric_real_mix"}
    ):
        raise ValueError("relative-RMS interferer scaling requires a paired real-mix method")
    if (
        args.method == "asymmetric_real_mix"
        and not args.score_group_symmetric_targets
        and args.maximum_anchor_polyphony <= 1
    ):
        raise ValueError(
            "asymmetric_real_mix requires retained dense partners above polyphony one"
        )
    if args.method == "threeway_real_mix":
        proposal_batch_size = args.threeway_candidate_pool_size or args.batch_size
        if proposal_batch_size < 3:
            raise ValueError("threeway_real_mix requires a batch size of at least three")
        if args.threeway_candidate_pool_size and proposal_batch_size < args.batch_size:
            raise ValueError("threeway-candidate-pool-size must be at least batch-size")
        if args.removal_completeness_weight or args.other_note_preservation_weight:
            raise ValueError(
                "threeway_real_mix has no validated three-way removal/preservation audit yet"
            )
        if args.nonlinear_property_view_probability:
            raise ValueError(
                "threeway_real_mix has no validated nonlinear-view attribution yet"
            )
    if args.synthetic_replay_weight:
        if (
            args.synthetic_replay_frozen_manifest is None
            or args.synthetic_replay_ordinal_view is None
        ):
            raise ValueError(
                "positive synthetic replay weight requires its frozen manifest and ordinal view"
            )
        if args.synthetic_replay_batch_size < 1:
            raise ValueError("synthetic replay batch size must be positive")
        if args.synthetic_replay_fixed_samples < 1024:
            raise ValueError("synthetic replay fixed samples must be at least 1024")
        if args.synthetic_replay_num_workers < 0:
            raise ValueError("synthetic replay num workers must be nonnegative")
    if args.gradient_conflict_every < 0:
        raise ValueError("gradient-conflict-every must be nonnegative")
    if args.maximum_replay_to_real_gradient_norm_ratio < 0.0:
        raise ValueError("maximum replay-to-real gradient norm ratio must be nonnegative")
    if args.replay_gradient_mode != "sum" and not args.synthetic_replay_weight:
        raise ValueError("protected replay gradient modes require synthetic replay")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    initializer = args.initializer.expanduser().resolve()
    student, initializer_provenance = load_dual_hpss_checkpoint(initializer, device)
    teacher, _ = load_dual_hpss_checkpoint(initializer, device)
    # Real-domain pseudo-labels may be updated slowly, but synthetic retention
    # must always be anchored to the original 30k authority rather than drift
    # with real-domain adaptation.
    retention_teacher, _ = load_dual_hpss_checkpoint(initializer, device)
    teacher.requires_grad_(False).eval()
    retention_teacher.requires_grad_(False).eval()
    trainable_names, trainable_count = trainable_parameter_names(student, args.trainable_scope)
    student_parameter_map = dict(student.named_parameters())
    teacher_parameter_map = dict(teacher.named_parameters())
    if any(name not in teacher_parameter_map for name in trainable_names):
        raise ValueError("teacher and student parameter names do not align")
    student.train()
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("schema") not in RESUME_SCHEMAS:
            raise ValueError("resume state has the wrong schema")
        student.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_teacher_state(teacher, state)
        start_step = int(state["adaptation_step"])
    real_corpus_binding: dict[str, Any] | None = None
    real_corpus_weights: dict[str, float] | None = None
    configured_tracks: list[Track] | None = None
    if args.real_corpus_config is not None:
        configured_tracks, real_corpus_weights, real_corpus_binding = (
            _load_configured_real_tracks(args.real_corpus_config)
        )
    dataset = MaestroScoreCropDataset(
        args.maestro_root,
        split="train",
        crop_samples=args.crop_samples,
        lead_samples=args.lead_samples,
        release_samples=args.release_samples,
        max_tracks=(0 if configured_tracks is not None else args.max_tracks),
        minimum_note_sec=args.minimum_note_sec,
        maximum_anchor_polyphony=args.maximum_anchor_polyphony,
        seed=args.seed,
        tracks_override=configured_tracks,
    )
    if real_corpus_weights is not None:
        if set(dataset.query_indices_by_corpus) != set(real_corpus_weights):
            raise ValueError(
                "one or more configured real corpora retained no eligible queries"
            )
        assert real_corpus_binding is not None
        real_corpus_binding["corpus_query_counts"] = {
            corpus: len(indices)
            for corpus, indices in dataset.query_indices_by_corpus.items()
        }
    batch_strata: dict[str, dict[str, tuple[int, ...]]] | None = None
    batch_stratum_weights: dict[str, float] | None = None
    pair_anchor_groups: dict[str, tuple[int, ...]] | None = None
    pair_partner_groups: dict[str, tuple[int, ...]] | None = None
    sampler_coverage_groups = (
        dataset.query_indices_by_corpus
        if real_corpus_weights is not None or args.sparse_batch_fraction
        else None
    )
    sampler_coverage_weights = real_corpus_weights
    if args.sparse_batch_fraction:
        if sampler_coverage_weights is None:
            corpus_count = len(dataset.query_indices_by_corpus)
            sampler_coverage_weights = {
                corpus: 1.0 / corpus_count
                for corpus in dataset.query_indices_by_corpus
            }
        batch_strata = {
            "sparse_polyphony_one": {
                corpus: tuple(
                    index
                    for index in indices
                    if dataset.queries[index].onset_polyphony <= 1
                )
                for corpus, indices in dataset.query_indices_by_corpus.items()
            },
            "dense_polyphony_above_one": {
                corpus: tuple(
                    index
                    for index in indices
                    if dataset.queries[index].onset_polyphony > 1
                )
                for corpus, indices in dataset.query_indices_by_corpus.items()
            },
        }
        batch_stratum_weights = {
            "sparse_polyphony_one": args.sparse_batch_fraction,
            "dense_polyphony_above_one": 1.0 - args.sparse_batch_fraction,
        }
    if args.method == "asymmetric_real_mix":
        if args.sparse_batch_fraction:
            raise ValueError(
                "asymmetric_real_mix defines its own sparse-anchor/dense-partner strata"
            )
        if sampler_coverage_weights is None:
            corpus_count = len(dataset.query_indices_by_corpus)
            sampler_coverage_weights = {
                corpus: 1.0 / corpus_count
                for corpus in dataset.query_indices_by_corpus
            }
        sampler_coverage_groups = dataset.query_indices_by_corpus
        if not args.score_group_symmetric_targets:
            pair_anchor_groups = {
                corpus: tuple(
                    index
                    for index in indices
                    if dataset.queries[index].onset_polyphony <= 1
                )
                for corpus, indices in dataset.query_indices_by_corpus.items()
            }
            pair_partner_groups = {
                corpus: tuple(
                    index
                    for index in indices
                    if dataset.queries[index].onset_polyphony > 1
                )
                for corpus, indices in dataset.query_indices_by_corpus.items()
            }
    proposal_batch_size = args.batch_size
    if args.method == "threeway_real_mix" and args.threeway_candidate_pool_size:
        proposal_batch_size = args.threeway_candidate_pool_size
    if args.score_group_candidate_pool_size:
        proposal_batch_size = args.score_group_candidate_pool_size
    sampler = DeterministicStepBatchSampler(
        len(dataset), batch_size=proposal_batch_size, start_step=start_step,
        stop_step=args.steps, seed=args.seed,
        collision_free_queries=(
            dataset.queries
            if args.method in {"symmetric_real_mix", "asymmetric_real_mix"}
            else None
        ),
        coverage_groups=sampler_coverage_groups,
        coverage_weights=sampler_coverage_weights,
        batch_strata=batch_strata,
        batch_stratum_weights=batch_stratum_weights,
        pair_anchor_groups=pair_anchor_groups,
        pair_partner_groups=pair_partner_groups,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_score_crops,
    )
    replay_loader: DataLoader[dict[str, Any]] | None = None
    replay_binding: dict[str, Any] | None = None
    if args.synthetic_replay_weight:
        replay_contract = FrozenV13Contract.load(
            args.synthetic_replay_frozen_manifest
        )
        replay_train_ordinals, _, replay_view_binding = (
            synthetic_corpus.load_synthetic_ordinal_view(
                args.synthetic_replay_ordinal_view,
                contract=replay_contract,
            )
        )
        if replay_train_ordinals is None or replay_view_binding is None:
            raise ValueError("synthetic replay view has no bound training ordinals")
        replay_base = V13StreamingResidualDataset(
            replay_contract,
            split="train",
            fixed_samples=args.synthetic_replay_fixed_samples,
            reader_cache_size=8,
            preserve_complete_note=False,
        )
        replay_dataset = Subset(replay_base, replay_train_ordinals)
        replay_sampler = ResumeSafeBatchSampler(
            len(replay_dataset),
            batch_size=args.synthetic_replay_batch_size,
            start_step=start_step,
            stop_step=args.steps,
            seed=args.seed + 8_191,
        )
        replay_loader = DataLoader(
            replay_dataset,
            batch_sampler=replay_sampler,
            num_workers=args.synthetic_replay_num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.synthetic_replay_num_workers > 0,
        )
        replay_binding = {
            "kind": (
                "original_distribution_frozen_teacher_prediction_replay"
                if args.synthetic_replay_target == "frozen_teacher"
                else "exact_original_distribution_target_replay"
            ),
            "target": args.synthetic_replay_target,
            "frozen_manifest": str(
                args.synthetic_replay_frozen_manifest.expanduser().resolve()
            ),
            "frozen_manifest_sha256": file_sha256(
                args.synthetic_replay_frozen_manifest.expanduser().resolve()
            ),
            "ordinal_view": str(
                args.synthetic_replay_ordinal_view.expanduser().resolve()
            ),
            "ordinal_view_sha256": file_sha256(
                args.synthetic_replay_ordinal_view.expanduser().resolve()
            ),
            "view_binding": replay_view_binding,
            "training_examples": int(len(replay_dataset)),
            "validation_examples_used_for_training": 0,
            "batch_size": args.synthetic_replay_batch_size,
            "fixed_samples": args.synthetic_replay_fixed_samples,
            "sampler": "resume_safe_affine_without_materialized_permutations",
        }
    source_files = [
        ROOT / "src/model_exp/dna_residual_v13_streaming.py",
        ROOT / "src/model_exp/occurrence_score_remix_ssl.py",
        ROOT / "src/model_exp/occurrence_nonlinear_property_view.py",
        ROOT / "scripts/train_dna_residual_v13_streaming.py",
        ROOT / "scripts/train_occurrence_score_remix_ssl.py",
    ]
    contract = {
        "schema": SCHEMA,
        "method": args.method,
        "real_audio_supervision": "aligned_score_only_no_source_stems",
        "teacher": {
            **initializer_provenance,
            "update": (
                {"type": "ema_trainable_scope", "decay": args.teacher_ema_decay}
                if args.teacher_ema_decay
                else "frozen"
            ),
        },
        "retention_teacher": {
            **initializer_provenance,
            "update": "frozen_initializer_authority",
        },
        "student_initialization": initializer_provenance,
        "data": {
            "dataset": (
                "QC-gated transcription-aligned real corpus mixture"
                if real_corpus_binding is not None
                else "MAESTRO v3.0.0"
            ),
            "root": str(args.maestro_root.resolve()),
            "corpus_mixture": real_corpus_binding,
            "split": "train_only",
            "max_tracks": (None if real_corpus_binding is not None else args.max_tracks),
            "maximum_anchor_polyphony": args.maximum_anchor_polyphony,
            "batch_polyphony_strata": (
                {
                    stratum: {
                        "weight": batch_stratum_weights[stratum],
                        "corpus_query_counts": {
                            corpus: len(indices)
                            for corpus, indices in groups.items()
                        },
                    }
                    for stratum, groups in batch_strata.items()
                }
                if batch_strata is not None and batch_stratum_weights is not None
                else None
            ),
            "asymmetric_pair_polyphony_strata": (
                {
                    "anchor": "onset_polyphony_one",
                    "partner": "onset_polyphony_above_one",
                    "anchor_corpus_query_counts": {
                        corpus: len(indices)
                        for corpus, indices in pair_anchor_groups.items()
                    },
                    "partner_corpus_query_counts": {
                        corpus: len(indices)
                        for corpus, indices in pair_partner_groups.items()
                    },
                }
                if pair_anchor_groups is not None and pair_partner_groups is not None
                else None
            ),
            "retained_queries": len(dataset),
            "corpus_query_counts": {
                corpus: len(indices)
                for corpus, indices in dataset.query_indices_by_corpus.items()
            },
            "onset_polyphony_histogram": {
                str(density): sum(
                    query.onset_polyphony == density for query in dataset.queries
                )
                for density in sorted(
                    {query.onset_polyphony for query in dataset.queries}
                )
            },
            "crop_samples": args.crop_samples,
            "sample_rate": SAMPLE_RATE,
            "score_complete_closure": {
                "weight": args.score_complete_closure_weight,
                "maximum_events": args.score_complete_max_events,
                "maximum_crops_per_update": args.score_complete_max_crops,
                "forward_chunk_size": args.score_closure_forward_chunk_size,
                "every": args.score_complete_closure_every,
                "phase": args.score_complete_closure_phase,
                "residual_channel": "forbidden",
                "authority": "complete_transcribed_events_overlapping_valid_crop",
            },
            "score_group_closure": {
                "weight": args.score_group_closure_weight,
                "symmetric_targets": args.score_group_symmetric_targets,
                "maximum_events": args.score_group_max_events,
                "maximum_groups_per_update": args.score_group_max_crops,
                "candidate_pool_size": args.score_group_candidate_pool_size,
                "gradient_batch_size": args.batch_size,
                "target": "exact_original_real_recording_before_mixture_of_mixtures",
                "residual_channel": "forbidden",
                "collision_policy": (
                    "exclude_only_same_pitch_temporally_overlapping_cross_recording_events"
                ),
                "real_source_stems": "not_used",
            },
            "output_purity": {
                "idempotence_weight": args.output_idempotence_weight,
                "competing_pitch_weight": args.output_purity_weight,
                "input_gradient": "stopped_at_first_pass_note_estimate",
                "competing_event_authority": (
                    "maximally_overlapping_different_pitch_score_event_in_origin_crop"
                ),
                "direct_first_pass_competing_harmonic_weight": (
                    args.direct_competing_harmonic_weight
                ),
                "direct_first_pass_authority": (
                    "competitor_exclusive_harmonics_from_both_complete_or_partial_score_crops"
                ),
            },
        },
        "synthetic_replay": replay_binding,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": trainable_count,
        "trainable_parameter_names": trainable_names,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files},
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_QOS", "SLURM_JOB_PARTITION")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(contract, args.output_dir / "contract.json")
    metrics_path = args.output_dir / "metrics.jsonl"
    preempt_requested = False
    source_checkpoint = torch.load(initializer, map_location="cpu", weights_only=False)

    def request_preempt(_signum: int, _frame: Any) -> None:
        nonlocal preempt_requested
        preempt_requested = True

    signal.signal(signal.SIGUSR1, request_preempt)

    def save_state(step: int) -> Path:
        path = args.output_dir / "checkpoints/state-last.pt"
        atomic_torch_save(
            {
                "schema": SCHEMA,
                "adaptation_step": step,
                "model": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "contract": contract,
            },
            path,
        )
        return path

    def save_model_checkpoint(step: int) -> Path:
        checkpoint = {
            key: source_checkpoint.get(key)
            for key in (
                "schema",
                "global_step",
                "model_contract",
                "loss_contract",
                "corpus_binding",
            )
        }
        checkpoint.update(
            model=student.state_dict(),
            ssl_adaptation_contract=contract,
            ssl_adaptation_step=step,
            initialization_checkpoint={
                "path": str(initializer),
                "sha256": initializer_provenance["sha256"],
                "global_step": initializer_provenance["global_step"],
            },
        )
        path = args.output_dir / f"checkpoints/model-adapt-{step:06d}.pt"
        atomic_torch_save(checkpoint, path)
        return path

    started = time.time()
    final_step = start_step
    replay_iterator = iter(replay_loader) if replay_loader is not None else None
    for step, batch in enumerate(loader, start=start_step + 1):
        final_step = step
        real_corpora = tuple(str(corpus) for corpus in batch["corpus"])
        real_corpus_composition = "+".join(sorted(real_corpora))
        mixture = batch["mixture"].to(device, non_blocking=True)
        current_batch_size = mixture.shape[0]
        pitch = batch["pitch"].to(device, non_blocking=True)
        absent_pitch = batch["absent_pitch"].to(device, non_blocking=True)
        valid = batch["valid_samples"].to(device, non_blocking=True)
        onset = batch["onset"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        active_pitch_mask = batch["active_pitch_mask"]
        score_complete_mask = batch["score_complete"].to(device, non_blocking=True).bool()
        query_onset_polyphony = batch["query_onset_polyphony"].to(
            device, non_blocking=True
        )
        replay_batch = next(replay_iterator) if replay_iterator is not None else None
        if replay_batch is not None:
            replay_mixture = replay_batch["mixture"].to(device, non_blocking=True)
            replay_target = replay_batch["target"].to(device, non_blocking=True)
            replay_pitch = replay_batch["pitch"].to(device, non_blocking=True)
            replay_valid = replay_batch["valid_samples"].to(device, non_blocking=True)
            replay_onset = replay_batch["note_onset_samples"].to(
                device, non_blocking=True
            )
            replay_offset = replay_batch["note_offset_samples"].to(
                device, non_blocking=True
            )
        generator = torch.Generator().manual_seed(args.seed + step * 97_409)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            teacher_target = forward_model(teacher, mixture, pitch, valid, onset, offset).float()
            if args.confidence_filter:
                noise = torch.randn(
                    mixture.shape, generator=generator, dtype=torch.float32
                ).to(device)
                support = (
                    torch.arange(mixture.shape[-1], device=device)[None]
                    < valid.long()[:, None]
                )
                noise = noise * support
                noise = noise / masked_rms(noise, valid).clamp_min(1e-5)[:, None]
                noise = noise * (
                    masked_rms(mixture, valid)
                    * (10.0 ** (-args.view_noise_snr_db / 20.0))
                )[:, None]
                second_view_target = forward_model(
                    teacher, mixture + noise, pitch, valid, onset, offset
                ).float()
            else:
                second_view_target = teacher_target
        if args.confidence_filter:
            accepted, confidence_diagnostics = confidence_filter(
                mixture,
                teacher_target,
                second_view_target,
                pitch,
                valid,
                maximum_relative_disagreement=args.maximum_relative_disagreement,
                minimum_harmonic_concentration=args.minimum_harmonic_concentration,
                minimum_target_to_mixture_rms=args.minimum_target_to_mixture_rms,
                maximum_target_to_mixture_rms=args.maximum_target_to_mixture_rms,
            )
        else:
            accepted = torch.ones(current_batch_size, device=device, dtype=torch.bool)
            target_to_mixture = masked_rms(teacher_target, valid) / masked_rms(
                mixture, valid
            ).clamp_min(1e-5)
            confidence_diagnostics = {
                "relative_disagreement": torch.zeros_like(target_to_mixture),
                "harmonic_concentration": torch.ones_like(target_to_mixture),
                "target_to_mixture_rms": target_to_mixture,
            }
        confidence_weights = accepted.float()
        remix_pitch = pitch
        remix_valid = valid
        remix_onset = onset
        remix_offset = offset
        remix_confidence_weights = confidence_weights
        consistency_rows = torch.arange(current_batch_size, device=device)
        # Each remixed query retains the row whose full score crop supplies
        # competing-event metadata for C7 output-purity auditing.
        remix_source_rows = torch.arange(current_batch_size, device=device)
        collision_free_pairs = 0
        score_group_proposal_pairs = 0
        score_group_proposal_eligible_pairs = 0
        if args.method in {
            "remix",
            "real_mix_invariance",
            "symmetric_real_mix",
            "asymmetric_real_mix",
            "threeway_real_mix",
        }:
            if args.method == "threeway_real_mix":
                triplet_all = collision_free_triplet_indices(
                    active_pitch_mask,
                    batch["pitch"],
                    generator=generator,
                )
                pair_rows_cpu = select_collision_free_triplet_anchors(
                    triplet_all,
                    maximum_anchors=args.threeway_max_anchors,
                    generator=generator,
                )
                if pair_rows_cpu.numel() < 1:
                    row = {
                        "adaptation_step": step,
                        "skipped_no_collision_free_triplets": True,
                        "real_corpora": list(real_corpora),
                        "real_corpus_composition": real_corpus_composition,
                        "triplet_examples": int(pair_rows_cpu.numel()),
                        "query_onset_polyphony_mean": float(
                            query_onset_polyphony.float().mean()
                        ),
                        "elapsed_sec": time.time() - started,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    if step % args.log_every == 0 or step == start_step + 1:
                        print(json.dumps(row, sort_keys=True), flush=True)
                    if step % args.save_every == 0 or preempt_requested:
                        save_state(step)
                    if preempt_requested:
                        print(f"USR1 checkpoint saved at adaptation step {step}", flush=True)
                        return
                    continue
                pair_rows = pair_rows_cpu.to(device=device)
                partner = triplet_all[pair_rows_cpu, 0].to(device=device)
                third_partner = triplet_all[pair_rows_cpu, 1].to(device=device)
                collision_free_pairs = int(pair_rows.numel())
            elif args.method in {"symmetric_real_mix", "asymmetric_real_mix"}:
                if args.method == "asymmetric_real_mix":
                    pair_rows_cpu = torch.arange(0, current_batch_size, 2)
                    partner_cpu = pair_rows_cpu + 1
                    batch_pitch_cpu = batch["pitch"].long()
                    partner_contains_anchor = active_pitch_mask[
                        partner_cpu,
                        batch_pitch_cpu.index_select(0, pair_rows_cpu),
                    ]
                    anchor_contains_partner = active_pitch_mask[
                        pair_rows_cpu,
                        batch_pitch_cpu.index_select(0, partner_cpu),
                    ]
                    if bool(
                        partner_contains_anchor.any()
                        or anchor_contains_partner.any()
                    ):
                        raise AssertionError("asymmetric sampler emitted a pitch collision")
                    if args.score_group_candidate_pool_size:
                        score_group_proposal_pairs = int(pair_rows_cpu.numel())
                        eligible_pair_positions = [
                            position
                            for position, (anchor_row, interferer_row) in enumerate(
                                zip(
                                    pair_rows_cpu.tolist(),
                                    partner_cpu.tolist(),
                                    strict=True,
                                )
                            )
                            if (
                                bool(batch["score_complete"][anchor_row])
                                and bool(batch["score_complete"][interferer_row])
                                and 1
                                <= len(batch["score_events"][anchor_row])
                                <= args.score_group_max_events
                                and (
                                    not args.score_group_symmetric_targets
                                    or 1
                                    <= len(batch["score_events"][interferer_row])
                                    <= args.score_group_max_events
                                )
                                and not score_event_groups_temporally_collide(
                                    batch["score_events"][anchor_row],
                                    batch["score_events"][interferer_row],
                                )
                            )
                        ]
                        score_group_proposal_eligible_pairs = len(
                            eligible_pair_positions
                        )
                        eligible_mask = torch.zeros(
                            score_group_proposal_pairs, dtype=torch.bool
                        )
                        eligible_mask[eligible_pair_positions] = True
                        selected_tensor = select_score_group_pair_positions(
                            eligible_mask,
                            number_of_pairs=args.batch_size // 2,
                            generator=generator,
                        )
                        pair_rows_cpu = pair_rows_cpu.index_select(
                            0, selected_tensor
                        )
                        partner_cpu = partner_cpu.index_select(0, selected_tensor)
                    pair_rows = pair_rows_cpu.to(device=device)
                    partner = partner_cpu.to(device=device)
                    collision_free_pairs = int(pair_rows.numel())
                else:
                    partner_all = collision_free_partner_indices(
                        active_pitch_mask,
                        batch["pitch"],
                        generator=generator,
                    )
                    pair_rows_cpu = torch.nonzero(
                        partner_all >= 0, as_tuple=False
                    ).flatten()
                    if pair_rows_cpu.numel() != current_batch_size:
                        row = {
                            "adaptation_step": step,
                            "skipped_incomplete_collision_free_pairing": True,
                            "real_corpora": list(real_corpora),
                            "real_corpus_composition": real_corpus_composition,
                            "paired_examples": int(pair_rows_cpu.numel()),
                            "query_onset_polyphony_mean": float(
                                query_onset_polyphony.float().mean()
                            ),
                            "elapsed_sec": time.time() - started,
                        }
                        with metrics_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        if step % args.log_every == 0 or step == start_step + 1:
                            print(json.dumps(row, sort_keys=True), flush=True)
                        if step % args.save_every == 0 or preempt_requested:
                            save_state(step)
                        if preempt_requested:
                            print(
                                f"USR1 checkpoint saved at adaptation step {step}",
                                flush=True,
                            )
                            return
                        continue
                    pair_rows = pair_rows_cpu.to(device=device)
                    partner = partner_all.index_select(0, pair_rows_cpu).to(device=device)
                    collision_free_pairs = int(pair_rows.numel())
            else:
                permutation = batch_derangement(
                    current_batch_size, generator=generator
                )
            uniform = torch.rand(current_batch_size, generator=generator)
            log_scale = (
                math.log(args.residual_scale_min)
                + uniform * math.log(args.residual_scale_max / args.residual_scale_min)
            )
            residual_scale = log_scale.exp().to(device)
            if args.method == "remix":
                remixed, pseudo_target, common_gain = self_remix(
                    mixture,
                    teacher_target,
                    permutation,
                    residual_scale=residual_scale,
                    peak_limit=0.98,
                )
            elif args.method == "real_mix_invariance":
                remixed, pseudo_target, common_gain = real_mixture_invariance_remix(
                    mixture,
                    teacher_target,
                    permutation,
                    interferer_scale=residual_scale,
                    peak_limit=0.98,
                )
            elif args.method == "threeway_real_mix":
                second_uniform = torch.rand(
                    pair_rows.shape[0], generator=generator
                )
                second_scale = (
                    math.log(args.residual_scale_min)
                    + second_uniform
                    * math.log(args.residual_scale_max / args.residual_scale_min)
                ).exp().to(device)
                triplet_scale = torch.stack(
                    (residual_scale.index_select(0, pair_rows), second_scale), dim=1
                )
                if args.interferer_scale_mode == "relative_rms":
                    anchor_rms = masked_rms(
                        mixture.index_select(0, pair_rows),
                        valid.index_select(0, pair_rows),
                    )
                    left_rms = masked_rms(
                        mixture.index_select(0, partner),
                        valid.index_select(0, partner),
                    )
                    right_rms = masked_rms(
                        mixture.index_select(0, third_partner),
                        valid.index_select(0, third_partner),
                    )
                    triplet_scale = torch.stack(
                        (
                            relative_rms_interferer_scale(
                                anchor_rms, left_rms, triplet_scale[:, 0]
                            ),
                            relative_rms_interferer_scale(
                                anchor_rms, right_rms, triplet_scale[:, 1]
                            ),
                        ),
                        dim=1,
                    )
                remixed, pseudo_target, common_gain = (
                    threeway_real_mixture_invariance_remix(
                        mixture,
                        teacher_target,
                        torch.stack((partner, third_partner), dim=1),
                        interferer_scales=triplet_scale,
                        anchor_rows=pair_rows,
                        peak_limit=0.98,
                    )
                )
                mixed_once = remixed
                mixed_valid = torch.maximum(
                    valid.index_select(0, pair_rows),
                    torch.maximum(
                        valid.index_select(0, partner),
                        valid.index_select(0, third_partner),
                    ),
                )
                remix_pitch = pitch.index_select(0, pair_rows)
                remix_valid = mixed_valid
                remix_onset = onset.index_select(0, pair_rows)
                remix_offset = offset.index_select(0, pair_rows)
                remix_confidence_weights = confidence_weights.index_select(0, pair_rows)
                consistency_rows = pair_rows
                remix_source_rows = pair_rows
            else:
                pair_scale = residual_scale.index_select(0, pair_rows)
                anchor_mixture_rms = masked_rms(
                    mixture.index_select(0, pair_rows),
                    valid.index_select(0, pair_rows),
                )
                partner_mixture_rms = masked_rms(
                    mixture.index_select(0, partner),
                    valid.index_select(0, partner),
                )
                if args.interferer_scale_mode == "relative_rms":
                    pair_scale = relative_rms_interferer_scale(
                        anchor_mixture_rms,
                        partner_mixture_rms,
                        pair_scale,
                    )
                interferer_to_anchor_rms_ratio = (
                    pair_scale
                    * partner_mixture_rms
                    / anchor_mixture_rms.clamp_min(1e-4)
                )
                if args.method == "asymmetric_real_mix":
                    anchor_mixture = mixture.index_select(0, pair_rows)
                    partner_mixture = mixture.index_select(0, partner)
                    raw_mixture = anchor_mixture + partner_mixture * pair_scale[:, None]
                    peak = raw_mixture.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
                    common_gain = torch.minimum(
                        torch.ones_like(peak), 0.98 / peak
                    )
                    mixed_once = raw_mixture * common_gain
                    first_target = teacher_target.index_select(
                        0, pair_rows
                    ) * common_gain
                    second_target = teacher_target.index_select(
                        0, partner
                    ) * pair_scale[:, None] * common_gain
                else:
                    mixed_once, first_target, second_target, common_gain = (
                        symmetric_real_mixture_remix(
                            mixture.index_select(0, pair_rows),
                            teacher_target.index_select(0, pair_rows),
                            partner,
                            interferer_scale=pair_scale,
                            peak_limit=0.98,
                        )
                    )
                partner_pitch = pitch.index_select(0, partner)
                partner_valid = valid.index_select(0, partner)
                mixed_valid = torch.maximum(
                    valid.index_select(0, pair_rows), partner_valid
                )
                if args.method == "asymmetric_real_mix":
                    remixed = mixed_once
                    pseudo_target = first_target
                    remix_pitch = pitch.index_select(0, pair_rows)
                    remix_valid = mixed_valid
                    remix_onset = onset.index_select(0, pair_rows)
                    remix_offset = offset.index_select(0, pair_rows)
                    remix_confidence_weights = confidence_weights.index_select(
                        0, pair_rows
                    )
                    consistency_rows = pair_rows
                    remix_source_rows = pair_rows
                else:
                    remixed = torch.cat((mixed_once, mixed_once), dim=0)
                    pseudo_target = torch.cat(
                        (first_target, second_target), dim=0
                    )
                    remix_pitch = torch.cat(
                        (pitch.index_select(0, pair_rows), partner_pitch), dim=0
                    )
                    remix_valid = torch.cat((mixed_valid, mixed_valid), dim=0)
                    remix_onset = torch.cat(
                        (
                            onset.index_select(0, pair_rows),
                            onset.index_select(0, partner),
                        ),
                        dim=0,
                    )
                    remix_offset = torch.cat(
                        (
                            offset.index_select(0, pair_rows),
                            offset.index_select(0, partner),
                        ),
                        dim=0,
                    )
                    remix_confidence_weights = torch.cat(
                        (
                            confidence_weights.index_select(0, pair_rows),
                            confidence_weights.index_select(0, partner),
                        ),
                        dim=0,
                    )
                    remix_source_rows = torch.cat((pair_rows, partner), dim=0)
        else:
            remixed, pseudo_target = mixture, teacher_target
            common_gain = torch.ones((current_batch_size, 1), device=device)
        nonlinear_property_view_fraction = 0.0
        nonlinear_property_view_probability_effective = (
            args.nonlinear_property_view_probability
        )
        if args.nonlinear_property_view_warmup_steps:
            nonlinear_property_view_probability_effective *= min(
                1.0,
                float(step) / float(args.nonlinear_property_view_warmup_steps),
            )
        nonlinear_shared_gain_mean = 1.0
        nonlinear_shared_gain_min = 1.0
        nonlinear_view_consistency_max_abs = 0.0
        if (
            nonlinear_property_view_probability_effective
            and args.method in {"symmetric_real_mix", "asymmetric_real_mix"}
        ):
            selected_cpu = torch.nonzero(
                torch.rand(remixed.shape[0], generator=generator)
                < nonlinear_property_view_probability_effective,
                as_tuple=False,
            ).flatten()
            if selected_cpu.numel():
                selected = selected_cpu.to(device=device)
                (
                    viewed_mixture,
                    viewed_target,
                    viewed_complement,
                    nonlinear_diagnostics,
                ) = apply_nonlinear_property_view(
                    remixed.index_select(0, selected),
                    pseudo_target.index_select(0, selected),
                    shared_dynamics_strength=args.shared_dynamics_strength,
                    shared_dynamics_window_samples=args.shared_dynamics_window_samples,
                    saturation_drive=args.saturation_attribution_drive,
                )
                remixed = remixed.clone()
                pseudo_target = pseudo_target.clone()
                remixed.index_copy_(0, selected, viewed_mixture)
                pseudo_target.index_copy_(0, selected, viewed_target)
                # Keep the paired auxiliary losses in the same physical view
                # as the main SSL target.  Without this propagation,
                # removal/preservation would audit a raw mixture while the
                # student was trained on the nonlinear view, so the claimed
                # mixture property would not actually constrain the loss.
                if args.method == "asymmetric_real_mix":
                    mixed_once = mixed_once.clone()
                    first_target = first_target.clone()
                    second_target = second_target.clone()
                    mixed_once.index_copy_(0, selected, viewed_mixture)
                    first_target.index_copy_(0, selected, viewed_target)
                    second_target.index_copy_(0, selected, viewed_complement)
                nonlinear_property_view_fraction = float(
                    selected_cpu.numel() / remixed.shape[0]
                )
                nonlinear_shared_gain_mean = float(
                    nonlinear_diagnostics["shared_gain_mean"]
                )
                nonlinear_shared_gain_min = float(
                    nonlinear_diagnostics["shared_gain_min"]
                )
                nonlinear_view_consistency_max_abs = float(
                    nonlinear_diagnostics["mixture_consistency_max_abs"]
                )
        if not bool(remix_confidence_weights.any()):
            row = {
                "adaptation_step": step,
                "skipped_no_confident_examples": True,
                "accepted_examples": 0,
                "accepted_fraction": 0.0,
                "relative_disagreement_mean": float(
                    confidence_diagnostics["relative_disagreement"].mean()
                ),
                "harmonic_concentration_mean": float(
                    confidence_diagnostics["harmonic_concentration"].mean()
                ),
                "teacher_target_to_mixture_rms": float(
                    confidence_diagnostics["target_to_mixture_rms"].mean()
                ),
                "elapsed_sec": time.time() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step % args.log_every == 0 or step == start_step + 1:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step % args.save_every == 0 or preempt_requested:
                save_state(step)
            if preempt_requested:
                print(f"USR1 checkpoint saved at adaptation step {step}", flush=True)
                return
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            estimate = forward_model(
                student,
                remixed,
                remix_pitch,
                remix_valid,
                remix_onset,
                remix_offset,
            )
            target_rms = masked_rms(pseudo_target, remix_valid).clamp_min(1e-4)
            remix_waveform = weighted_mean(
                per_example_masked_waveform_l1(
                    estimate, pseudo_target, remix_valid
                )
                / target_rms,
                remix_confidence_weights,
            )
            remix_spectral = weighted_mean(
                per_example_multiresolution_logmag_l1(estimate, pseudo_target),
                remix_confidence_weights,
            )
            loss = args.remix_weight * (
                remix_waveform + args.spectral_weight * remix_spectral
            )
            pre_onset_leakage = torch.zeros((), device=device)
            pre_onset_leakage_eligible_fraction = 0.0
            if args.pre_onset_leakage_weight:
                pre_onset_rms, pre_onset_samples = per_example_pre_onset_rms(
                    estimate,
                    remix_onset,
                    remix_valid,
                    guard_samples=args.pre_onset_leakage_guard_samples,
                )
                pre_onset_weights = remix_confidence_weights * (
                    pre_onset_samples >= args.pre_onset_leakage_min_samples
                ).to(dtype=remix_confidence_weights.dtype)
                pre_onset_leakage_eligible_fraction = float(
                    (pre_onset_weights > 0).float().mean()
                )
                if bool(torch.any(pre_onset_weights > 0)):
                    pre_onset_leakage = weighted_mean(
                        pre_onset_rms / target_rms,
                        pre_onset_weights,
                    )
                    loss = loss + (
                        args.pre_onset_leakage_weight * pre_onset_leakage
                    )
            remix_projection = torch.zeros((), device=device)
            if args.remix_projection_weight:
                remix_projection = weighted_mean(
                    (
                        per_example_projection_gain(
                            estimate, pseudo_target, remix_valid
                        )
                        - 1.0
                    ).abs(),
                    remix_confidence_weights,
                )
                loss = loss + args.remix_projection_weight * remix_projection
            removal_completeness = torch.zeros((), device=device)
            other_note_preservation = torch.zeros((), device=device)
            if (
                args.removal_completeness_weight
                or args.other_note_preservation_weight
            ):
                pair_count = collision_free_pairs
                expected_estimates = (
                    pair_count
                    if args.method == "asymmetric_real_mix"
                    else 2 * pair_count
                )
                if pair_count < 1 or estimate.shape[0] != expected_estimates:
                    raise AssertionError("paired real-mix layout changed")
                first_estimate = estimate[:pair_count]
                first_complement = mixed_once - first_estimate
                first_pitch = pitch.index_select(0, pair_rows)
                first_onset = onset.index_select(0, pair_rows)
                first_offset = offset.index_select(0, pair_rows)
                second_pitch = pitch.index_select(0, partner)
                second_onset = onset.index_select(0, partner)
                second_offset = offset.index_select(0, partner)
                if args.method == "asymmetric_real_mix":
                    audit_mixture = torch.cat(
                        (first_complement, first_complement), dim=0
                    )
                    audit_pitch = torch.cat((first_pitch, second_pitch), dim=0)
                    audit_valid = torch.cat((mixed_valid,) * 2, dim=0)
                    audit_onset = torch.cat((first_onset, second_onset), dim=0)
                    audit_offset = torch.cat((first_offset, second_offset), dim=0)
                else:
                    second_estimate = estimate[pair_count:]
                    second_complement = mixed_once - second_estimate
                    audit_mixture = torch.cat(
                        (
                            first_complement,
                            second_complement,
                            first_complement,
                            second_complement,
                        ),
                        dim=0,
                    )
                    audit_pitch = torch.cat(
                        (first_pitch, second_pitch, second_pitch, first_pitch), dim=0
                    )
                    audit_valid = torch.cat((mixed_valid,) * 4, dim=0)
                    audit_onset = torch.cat(
                        (first_onset, second_onset, second_onset, first_onset), dim=0
                    )
                    audit_offset = torch.cat(
                        (first_offset, second_offset, second_offset, first_offset),
                        dim=0,
                    )
                audit_estimate = forward_model(
                    teacher,
                    audit_mixture,
                    audit_pitch,
                    audit_valid,
                    audit_onset,
                    audit_offset,
                )
                removed_first = audit_estimate[:pair_count]
                first_rms = masked_rms(first_target, mixed_valid).clamp_min(1e-4)
                second_rms = masked_rms(second_target, mixed_valid).clamp_min(1e-4)
                pair_weights = confidence_weights.index_select(0, pair_rows)
                partner_weights = confidence_weights.index_select(0, partner)
                if args.method == "asymmetric_real_mix":
                    removed_second = None
                    preserved_second = audit_estimate[pair_count:]
                    preserved_first = None
                else:
                    removed_second = audit_estimate[pair_count : 2 * pair_count]
                    preserved_second = audit_estimate[
                        2 * pair_count : 3 * pair_count
                    ]
                    preserved_first = audit_estimate[3 * pair_count :]
                if args.removal_completeness_weight:
                    removal_values = masked_rms(
                        removed_first, mixed_valid
                    ) / first_rms
                    removal_weights = pair_weights
                    if removed_second is not None:
                        removal_values = torch.cat(
                            (
                                removal_values,
                                masked_rms(removed_second, mixed_valid) / second_rms,
                            ),
                            dim=0,
                        )
                        removal_weights = torch.cat(
                            (pair_weights, partner_weights), dim=0
                        )
                    removal_completeness = weighted_mean(
                        removal_values,
                        removal_weights,
                    )
                    loss = (
                        loss
                        + args.removal_completeness_weight
                        * removal_completeness
                    )
                if args.other_note_preservation_weight:
                    preservation_values = per_example_masked_waveform_l1(
                        preserved_second, second_target, mixed_valid
                    ) / second_rms
                    preservation_weights = partner_weights
                    if preserved_first is not None:
                        preservation_values = torch.cat(
                            (
                                preservation_values,
                                per_example_masked_waveform_l1(
                                    preserved_first, first_target, mixed_valid
                                )
                                / first_rms,
                            ),
                            dim=0,
                        )
                        preservation_weights = torch.cat(
                            (partner_weights, pair_weights), dim=0
                        )
                    other_note_preservation = weighted_mean(
                        preservation_values,
                        preservation_weights,
                    )
                    loss = (
                        loss
                        + args.other_note_preservation_weight
                        * other_note_preservation
                    )
            consistency = torch.zeros((), device=device)
            consistency_spectral = torch.zeros((), device=device)
            consistency_projection = torch.zeros((), device=device)
            if (
                args.consistency_weight
                or args.consistency_spectral_weight
                or args.consistency_projection_weight
            ):
                consistency_mixture = mixture.index_select(0, consistency_rows)
                consistency_pitch = pitch.index_select(0, consistency_rows)
                consistency_valid = valid.index_select(0, consistency_rows)
                consistency_onset = onset.index_select(0, consistency_rows)
                consistency_offset = offset.index_select(0, consistency_rows)
                consistency_target = teacher_target.index_select(0, consistency_rows)
                consistency_confidence = confidence_weights.index_select(
                    0, consistency_rows
                )
                original_estimate = forward_model(
                    student,
                    consistency_mixture,
                    consistency_pitch,
                    consistency_valid,
                    consistency_onset,
                    consistency_offset,
                )
                if args.consistency_weight:
                    consistency = weighted_mean(
                        per_example_masked_waveform_l1(
                            original_estimate, consistency_target, consistency_valid
                        )
                        / masked_rms(
                            consistency_target, consistency_valid
                        ).clamp_min(1e-4),
                        consistency_confidence,
                    )
                    loss = loss + args.consistency_weight * consistency
                if args.consistency_spectral_weight:
                    consistency_spectral = weighted_mean(
                        per_example_multiresolution_logmag_l1(
                            original_estimate, consistency_target
                        ),
                        consistency_confidence,
                    )
                    loss = (
                        loss
                        + args.consistency_spectral_weight * consistency_spectral
                    )
                if args.consistency_projection_weight:
                    consistency_projection = weighted_mean(
                        (
                            per_example_projection_gain(
                                original_estimate,
                                consistency_target,
                                consistency_valid,
                            )
                            - 1.0
                        ).abs(),
                        consistency_confidence,
                    )
                    loss = (
                        loss
                        + args.consistency_projection_weight
                        * consistency_projection
                    )
            score_complete_closure = torch.zeros((), device=device)
            score_complete_closure_gradients: tuple[
                torch.Tensor | None, ...
            ] | None = None
            score_complete_eligible_crops = 0
            score_complete_events = 0
            score_complete_closure_scheduled = bool(
                args.score_complete_closure_weight
                and objective_scheduled(
                    step,
                    every=args.score_complete_closure_every,
                    phase=args.score_complete_closure_phase,
                )
            )
            if score_complete_closure_scheduled:
                eligible_rows_cpu = torch.tensor(
                    [
                        index
                        for index, events in enumerate(batch["score_events"])
                        if bool(batch["score_complete"][index])
                        and 1 <= len(events) <= args.score_complete_max_events
                    ],
                    dtype=torch.long,
                )
                if eligible_rows_cpu.numel() > args.score_complete_max_crops:
                    order = torch.randperm(
                        eligible_rows_cpu.numel(), generator=generator
                    )
                    eligible_rows_cpu = eligible_rows_cpu.index_select(
                        0, order[: args.score_complete_max_crops]
                    ).sort().values
                score_complete_eligible_crops = int(eligible_rows_cpu.numel())
                if score_complete_eligible_crops:
                    closure_rows = eligible_rows_cpu.to(device=device)
                    closure_mixture = mixture.index_select(0, closure_rows)
                    closure_valid = valid.index_select(0, closure_rows)
                    closure_pitches: list[int] = []
                    closure_onsets: list[int] = []
                    closure_offsets: list[int] = []
                    note_to_mixture: list[int] = []
                    for local_row, original_row in enumerate(eligible_rows_cpu.tolist()):
                        for event_pitch, event_onset, event_offset in batch["score_events"][original_row]:
                            closure_pitches.append(int(event_pitch))
                            closure_onsets.append(int(event_onset))
                            closure_offsets.append(int(event_offset))
                            note_to_mixture.append(local_row)
                    score_complete_events = len(closure_pitches)
                    if score_complete_events < score_complete_eligible_crops:
                        raise AssertionError("eligible crop has no score occurrence")
                    note_to_mixture_tensor = torch.tensor(
                        note_to_mixture, device=device, dtype=torch.long
                    )
                    score_complete_closure, score_complete_closure_gradients = (
                        streaming_score_complete_closure_gradients(
                        student,
                        closure_mixture,
                        torch.tensor(closure_pitches, device=device, dtype=torch.long),
                        closure_valid,
                        torch.tensor(closure_onsets, device=device, dtype=torch.long),
                        torch.tensor(closure_offsets, device=device, dtype=torch.long),
                        note_to_mixture_tensor,
                        trainable_parameters,
                        chunk_size=args.score_closure_forward_chunk_size,
                        )
                    )
            score_group_closure = torch.zeros((), device=device)
            score_group_closure_gradients: tuple[
                torch.Tensor | None, ...
            ] | None = None
            score_group_eligible_crops = 0
            score_group_events = 0
            score_group_collision_rejections = 0
            score_group_corpora: list[str] = []
            if args.score_group_closure_weight:
                if args.method != "asymmetric_real_mix":
                    raise AssertionError("validated score-group layout changed")
                eligible_local_rows: list[int] = []
                for local_row, (anchor_row, interferer_row) in enumerate(
                    zip(
                        pair_rows.detach().cpu().tolist(),
                        partner.detach().cpu().tolist(),
                        strict=True,
                    )
                ):
                    anchor_events = batch["score_events"][anchor_row]
                    interferer_events = batch["score_events"][interferer_row]
                    if not bool(batch["score_complete"][anchor_row]) or not bool(
                        batch["score_complete"][interferer_row]
                    ):
                        continue
                    if not 1 <= len(anchor_events) <= args.score_group_max_events:
                        continue
                    if (
                        args.score_group_symmetric_targets
                        and not 1
                        <= len(interferer_events)
                        <= args.score_group_max_events
                    ):
                        continue
                    if score_event_groups_temporally_collide(
                        anchor_events, interferer_events
                    ):
                        score_group_collision_rejections += 1
                        continue
                    eligible_local_rows.append(local_row)
                if len(eligible_local_rows) > args.score_group_max_crops:
                    order = torch.randperm(
                        len(eligible_local_rows), generator=generator
                    )[: args.score_group_max_crops]
                    eligible_local_rows = sorted(
                        eligible_local_rows[index] for index in order.tolist()
                    )
                score_group_eligible_crops = len(eligible_local_rows) * (
                    2 if args.score_group_symmetric_targets else 1
                )
                if eligible_local_rows:
                    group_mixtures: list[torch.Tensor] = []
                    group_targets: list[torch.Tensor] = []
                    group_valid: list[torch.Tensor] = []
                    group_pitches: list[int] = []
                    group_onsets: list[int] = []
                    group_offsets: list[int] = []
                    note_to_group: list[int] = []
                    for local_row in eligible_local_rows:
                        anchor_row = int(pair_rows[local_row].item())
                        interferer_row = int(partner[local_row].item())
                        group_index = len(group_mixtures)
                        score_group_corpora.append(str(batch["corpus"][anchor_row]))
                        group_mixtures.append(mixed_once[local_row])
                        group_targets.append(
                            mixture[anchor_row] * common_gain[local_row]
                        )
                        group_valid.append(mixed_valid[local_row])
                        for event_pitch, event_onset, event_offset in batch["score_events"][anchor_row]:
                            group_pitches.append(int(event_pitch))
                            group_onsets.append(int(event_onset))
                            group_offsets.append(int(event_offset))
                            note_to_group.append(group_index)
                        if args.score_group_symmetric_targets:
                            group_index = len(group_mixtures)
                            score_group_corpora.append(
                                str(batch["corpus"][interferer_row])
                            )
                            group_mixtures.append(mixed_once[local_row])
                            group_targets.append(
                                mixture[interferer_row]
                                * pair_scale[local_row]
                                * common_gain[local_row]
                            )
                            group_valid.append(mixed_valid[local_row])
                            for event_pitch, event_onset, event_offset in batch[
                                "score_events"
                            ][interferer_row]:
                                group_pitches.append(int(event_pitch))
                                group_onsets.append(int(event_onset))
                                group_offsets.append(int(event_offset))
                                note_to_group.append(group_index)
                    score_group_events = len(group_pitches)
                    group_mixture_tensor = torch.stack(group_mixtures)
                    group_target_tensor = torch.stack(group_targets)
                    group_valid_tensor = torch.stack(group_valid)
                    note_to_group_tensor = torch.tensor(
                        note_to_group, device=device, dtype=torch.long
                    )
                    score_group_closure, score_group_closure_gradients = (
                        streaming_score_complete_closure_gradients(
                            student,
                            group_mixture_tensor,
                            torch.tensor(
                                group_pitches, device=device, dtype=torch.long
                            ),
                            group_valid_tensor,
                            torch.tensor(
                                group_onsets, device=device, dtype=torch.long
                            ),
                            torch.tensor(
                                group_offsets, device=device, dtype=torch.long
                            ),
                            note_to_group_tensor,
                            trainable_parameters,
                            chunk_size=args.score_closure_forward_chunk_size,
                            closure_target=group_target_tensor,
                        )
                    )
            output_idempotence = torch.zeros((), device=device)
            output_purity = torch.zeros((), device=device)
            output_purity_eligible = 0
            direct_competing_harmonic = torch.zeros((), device=device)
            direct_competing_harmonic_eligible = 0
            direct_competing_harmonic_gradient_norm: float | None = None
            direct_competing_harmonic_gradients: (
                tuple[torch.Tensor | None, ...] | None
            ) = None
            direct_competing_to_replay_gradient_cosine: float | None = None
            direct_competing_replay_conflicting: bool | None = None
            direct_competing_pcgrad_survival_fraction: float | None = None
            if args.output_idempotence_weight or args.output_purity_weight:
                # The first-pass estimate is deliberately detached before it
                # becomes an input.  Thus C7 cannot win by changing the
                # primary pseudo-target prediction itself; it trains the
                # shared extractor to recognize a self-contained note object.
                detached_estimate = estimate.detach()
                estimate_rms = masked_rms(detached_estimate, remix_valid).clamp_min(1e-4)
                if args.output_idempotence_weight:
                    reextracted = forward_model(
                        student,
                        detached_estimate,
                        remix_pitch,
                        remix_valid,
                        remix_onset,
                        remix_offset,
                    )
                    output_idempotence = weighted_mean(
                        per_example_masked_waveform_l1(
                            reextracted, detached_estimate, remix_valid
                        ) / estimate_rms,
                        remix_confidence_weights,
                    )
                    loss = loss + args.output_idempotence_weight * output_idempotence
                if args.output_purity_weight:
                    source_events = [
                        batch["score_events"][index]
                        for index in remix_source_rows.detach().cpu().tolist()
                    ]
                    competing_events = select_overlapping_competing_score_events(
                        source_events,
                        remix_pitch,
                        remix_onset,
                        remix_offset,
                    )
                    eligible_cpu = torch.tensor(
                        [index for index, event in enumerate(competing_events) if event is not None],
                        dtype=torch.long,
                    )
                    output_purity_eligible = int(eligible_cpu.numel())
                    if output_purity_eligible:
                        eligible = eligible_cpu.to(device=device)
                        events = [competing_events[index] for index in eligible_cpu.tolist()]
                        if any(event is None for event in events):
                            raise AssertionError("purity selector returned a missing eligible event")
                        competing_pitch = torch.tensor(
                            [event[0] for event in events], device=device, dtype=torch.long
                        )
                        competing_onset = torch.tensor(
                            [event[1] for event in events], device=device, dtype=torch.long
                        )
                        competing_offset = torch.tensor(
                            [event[2] for event in events], device=device, dtype=torch.long
                        )
                        competing_estimate = forward_model(
                            student,
                            detached_estimate.index_select(0, eligible),
                            competing_pitch,
                            remix_valid.index_select(0, eligible),
                            competing_onset,
                            competing_offset,
                        )
                        output_purity = weighted_mean(
                            masked_rms(
                                competing_estimate,
                                remix_valid.index_select(0, eligible),
                            )
                            / estimate_rms.index_select(0, eligible),
                            remix_confidence_weights.index_select(0, eligible),
                        )
                        loss = loss + args.output_purity_weight * output_purity
            if args.direct_competing_harmonic_weight:
                # Unlike the detached re-extraction diagnostic above, this
                # term differentiates the actual first-pass note output.  Both
                # anchor and interferer scores describe the asymmetric mixed
                # timeline, so differently pitched overlapping events from
                # either recording are valid competitors.  Harmonic bins also
                # belonging to the query are excluded by the metric helper.
                remixed_event_groups = [
                    tuple(batch["score_events"][anchor_row])
                    + tuple(batch["score_events"][interferer_row])
                    for anchor_row, interferer_row in zip(
                        pair_rows.detach().cpu().tolist(),
                        partner.detach().cpu().tolist(),
                        strict=True,
                    )
                ]
                query_events = [
                    (int(query_pitch), int(query_onset), int(query_offset))
                    for query_pitch, query_onset, query_offset in zip(
                        remix_pitch.detach().cpu().tolist(),
                        remix_onset.detach().cpu().tolist(),
                        remix_offset.detach().cpu().tolist(),
                        strict=True,
                    )
                ]
                competing_groups = [
                    tuple(
                        event
                        for event in events
                        if int(event[0]) != query_pitch
                        and max(int(event[1]), query_onset)
                        < min(int(event[2]), query_offset)
                    )
                    for (query_pitch, query_onset, query_offset), events in zip(
                        query_events, remixed_event_groups, strict=True
                    )
                ]
                direct_ratios = competing_harmonic_leakage_ratio(
                    estimate.float(), query_events, competing_groups
                )
                direct_eligible = torch.isfinite(direct_ratios)
                direct_competing_harmonic_eligible = int(direct_eligible.sum())
                if direct_competing_harmonic_eligible:
                    direct_competing_harmonic = weighted_mean(
                        direct_ratios[direct_eligible],
                        remix_confidence_weights[direct_eligible],
                    )
                    loss = loss + (
                        args.direct_competing_harmonic_weight
                        * direct_competing_harmonic
                    )
                    if (
                        args.direct_competing_gradient_diagnostic_every
                        and objective_scheduled(
                            step,
                            every=(
                                args.direct_competing_gradient_diagnostic_every
                            ),
                            phase=(
                                args.direct_competing_gradient_diagnostic_phase
                            ),
                        )
                    ):
                        direct_competing_harmonic_gradients = torch.autograd.grad(
                            direct_competing_harmonic,
                            trainable_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        direct_gradient_squared = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                        for gradient in direct_competing_harmonic_gradients:
                            if gradient is not None:
                                direct_gradient_squared = (
                                    direct_gradient_squared
                                    + gradient.float().square().sum()
                                )
                        direct_competing_harmonic_gradient_norm = float(
                            direct_gradient_squared.sqrt()
                        )
            absent = torch.zeros((), device=device)
            absent_gradient_norm: float | None = None
            absent_gradients: tuple[torch.Tensor | None, ...] | None = None
            absent_to_replay_gradient_cosine: float | None = None
            absent_replay_conflicting: bool | None = None
            absent_pcgrad_survival_fraction: float | None = None
            absent_scheduled = bool(
                args.absent_weight
                and objective_scheduled(
                    step, every=args.absent_every, phase=args.absent_phase
                )
            )
            if absent_scheduled:
                complete = torch.nonzero(score_complete_mask, as_tuple=False).flatten()
                if complete.numel():
                    complete_onset = onset.index_select(0, complete)
                    absent_onset, absent_offset = rt2_absent_bounds(complete_onset)
                    absent_estimate = forward_model(
                        student,
                        mixture.index_select(0, complete),
                        absent_pitch.index_select(0, complete),
                        valid.index_select(0, complete),
                        absent_onset,
                        absent_offset,
                    )
                    complete_valid = valid.index_select(0, complete)
                    absent = masked_rms(absent_estimate, complete_valid).mean() / masked_rms(
                        mixture.index_select(0, complete), complete_valid
                    ).clamp_min(1e-4).mean()
                    loss = loss + args.absent_weight * absent
                    if (
                        args.absent_gradient_diagnostic_every
                        and objective_scheduled(
                            step,
                            every=args.absent_gradient_diagnostic_every,
                            phase=args.absent_gradient_diagnostic_phase,
                        )
                    ):
                        absent_gradients = torch.autograd.grad(
                            absent,
                            trainable_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        absent_gradient_squared = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                        for gradient in absent_gradients:
                            if gradient is not None:
                                absent_gradient_squared = (
                                    absent_gradient_squared
                                    + gradient.float().square().sum()
                                )
                        absent_gradient_norm = float(absent_gradient_squared.sqrt())
            synthetic_replay_waveform = torch.zeros((), device=device)
            synthetic_replay_spectral = torch.zeros((), device=device)
            synthetic_replay_spectral_excess = torch.zeros((), device=device)
            synthetic_replay_projection = torch.zeros((), device=device)
            synthetic_replay_raw_loss = torch.zeros((), device=device)
            synthetic_replay_loss = torch.zeros((), device=device)
            synthetic_replay_spectral_guard_loss = torch.zeros((), device=device)
            real_ssl_loss = loss
            reporting_loss = real_ssl_loss + (
                args.score_complete_closure_weight * score_complete_closure.detach()
            ) + (
                args.score_group_closure_weight * score_group_closure.detach()
            )
            if replay_batch is not None:
                replay_supervision = replay_target
                if args.synthetic_replay_target == "frozen_teacher":
                    with torch.inference_mode():
                        replay_supervision = forward_model(
                            # Keep original-distribution replay bound to the
                            # immutable initializer even when real SSL uses an
                            # EMA teacher.
                            retention_teacher,
                            replay_mixture,
                            replay_pitch,
                            replay_valid,
                            replay_onset,
                            replay_offset,
                        ).float()
                replay_estimate = forward_model(
                    student,
                    replay_mixture,
                    replay_pitch,
                    replay_valid,
                    replay_onset,
                    replay_offset,
                )
                replay_target_rms = masked_rms(
                    replay_supervision, replay_valid
                ).clamp_min(1e-4)
                synthetic_replay_waveform = (
                    per_example_masked_waveform_l1(
                        replay_estimate, replay_supervision, replay_valid
                    )
                    / replay_target_rms
                ).mean()
                if args.synthetic_replay_spectral_weight:
                    synthetic_replay_spectral = (
                        per_example_multiresolution_logmag_l1(
                            replay_estimate, replay_supervision
                        ).mean()
                    )
                if args.synthetic_replay_projection_weight:
                    synthetic_replay_projection = (
                        per_example_projection_gain(
                            replay_estimate, replay_supervision, replay_valid
                        )
                        - 1.0
                    ).abs().mean()
                synthetic_replay_spectral_excess = torch.relu(
                    synthetic_replay_spectral
                    - args.synthetic_replay_spectral_margin
                )
                synthetic_replay_raw_loss = (
                    synthetic_replay_waveform
                    + args.synthetic_replay_spectral_weight
                    * synthetic_replay_spectral_excess
                    + args.synthetic_replay_projection_weight
                    * synthetic_replay_projection
                )
                synthetic_replay_loss = (
                    synthetic_replay_raw_loss
                    if args.synthetic_replay_margin == 0.0
                    else torch.relu(
                        synthetic_replay_raw_loss - args.synthetic_replay_margin
                    )
                )
                synthetic_replay_spectral_guard_loss = (
                    args.synthetic_replay_spectral_guard_weight
                    * synthetic_replay_spectral_excess
                )
                synthetic_replay_loss = synthetic_replay_loss + synthetic_replay_spectral_guard_loss
                loss = reporting_loss + args.synthetic_replay_weight * synthetic_replay_loss
            else:
                loss = reporting_loss
        gradient_conflict_cosine: float | None = None
        real_gradient_norm: float | None = None
        replay_gradient_norm: float | None = None
        replay_to_real_gradient_norm_ratio: float | None = None
        applied_replay_to_real_gradient_norm_ratio: float | None = None
        replay_gradient_applied_weight: float | None = None
        gradient_conflicting: bool | None = None
        replay_gradient_projected = False
        combined_gradients: tuple[torch.Tensor | None, ...] | None = None
        if (
            replay_batch is not None
            and (
                args.replay_gradient_mode != "sum"
                or (
                    args.gradient_conflict_every
                    and step % args.gradient_conflict_every == 0
                )
            )
        ):
            real_gradients = torch.autograd.grad(
                real_ssl_loss,
                trainable_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if score_complete_closure_gradients is not None:
                real_gradients = add_parameter_gradients(
                    real_gradients,
                    score_complete_closure_gradients,
                    right_weight=args.score_complete_closure_weight,
                )
            if score_group_closure_gradients is not None:
                real_gradients = add_parameter_gradients(
                    real_gradients,
                    score_group_closure_gradients,
                    right_weight=args.score_group_closure_weight,
                )
            replay_gradients = torch.autograd.grad(
                synthetic_replay_loss,
                trainable_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if direct_competing_harmonic_gradients is not None:
                direct_replay_geometry = gradient_pair_geometry(
                    direct_competing_harmonic_gradients,
                    replay_gradients,
                )
                direct_competing_to_replay_gradient_cosine = float(
                    direct_replay_geometry["gradient_cosine"]
                )
                direct_competing_replay_conflicting = bool(
                    direct_replay_geometry["gradient_conflicting"]
                )
                direct_competing_pcgrad_survival_fraction = float(
                    direct_replay_geometry["left_pcgrad_survival_fraction"]
                )
            if absent_gradients is not None:
                absent_replay_geometry = gradient_pair_geometry(
                    absent_gradients,
                    replay_gradients,
                )
                absent_to_replay_gradient_cosine = float(
                    absent_replay_geometry["gradient_cosine"]
                )
                absent_replay_conflicting = bool(
                    absent_replay_geometry["gradient_conflicting"]
                )
                absent_pcgrad_survival_fraction = float(
                    absent_replay_geometry["left_pcgrad_survival_fraction"]
                )
            combined_gradients, gradient_diagnostics = (
                compose_real_protected_replay_gradients(
                    real_gradients,
                    replay_gradients,
                    replay_weight=args.synthetic_replay_weight,
                    mode=args.replay_gradient_mode,
                    maximum_replay_to_real_norm_ratio=(
                        args.maximum_replay_to_real_gradient_norm_ratio
                    ),
                )
            )
            gradient_conflict_cosine = float(
                gradient_diagnostics["gradient_conflict_cosine"]
            )
            real_gradient_norm = float(gradient_diagnostics["real_gradient_norm"])
            replay_gradient_norm = float(
                gradient_diagnostics["replay_gradient_norm"]
            )
            replay_to_real_gradient_norm_ratio = float(
                gradient_diagnostics["replay_to_real_gradient_norm_ratio"]
            )
            applied_replay_to_real_gradient_norm_ratio = float(
                gradient_diagnostics[
                    "applied_replay_to_real_gradient_norm_ratio"
                ]
            )
            replay_gradient_applied_weight = float(
                gradient_diagnostics["replay_gradient_applied_weight"]
            )
            gradient_conflicting = bool(
                gradient_diagnostics["gradient_conflicting"]
            )
            replay_gradient_projected = bool(
                gradient_diagnostics["replay_gradient_projected"]
            )
        if args.replay_gradient_mode == "sum":
            auxiliary_gradients: tuple[torch.Tensor | None, ...] | None = None
            if score_complete_closure_gradients is not None:
                auxiliary_gradients = tuple(
                    None
                    if gradient is None
                    else gradient.detach().mul(args.score_complete_closure_weight)
                    for gradient in score_complete_closure_gradients
                )
            if score_group_closure_gradients is not None:
                if auxiliary_gradients is None:
                    auxiliary_gradients = tuple(
                        None
                        if gradient is None
                        else gradient.detach().mul(args.score_group_closure_weight)
                        for gradient in score_group_closure_gradients
                    )
                else:
                    auxiliary_gradients = add_parameter_gradients(
                        auxiliary_gradients,
                        score_group_closure_gradients,
                        right_weight=args.score_group_closure_weight,
                    )
            if auxiliary_gradients is not None:
                for parameter, gradient in zip(
                    trainable_parameters, auxiliary_gradients, strict=True
                ):
                    parameter.grad = None if gradient is None else gradient.detach()
            loss.backward()
        else:
            if combined_gradients is None:
                raise AssertionError("protected replay gradients were not computed")
            for parameter, gradient in zip(
                trainable_parameters, combined_gradients, strict=True
            ):
                parameter.grad = None if gradient is None else gradient.detach()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            args.gradient_clip,
        )
        optimizer.step()
        if args.teacher_ema_decay:
            with torch.no_grad():
                for name in trainable_names:
                    teacher_parameter_map[name].mul_(args.teacher_ema_decay).add_(
                        student_parameter_map[name], alpha=1.0 - args.teacher_ema_decay
                    )
        row = {
            "adaptation_step": step,
            "real_corpora": list(real_corpora),
            "real_corpus_composition": real_corpus_composition,
            "skipped_no_confident_examples": False,
            "accepted_examples": int(accepted.sum()),
            "accepted_fraction": float(accepted.float().mean()),
            "collision_free_pairs": collision_free_pairs,
            "query_onset_polyphony_mean": float(
                query_onset_polyphony.float().mean()
            ),
            "relative_disagreement_mean": float(
                confidence_diagnostics["relative_disagreement"].mean()
            ),
            "harmonic_concentration_mean": float(
                confidence_diagnostics["harmonic_concentration"].mean()
            ),
            "loss": float(loss.detach()),
            "remix_waveform": float(remix_waveform.detach()),
            "remix_spectral": float(remix_spectral.detach()),
            "pre_onset_leakage": float(pre_onset_leakage.detach()),
            "pre_onset_leakage_eligible_fraction": pre_onset_leakage_eligible_fraction,
            "remix_projection": float(remix_projection.detach()),
            "removal_completeness": float(removal_completeness.detach()),
            "other_note_preservation": float(other_note_preservation.detach()),
            "nonlinear_property_view_fraction": nonlinear_property_view_fraction,
            "nonlinear_property_view_probability_effective": nonlinear_property_view_probability_effective,
            "nonlinear_shared_gain_mean": nonlinear_shared_gain_mean,
            "nonlinear_shared_gain_min": nonlinear_shared_gain_min,
            "nonlinear_view_consistency_max_abs": nonlinear_view_consistency_max_abs,
            "consistency": float(consistency.detach()),
            "consistency_spectral": float(consistency_spectral.detach()),
            "consistency_projection": float(consistency_projection.detach()),
            "score_complete_closure": float(score_complete_closure.detach()),
            "score_complete_closure_scheduled": score_complete_closure_scheduled,
            "score_complete_eligible_crops": score_complete_eligible_crops,
            "score_complete_events": score_complete_events,
            "score_group_closure": float(score_group_closure.detach()),
            "score_group_eligible_crops": score_group_eligible_crops,
            "score_group_events": score_group_events,
            "score_group_collision_rejections": score_group_collision_rejections,
            "score_group_corpora": score_group_corpora,
            "score_group_proposal_pairs": score_group_proposal_pairs,
            "score_group_proposal_eligible_pairs": score_group_proposal_eligible_pairs,
            "output_idempotence": float(output_idempotence.detach()),
            "output_purity": float(output_purity.detach()),
            "output_purity_eligible": output_purity_eligible,
            "direct_competing_harmonic": float(
                direct_competing_harmonic.detach()
            ),
            "direct_competing_harmonic_eligible": (
                direct_competing_harmonic_eligible
            ),
            "direct_competing_harmonic_gradient_norm": (
                direct_competing_harmonic_gradient_norm
            ),
            "direct_competing_harmonic_weighted_gradient_norm": (
                None
                if direct_competing_harmonic_gradient_norm is None
                else args.direct_competing_harmonic_weight
                * direct_competing_harmonic_gradient_norm
            ),
            "direct_competing_to_replay_gradient_cosine": (
                direct_competing_to_replay_gradient_cosine
            ),
            "direct_competing_replay_conflicting": (
                direct_competing_replay_conflicting
            ),
            "direct_competing_pcgrad_survival_fraction": (
                direct_competing_pcgrad_survival_fraction
            ),
            "absent_energy_ratio": float(absent.detach()),
            "absent_scheduled": absent_scheduled,
            "absent_gradient_norm": absent_gradient_norm,
            "absent_weighted_gradient_norm": (
                None
                if absent_gradient_norm is None
                else args.absent_weight * absent_gradient_norm
            ),
            "absent_to_replay_gradient_cosine": absent_to_replay_gradient_cosine,
            "absent_replay_conflicting": absent_replay_conflicting,
            "absent_pcgrad_survival_fraction": absent_pcgrad_survival_fraction,
            "synthetic_replay_loss": float(synthetic_replay_loss.detach()),
            "synthetic_replay_raw_loss": float(synthetic_replay_raw_loss.detach()),
            "synthetic_replay_margin": args.synthetic_replay_margin,
            "synthetic_replay_spectral_margin": args.synthetic_replay_spectral_margin,
            "synthetic_replay_spectral_guard_weight": (
                args.synthetic_replay_spectral_guard_weight
            ),
            "synthetic_replay_waveform": float(
                synthetic_replay_waveform.detach()
            ),
            "synthetic_replay_spectral": float(
                synthetic_replay_spectral.detach()
            ),
            "synthetic_replay_spectral_excess": float(
                synthetic_replay_spectral_excess.detach()
            ),
            "synthetic_replay_spectral_guard_loss": float(
                synthetic_replay_spectral_guard_loss.detach()
            ),
            "synthetic_replay_projection": float(
                synthetic_replay_projection.detach()
            ),
            "gradient_conflict_cosine": gradient_conflict_cosine,
            "real_gradient_norm": real_gradient_norm,
            "replay_gradient_norm": replay_gradient_norm,
            "gradient_conflicting": gradient_conflicting,
            "replay_to_real_gradient_norm_ratio": replay_to_real_gradient_norm_ratio,
            "applied_replay_to_real_gradient_norm_ratio": (
                applied_replay_to_real_gradient_norm_ratio
            ),
            "replay_gradient_applied_weight": replay_gradient_applied_weight,
            "replay_gradient_projected": replay_gradient_projected,
            "teacher_target_rms": float(masked_rms(teacher_target, valid).mean()),
            "teacher_target_to_mixture_rms": float(
                (masked_rms(teacher_target, valid) / masked_rms(mixture, valid).clamp_min(1e-5)).mean()
            ),
            "common_gain": float(common_gain.mean()),
            "interferer_scale_mean": (
                float(pair_scale.mean())
                if args.method == "symmetric_real_mix"
                else None
            ),
            "interferer_to_anchor_rms_ratio_mean": (
                float(interferer_to_anchor_rms_ratio.mean())
                if args.method == "symmetric_real_mix"
                else None
            ),
            "gradient_norm": float(gradient_norm),
            "elapsed_sec": time.time() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if step % args.log_every == 0 or step == start_step + 1:
            print(json.dumps(row, sort_keys=True), flush=True)
        if step % args.save_every == 0 or preempt_requested:
            save_state(step)
        if args.model_save_every and step % args.model_save_every == 0:
            save_model_checkpoint(step)
        if preempt_requested:
            print(f"USR1 checkpoint saved at adaptation step {step}", flush=True)
            return
    state_path = save_state(final_step)
    final_path = save_model_checkpoint(final_step)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "adaptation_step": final_step,
        "initializer": initializer_provenance,
        "final_checkpoint": {
            "path": str(final_path.resolve()),
            "sha256": file_sha256(final_path),
        },
        "resume_state": str(state_path.resolve()),
        "elapsed_sec": time.time() - started,
    }
    atomic_json(report, args.output_dir / "report.json")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
