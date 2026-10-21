#!/usr/bin/env python3
"""Train complementary pitch-comb or HPSS occurrence-separation branches."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Callable
import math
import hashlib
from copy import deepcopy

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.harmonic_target import (  # noqa: E402
    gt_adaptive_harmonic_comb_loss,
    project_harmonic_target_batch,
)
from model_exp.hpss_target import project_hpss_batch  # noqa: E402
from model_exp.occurrence_dual_hpss_separator import (  # noqa: E402
    CONDITIONING_MODES,
    HARMONIC_PARAMETERIZATIONS,
    OccurrenceDualHPSSSeparator,
    PERCUSSIVE_PARAMETERIZATIONS,
)
from model_exp.occurrence_dual_hpss_symmetric_alignment import (  # noqa: E402
    OccurrenceDualHPSSSymmetricAlignment,
)
from model_exp.occurrence_audio_augmentation import (  # noqa: E402
    OccurrenceAugmentationConfig,
    apply_occurrence_augmentation,
)
from model_exp.occurrence_full_score_data import (  # noqa: E402
    FullScoreSyntheticDataset,
    collate_full_score,
)
from model_exp.directional_selective_gate import (  # noqa: E402
    lower_member_exact_octave_gate,
)
from model_exp.occurrence_subtraction_diagnostics import (  # noqa: E402
    batch_anchor_quality_metrics,
    projection_underprediction_loss,
)
from model_exp.occurrence_dual_harmonic_residual_separator import (  # noqa: E402
    OccurrenceDualHarmonicResidualSeparator,
)
from model_exp.occurrence_joint_tfc_tdf_separator import (  # noqa: E402
    bounded_complex_ratio_mask_loss,
    magnitude_source_ownership_loss,
    unbounded_complex_ratio_mask_loss,
)
from model_exp.pitch_relative_harmonic_gate import (  # noqa: E402
    GATE_MODES,
    gate_applies_to_target_and_prediction,
    pitch_relative_harmonic_gate,
)
import scripts.train_separator as base  # noqa: E402


def _source_losses(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    complex_normalizer: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    zeros = torch.zeros_like(target)
    return dna_residual_losses(
        estimate,
        target,
        zeros,
        estimate,
        valid_samples=valid_samples,
        complex_normalizer=complex_normalizer,
        **base.ACTIVE_BASE_LOSS_WEIGHTS,
    )


PRESET = os.environ.get("OCCURRENCE_DUAL_HARMONIC_RESIDUAL_PRESET", "30m")
HARMONIC_TARGET_DEVICE = os.environ.get("OCCURRENCE_HARMONIC_TARGET_DEVICE", "cuda")
DECOMPOSITION = os.environ.get("OCCURRENCE_DUAL_DECOMPOSITION", "pitch_comb")
if DECOMPOSITION not in {"pitch_comb", "hpss"}:
    raise ValueError("OCCURRENCE_DUAL_DECOMPOSITION must be pitch_comb or hpss")
SECOND_COMPONENT = "percussive" if DECOMPOSITION == "hpss" else "residual"
HPSS_HARMONIC_GATE = os.environ.get("OCCURRENCE_HPSS_HARMONIC_GATE", "none")
if HPSS_HARMONIC_GATE not in GATE_MODES:
    raise ValueError(f"Unknown OCCURRENCE_HPSS_HARMONIC_GATE: {HPSS_HARMONIC_GATE}")
HPSS_PERCUSSIVE_PARAMETERIZATION = os.environ.get(
    "OCCURRENCE_HPSS_PERCUSSIVE_PARAMETERIZATION", "ownership"
)
HPSS_HARMONIC_PARAMETERIZATION = os.environ.get(
    "OCCURRENCE_HPSS_HARMONIC_PARAMETERIZATION", "ownership"
)
CONDITIONING_MODE = os.environ.get(
    "OCCURRENCE_CONDITIONING_MODE", "rt2_target_roll"
)
ALLOW_LEGACY_CONDITIONING_FINETUNE = (
    os.environ.get("OCCURRENCE_ALLOW_LEGACY_CONDITIONING_FINETUNE", "0") == "1"
)
if CONDITIONING_MODE not in CONDITIONING_MODES:
    raise ValueError(f"Unknown OCCURRENCE_CONDITIONING_MODE: {CONDITIONING_MODE}")
if CONDITIONING_MODE == "harmonic_lifecycle" and not ALLOW_LEGACY_CONDITIONING_FINETUNE:
    raise ValueError(
        "harmonic_lifecycle conditioning is deprecated for training; use exact "
        "RT2 target-roll or rt2_onset_only conditioning. Legacy checkpoints "
        "require OCCURRENCE_ALLOW_LEGACY_CONDITIONING_FINETUNE=1 for an explicit "
        "model-only fine-tune."
    )
ROLL_HOLD_MIN_MS = float(os.environ.get("OCCURRENCE_ROLL_HOLD_MIN_MS", "0"))
ROLL_HOLD_MAX_MS = float(os.environ.get("OCCURRENCE_ROLL_HOLD_MAX_MS", "0"))
DUAL_COMMUNICATION = os.environ.get("OCCURRENCE_DUAL_COMMUNICATION", "none")
if DUAL_COMMUNICATION not in {"none", "symmetric_lowres"}:
    raise ValueError(f"Unknown OCCURRENCE_DUAL_COMMUNICATION: {DUAL_COMMUNICATION}")
DUAL_BACKBONE = os.environ.get("OCCURRENCE_DUAL_BACKBONE", "tfc_tdf")
if DUAL_BACKBONE != "tfc_tdf":
    raise ValueError("The paper recipe uses the TFC-TDF backbone")
COMMUNICATION_INITIALIZATION = os.environ.get(
    "OCCURRENCE_COMMUNICATION_INITIALIZATION_CHECKPOINT", ""
)
OUTPUT_CARRIER_MODE = os.environ.get(
    "OCCURRENCE_HPSS_OUTPUT_CARRIER_MODE", "encoder_input"
)
if OUTPUT_CARRIER_MODE not in {"encoder_input", "ungated_hpss"}:
    raise ValueError(f"Unknown OCCURRENCE_HPSS_OUTPUT_CARRIER_MODE: {OUTPUT_CARRIER_MODE}")
COMPLEX_MASK_MODE = os.environ.get(
    "OCCURRENCE_HPSS_COMPLEX_MASK_MODE", "bounded"
)
if COMPLEX_MASK_MODE not in {"bounded", "unbounded"}:
    raise ValueError(f"Unknown OCCURRENCE_HPSS_COMPLEX_MASK_MODE: {COMPLEX_MASK_MODE}")
HEAD_ABLATION_INITIALIZATION = os.environ.get(
    "OCCURRENCE_HEAD_ABLATION_INITIALIZATION_CHECKPOINT", ""
)
HEAD_ABLATION_INITIALIZATION_SHA256 = os.environ.get(
    "OCCURRENCE_HEAD_ABLATION_INITIALIZATION_SHA256", ""
)
if HPSS_PERCUSSIVE_PARAMETERIZATION not in PERCUSSIVE_PARAMETERIZATIONS:
    raise ValueError(
        "OCCURRENCE_HPSS_PERCUSSIVE_PARAMETERIZATION must be ownership, "
        "magnitude_mask, or complex_mask"
    )
if HPSS_HARMONIC_PARAMETERIZATION not in HARMONIC_PARAMETERIZATIONS:
    raise ValueError(
        "OCCURRENCE_HPSS_HARMONIC_PARAMETERIZATION must be ownership or "
        "complex_mask"
    )
HARMONIC_LOSS_WEIGHT = float(os.environ.get("OCCURRENCE_DUAL_HARMONIC_LOSS_WEIGHT", "0.3333333333333333"))
SECOND_LOSS_WEIGHT = float(os.environ.get("OCCURRENCE_DUAL_SECOND_LOSS_WEIGHT", "0.3333333333333333"))
COMBINED_LOSS_WEIGHT = float(os.environ.get("OCCURRENCE_DUAL_COMBINED_LOSS_WEIGHT", "0.3333333333333333"))
GT_HARMONIC_COMB_LOSS_WEIGHT = float(
    os.environ.get("OCCURRENCE_GT_HARMONIC_COMB_LOSS_WEIGHT", "0")
)
GT_HARMONIC_COMB_LOGMAG_WEIGHT = float(
    os.environ.get("OCCURRENCE_GT_HARMONIC_COMB_LOGMAG_WEIGHT", "0.5")
)
GT_HARMONIC_COMB_COMPLEX_CAP = float(
    os.environ.get("OCCURRENCE_GT_HARMONIC_COMB_COMPLEX_CAP", "0")
)
PREPROJECTION_TARGET_COMPLEX_WEIGHT = float(
    os.environ.get("OCCURRENCE_PREPROJECTION_TARGET_COMPLEX_WEIGHT", "0")
)
PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT = float(
    os.environ.get("OCCURRENCE_PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT", "0")
)
if GT_HARMONIC_COMB_LOSS_WEIGHT < 0.0:
    raise ValueError("OCCURRENCE_GT_HARMONIC_COMB_LOSS_WEIGHT must be non-negative")
if GT_HARMONIC_COMB_LOGMAG_WEIGHT < 0.0:
    raise ValueError("OCCURRENCE_GT_HARMONIC_COMB_LOGMAG_WEIGHT must be non-negative")
if GT_HARMONIC_COMB_COMPLEX_CAP < 0.0:
    raise ValueError("OCCURRENCE_GT_HARMONIC_COMB_COMPLEX_CAP must be non-negative")
OFFSET_JITTER_MS = float(os.environ.get("OCCURRENCE_OFFSET_JITTER_MS", "0"))
MIDI_TAIL_ENERGY_WEIGHT = float(os.environ.get("OCCURRENCE_MIDI_TAIL_ENERGY_WEIGHT", "0"))
MIDI_TAIL_CVAR_WEIGHT = float(os.environ.get("OCCURRENCE_MIDI_TAIL_CVAR_WEIGHT", "0"))
MIDI_TAIL_EXCESS_WEIGHT = float(os.environ.get("OCCURRENCE_MIDI_TAIL_EXCESS_WEIGHT", "0"))
MIDI_TAIL_CVAR_FRACTION = float(os.environ.get("OCCURRENCE_MIDI_TAIL_CVAR_FRACTION", "0.25"))
MIDI_TAIL_ENVELOPE_WEIGHT = float(
    os.environ.get("OCCURRENCE_MIDI_TAIL_ENVELOPE_WEIGHT", "0")
)
MIDI_TAIL_ENVELOPE_CVAR_WEIGHT = float(
    os.environ.get("OCCURRENCE_MIDI_TAIL_ENVELOPE_CVAR_WEIGHT", "0")
)
MIDI_TAIL_ENVELOPE_WINDOWS_MS = tuple(
    float(value)
    for value in os.environ.get(
        "OCCURRENCE_MIDI_TAIL_ENVELOPE_WINDOWS_MS", "20,80"
    ).split(",")
    if value.strip()
)
HIGH_BAND_LOG_MAG_WEIGHT = float(os.environ.get("OCCURRENCE_HIGH_BAND_LOG_MAG_WEIGHT", "0"))
HIGH_BAND_BEGIN_HZ = float(os.environ.get("OCCURRENCE_HIGH_BAND_BEGIN_HZ", "4000"))
PROJECTION_UNDER_WEIGHT = float(
    os.environ.get("OCCURRENCE_PROJECTION_UNDER_WEIGHT", "0")
)
INPUT_GATE_PROBABILITY = float(
    os.environ.get("OCCURRENCE_HPSS_INPUT_GATE_PROBABILITY", "1")
)
INPUT_GATE_POLICY = os.environ.get(
    "OCCURRENCE_HPSS_INPUT_GATE_POLICY", "static"
)
FULL_SCORE_INDEX = Path(
    os.environ.get("OCCURRENCE_FULL_SCORE_INDEX", "")
).expanduser().resolve()
SELECTIVE_OVERLAP_SAMPLES = int(
    os.environ.get("OCCURRENCE_SELECTIVE_GATE_MIN_OVERLAP_SAMPLES", "1280")
)
SELECTIVE_OVERLAP_FRACTION = float(
    os.environ.get("OCCURRENCE_SELECTIVE_GATE_MIN_OVERLAP_FRACTION", "0.25")
)
ROBUSTNESS_AUGMENTATION = os.environ.get(
    "OCCURRENCE_ROBUSTNESS_AUGMENTATION", "none"
)
if ROBUSTNESS_AUGMENTATION not in {"none", "coherent_v1", "coherent_v2"}:
    raise ValueError(
        "OCCURRENCE_ROBUSTNESS_AUGMENTATION must be none, coherent_v1, or coherent_v2"
    )
ROBUSTNESS_CONFIG = OccurrenceAugmentationConfig(
    gain_probability=float(os.environ.get("OCCURRENCE_AUG_GAIN_PROBABILITY", "0.8")),
    gain_db_limit=float(os.environ.get("OCCURRENCE_AUG_GAIN_DB_LIMIT", "6")),
    pitch_shift_probability=(
        float(os.environ.get("OCCURRENCE_AUG_PITCH_SHIFT_PROBABILITY", "0.25"))
        if ROBUSTNESS_AUGMENTATION == "coherent_v2"
        else 0.0
    ),
    pitch_shift_semitones_limit=(
        int(os.environ.get("OCCURRENCE_AUG_PITCH_SHIFT_SEMITONES_LIMIT", "2"))
        if ROBUSTNESS_AUGMENTATION == "coherent_v2"
        else 0
    ),
    detune_probability=float(os.environ.get("OCCURRENCE_AUG_DETUNE_PROBABILITY", "0.35")),
    detune_cents_limit=float(os.environ.get("OCCURRENCE_AUG_DETUNE_CENTS_LIMIT", "20")),
    linear_echo_probability=float(os.environ.get("OCCURRENCE_AUG_LINEAR_PROBABILITY", "0.25")),
    linear_echo_wet_limit=float(os.environ.get("OCCURRENCE_AUG_LINEAR_WET_LIMIT", "0.16")),
    noise_probability=float(os.environ.get("OCCURRENCE_AUG_NOISE_PROBABILITY", "0.35")),
    noise_snr_db_min=float(os.environ.get("OCCURRENCE_AUG_NOISE_SNR_DB_MIN", "20")),
    noise_snr_db_max=float(os.environ.get("OCCURRENCE_AUG_NOISE_SNR_DB_MAX", "40")),
)
ROBUSTNESS_CONFIG.validate()
_STEP_GENERATOR: torch.Generator | None = None


def set_training_step(step: int, seed: int) -> None:
    """Freeze stochastic interventions by optimizer step, independent of model RNG."""

    global _STEP_GENERATOR
    if step < 1:
        raise ValueError("training step must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _STEP_GENERATOR = torch.Generator(device=device)
    _STEP_GENERATOR.manual_seed(int(seed) * 1_000_003 + int(step))
if min(HARMONIC_LOSS_WEIGHT, SECOND_LOSS_WEIGHT, COMBINED_LOSS_WEIGHT) < 0 or not abs(
    HARMONIC_LOSS_WEIGHT + SECOND_LOSS_WEIGHT + COMBINED_LOSS_WEIGHT - 1.0
) < 1e-6:
    raise ValueError("dual source-loss weights must be non-negative and sum to one")
for name, value in {
    "OCCURRENCE_OFFSET_JITTER_MS": OFFSET_JITTER_MS,
    "OCCURRENCE_MIDI_TAIL_ENERGY_WEIGHT": MIDI_TAIL_ENERGY_WEIGHT,
    "OCCURRENCE_MIDI_TAIL_CVAR_WEIGHT": MIDI_TAIL_CVAR_WEIGHT,
    "OCCURRENCE_MIDI_TAIL_EXCESS_WEIGHT": MIDI_TAIL_EXCESS_WEIGHT,
    "OCCURRENCE_MIDI_TAIL_ENVELOPE_WEIGHT": MIDI_TAIL_ENVELOPE_WEIGHT,
    "OCCURRENCE_MIDI_TAIL_ENVELOPE_CVAR_WEIGHT": (
        MIDI_TAIL_ENVELOPE_CVAR_WEIGHT
    ),
    "OCCURRENCE_HIGH_BAND_LOG_MAG_WEIGHT": HIGH_BAND_LOG_MAG_WEIGHT,
    "OCCURRENCE_PROJECTION_UNDER_WEIGHT": PROJECTION_UNDER_WEIGHT,
    "OCCURRENCE_PREPROJECTION_TARGET_COMPLEX_WEIGHT": (
        PREPROJECTION_TARGET_COMPLEX_WEIGHT
    ),
    "OCCURRENCE_PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT": (
        PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT
    ),
}.items():
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
if not 0 < MIDI_TAIL_CVAR_FRACTION <= 1:
    raise ValueError("OCCURRENCE_MIDI_TAIL_CVAR_FRACTION must be in (0, 1]")
if not MIDI_TAIL_ENVELOPE_WINDOWS_MS or any(
    not math.isfinite(value) or value <= 0
    for value in MIDI_TAIL_ENVELOPE_WINDOWS_MS
):
    raise ValueError(
        "OCCURRENCE_MIDI_TAIL_ENVELOPE_WINDOWS_MS must contain positive values"
    )
if not math.isfinite(HIGH_BAND_BEGIN_HZ) or not 0 < HIGH_BAND_BEGIN_HZ < 8000:
    raise ValueError("OCCURRENCE_HIGH_BAND_BEGIN_HZ must be in (0, 8000)")
if not 0 <= INPUT_GATE_PROBABILITY <= 1:
    raise ValueError("OCCURRENCE_HPSS_INPUT_GATE_PROBABILITY must be in [0, 1]")
if INPUT_GATE_PROBABILITY < 1 and HPSS_HARMONIC_GATE != "below_f0_input_only":
    raise ValueError(
        "mixed input gating requires OCCURRENCE_HPSS_HARMONIC_GATE=below_f0_input_only"
    )
if INPUT_GATE_POLICY not in {"static", "lower_member_exact_octave"}:
    raise ValueError(f"Unknown OCCURRENCE_HPSS_INPUT_GATE_POLICY: {INPUT_GATE_POLICY}")
if INPUT_GATE_POLICY == "lower_member_exact_octave":
    if HPSS_HARMONIC_GATE != "below_f0_input_only":
        raise ValueError("directional selective gating requires below_f0_input_only")
    if INPUT_GATE_PROBABILITY != 1:
        raise ValueError("directional policy cannot be combined with random mixed gating")
    if not FULL_SCORE_INDEX.is_file():
        raise FileNotFoundError(f"full-score index is unavailable: {FULL_SCORE_INDEX}")
if SELECTIVE_OVERLAP_SAMPLES < 1 or not 0 < SELECTIVE_OVERLAP_FRACTION <= 1:
    raise ValueError("invalid directional selective-gate overlap threshold")


def directional_gate_strength(batch: dict[str, Any]) -> torch.Tensor:
    return lower_member_exact_octave_gate(
        batch,
        minimum_overlap_samples=SELECTIVE_OVERLAP_SAMPLES,
        minimum_overlap_fraction=SELECTIVE_OVERLAP_FRACTION,
    )


# Narrow extension point for query-ownership pilots.  The production objective
# remains byte-for-byte unchanged unless a dedicated wrapper installs a hook.
QUERY_OWNERSHIP_AUXILIARY: Callable[..., dict[str, torch.Tensor]] | None = None


def jitter_occurrence_offsets(
    onset: torch.Tensor,
    offset: torch.Tensor,
    valid: torch.Tensor,
    *,
    maximum_ms: float,
    sample_rate: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb only conditioning offsets while retaining valid geometry."""

    maximum_samples = int(round(float(maximum_ms) * int(sample_rate) / 1000.0))
    if maximum_samples <= 0:
        return offset, torch.zeros_like(offset)
    delta = torch.randint(
        -maximum_samples,
        maximum_samples + 1,
        offset.shape,
        device=offset.device,
        generator=generator,
    )
    shifted = torch.minimum(torch.maximum(offset + delta, onset + 1), valid)
    return shifted, shifted - offset


def midi_offset_tail_losses(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    midi_offset: torch.Tensor,
    *,
    cvar_fraction: float,
) -> dict[str, torch.Tensor]:
    """Compare target and estimate from MIDI note-off through the valid crop."""

    positions = torch.arange(estimate.shape[-1], device=estimate.device)[None]
    support = (positions >= midi_offset.long()[:, None]) & (
        positions < valid.long()[:, None]
    )
    count = support.sum(dim=-1)
    denominator = count.clamp_min(1).to(torch.float32)
    estimate_power = (estimate.float().square() * support).sum(dim=-1) / denominator
    target_power = (target.float().square() * support).sum(dim=-1) / denominator
    scale = 10.0 / math.log(10.0)
    signed_db = scale * (
        torch.log(estimate_power + 1e-12) - torch.log(target_power + 1e-12)
    )
    absolute_db = signed_db.abs()
    excess_db = signed_db.clamp_min(0.0)
    present = count > 0
    selected = absolute_db[present]
    if selected.numel():
        hard_count = max(1, int(math.ceil(selected.numel() * float(cvar_fraction))))
        cvar = torch.topk(selected, k=hard_count).values.mean()
        mean = selected.mean()
        excess = excess_db[present].mean()
    else:
        mean = estimate.new_zeros((), dtype=torch.float32)
        cvar = mean
        excess = mean
    return {
        "midi_tail_energy_error_db": mean,
        "midi_tail_cvar_error_db": cvar,
        "midi_tail_excess_energy_db": excess,
    }


def midi_offset_tail_envelope_losses(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    midi_offset: torch.Tensor,
    *,
    window_samples: tuple[int, ...],
    cvar_fraction: float,
) -> dict[str, torch.Tensor]:
    """Region-balanced multiscale log-RMS error after MIDI note-off.

    The legacy tail-energy term collapses the entire release to one scalar, so
    two envelopes with equal power but reversed decay receive no penalty.  This
    loss retains temporal correspondence at 50% overlapping scales, averages
    frames within each example before averaging the batch, and never pools
    samples from before the MIDI offset into the tail objective.
    """

    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate and target must have equal [batch, samples] shape")
    if valid.shape != midi_offset.shape or valid.shape != (estimate.shape[0],):
        raise ValueError("valid and midi_offset must have shape [batch]")
    if not 0.0 < float(cvar_fraction) <= 1.0:
        raise ValueError("cvar_fraction must be in (0, 1]")
    if not window_samples or any(int(window) <= 0 for window in window_samples):
        raise ValueError("window_samples must contain positive integers")

    estimate = estimate.float()
    target = target.float()
    positions = torch.arange(estimate.shape[-1], device=estimate.device)[None]
    sample_valid = positions < valid.long()[:, None]
    target_rms = torch.sqrt(
        (target.square() * sample_valid).sum(dim=-1)
        / sample_valid.sum(dim=-1).clamp_min(1)
    )
    # A target-relative -80 dB floor keeps gradients finite without letting
    # padded or truly silent frames dominate the logarithm.
    floor = (target_rms * 1e-4).clamp_min(1e-7)
    scale = 20.0 / math.log(10.0)
    per_scale_error: list[torch.Tensor] = []
    per_scale_under: list[torch.Tensor] = []
    per_scale_over: list[torch.Tensor] = []
    per_scale_present: list[torch.Tensor] = []

    for raw_window in window_samples:
        window = int(raw_window)
        if window > estimate.shape[-1]:
            continue
        hop = max(1, window // 2)
        estimate_frames = estimate.unfold(-1, window, hop)
        target_frames = target.unfold(-1, window, hop)
        starts = torch.arange(
            estimate_frames.shape[-2], device=estimate.device
        ) * hop
        support = (
            (starts[None] >= midi_offset.long()[:, None])
            & (starts[None] + window <= valid.long()[:, None])
        )
        count = support.sum(dim=-1)
        present = count > 0
        estimate_rms = torch.sqrt(estimate_frames.square().mean(dim=-1))
        target_frame_rms = torch.sqrt(target_frames.square().mean(dim=-1))
        estimate_db = scale * torch.log(estimate_rms + floor[:, None])
        target_db = scale * torch.log(target_frame_rms + floor[:, None])
        signed = estimate_db - target_db
        denominator = count.clamp_min(1).to(estimate.dtype)
        per_scale_error.append((signed.abs() * support).sum(dim=-1) / denominator)
        per_scale_under.append(((-signed).clamp_min(0.0) * support).sum(dim=-1) / denominator)
        per_scale_over.append((signed.clamp_min(0.0) * support).sum(dim=-1) / denominator)
        per_scale_present.append(present)

    zero = estimate.new_zeros(())
    if not per_scale_error:
        return {
            "midi_tail_envelope_mae_db": zero,
            "midi_tail_envelope_cvar_db": zero,
            "midi_tail_envelope_under_db": zero,
            "midi_tail_envelope_over_db": zero,
        }

    present = torch.stack(per_scale_present, dim=1)
    scale_count = present.sum(dim=1)
    example_present = scale_count > 0

    def combine(values: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(values, dim=1)
        return (stacked * present).sum(dim=1) / scale_count.clamp_min(1)

    example_error = combine(per_scale_error)
    example_under = combine(per_scale_under)
    example_over = combine(per_scale_over)
    selected = example_error[example_present]
    if selected.numel() == 0:
        return {
            "midi_tail_envelope_mae_db": zero,
            "midi_tail_envelope_cvar_db": zero,
            "midi_tail_envelope_under_db": zero,
            "midi_tail_envelope_over_db": zero,
        }
    hard_count = max(1, int(math.ceil(selected.numel() * float(cvar_fraction))))
    return {
        "midi_tail_envelope_mae_db": selected.mean(),
        "midi_tail_envelope_cvar_db": torch.topk(
            selected, k=hard_count
        ).values.mean(),
        "midi_tail_envelope_under_db": example_under[example_present].mean(),
        "midi_tail_envelope_over_db": example_over[example_present].mean(),
    }


def high_band_log_magnitude_loss(
    branch: torch.nn.Module,
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    begin_hz: float,
) -> torch.Tensor:
    estimate_spectrum = branch._stft(estimate.float())
    target_spectrum = branch._stft(target.float())
    frequencies = (
        torch.arange(estimate_spectrum.shape[-2], device=estimate.device)
        * float(branch.config.sample_rate)
        / float(branch.config.n_fft)
    )
    frequency_support = frequencies >= float(begin_hz)
    frame_samples = (
        torch.arange(estimate_spectrum.shape[-1], device=estimate.device)[None]
        * int(branch.config.hop_length)
    )
    support = frequency_support[None, :, None] & (
        frame_samples[:, None, :] < valid.long()[:, None, None]
    )
    error = (
        torch.log1p(estimate_spectrum.abs()) - torch.log1p(target_spectrum.abs())
    ).abs()
    return (error * support).sum() / support.sum().clamp_min(1)


_HEAD_ABLATION_OUTPUT_PREFIXES = (
    "harmonic_branch.target_head.",
    "harmonic_branch.background_head.",
    "harmonic_branch.ownership_head.",
    "percussive_branch.target_head.",
    "percussive_branch.background_head.",
    "percussive_branch.ownership_head.",
)


def _is_head_ablation_output_key(key: str) -> bool:
    return key.startswith(_HEAD_ABLATION_OUTPUT_PREFIXES)


def initialize_head_ablation_shared_state(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Copy every shared trunk/communication tensor and reset every head."""

    source = checkpoint_path.expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_sha256 = digest.hexdigest()
    if (
        HEAD_ABLATION_INITIALIZATION_SHA256
        and checkpoint_sha256 != HEAD_ABLATION_INITIALIZATION_SHA256
    ):
        raise ValueError("Head-ablation initialization checkpoint SHA256 changed")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != base.CHECKPOINT_SCHEMA:
        raise ValueError("Head-ablation initialization checkpoint schema changed")
    if int(checkpoint.get("global_step", -1)) != 30_000:
        raise ValueError("Head-ablation initialization must be the mature 30k checkpoint")
    source_contract = checkpoint.get("model_contract")
    if not isinstance(source_contract, dict):
        raise ValueError("Head-ablation checkpoint lacks a model contract")
    source_architecture = source_contract.get("architecture")
    if not isinstance(source_architecture, dict):
        raise ValueError("Head-ablation checkpoint lacks an architecture contract")
    source_communication = source_architecture.get("communication")
    if not isinstance(source_communication, dict) or source_communication.get("type") != (
        "simultaneous_bidirectional_cross_attention"
    ):
        raise ValueError("Head-ablation source is not the symmetric alignment model")
    if str(source_contract.get("preset")) != PRESET:
        raise ValueError("Head-ablation source preset changed")
    if str((source_contract.get("harmonic_gate") or {}).get("mode")) != HPSS_HARMONIC_GATE:
        raise ValueError("Head-ablation source harmonic gate changed")
    for field in ("harmonic_parameterization", "percussive_parameterization"):
        if str(source_architecture.get(field, "ownership")) != "ownership":
            raise ValueError("Head-ablation source must use ownership heads")

    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("Head-ablation checkpoint lacks model state")
    target_state = model.state_dict()
    copied_keys: list[str] = []
    copied_elements = 0
    for key, target_value in target_state.items():
        if _is_head_ablation_output_key(key):
            continue
        source_value = source_state.get(key)
        if not isinstance(source_value, torch.Tensor):
            raise ValueError(f"Head-ablation source lacks shared tensor {key}")
        if source_value.shape != target_value.shape:
            raise ValueError(f"Head-ablation shared tensor shape changed for {key}")
        target_state[key] = source_value.to(dtype=target_value.dtype)
        copied_keys.append(key)
        copied_elements += int(target_value.numel())
    non_head_source_only = sorted(
        key
        for key in source_state
        if key not in target_state and not _is_head_ablation_output_key(key)
    )
    if non_head_source_only:
        raise ValueError(
            f"Head-ablation source has unmatched shared tensors: {non_head_source_only}"
        )
    model.load_state_dict(target_state, strict=True)
    return {
        "schema": "occurrence_head_ablation_shared_trunk_initialization_v1",
        "checkpoint": str(source),
        "checkpoint_sha256": checkpoint_sha256,
        "source_global_step": 30_000,
        "model_only": True,
        "optimizer_reset": True,
        "shared_tensors_copied": len(copied_keys),
        "shared_elements_copied": copied_elements,
        "output_head_policy": "all branch output heads deterministically reinitialized",
        "excluded_output_prefixes": list(_HEAD_ABLATION_OUTPUT_PREFIXES),
    }


def make_model() -> torch.nn.Module:
    if DECOMPOSITION == "hpss":
        model_type = (
            OccurrenceDualHPSSSymmetricAlignment
            if DUAL_COMMUNICATION == "symmetric_lowres"
            else OccurrenceDualHPSSSeparator
        )
        if COMMUNICATION_INITIALIZATION and HEAD_ABLATION_INITIALIZATION:
            raise ValueError("Communication and head-ablation initialization are exclusive")
        model = model_type.from_preset(
            PRESET,
            HPSS_HARMONIC_GATE,
            HPSS_PERCUSSIVE_PARAMETERIZATION,
            CONDITIONING_MODE,
            ROLL_HOLD_MIN_MS,
            ROLL_HOLD_MAX_MS,
            harmonic_parameterization=HPSS_HARMONIC_PARAMETERIZATION,
            output_carrier_mode=OUTPUT_CARRIER_MODE,
            complex_mask_mode=COMPLEX_MASK_MODE,
        )
        if DUAL_COMMUNICATION == "symmetric_lowres" and COMMUNICATION_INITIALIZATION:
            source = Path(COMMUNICATION_INITIALIZATION).expanduser().resolve()
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            baseline = OccurrenceDualHPSSSeparator.from_preset(
                PRESET,
                HPSS_HARMONIC_GATE,
                HPSS_PERCUSSIVE_PARAMETERIZATION,
                CONDITIONING_MODE,
                ROLL_HOLD_MIN_MS,
                ROLL_HOLD_MAX_MS,
                harmonic_parameterization=HPSS_HARMONIC_PARAMETERIZATION,
            )
            source_contract = deepcopy(checkpoint.get("model_contract"))
            legacy_normalizations: list[str] = []
            if isinstance(source_contract, dict):
                architecture = source_contract.get("architecture")
                if isinstance(architecture, dict):
                    for field in (
                        "harmonic_parameterization",
                        "percussive_parameterization",
                    ):
                        if field not in architecture:
                            architecture[field] = "ownership"
                            legacy_normalizations.append(
                                f"architecture.{field}=ownership"
                            )
            if source_contract != baseline.contract():
                difference = base.first_contract_difference(
                    source_contract, baseline.contract()
                )
                raise ValueError(
                    "Communication initialization is not the matched baseline: "
                    f"first_difference={difference}"
                )
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            allowed = ("harmonic_to_percussive.", "percussive_to_harmonic.")
            if unexpected or not missing or any(
                not key.startswith(allowed) for key in missing
            ):
                raise ValueError(
                    f"Invalid communication initialization keys: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            model.communication_initialization = {
                "checkpoint": str(source),
                "checkpoint_sha256": digest.hexdigest(),
                "source_global_step": int(checkpoint.get("global_step", 0)),
                "model_only": True,
                "optimizer_reset": True,
                "accepted_missing_keys": sorted(missing),
                "accepted_missing_prefixes": list(allowed),
                "unexpected_keys": [],
                "legacy_contract_normalizations": legacy_normalizations,
            }
            del checkpoint, baseline
        if HEAD_ABLATION_INITIALIZATION:
            model.communication_initialization = initialize_head_ablation_shared_state(
                model, Path(HEAD_ABLATION_INITIALIZATION)
            )
        return model
    return OccurrenceDualHarmonicResidualSeparator.from_preset(PRESET)


def decompose_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    projection_device = torch.device(HARMONIC_TARGET_DEVICE)
    projection_kwargs = {
        "sample_rate": model.config.sample_rate,
        "n_fft": model.config.n_fft,
        "hop_length": model.config.hop_length,
    }
    valid_samples = batch["valid_samples"].detach().to(projection_device)
    if DECOMPOSITION == "hpss":
        joined = torch.cat((batch["mixture"], batch["target"]), dim=0).detach().to(projection_device)
        joined_valid = torch.cat((valid_samples, valid_samples), dim=0)
        harmonic, second = project_hpss_batch(
            joined,
            joined_valid,
            **projection_kwargs,
            kernel_size=31,
            power=2.0,
            margin=1.0,
        )
        harmonic_mixture, harmonic_target = harmonic.chunk(2, dim=0)
        second_mixture, second_target = second.chunk(2, dim=0)
        target_pitch = torch.cat((batch["pitch"], batch["pitch"]), dim=0).detach().to(projection_device)
        if gate_applies_to_target_and_prediction(HPSS_HARMONIC_GATE):
            gated_targets = pitch_relative_harmonic_gate(
                harmonic,
                target_pitch,
                joined_valid,
                **projection_kwargs,
                mode=HPSS_HARMONIC_GATE,
            )
            _, harmonic_target = gated_targets.chunk(2, dim=0)
        return {
            "harmonic_mixture": harmonic_mixture.to(batch["mixture"].device),
            "second_mixture": second_mixture.to(batch["mixture"].device),
            "harmonic_target": harmonic_target.to(batch["target"].device),
            "second_target": second_target.to(batch["target"].device),
        }
    pitch = batch["pitch"].detach().to(projection_device)
    harmonic_mixture = project_harmonic_target_batch(
        batch["mixture"].detach().to(projection_device),
        pitch,
        valid_samples,
        **projection_kwargs,
    ).to(batch["mixture"].device)
    harmonic_target = project_harmonic_target_batch(
        batch["target"].detach().to(projection_device),
        pitch,
        valid_samples,
        **projection_kwargs,
    ).to(batch["target"].device)
    return {
        "harmonic_mixture": harmonic_mixture,
        "second_mixture": batch["mixture"] - harmonic_mixture,
        "harmonic_target": harmonic_target,
        "second_target": batch["target"] - harmonic_target,
    }


def _allocation_loss(
    branch: torch.nn.Module,
    diagnostics: dict[str, torch.Tensor],
    mixture: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    frequency_bins = int(branch.config.processed_frequency_bins)
    target_spectrum = branch._stft(target.float())[..., :frequency_bins, :]
    background_spectrum = branch._stft((mixture - target).float())[
        ..., :frequency_bins, :
    ]
    if bool(getattr(branch, "complex_mask_parameterization", False)):
        loss_function = (
            unbounded_complex_ratio_mask_loss
            if bool(
                getattr(branch, "unbounded_complex_mask_parameterization", False)
            )
            else bounded_complex_ratio_mask_loss
        )
        return loss_function(
            diagnostics["complex_mask"],
            target_spectrum + background_spectrum,
            target_spectrum,
            diagnostics["frame_support"],
        )
    return magnitude_source_ownership_loss(
        diagnostics["ownership_logits"],
        target_spectrum,
        background_spectrum,
        diagnostics["frame_support"],
    )


def _relative_complex_spectrum_l1(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    frame_support: torch.Tensor,
    *,
    normalizer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a per-example relative complex error over supported STFT bins."""

    if estimate.shape != reference.shape or estimate.ndim != 3:
        raise ValueError("estimate and reference spectra must share [B,F,T] shape")
    if frame_support.shape != (estimate.shape[0], estimate.shape[-1]):
        raise ValueError("frame_support must have shape [B,T]")
    if normalizer is not None and normalizer.shape != reference.shape:
        raise ValueError("complex normalizer must match reference spectrum")
    support = frame_support.to(dtype=estimate.real.dtype)
    count = (support.sum(dim=-1) * estimate.shape[-2]).clamp_min(1.0)
    numerator = (
        (estimate - reference).abs() * support[:, None]
    ).sum(dim=(-2, -1)) / count
    denominator_spectrum = reference if normalizer is None else normalizer
    denominator = (
        denominator_spectrum.abs() * support[:, None]
    ).sum(dim=(-2, -1)) / count
    return (numerator / denominator.clamp_min(1e-5)).mean()


def _preprojection_target_rest_losses(
    model: torch.nn.Module,
    diagnostics: dict[str, Any],
    mixture: torch.Tensor,
    effective_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise the two raw heads without relying on mixture projection."""

    harmonic = diagnostics["harmonic_branch"]
    second = diagnostics[f"{SECOND_COMPONENT}_branch"]
    raw_target = harmonic["raw_target_spectrum"] + second["raw_target_spectrum"]
    raw_background = (
        harmonic["raw_background_spectrum"] + second["raw_background_spectrum"]
    )
    target_spectrum = model.harmonic_branch._stft(effective_target.float())
    mixture_spectrum = model.harmonic_branch._stft(mixture.float())
    background_spectrum = mixture_spectrum - target_spectrum
    support = harmonic["frame_support"]
    target_loss = _relative_complex_spectrum_l1(
        raw_target, target_spectrum, support
    )
    background_loss = _relative_complex_spectrum_l1(
        raw_background,
        background_spectrum,
        support,
        normalizer=mixture_spectrum,
    )
    return target_loss, background_loss


def forward_metrics(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    batch, augmentation_metrics = apply_occurrence_augmentation(
        batch,
        ROBUSTNESS_CONFIG,
        enabled=model.training and ROBUSTNESS_AUGMENTATION != "none",
        generator=_STEP_GENERATOR,
    )
    components = decompose_batch(model, batch)
    original_offset = batch["note_offset_samples"].long()
    conditioned_offset, offset_delta = jitter_occurrence_offsets(
        batch["note_onset_samples"].long(),
        original_offset,
        batch["valid_samples"].long(),
        maximum_ms=OFFSET_JITTER_MS,
        sample_rate=int(model.config.sample_rate),
        generator=_STEP_GENERATOR,
    )
    input_gate_strength = None
    if INPUT_GATE_POLICY == "lower_member_exact_octave":
        input_gate_strength = directional_gate_strength(batch)
    elif INPUT_GATE_PROBABILITY < 1:
        input_gate_strength = (
            torch.rand(
                batch["pitch"].shape,
                device=batch["pitch"].device,
                generator=_STEP_GENERATOR,
            )
            < INPUT_GATE_PROBABILITY
        ).to(batch["mixture"].dtype)
    gate_kwargs = (
        {"input_gate_strength": input_gate_strength}
        if DECOMPOSITION == "hpss"
        else {}
    )
    estimate, diagnostics = model(
        components["harmonic_mixture"],
        components["second_mixture"],
        batch["pitch"].long(),
        batch["valid_samples"].long(),
        batch["note_onset_samples"].long(),
        conditioned_offset,
        **gate_kwargs,
    )
    valid = batch["valid_samples"].long()
    harmonic = _source_losses(
        diagnostics["harmonic"], components["harmonic_target"], valid
    )
    second = _source_losses(
        diagnostics[SECOND_COMPONENT], components["second_target"], valid
    )
    effective_target = components["harmonic_target"] + components["second_target"]
    combined = _source_losses(estimate, effective_target, valid)
    preprojection_target_complex, preprojection_background_complex = (
        _preprojection_target_rest_losses(
            model, diagnostics, batch["mixture"], effective_target
        )
        if (
            PREPROJECTION_TARGET_COMPLEX_WEIGHT > 0
            or PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT > 0
        )
        else (estimate.new_zeros(()), estimate.new_zeros(()))
    )
    if any(
        weight > 0
        for weight in (
            MIDI_TAIL_ENERGY_WEIGHT,
            MIDI_TAIL_CVAR_WEIGHT,
            MIDI_TAIL_EXCESS_WEIGHT,
        )
    ):
        tail = midi_offset_tail_losses(
            estimate,
            effective_target,
            valid,
            original_offset,
            cvar_fraction=MIDI_TAIL_CVAR_FRACTION,
        )
    else:
        zero = estimate.new_zeros((), dtype=torch.float32)
        tail = {
            "midi_tail_energy_error_db": zero,
            "midi_tail_cvar_error_db": zero,
            "midi_tail_excess_energy_db": zero,
        }
    high_band = (
        high_band_log_magnitude_loss(
            model.harmonic_branch,
            estimate,
            effective_target,
            valid,
            begin_hz=HIGH_BAND_BEGIN_HZ,
        )
        if HIGH_BAND_LOG_MAG_WEIGHT > 0
        else estimate.new_zeros((), dtype=torch.float32)
    )
    projection_under = (
        projection_underprediction_loss(estimate, effective_target, valid)
        if PROJECTION_UNDER_WEIGHT > 0
        else estimate.new_zeros((), dtype=torch.float32)
    )
    harmonic_comb = estimate.new_zeros((), dtype=torch.float32)
    harmonic_comb_metrics: dict[str, torch.Tensor] = {}
    if GT_HARMONIC_COMB_LOSS_WEIGHT > 0.0:
        frequency_bins = int(model.harmonic_branch.config.processed_frequency_bins)
        estimate_spectrum = model.harmonic_branch._stft(estimate.float())[
            ..., :frequency_bins, :
        ]
        target_spectrum = model.harmonic_branch._stft(effective_target.float())[
            ..., :frequency_bins, :
        ]
        harmonic_comb, harmonic_comb_metrics = gt_adaptive_harmonic_comb_loss(
            estimate_spectrum,
            target_spectrum,
            batch["pitch"].long(),
            diagnostics["harmonic_branch"]["frame_support"],
            sample_rate=model.harmonic_branch.config.sample_rate,
            n_fft=model.harmonic_branch.config.n_fft,
            inner_hz=15.0,
            outer_hz=60.0,
            maximum_harmonics=60,
            minimum_partial_db=-60.0,
            log_magnitude_weight=GT_HARMONIC_COMB_LOGMAG_WEIGHT,
            relative_complex_cap=GT_HARMONIC_COMB_COMPLEX_CAP,
        )
    harmonic_ownership, harmonic_ownership_metrics = _allocation_loss(
        model.harmonic_branch,
        diagnostics["harmonic_branch"],
        diagnostics.get(
            "harmonic_carrier",
            diagnostics.get("harmonic_input", components["harmonic_mixture"]),
        ),
        components["harmonic_target"],
    )
    second_branch = getattr(model, f"{SECOND_COMPONENT}_branch")
    second_ownership, second_ownership_metrics = _allocation_loss(
        second_branch,
        diagnostics[f"{SECOND_COMPONENT}_branch"],
        diagnostics.get(f"{SECOND_COMPONENT}_carrier", components["second_mixture"]),
        components["second_target"],
    )
    with torch.no_grad():
        anchor_metrics = batch_anchor_quality_metrics(
            estimate,
            effective_target,
            batch["mixture"],
            valid,
        )
    metrics: dict[str, torch.Tensor] = {
        "loss": (
            HARMONIC_LOSS_WEIGHT * harmonic["loss"]
            + SECOND_LOSS_WEIGHT * second["loss"]
            + COMBINED_LOSS_WEIGHT * combined["loss"]
        ) + 0.5 * (harmonic_ownership + second_ownership)
        + MIDI_TAIL_ENERGY_WEIGHT * tail["midi_tail_energy_error_db"]
        + MIDI_TAIL_CVAR_WEIGHT * tail["midi_tail_cvar_error_db"]
        + MIDI_TAIL_EXCESS_WEIGHT * tail["midi_tail_excess_energy_db"]
        + HIGH_BAND_LOG_MAG_WEIGHT * high_band
        + PROJECTION_UNDER_WEIGHT * projection_under
        + GT_HARMONIC_COMB_LOSS_WEIGHT * harmonic_comb
        + PREPROJECTION_TARGET_COMPLEX_WEIGHT * preprojection_target_complex
        + PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT * preprojection_background_complex,
        # The shared trainer uses these stable top-level names for monitor
        # selection and checkpointing. They describe the summed full-target
        # output; branch-qualified copies are logged below.
        "complex": combined["complex"],
        "log_magnitude": combined["log_magnitude"],
        "waveform_l1": combined["waveform_l1"],
        "envelope": combined["envelope"],
        "harmonic_loss": harmonic["loss"],
        f"{SECOND_COMPONENT}_loss": second["loss"],
        "combined_loss": combined["loss"],
        "harmonic_ownership": harmonic_ownership,
        f"{SECOND_COMPONENT}_ownership": second_ownership,
        **tail,
        "high_band_log_magnitude": high_band,
        "projection_underprediction": projection_under,
        "preprojection_target_complex": preprojection_target_complex,
        "preprojection_background_complex": preprojection_background_complex,
        **harmonic_comb_metrics,
        **anchor_metrics,
        "conditioning_offset_abs_jitter_ms": (
            offset_delta.float().abs().mean()
            * 1000.0
            / float(model.config.sample_rate)
        ),
        "conditioning_positive_hold_ms": (
            diagnostics.get(
                "conditioning_hold_extension_samples",
                torch.zeros_like(original_offset),
            ).float().mean()
            * 1000.0
            / float(model.config.sample_rate)
        ),
        "input_gate_fraction": diagnostics.get(
            "input_gate_strength",
            torch.ones_like(batch["pitch"], dtype=batch["mixture"].dtype),
        ).float().mean(),
        **augmentation_metrics,
    }
    if QUERY_OWNERSHIP_AUXILIARY is not None:
        auxiliary = QUERY_OWNERSHIP_AUXILIARY(
            model=model,
            batch=batch,
            components=components,
            positive_estimate=estimate,
            positive_diagnostics=diagnostics,
            conditioned_offset=conditioned_offset,
            gate_kwargs=gate_kwargs,
        )
        auxiliary_loss = auxiliary.pop("loss")
        metrics["loss"] = metrics["loss"] + auxiliary_loss
        metrics["query_ownership_auxiliary_loss"] = auxiliary_loss
        for key, value in auxiliary.items():
            metrics[f"query_ownership_{key}"] = value
    for key, value in diagnostics.get("alignment", {}).items():
        metrics[f"alignment_{key}"] = value
    for prefix, values in (
        ("harmonic", harmonic),
        (SECOND_COMPONENT, second),
        ("combined", combined),
        ("harmonic_ownership", harmonic_ownership_metrics),
        (f"{SECOND_COMPONENT}_ownership", second_ownership_metrics),
    ):
        for key, value in values.items():
            metrics.setdefault(f"{prefix}_{key}", value)
    return metrics


def active_loss_contract() -> dict[str, Any]:
    # Keep the pitch-comb contract byte-for-byte compatible with the original
    # 3k checkpoint.  This is the same decomposition and loss; only the later
    # HPSS arm needs the richer typed decomposition metadata below.
    if DECOMPOSITION == "pitch_comb":
        return {
            "name": "dual_harmonic_residual_target_ownership_v1",
            "task": "direct_occurrence_dual_harmonic_residual",
            "branch_source_losses": {"harmonic": 1 / 3, "residual": 1 / 3},
            "combined_full_target_loss": 1 / 3,
            "branch_ownership_weights": {"harmonic": 0.5, "residual": 0.5},
            **base.ACTIVE_BASE_LOSS_WEIGHTS,
            "dna_conditioning": False,
            "harmonic_projection": {
                "inner_hz": 15.0,
                "outer_hz": 60.0,
                "maximum_harmonics": 60,
                "device": HARMONIC_TARGET_DEVICE,
            },
            "decomposition": {
                "harmonic_mixture": "H_pitch(mixture)",
                "residual_mixture": "mixture-H_pitch(mixture)",
                "harmonic_target": "H_pitch(target)",
                "residual_target": "target-H_pitch(target)",
                "reconstruction": "predicted_harmonic+predicted_residual",
            },
        }
    if DECOMPOSITION == "hpss":
        decomposition = {
            "type": "ordinary_hpss_harmonic_and_percussive",
            "kernel_size": 31,
            "power": 2.0,
            "margin": 1.0,
            "percussive": "exact waveform complement target-H_hpss(target)",
            "explicit_residual_removal": False,
            "reconstruction": "predicted_harmonic+predicted_percussive",
            "harmonic_gate": HPSS_HARMONIC_GATE,
            "input_gate_probability": INPUT_GATE_PROBABILITY,
            "input_gate_policy": INPUT_GATE_POLICY,
            "selective_gate": (
                {
                    "relation": "target_is_lower_member_of_exact_octave_pair",
                    "competitor_semitones": 12,
                    "minimum_overlap_samples": SELECTIVE_OVERLAP_SAMPLES,
                    "minimum_overlap_fraction_of_shorter_note": SELECTIVE_OVERLAP_FRACTION,
                    "score_index": str(FULL_SCORE_INDEX),
                    "score_index_sha256": base.sha256_file(FULL_SCORE_INDEX),
                }
                if INPUT_GATE_POLICY == "lower_member_exact_octave"
                else None
            ),
            "input_gate_sampling": (
                "deterministic_score_directional"
                if INPUT_GATE_POLICY == "lower_member_exact_octave"
                else "independent_per_example_bernoulli"
                if INPUT_GATE_PROBABILITY < 1
                else "always_static_model_gate"
            ),
            "percussive_component_after_gate": "unchanged",
            "percussive_parameterization": HPSS_PERCUSSIVE_PARAMETERIZATION,
            "output_carrier_mode": OUTPUT_CARRIER_MODE,
            "backbone": DUAL_BACKBONE,
            "percussive_phase_policy": (
                "reuse_hpss_percussive_mixture_phase"
                if HPSS_PERCUSSIVE_PARAMETERIZATION == "magnitude_mask"
                else f"{COMPLEX_MASK_MODE}_complex_ratio_mask"
                if HPSS_PERCUSSIVE_PARAMETERIZATION == "complex_mask"
                else "learned_complex_residual_plus_ownership"
            ),
            "optimized_sum_target": (
                "gated_hpss_harmonic_target+ordinary_hpss_percussive_target"
                if gate_applies_to_target_and_prediction(HPSS_HARMONIC_GATE)
                else "ungated_hpss_harmonic_target+ordinary_hpss_percussive_target"
            ),
        }
    complex_mask_ablation = "complex_mask" in {
        HPSS_HARMONIC_PARAMETERIZATION,
        HPSS_PERCUSSIVE_PARAMETERIZATION,
    }
    if complex_mask_ablation:
        decomposition["harmonic_parameterization"] = HPSS_HARMONIC_PARAMETERIZATION
    contract: dict[str, Any] = {
        "name": (
            f"dual_harmonic_{SECOND_COMPONENT}_target_{COMPLEX_MASK_MODE}_complex_mask_v1"
            if complex_mask_ablation
            else f"dual_harmonic_{SECOND_COMPONENT}_target_ownership_v1"
        ),
        "task": f"direct_occurrence_dual_harmonic_{SECOND_COMPONENT}",
        "branch_source_losses": {"harmonic": HARMONIC_LOSS_WEIGHT, SECOND_COMPONENT: SECOND_LOSS_WEIGHT},
        "combined_effective_target_loss": COMBINED_LOSS_WEIGHT,
        **base.ACTIVE_BASE_LOSS_WEIGHTS,
        "dna_conditioning": False,
        "decomposition_device": HARMONIC_TARGET_DEVICE,
        "decomposition": decomposition,
    }
    if CONDITIONING_MODE == "harmonic_lifecycle":
        contract["legacy_conditioning_finetune_opt_in"] = True
    if complex_mask_ablation:
        contract["branch_allocation_supervision"] = {
            "harmonic": {
                "parameterization": HPSS_HARMONIC_PARAMETERIZATION,
                "weight": 0.5,
            },
            SECOND_COMPONENT: {
                "parameterization": HPSS_PERCUSSIVE_PARAMETERIZATION,
                "weight": 0.5,
            },
            "complex_mask_oracle": (
                "regularized_unbounded_target_times_carrier_conjugate_over_power"
                if COMPLEX_MASK_MODE == "unbounded"
                else "clipped_target_times_mixture_conjugate_over_power"
            ),
            "complex_mask_loss": "fourth_root_mixture_power_weighted_smooth_l1",
        }
    else:
        contract["branch_ownership_weights"] = {
            "harmonic": 0.5,
            SECOND_COMPONENT: 0.5,
        }
    if ROBUSTNESS_AUGMENTATION != "none":
        contract["training_augmentation"] = ROBUSTNESS_CONFIG.contract()
    contract["gt_adaptive_harmonic_comb_loss"] = (
        None
        if GT_HARMONIC_COMB_LOSS_WEIGHT == 0.0
        else {
            "weight": GT_HARMONIC_COMB_LOSS_WEIGHT,
            "target": "summed_effective_ground_truth_target",
            "estimate": "summed_harmonic_plus_second_prediction",
            "partial_estimator": (
                "per_example_time_averaged_gt_magnitude_local_argmax_"
                "within_integer_harmonic_midpoints"
            ),
            "minimum_partial_db": -60.0,
            "inner_hz": 15.0,
            "outer_hz": 60.0,
            "maximum_harmonics": 60,
            "complex": "per_example_target_magnitude_normalized_l1",
            "relative_complex_cap": GT_HARMONIC_COMB_COMPLEX_CAP,
            "log_magnitude": "log1p_100_magnitude_l1",
            "log_magnitude_weight": GT_HARMONIC_COMB_LOGMAG_WEIGHT,
            "input_operation": "none",
            "output_operation": "none",
        }
    )
    if (
        PREPROJECTION_TARGET_COMPLEX_WEIGHT > 0
        or PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT > 0
    ):
        contract["preprojection_target_rest_supervision"] = {
            "target_weight": PREPROJECTION_TARGET_COMPLEX_WEIGHT,
            "background_weight": PREPROJECTION_BACKGROUND_COMPLEX_WEIGHT,
            "target": "sum_of_raw_branch_target_spectra",
            "background": "sum_of_raw_branch_background_spectra",
            "target_normalization": "per_example_target_stft_magnitude",
            "background_normalization": "per_example_mixture_stft_magnitude",
            "projection_dependency": "none",
        }
    if any(
        value > 0
        for value in (
            OFFSET_JITTER_MS,
            MIDI_TAIL_ENERGY_WEIGHT,
            MIDI_TAIL_CVAR_WEIGHT,
            MIDI_TAIL_EXCESS_WEIGHT,
            HIGH_BAND_LOG_MAG_WEIGHT,
            PROJECTION_UNDER_WEIGHT,
        )
    ):
        contract["auditory_quality_ablation"] = {
            "condition_only_offset_jitter_ms": OFFSET_JITTER_MS,
            "midi_offset_tail_energy_weight": MIDI_TAIL_ENERGY_WEIGHT,
            "midi_offset_tail_cvar_weight": MIDI_TAIL_CVAR_WEIGHT,
            "midi_offset_tail_excess_weight": MIDI_TAIL_EXCESS_WEIGHT,
            "midi_offset_tail_cvar_fraction": MIDI_TAIL_CVAR_FRACTION,
            "high_band_log_magnitude_weight": HIGH_BAND_LOG_MAG_WEIGHT,
            "high_band_begin_hz": HIGH_BAND_BEGIN_HZ,
            "projection_underprediction_weight": PROJECTION_UNDER_WEIGHT,
            "projection_underprediction": (
                "mean(relu(1-<estimate,target>/||target||^2)^2) over valid support"
            ),
            "tail_region": "midi_note_off_through_valid_acoustic_crop",
        }
    return contract


@torch.inference_mode()
def build_monitor_media(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    autocast: Any,
    wandb: Any,
) -> dict[str, Any]:
    raw = next(iter(loader))
    batch = base.to_device(raw, device)
    components = decompose_batch(model, batch)
    model.eval()
    try:
        with autocast():
            gate_kwargs = {}
            if INPUT_GATE_POLICY == "lower_member_exact_octave":
                gate_kwargs["input_gate_strength"] = directional_gate_strength(batch)
            estimate, diagnostics = model(
                components["harmonic_mixture"],
                components["second_mixture"],
                batch["pitch"].long(),
                batch["valid_samples"].long(),
                batch["note_onset_samples"].long(),
                batch["note_offset_samples"].long(),
                **gate_kwargs,
            )
    finally:
        model.train()

    from scripts.reformat_occurrence_media_smoke import figure_for
    import matplotlib.pyplot as plt

    payload: dict[str, Any] = {}
    for index in range(min(2, int(batch["mixture"].shape[0]))):
        length = int(batch["valid_samples"][index])
        signals = {
            "mixture": batch["mixture"][index, :length].float().cpu().numpy(),
            "harmonic_mixture": components["harmonic_mixture"][index, :length].float().cpu().numpy(),
            f"{SECOND_COMPONENT}_mixture": components["second_mixture"][index, :length].float().cpu().numpy(),
            "target": batch["target"][index, :length].float().cpu().numpy(),
            "harmonic_target": components["harmonic_target"][index, :length].float().cpu().numpy(),
            f"{SECOND_COMPONENT}_target": components["second_target"][index, :length].float().cpu().numpy(),
            "harmonic_prediction": diagnostics["harmonic"][index, :length].float().cpu().numpy(),
            f"{SECOND_COMPONENT}_prediction": diagnostics[SECOND_COMPONENT][index, :length].float().cpu().numpy(),
            "combined_prediction": estimate[index, :length].float().cpu().numpy(),
        }
        figure = figure_for([(name.replace("_", " ").title(), audio) for name, audio in signals.items()])
        prefix = f"monitor_media/example_{index}_p{int(batch['pitch'][index])}"
        for name, audio in signals.items():
            payload[f"{prefix}/{name}"] = wandb.Audio(audio, sample_rate=16_000)
        payload[f"{prefix}/spectrogram_comparison"] = wandb.Image(figure)
        plt.close(figure)
    return payload


prototype = make_model()
base.ACTIVE_MODEL_CONTRACT = prototype.contract()
base.MODEL_KIND = f"dual_harmonic_{SECOND_COMPONENT}_tfc_tdf"
base.selected_model = make_model
base.forward_metrics = forward_metrics
base.build_monitor_media = build_monitor_media
base.active_loss_contract = active_loss_contract
base.set_training_step = set_training_step
if INPUT_GATE_POLICY == "lower_member_exact_octave":
    _base_collate_v13 = base.collate_v13
    FullScoreSyntheticDataset.score_index = FULL_SCORE_INDEX

    def selective_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_full_score(rows, _base_collate_v13)

    base.V13StreamingResidualDataset = FullScoreSyntheticDataset
    base.collate_v13 = selective_collate


if __name__ == "__main__":
    base.main()
