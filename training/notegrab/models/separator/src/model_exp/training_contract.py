"""Model and loss settings for the NoteSep training recipe."""

from __future__ import annotations

import math
import os

MODEL_CONTRACT = {
    "n_fft": 512,
    "hop_length": 128,
    "channels": 96,
    "blocks": 8,
    "pitch_embed_dim": 16,
    "use_rt2_prior": False,
    "use_region_context": True,
    "maximum_correction_scale": 2.0,
    "output_mode": "ungated_residual",
    "soft_replace_gate_bias": -12.0,
    "sample_rate": 16_000,
    "use_harmonic_features": False,
    "use_frequency_dilated_blocks": False,
}


def _environment_loss_weight(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


FREE_TARGET_LOSS_WEIGHTS = {
    "complex_weight": _environment_loss_weight("DNA_V13_COMPLEX_WEIGHT", 1.0),
    "complex_error_cap": _environment_loss_weight("DNA_V13_COMPLEX_CAP", 0.0),
    "log_magnitude_weight": _environment_loss_weight(
        "DNA_V13_LOGMAG_WEIGHT", 0.5
    ),
    "waveform_weight": _environment_loss_weight("DNA_V13_WAVEFORM_WEIGHT", 0.2),
    "envelope_weight": _environment_loss_weight("DNA_V13_ENVELOPE_WEIGHT", 0.2),
    "phase_coherence_weight": _environment_loss_weight(
        "DNA_V13_PHASE_WEIGHT", 0.0
    ),
    "do_no_harm_weight": 0.0,
    "correction_weight": 0.0,
}
