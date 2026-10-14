"""Score-onset support applied to raw candidates BEFORE ASO allocation.

This inference adapter does not change checkpoint weights or the frozen model
source. Offsets and the final waveform are not gated. With a centered STFT,
waveform support can begin n_fft/2 samples before the first allowed frame.
"""
from __future__ import annotations

import math
import torch


def bound_raw_onsets(raw, onsets, *, sample_rate=16000, hop_length=512,
                     lead_seconds=0.100):
    """Return bounded [B,N,F,T] complex candidates; onsets are local samples.

    Negative onsets are allowed for notes carried in from a preceding window.
    This is a centered-STFT API: frame zero is centered at sample zero.
    """
    if raw.ndim != 4 or not raw.is_complex():
        raise ValueError("raw must be complex [batch, notes, frequency, time]")
    if onsets.shape != raw.shape[:2] or not torch.isfinite(onsets).all():
        raise ValueError("finite onset samples must have shape [batch, notes]")
    if (not math.isfinite(lead_seconds) or lead_seconds < 0
            or sample_rate <= 0 or hop_length <= 0):
        raise ValueError("lead must be nonnegative; sample rate and hop positive")
    centers = torch.arange(raw.shape[-1], device=raw.device) * hop_length
    allowed = centers[None, None, :] >= (
        onsets.to(raw.device)[..., None] - round(lead_seconds * sample_rate))
    return raw * allowed[:, :, None, :]


def onset_supported_forward(model, mixture, raw, pitches, *, onsets,
                            acoustic_offsets, lead_seconds=0.100,
                            sample_rate=16000, hop_length=512,
                            support_onsets=None, **kwargs):
    """Run frozen ASO with onset-supported raw features AND magnitude shares.

    Validated for AcousticMultiplicativeTemporalOwnership (step 29058).
    Zero raw proposals cannot acquire ownership in that allocator. No
    all-empty-bin repair, post-output crop, phase change, or gain is added.
    Old benchmark callers can still call the frozen model directly.

    `onsets` are the model's conditioning coordinates. When conditioning has
    been widened, pass the original local score onsets as `support_onsets`:
    the 100 ms support allowance must not be added to an already early onset.
    Omitting it preserves existing callers' behavior exactly.
    """
    if getattr(model, "hop_length", hop_length) != hop_length:
        raise ValueError("support hop must match ASO's temporal coordinates")
    if mixture.shape != (raw.shape[0], *raw.shape[-2:]):
        raise ValueError("mixture must be [batch, frequency, time]")
    if onsets.shape != raw.shape[:2] or not torch.isfinite(onsets).all():
        raise ValueError("finite conditioning onsets must have shape [batch, notes]")
    score_onsets = onsets if support_onsets is None else support_onsets
    candidates = bound_raw_onsets(raw, score_onsets, sample_rate=sample_rate,
                                 hop_length=hop_length, lead_seconds=lead_seconds)
    magnitude = candidates.abs()
    total = magnitude.sum(1, keepdim=True)
    shares = torch.where(total > 1e-20, magnitude / total.clamp_min(1e-20), 0)
    anchors = shares * mixture[:, None]
    return model(mixture, anchors, candidates, pitches, onsets=onsets,
                 acoustic_offsets=acoustic_offsets, **kwargs)
