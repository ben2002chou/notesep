"""Pitch-conditioned harmonic target projection for occurrence separation."""

from __future__ import annotations

import math

import torch


def gt_adaptive_harmonic_partial_masks(
    target_spectrum: torch.Tensor,
    pitch: torch.Tensor,
    frame_support: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    inner_hz: float = 15.0,
    outer_hz: float = 60.0,
    maximum_harmonics: int = 60,
    minimum_partial_db: float = -60.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locate target partials and return one broad mask per partial.

    Each expected integer-harmonic cell searches the clean target's
    time-averaged magnitude for its strongest bin.  Searching within the
    midpoints between adjacent ideal harmonics allows instrument-specific
    detuning and inharmonicity without assigning the same peak to neighboring
    partials.  The resulting centers drive the same raised-cosine 15--60 Hz
    neighborhoods as the existing query-pitch comb.

    Returns ``(partial_masks, centers_hz, partial_valid)`` with shapes
    ``[B, F, K]``, ``[B, K]``, and ``[B, K]``.  This is a target-derived loss
    weight only; all returned tensors are detached from autograd.
    """

    if target_spectrum.ndim != 3 or not target_spectrum.is_complex():
        raise ValueError("target_spectrum must be complex [batch, frequency, frame]")
    if pitch.shape != target_spectrum.shape[:1]:
        raise ValueError("pitch must have shape [batch]")
    if frame_support.shape != target_spectrum.shape[::2]:
        raise ValueError("frame_support must have shape [batch, frame]")
    if not 0.0 < inner_hz < outer_hz:
        raise ValueError("template widths must satisfy 0 < inner < outer")
    if maximum_harmonics < 1:
        raise ValueError("maximum_harmonics must be positive")
    if minimum_partial_db >= 0.0:
        raise ValueError("minimum_partial_db must be negative")

    device = target_spectrum.device
    frequency_bins = target_spectrum.shape[-2]
    frequencies = torch.arange(
        frequency_bins, device=device, dtype=torch.float32
    ) * (float(sample_rate) / float(n_fft))
    fundamental = 440.0 * torch.pow(
        2.0, (pitch.detach().float() - 69.0) / 12.0
    )
    harmonic_number = torch.arange(
        1, maximum_harmonics + 1, device=device, dtype=torch.float32
    )
    expected = fundamental[:, None] * harmonic_number[None]

    valid_frames = frame_support.to(torch.float32)
    frame_count = valid_frames.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_magnitude = (
        target_spectrum.detach().abs().float() * valid_frames[:, None]
    ).sum(dim=-1) / frame_count

    # Ensure every cell contains at least its nearest STFT bin. This matters
    # for low fundamentals whose half-spacing is below one FFT bin.
    bin_hz = float(sample_rate) / float(n_fft)
    search_radius = torch.maximum(
        0.5 * fundamental,
        torch.full_like(fundamental, 0.51 * bin_hz),
    )
    distance_to_expected = (
        frequencies[None, None] - expected[:, :, None]
    ).abs()
    search = distance_to_expected <= search_radius[:, None, None]
    search = search & (frequencies[None, None] > 0.0)
    center_valid = expected <= float(sample_rate) / 2.0 + float(outer_hz)
    search = search & center_valid[:, :, None]
    search_scores = mean_magnitude[:, None].expand(
        -1, maximum_harmonics, -1
    ).masked_fill(~search, -1.0)
    peak_bin = search_scores.argmax(dim=-1)
    centers_hz = frequencies[peak_bin]
    peak_magnitude = mean_magnitude.gather(1, peak_bin)
    relative_floor = mean_magnitude.amax(dim=-1, keepdim=True) * (
        10.0 ** (float(minimum_partial_db) / 20.0)
    )
    partial_valid = center_valid & (peak_magnitude >= relative_floor.clamp_min(1e-10))

    distance_to_peak = (frequencies[:, None] - centers_hz[:, None]).abs()
    transition = (
        (distance_to_peak - float(inner_hz))
        / (float(outer_hz) - float(inner_hz))
    ).clamp(0.0, 1.0)
    partial_masks = 0.5 + 0.5 * torch.cos(math.pi * transition)
    partial_masks = partial_masks * partial_valid[:, None].to(partial_masks.dtype)
    return partial_masks.detach(), centers_hz.detach(), partial_valid.detach()


def gt_adaptive_harmonic_template(
    target_spectrum: torch.Tensor,
    pitch: torch.Tensor,
    frame_support: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    inner_hz: float = 15.0,
    outer_hz: float = 60.0,
    maximum_harmonics: int = 60,
    minimum_partial_db: float = -60.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locate target partials and return their detached union template."""

    partial_masks, centers_hz, partial_valid = gt_adaptive_harmonic_partial_masks(
        target_spectrum,
        pitch,
        frame_support,
        sample_rate=sample_rate,
        n_fft=n_fft,
        inner_hz=inner_hz,
        outer_hz=outer_hz,
        maximum_harmonics=maximum_harmonics,
        minimum_partial_db=minimum_partial_db,
    )
    return partial_masks.amax(dim=-1), centers_hz, partial_valid


def gt_adaptive_harmonic_comb_loss(
    estimate_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    pitch: torch.Tensor,
    frame_support: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    inner_hz: float = 15.0,
    outer_hz: float = 60.0,
    maximum_harmonics: int = 60,
    minimum_partial_db: float = -60.0,
    log_magnitude_weight: float = 0.5,
    relative_complex_cap: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Relative target reconstruction loss inside GT-located partial bands."""

    if estimate_spectrum.shape != target_spectrum.shape:
        raise ValueError("estimate and target spectra must have matching shapes")
    if log_magnitude_weight < 0.0:
        raise ValueError("log_magnitude_weight must be non-negative")
    if relative_complex_cap < 0.0:
        raise ValueError("relative_complex_cap must be non-negative")
    template, _, partial_valid = gt_adaptive_harmonic_template(
        target_spectrum,
        pitch,
        frame_support,
        sample_rate=sample_rate,
        n_fft=n_fft,
        inner_hz=inner_hz,
        outer_hz=outer_hz,
        maximum_harmonics=maximum_harmonics,
        minimum_partial_db=minimum_partial_db,
    )
    weights = template[:, :, None] * frame_support[:, None].to(template.dtype)
    weight_sum = weights.sum(dim=(-2, -1))
    has_support = weight_sum > 0.0
    denominator = weight_sum.clamp_min(1.0)

    estimate_magnitude = estimate_spectrum.abs().float()
    target_magnitude = target_spectrum.detach().abs().float()
    absolute_complex = (
        (estimate_spectrum - target_spectrum.detach()).abs().float() * weights
    ).sum(dim=(-2, -1)) / denominator
    target_scale = (target_magnitude * weights).sum(dim=(-2, -1)) / denominator
    relative_complex = absolute_complex / target_scale.clamp_min(1e-5)
    log_error = (
        torch.log1p(100.0 * estimate_magnitude)
        - torch.log1p(100.0 * target_magnitude)
    ).abs()
    log_magnitude = (log_error * weights).sum(dim=(-2, -1)) / denominator
    relative_complex = torch.where(
        has_support, relative_complex, torch.zeros_like(relative_complex)
    )
    log_magnitude = torch.where(
        has_support, log_magnitude, torch.zeros_like(log_magnitude)
    )

    valid = frame_support[:, None].to(target_magnitude.dtype)
    total_target_power = (target_magnitude.square() * valid).sum(
        dim=(-2, -1)
    ).clamp_min(1e-10)
    harmonic_target_fraction = (
        target_magnitude.square() * weights
    ).sum(dim=(-2, -1)) / total_target_power
    complex_mean = relative_complex.mean()
    optimized_complex = (
        relative_complex.clamp_max(float(relative_complex_cap)).mean()
        if float(relative_complex_cap) > 0.0
        else complex_mean
    )
    log_mean = log_magnitude.mean()
    loss = optimized_complex + float(log_magnitude_weight) * log_mean
    return loss, {
        "gt_harmonic_comb_complex": complex_mean,
        "gt_harmonic_comb_complex_optimized": optimized_complex,
        "gt_harmonic_comb_complex_capped_fraction": (
            (relative_complex > float(relative_complex_cap)).float().mean()
            if float(relative_complex_cap) > 0.0
            else relative_complex.new_zeros(())
        ),
        "gt_harmonic_comb_log_magnitude": log_mean,
        "gt_harmonic_comb_loss": loss,
        "gt_harmonic_comb_support_fraction": template.mean(),
        "gt_harmonic_comb_target_energy_fraction": harmonic_target_fraction.mean(),
        "gt_harmonic_comb_partial_count": partial_valid.sum(dim=-1).float().mean(),
    }


def gt_adaptive_harmonic_group_balanced_loss(
    estimate_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    pitch: torch.Tensor,
    frame_support: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    inner_hz: float = 15.0,
    outer_hz: float = 60.0,
    maximum_harmonics: int = 60,
    minimum_partial_db: float = -60.0,
    log_magnitude_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Give H1, H2--H3, H4--H7, and H8+ equal reconstruction weight.

    Every group is normalized by its own GT magnitude and averaged only when
    that example has a detected partial in the group.  This prevents a loud
    fundamental or H2 from hiding attenuated upper partials while retaining
    GT-localized, instrument-specific partial centers.
    """

    if estimate_spectrum.shape != target_spectrum.shape:
        raise ValueError("estimate and target spectra must have matching shapes")
    if maximum_harmonics < 8:
        raise ValueError("group-balanced loss requires at least 8 harmonics")
    if log_magnitude_weight < 0.0:
        raise ValueError("log_magnitude_weight must be non-negative")
    partial_masks, _, partial_valid = gt_adaptive_harmonic_partial_masks(
        target_spectrum,
        pitch,
        frame_support,
        sample_rate=sample_rate,
        n_fft=n_fft,
        inner_hz=inner_hz,
        outer_hz=outer_hz,
        maximum_harmonics=maximum_harmonics,
        minimum_partial_db=minimum_partial_db,
    )
    estimate_magnitude = estimate_spectrum.abs().float()
    target_magnitude = target_spectrum.detach().abs().float()
    complex_error = (estimate_spectrum - target_spectrum.detach()).abs().float()
    log_error = (
        torch.log1p(100.0 * estimate_magnitude)
        - torch.log1p(100.0 * target_magnitude)
    ).abs()
    target_log_magnitude = torch.log1p(100.0 * target_magnitude)
    frame_weights = frame_support[:, None].to(target_magnitude.dtype)
    group_ranges = {
        "h1": (0, 1),
        "h2_h3": (1, 3),
        "h4_h7": (3, 7),
        "h8_plus": (7, maximum_harmonics),
    }
    group_losses: list[torch.Tensor] = []
    group_supported: list[torch.Tensor] = []
    metrics: dict[str, torch.Tensor] = {}
    for name, (begin, end) in group_ranges.items():
        group_template = partial_masks[..., begin:end].amax(dim=-1)
        weights = group_template[:, :, None] * frame_weights
        weight_sum = weights.sum(dim=(-2, -1))
        supported = partial_valid[:, begin:end].any(dim=-1) & (weight_sum > 0.0)
        denominator = weight_sum.clamp_min(1.0)
        absolute_complex = (complex_error * weights).sum(dim=(-2, -1)) / denominator
        target_scale = (target_magnitude * weights).sum(dim=(-2, -1)) / denominator
        relative_complex = absolute_complex / target_scale.clamp_min(1e-5)
        absolute_log_magnitude = (log_error * weights).sum(
            dim=(-2, -1)
        ) / denominator
        target_log_scale = (target_log_magnitude * weights).sum(
            dim=(-2, -1)
        ) / denominator
        relative_log_magnitude = absolute_log_magnitude / target_log_scale.clamp_min(
            1e-5
        )
        combined = (
            relative_complex
            + float(log_magnitude_weight) * relative_log_magnitude
        )
        combined = torch.where(supported, combined, torch.zeros_like(combined))
        group_losses.append(combined)
        group_supported.append(supported)
        support_count = supported.float().sum().clamp_min(1.0)
        metrics[f"gt_harmonic_group_{name}_complex"] = (
            torch.where(supported, relative_complex, torch.zeros_like(relative_complex)).sum()
            / support_count
        )
        metrics[f"gt_harmonic_group_{name}_log_magnitude"] = (
            torch.where(
                supported,
                relative_log_magnitude,
                torch.zeros_like(relative_log_magnitude),
            ).sum()
            / support_count
        )
        metrics[f"gt_harmonic_group_{name}_support_fraction"] = supported.float().mean()

    stacked_loss = torch.stack(group_losses, dim=-1)
    stacked_supported = torch.stack(group_supported, dim=-1)
    loss = stacked_loss.sum() / stacked_supported.float().sum().clamp_min(1.0)
    metrics.update(
        {
            "gt_harmonic_group_balanced_loss": loss,
            "gt_harmonic_group_balanced_supported_groups": (
                stacked_supported.float().sum(dim=-1).mean()
            ),
            "gt_harmonic_group_balanced_partial_count": (
                partial_valid.sum(dim=-1).float().mean()
            ),
        }
    )
    return loss, metrics


def harmonic_soft_mask_torch(
    pitch: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    inner_hz: float = 15.0,
    outer_hz: float = 60.0,
    maximum_harmonics: int = 60,
) -> torch.Tensor:
    """Return raised-cosine masks shaped ``[batch, n_fft // 2 + 1]``."""

    if pitch.ndim != 1:
        raise ValueError("pitch must have shape [batch]")
    if not 0.0 < inner_hz < outer_hz:
        raise ValueError("mask widths must satisfy 0 < inner < outer")
    if maximum_harmonics < 1:
        raise ValueError("maximum_harmonics must be positive")

    # Match the NumPy audit's float64 frequency geometry before returning the
    # model-friendly float mask.  High harmonics otherwise accumulate enough
    # float32 center error to perturb bins near the raised-cosine boundary.
    dtype = torch.float64
    frequencies = torch.arange(
        n_fft // 2 + 1, device=pitch.device, dtype=dtype
    ) * (float(sample_rate) / float(n_fft))
    fundamental = 440.0 * torch.pow(
        torch.tensor(2.0, device=pitch.device, dtype=dtype),
        (pitch.to(dtype) - 69.0) / 12.0,
    )
    harmonic_number = torch.arange(
        1, maximum_harmonics + 1, device=pitch.device, dtype=dtype
    )
    centers = fundamental[:, None] * harmonic_number[None, :]
    valid_center = centers <= float(sample_rate) / 2.0 + outer_hz
    distance = (frequencies[None, :, None] - centers[:, None, :]).abs()
    distance = distance.masked_fill(~valid_center[:, None, :], float("inf"))
    nearest = distance.amin(dim=-1)
    transition = ((nearest - inner_hz) / (outer_hz - inner_hz)).clamp(0.0, 1.0)
    return (0.5 + 0.5 * torch.cos(math.pi * transition)).float()


def project_harmonic_target_batch(
    audio: torch.Tensor,
    pitch: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    inner_hz: float = 15.0,
    outer_hz: float = 60.0,
    maximum_harmonics: int = 60,
) -> torch.Tensor:
    """Project padded waveforms onto smooth neighborhoods of their harmonics."""

    if audio.ndim != 2:
        raise ValueError("audio must have shape [batch, samples]")
    if pitch.shape != audio.shape[:1] or valid_samples.shape != audio.shape[:1]:
        raise ValueError("pitch and valid_samples must have shape [batch]")
    samples = audio.shape[-1]
    support = torch.arange(samples, device=audio.device)[None] < valid_samples.long()[:, None]
    values = audio.float() * support
    window = torch.hann_window(n_fft, device=audio.device, dtype=values.dtype)
    spectrum = torch.stft(
        values,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    mask = harmonic_soft_mask_torch(
        pitch,
        sample_rate=sample_rate,
        n_fft=n_fft,
        inner_hz=inner_hz,
        outer_hz=outer_hz,
        maximum_harmonics=maximum_harmonics,
    )
    projected = torch.istft(
        spectrum * mask[:, :, None],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=samples,
    )
    return projected * support
