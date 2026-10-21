"""Semantics-preserving waveform augmentations for occurrence separation.

The transformations in this module preserve the additive mixture/target
contract.  Shared transformations are applied to every available source
waveform, while additive noise is placed only in the mixture-side residual so
the task explicitly remains noise-to-clean occurrence separation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


_AUDIO_KEYS = ("mixture", "target", "dna", "other_dna", "unassigned_residual")


@dataclass(frozen=True)
class OccurrenceAugmentationConfig:
    gain_probability: float = 0.80
    gain_db_limit: float = 6.0
    pitch_shift_probability: float = 0.0
    pitch_shift_semitones_limit: int = 0
    detune_probability: float = 0.35
    detune_cents_limit: float = 20.0
    linear_echo_probability: float = 0.25
    linear_echo_wet_limit: float = 0.16
    linear_echo_delay_ms_min: float = 4.0
    linear_echo_delay_ms_max: float = 24.0
    noise_probability: float = 0.35
    noise_snr_db_min: float = 20.0
    noise_snr_db_max: float = 40.0
    sample_rate: int = 16_000

    def validate(self) -> None:
        for name in (
            "gain_probability",
            "pitch_shift_probability",
            "detune_probability",
            "linear_echo_probability",
            "noise_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "gain_db_limit",
            "detune_cents_limit",
            "linear_echo_wet_limit",
            "linear_echo_delay_ms_min",
            "linear_echo_delay_ms_max",
            "noise_snr_db_min",
            "noise_snr_db_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.linear_echo_delay_ms_min > self.linear_echo_delay_ms_max:
            raise ValueError("linear echo delay bounds are reversed")
        if self.noise_snr_db_min > self.noise_snr_db_max:
            raise ValueError("noise SNR bounds are reversed")
        if not 0.0 <= self.linear_echo_wet_limit < 1.0:
            raise ValueError("linear_echo_wet_limit must be in [0, 1)")
        if int(self.sample_rate) < 1:
            raise ValueError("sample_rate must be positive")
        if (
            isinstance(self.pitch_shift_semitones_limit, bool)
            or int(self.pitch_shift_semitones_limit)
            != self.pitch_shift_semitones_limit
            or int(self.pitch_shift_semitones_limit) < 0
        ):
            raise ValueError("pitch_shift_semitones_limit must be a non-negative integer")
        if self.pitch_shift_probability > 0 and self.pitch_shift_semitones_limit < 1:
            raise ValueError(
                "positive pitch_shift_probability requires a nonzero semitone limit"
            )

    def contract(self) -> dict[str, Any]:
        values = asdict(self)
        transposition_enabled = self.pitch_shift_probability > 0
        if not transposition_enabled:
            # Preserve the coherent_v1 checkpoint contract byte-for-byte so a
            # running or interrupted v1 job remains strictly resumable.
            values.pop("pitch_shift_probability")
            values.pop("pitch_shift_semitones_limit")
        contract = {
            "type": (
                "coherent_occurrence_waveform_robustness_v2"
                if transposition_enabled
                else "coherent_occurrence_waveform_robustness_v1"
            ),
            **values,
            "gain_policy": "same_per_example_gain_for_mixture_and_all_sources",
            "detune_policy": (
                "batch_shared_speed_perturbation_for_all_sources_with_rescaled_"
                "sample_geometry_and_nominal_midi_query"
            ),
            "linear_effect_policy": "same_causal_direct_plus_single_echo_per_example",
            "noise_policy": "colored_mixture_only_noise_with_clean_target",
            "validation_policy": "clean_unaugmented",
        }
        if transposition_enabled:
            contract["pitch_shift_policy"] = (
                "batch_shared_length_preserving_integer_semitone_shift_with_"
                "per_source_rms_preservation_and_same_shift_applied_to_target_"
                "mixture_components_and_midi_query"
            )
        return contract


def _sample_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    values = torch.rand(
        shape,
        device=reference.device,
        dtype=torch.float32,
        generator=generator,
    )
    return values * (float(high) - float(low)) + float(low)


def _masked_rms(values: torch.Tensor, valid_samples: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(values.shape[-1], device=values.device)[None]
    support = positions < valid_samples.long()[:, None]
    denominator = support.sum(dim=-1).clamp_min(1).float()
    power = (values.float().square() * support).sum(dim=-1) / denominator
    return power.sqrt()


def _resize_waveforms(values: torch.Tensor, length: int) -> torch.Tensor:
    resized = F.interpolate(
        values.float()[:, None],
        size=int(length),
        mode="linear",
        align_corners=False,
    )[:, 0]
    original_length = int(values.shape[-1])
    if int(length) < original_length:
        resized = F.pad(resized, (0, original_length - int(length)))
    return resized[..., :original_length].to(values.dtype)


def _length_preserving_pitch_shift(
    values: torch.Tensor,
    *,
    sample_rate: int,
    semitones: int,
) -> torch.Tensor:
    """Pitch-shift a waveform batch without changing sample geometry."""

    import torchaudio.functional as AF

    original = values.float()
    if not bool(torch.isfinite(original).all()):
        raise FloatingPointError("pitch-shift input contains non-finite samples")
    original_rms = original.square().mean(dim=-1, keepdim=True).sqrt()
    audible = original_rms[:, 0] > 1e-7
    shifted = torch.zeros_like(original)
    # torchaudio's phase-vocoder pitch shifter can emit non-finite samples for
    # an exactly silent row.  Source decompositions commonly contain such rows
    # (for example an absent other-DNA component), and zero is already the
    # exact pitch shift of silence.  Exclude those rows from the transform.
    if bool(audible.any()):
        # This preprocessing runs inside the trainer's model autocast block.
        # The phase-vocoder/resampler is not numerically safe in bfloat16 for
        # sparse notes, so make its float32 contract explicit.
        with torch.autocast(device_type=original.device.type, enabled=False):
            shifted_audible = AF.pitch_shift(
                original[audible].float(),
                sample_rate=int(sample_rate),
                n_steps=float(semitones),
                n_fft=512,
                win_length=512,
                hop_length=128,
            ).float()
        copy_samples = min(int(original.shape[-1]), int(shifted_audible.shape[-1]))
        shifted[audible, :copy_samples] = shifted_audible[:, :copy_samples]
    samples = int(values.shape[-1])
    if shifted.shape[-1] < samples:
        shifted = F.pad(shifted, (0, samples - int(shifted.shape[-1])))
    shifted = shifted[..., :samples]
    if not bool(torch.isfinite(shifted).all()):
        raise FloatingPointError(
            f"pitch-shift output contains non-finite samples at {semitones:+d} semitones"
        )

    # Phase-vocoder/resampler gain can vary sharply for short, sparse notes.
    # A pitch augmentation should move frequency, not invent a level cue, so
    # restore each source row's pre-shift RMS.  The gain ceiling prevents a
    # nearly empty shifted row from being amplified into an impulse outlier.
    shifted_rms = shifted.square().mean(dim=-1, keepdim=True).sqrt()
    scale = torch.where(
        original_rms > 1e-7,
        original_rms / shifted_rms.clamp_min(1e-7),
        torch.zeros_like(original_rms),
    ).clamp(max=4.0)
    shifted = shifted * scale
    return shifted.to(values.dtype)


def _transpose_audio_sources(
    audio: dict[str, torch.Tensor],
    *,
    sample_rate: int,
    semitones: int,
) -> dict[str, torch.Tensor]:
    """Transpose target and background coherently and retain both additive views."""

    background = audio["mixture"] - audio["target"]
    names = ["target", "background"]
    values = [audio["target"], background]
    for key in ("dna", "other_dna"):
        if key in audio:
            names.append(key)
            values.append(audio[key])
    stacked = torch.cat(values, dim=0)
    shifted = _length_preserving_pitch_shift(
        stacked,
        sample_rate=sample_rate,
        semitones=semitones,
    )
    batch_size = int(audio["mixture"].shape[0])
    transformed = {
        name: value
        for name, value in zip(names, shifted.split(batch_size, dim=0), strict=True)
    }
    result = dict(audio)
    result["target"] = transformed["target"]
    result["mixture"] = transformed["target"] + transformed["background"]
    for key in ("dna", "other_dna"):
        if key in transformed:
            result[key] = transformed[key]
    if "unassigned_residual" in result:
        assigned = torch.zeros_like(result["mixture"])
        for key in ("dna", "other_dna"):
            if key in result:
                assigned = assigned + result[key]
        result["unassigned_residual"] = result["mixture"] - assigned
    return result


def apply_occurrence_augmentation(
    batch: dict[str, Any],
    config: OccurrenceAugmentationConfig,
    *,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return an augmented shallow batch copy and scalar diagnostics."""

    config.validate()
    mixture = batch["mixture"]
    target = batch["target"]
    if mixture.ndim != 2 or target.shape != mixture.shape:
        raise ValueError("mixture and target must have equal [batch, samples] shapes")
    batch_size, samples = mixture.shape
    valid = batch["valid_samples"].long()
    if valid.shape != (batch_size,):
        raise ValueError("valid_samples must have shape [batch]")

    zero = mixture.new_zeros((), dtype=torch.float32)
    metrics = {
        "augmentation_gain_fraction": zero,
        "augmentation_gain_db_abs_mean": zero,
        "augmentation_pitch_shift_fraction": zero,
        "augmentation_pitch_shift_semitones_abs": zero,
        "augmentation_detune_fraction": zero,
        "augmentation_detune_cents_abs": zero,
        "augmentation_linear_effect_fraction": zero,
        "augmentation_linear_effect_wet_mean": zero,
        "augmentation_noise_fraction": zero,
        "augmentation_noise_snr_db_mean": zero,
    }
    if not enabled:
        return batch, metrics

    result = dict(batch)
    audio = {
        key: batch[key].clone()
        for key in _AUDIO_KEYS
        if key in batch and torch.is_tensor(batch[key])
    }
    if "mixture" not in audio or "target" not in audio:
        raise ValueError("augmentation requires mixture and target waveforms")

    # Integer transposition differs from detuning: both the audio and target
    # MIDI query move together, while onset/offset and valid-sample geometry do
    # not change.  The shifted mixture is explicitly rebuilt as shifted target
    # plus shifted background; the DNA compatibility decomposition is then
    # closed exactly through unassigned_residual.
    use_pitch_shift = bool(
        torch.rand((), device=mixture.device, generator=generator)
        < float(config.pitch_shift_probability)
    )
    semitones = 0
    if use_pitch_shift and config.pitch_shift_semitones_limit > 0:
        pitch = batch["pitch"].long()
        limit = int(config.pitch_shift_semitones_limit)
        candidates = torch.cat(
            (
                torch.arange(-limit, 0, device=pitch.device),
                torch.arange(1, limit + 1, device=pitch.device),
            )
        )
        candidates = candidates[
            (pitch.min() + candidates >= 0) & (pitch.max() + candidates <= 127)
        ]
        if candidates.numel() > 0:
            index = torch.randint(
                int(candidates.numel()),
                (),
                device=pitch.device,
                generator=generator,
            )
            semitones = int(candidates[index].item())
            audio = _transpose_audio_sources(
                audio,
                sample_rate=config.sample_rate,
                semitones=semitones,
            )
            result["pitch"] = pitch + semitones
            if "full_score_pitches" in batch:
                shifted_score = batch["full_score_pitches"].long() + semitones
                if "full_score_valid" in batch:
                    shifted_score = torch.where(
                        batch["full_score_valid"].bool(),
                        shifted_score.clamp(0, 127),
                        batch["full_score_pitches"].long(),
                    )
                result["full_score_pitches"] = shifted_score
            metrics["augmentation_pitch_shift_fraction"] = mixture.new_ones(
                (), dtype=torch.float32
            )
            metrics["augmentation_pitch_shift_semitones_abs"] = mixture.new_tensor(
                abs(semitones), dtype=torch.float32
            )

    # Small, batch-shared speed perturbation is cheap enough to run online and
    # retains exact mixture/source additivity.  Keeping the MIDI query nominal
    # makes this explicitly a tuning-tolerance augmentation rather than a
    # transposition augmentation.
    use_detune = bool(
        torch.rand((), device=mixture.device, generator=generator)
        < float(config.detune_probability)
    )
    detune_cents = 0.0
    if use_detune and config.detune_cents_limit > 0.0:
        detune_cents = float(
            _sample_uniform(
                (),
                -config.detune_cents_limit,
                config.detune_cents_limit,
                mixture,
                generator,
            )
        )
        pitch_ratio = 2.0 ** (detune_cents / 1200.0)
        time_scale = 1.0 / pitch_ratio
        resized_samples = max(2, int(round(samples * time_scale)))
        audio = {
            key: _resize_waveforms(values, resized_samples)
            for key, values in audio.items()
        }
        for key in (
            "valid_samples",
            "note_onset_samples",
            "note_offset_samples",
        ):
            if key in batch:
                geometry = torch.round(batch[key].float() * time_scale).long()
                if key == "valid_samples":
                    geometry = geometry.clamp(min=1, max=samples)
                result[key] = geometry
        result["note_onset_samples"] = torch.minimum(
            result["note_onset_samples"], result["valid_samples"] - 1
        ).clamp_min(0)
        result["note_offset_samples"] = torch.minimum(
            torch.maximum(
                result["note_offset_samples"], result["note_onset_samples"] + 1
            ),
            result["valid_samples"],
        )
        if (
            "full_score_onsets" in batch
            and "full_score_offsets" in batch
            and "full_score_valid" in batch
        ):
            score_valid = batch["full_score_valid"].bool()
            score_onsets = torch.round(
                batch["full_score_onsets"].float() * time_scale
            ).long()
            score_offsets = torch.round(
                batch["full_score_offsets"].float() * time_scale
            ).long()
            maximum = result["valid_samples"].long()[:, None]
            # Negative onset is meaningful: it marks a note already held at
            # crop entry. Clamping it to zero fabricates a new attack.
            score_onsets = torch.minimum(score_onsets, (maximum - 1).clamp_min(0))
            score_offsets = torch.minimum(
                torch.maximum(score_offsets, score_onsets + 1), maximum
            )
            result["full_score_onsets"] = torch.where(
                score_valid, score_onsets, batch["full_score_onsets"].long()
            )
            result["full_score_offsets"] = torch.where(
                score_valid, score_offsets, batch["full_score_offsets"].long()
            )
        valid = result["valid_samples"].long()
        metrics["augmentation_detune_fraction"] = mixture.new_ones((), dtype=torch.float32)
        metrics["augmentation_detune_cents_abs"] = mixture.new_tensor(
            abs(detune_cents), dtype=torch.float32
        )

    gain_mask = (
        torch.rand(batch_size, device=mixture.device, generator=generator)
        < float(config.gain_probability)
    )
    gain_db = _sample_uniform(
        (batch_size,),
        -config.gain_db_limit,
        config.gain_db_limit,
        mixture,
        generator,
    ) * gain_mask
    gain = torch.pow(10.0, gain_db / 20.0).to(mixture.dtype)
    audio = {key: values * gain[:, None] for key, values in audio.items()}
    metrics["augmentation_gain_fraction"] = gain_mask.float().mean()
    metrics["augmentation_gain_db_abs_mean"] = gain_db.abs().mean()

    echo_mask = (
        torch.rand(batch_size, device=mixture.device, generator=generator)
        < float(config.linear_echo_probability)
    )
    echo_wet = _sample_uniform(
        (batch_size,),
        0.0,
        config.linear_echo_wet_limit,
        mixture,
        generator,
    ) * echo_mask
    if bool(echo_mask.any()) and config.linear_echo_wet_limit > 0.0:
        delay_ms = float(
            _sample_uniform(
                (),
                config.linear_echo_delay_ms_min,
                config.linear_echo_delay_ms_max,
                mixture,
                generator,
            )
        )
        delay = min(samples - 1, max(1, round(delay_ms * config.sample_rate / 1000.0)))
        for key, values in tuple(audio.items()):
            effected = values.clone()
            effected[:, delay:] = effected[:, delay:] + echo_wet[:, None] * values[:, :-delay]
            audio[key] = effected
    metrics["augmentation_linear_effect_fraction"] = echo_mask.float().mean()
    metrics["augmentation_linear_effect_wet_mean"] = echo_wet.mean()

    noise_mask = (
        torch.rand(batch_size, device=mixture.device, generator=generator)
        < float(config.noise_probability)
    )
    snr_db = _sample_uniform(
        (batch_size,),
        config.noise_snr_db_min,
        config.noise_snr_db_max,
        mixture,
        generator,
    )
    if bool(noise_mask.any()):
        white = torch.randn(
            audio["mixture"].shape,
            device=mixture.device,
            dtype=torch.float32,
            generator=generator,
        )
        colored = 0.65 * white + 0.35 * F.avg_pool1d(
            white[:, None], kernel_size=9, stride=1, padding=4
        )[:, 0]
        clean_rms = _masked_rms(audio["mixture"], valid).clamp_min(1e-7)
        noise_rms = _masked_rms(colored, valid).clamp_min(1e-7)
        noise_scale = clean_rms / (noise_rms * torch.pow(10.0, snr_db / 20.0))
        additive = colored.to(mixture.dtype) * noise_scale[:, None].to(mixture.dtype)
        additive = additive * noise_mask[:, None]
        positions = torch.arange(samples, device=mixture.device)[None]
        additive = additive * (positions < valid[:, None])
        audio["mixture"] = audio["mixture"] + additive
        if "unassigned_residual" in audio:
            audio["unassigned_residual"] = audio["unassigned_residual"] + additive
    metrics["augmentation_noise_fraction"] = noise_mask.float().mean()
    metrics["augmentation_noise_snr_db_mean"] = (
        (snr_db * noise_mask).sum() / noise_mask.sum().clamp_min(1)
    )

    result.update(audio)
    return result, metrics


__all__ = ["OccurrenceAugmentationConfig", "apply_occurrence_augmentation"]
