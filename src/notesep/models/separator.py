"""Joint two-source occurrence-conditioned TFC-TDF separator."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .conditioning import (
    DNAResidualTFCFramewiseConditioned,
    _SpatialFiLM,
)
from .unet import (
    DNAResidualTFCTDFUNetConfig,
    dna_residual_tfc_tdf_unet_config,
)


_CONTRACT_KIND = "occurrence_joint_tfc_tdf_separator_v1"
_OWNERSHIP_CONTRACT_KIND = "occurrence_ownership_tfc_tdf_separator_v1"
_COMPLEX_MASK_CONTRACT_KIND = "occurrence_complex_mask_tfc_tdf_separator_v1"
_UNBOUNDED_COMPLEX_MASK_CONTRACT_KIND = (
    "occurrence_unbounded_complex_mask_tfc_tdf_separator_v1"
)
_MAGNITUDE_MASK_CONTRACT_KIND = "occurrence_magnitude_mask_tfc_tdf_separator_v1"
_POLAR_COMPLEX_MASK_CONTRACT_KIND = (
    "occurrence_bounded_polar_complex_mask_tfc_tdf_separator_v1"
)
_SCORE_CHANNELS = ("target_harmonic", "onset", "sustain", "release", "silence")
_RT2_ROLL_SCORE_CHANNELS = (
    "target_harmonic_held",
    "onset",
    "post_offset_harmonic_uncertainty",
    "unused_3",
    "unused_4",
)
_RT2_ONSET_ONLY_SCORE_CHANNELS = (
    "target_harmonic_onset",
    "onset",
    "unused_2",
    "unused_3",
    "unused_4",
)
_FULL_ROLL_CONDITIONING_MODE = "rt2_full_encoder_target_decoder"
_NATIVE_FULL_ROLL_CONDITIONING_MODE = "rt2_native_full_encoder_target_decoder"
_ONSET_ONLY_CONDITIONING_MODE = "rt2_onset_only"
_RELATIVE_OCTAVE_INTERVALS = (-24, -12, 12, 24)
_TARGET_RELATIVE_FREQUENCY_CHANNELS = (
    "relative_log2_frequency",
    "fundamental_proximity_100c",
    "harmonic_proximity_75c",
)
_TARGET_RELATIVE_HARMONIC_POSITION_CHANNELS = (
    "harmonic_phase_sine",
    "harmonic_phase_cosine",
    "normalized_log_harmonic_order",
)
_CONDITIONING_MODES = {
    "harmonic_lifecycle",
    "rt2_target_roll",
    _ONSET_ONLY_CONDITIONING_MODE,
    _FULL_ROLL_CONDITIONING_MODE,
    _NATIVE_FULL_ROLL_CONDITIONING_MODE,
}
_ONSET_SECONDS = 0.100
_RELEASE_SECONDS = 0.250
_HARMONIC_COUNT = 60
_HARMONIC_BANDWIDTH_HZ = 7.5


def _complex_head(input_channels: int, hidden_channels: int, subbands: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
        nn.GELU(),
        nn.Conv2d(hidden_channels, 2 * subbands, 1, bias=False),
    )
    nn.init.zeros_(head[-1].weight)
    return head


def _ownership_head(
    input_channels: int, hidden_channels: int, subbands: int
) -> nn.Sequential:
    head = nn.Sequential(
        nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
        nn.GELU(),
        nn.Conv2d(hidden_channels, subbands, 1, bias=True),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, -2.0)
    return head


def magnitude_source_ownership_loss(
    predicted_logits: torch.Tensor,
    target_spectrum: torch.Tensor,
    background_spectrum: torch.Tensor,
    frame_support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise source allocation without constraining complex cancellation."""

    if not (
        predicted_logits.shape == target_spectrum.shape == background_spectrum.shape
    ):
        raise ValueError("Ownership tensors must have matching [batch, freq, frame] shapes")
    if frame_support.shape != predicted_logits.shape[::2]:
        expected = (predicted_logits.shape[0], predicted_logits.shape[-1])
        if frame_support.shape != expected:
            raise ValueError(f"frame_support must have shape {expected}")
    target_magnitude = target_spectrum.abs()
    background_magnitude = background_spectrum.abs()
    total_magnitude = target_magnitude + background_magnitude
    oracle = (target_magnitude / total_magnitude.clamp_min(1e-7)).detach()
    weights = total_magnitude.clamp_min(0.0).sqrt()
    weights = weights * frame_support[:, None].to(weights.dtype)
    denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-7)
    logits = predicted_logits.float()
    predicted_mask = torch.sigmoid(logits)
    error = F.binary_cross_entropy_with_logits(
        logits,
        oracle.float(),
        reduction="none",
    )
    per_example = (error * weights).sum(dim=(-2, -1)) / denominator
    weighted_oracle = (oracle * weights).sum(dim=(-2, -1)) / denominator
    weighted_prediction = (
        predicted_mask.float() * weights
    ).sum(dim=(-2, -1)) / denominator
    return per_example.mean(), {
        "ownership_oracle_mean": weighted_oracle.mean(),
        "ownership_prediction_mean": weighted_prediction.mean(),
        "ownership_absolute_error": (
            (predicted_mask.float() - oracle).abs() * weights
        ).sum(dim=(-2, -1)).div(denominator).mean(),
    }


class OccurrenceJointTFCTDFSeparator(DNAResidualTFCFramewiseConditioned):
    """Predict target and background jointly from a mixture and note occurrence.

    The inherited SDX23-derived TFC-TDF encoder/decoder is shared by both raw
    complex heads. The pair is then orthogonally projected onto the affine
    constraint ``target + background = mixture`` independently at every
    time-frequency bin.
    """

    def __init__(
        self,
        config: DNAResidualTFCTDFUNetConfig,
        *,
        activation_checkpointing: bool = True,
        ownership_parameterization: bool = False,
        complex_mask_parameterization: bool = False,
        unbounded_complex_mask_parameterization: bool = False,
        magnitude_mask_parameterization: bool = False,
        bounded_polar_complex_mask_parameterization: bool = False,
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        roll_tail_decay_min_ms: float = 0.0,
        roll_tail_decay_max_ms: float = 0.0,
        rt2_pianoroll_embedding: torch.Tensor | None = None,
        relative_octave_decoder_conditioning: bool = False,
        target_relative_frequency_decoder_conditioning: bool = False,
        target_relative_harmonic_position_conditioning: bool = False,
        multi_target_full_score: bool = False,
        multi_occurrence_target_roll: bool = False,
        target_score_input_routing: bool = False,
        full_score_contrast_input: bool = False,
        learned_full_score_rejection_input: bool = False,
        preserve_signed_score_times: bool = False,
    ) -> None:
        if sum(
            bool(value)
            for value in (
                ownership_parameterization,
                complex_mask_parameterization,
                magnitude_mask_parameterization,
                bounded_polar_complex_mask_parameterization,
            )
        ) > 1:
            raise ValueError("Output parameterizations are mutually exclusive")
        if unbounded_complex_mask_parameterization and not complex_mask_parameterization:
            raise ValueError("unbounded complex mask requires complex-mask parameterization")
        super().__init__(config, arm="g2_film")
        if conditioning_mode not in _CONDITIONING_MODES:
            raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")
        if not 0 <= roll_hold_min_ms <= roll_hold_max_ms:
            raise ValueError("RT2 roll hold bounds are invalid")
        if (
            conditioning_mode not in {
                _FULL_ROLL_CONDITIONING_MODE,
                _NATIVE_FULL_ROLL_CONDITIONING_MODE,
            }
            and roll_hold_max_ms != 0
        ):
            raise ValueError("Roll hold extension is only supported by full-roll mode")
        if not 0 <= roll_tail_decay_min_ms <= roll_tail_decay_max_ms:
            raise ValueError("RT2 roll tail-decay bounds are invalid")
        if (
            conditioning_mode != _FULL_ROLL_CONDITIONING_MODE
            and roll_tail_decay_max_ms != 0
        ):
            raise ValueError("Roll tail uncertainty is only supported by full-roll mode")
        if roll_hold_max_ms and roll_tail_decay_max_ms:
            raise ValueError("Hard roll holds and soft tail uncertainty are mutually exclusive")
        self.conditioning_mode = str(conditioning_mode)
        self.roll_hold_min_ms = float(roll_hold_min_ms)
        self.roll_hold_max_ms = float(roll_hold_max_ms)
        self.roll_tail_decay_min_ms = float(roll_tail_decay_min_ms)
        self.roll_tail_decay_max_ms = float(roll_tail_decay_max_ms)
        self.use_activation_checkpointing = bool(activation_checkpointing)
        self.ownership_parameterization = bool(ownership_parameterization)
        self.complex_mask_parameterization = bool(complex_mask_parameterization)
        self.unbounded_complex_mask_parameterization = bool(
            unbounded_complex_mask_parameterization
        )
        self.magnitude_mask_parameterization = bool(magnitude_mask_parameterization)
        self.bounded_polar_complex_mask_parameterization = bool(
            bounded_polar_complex_mask_parameterization
        )
        self.relative_octave_decoder_conditioning = bool(
            relative_octave_decoder_conditioning
        )
        self.target_relative_frequency_decoder_conditioning = bool(
            target_relative_frequency_decoder_conditioning
        )
        self.target_relative_harmonic_position_conditioning = bool(
            target_relative_harmonic_position_conditioning
        )
        self.multi_target_full_score = bool(multi_target_full_score)
        self.multi_occurrence_target_roll = bool(multi_occurrence_target_roll)
        self.auxiliary_carrier_conditioning = False
        if self.multi_target_full_score and self.multi_occurrence_target_roll:
            raise ValueError(
                "Multi-target and repeated-single-pitch score modes are exclusive"
            )
        if (
            self.multi_occurrence_target_roll
            and self.conditioning_mode != "rt2_target_roll"
        ):
            raise ValueError(
                "Repeated pitch tracks require RT2 target-roll conditioning"
            )
        self.target_score_input_routing = bool(target_score_input_routing)
        self.full_score_contrast_input = bool(full_score_contrast_input)
        self.learned_full_score_rejection_input = bool(
            learned_full_score_rejection_input
        )
        self.preserve_signed_score_times = bool(preserve_signed_score_times)
        if (
            self.full_score_contrast_input
            and self.learned_full_score_rejection_input
        ):
            raise ValueError("Fixed and learned full-score rejection are exclusive")
        if (
            self.full_score_contrast_input
            or self.learned_full_score_rejection_input
        ) and not self.multi_target_full_score:
            raise ValueError(
                "Full-score rejection input requires a multi-target score query"
            )
        self.full_score_rejection_scale = (
            nn.Parameter(torch.zeros(len(_RT2_ROLL_SCORE_CHANNELS)))
            if self.learned_full_score_rejection_input
            else None
        )
        if self.multi_target_full_score and self.conditioning_mode not in {
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            raise ValueError("Multi-target events require full-score conditioning")
        if self.multi_target_full_score and (
            self.relative_octave_decoder_conditioning
            or self.target_relative_frequency_decoder_conditioning
        ):
            raise ValueError(
                "Single-pitch-relative decoder maps are undefined for multi-target events"
            )
        if (
            (
                self.relative_octave_decoder_conditioning
                or self.target_relative_frequency_decoder_conditioning
            )
            and self.conditioning_mode != _NATIVE_FULL_ROLL_CONDITIONING_MODE
        ):
            raise ValueError(
                "Target-relative decoder conditioning requires native full-roll mode"
            )
        if (
            self.target_relative_harmonic_position_conditioning
            and self.conditioning_mode != "harmonic_lifecycle"
        ):
            raise ValueError(
                "Target-relative harmonic position conditioning currently requires "
                "harmonic-lifecycle mode"
            )
        self._full_score_events: tuple[torch.Tensor, ...] | None = None
        self._full_score_extension_params: tuple[torch.Tensor, ...] | None = None

        if self.conditioning_mode in {
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            self.skip_film = nn.ModuleList(
                _SpatialFiLM(
                    len(_RT2_ROLL_SCORE_CHANNELS),
                    config.initial_channels + index * config.growth,
                )
                for index in range(config.num_scales)
            )
            self.output_skip_film = _SpatialFiLM(
                len(_RT2_ROLL_SCORE_CHANNELS), config.initial_channels
            )
        if self.conditioning_mode == _NATIVE_FULL_ROLL_CONDITIONING_MODE:
            if rt2_pianoroll_embedding is None:
                # Checkpoint reconstruction installs the frozen value through
                # load_state_dict immediately after construction.
                rt2_pianoroll_embedding = torch.zeros(1536, 256)
            if tuple(rt2_pianoroll_embedding.shape) != (1536, 256):
                raise ValueError("RT2 piano-roll embedding must have shape [1536, 256]")
            self.rt2_pianoroll_embedding = nn.Embedding.from_pretrained(
                rt2_pianoroll_embedding.float(), freeze=True
            )
            self.rt2_score_projection = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, len(_RT2_ROLL_SCORE_CHANNELS)),
            )
            self.rt2_context_projection = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, config.conditioning_channels),
            )
        if (
            self.relative_octave_decoder_conditioning
            or self.target_relative_frequency_decoder_conditioning
        ):
            decoder_channels = len(_RT2_ROLL_SCORE_CHANNELS)
            if self.relative_octave_decoder_conditioning:
                decoder_channels += len(_RELATIVE_OCTAVE_INTERVALS)
            if self.target_relative_frequency_decoder_conditioning:
                decoder_channels += len(
                    _TARGET_RELATIVE_FREQUENCY_CHANNELS
                )
            self.decoder_film = nn.ModuleList(
                _SpatialFiLM(
                    decoder_channels,
                    config.initial_channels
                    + (config.num_scales - index - 1) * config.growth,
                )
                for index in range(config.num_scales)
            )
            self.skip_film = nn.ModuleList(
                _SpatialFiLM(
                    decoder_channels,
                    config.initial_channels + index * config.growth,
                )
                for index in range(config.num_scales)
            )
            self.output_skip_film = _SpatialFiLM(
                decoder_channels, config.initial_channels
            )

        if self.target_relative_harmonic_position_conditioning:
            extra_channels = len(_TARGET_RELATIVE_HARMONIC_POSITION_CHANNELS)
            old_first_conv = self.first_conv
            # Keep every shared parameter bit-identical under a matched seed.
            # The added path begins as a no-op and learns from zero, while the
            # random state seen by the target/background heads is unchanged.
            with torch.random.fork_rng(devices=[]):
                expanded_first_conv = nn.Conv2d(
                    old_first_conv.in_channels
                    + extra_channels * config.num_subbands,
                    old_first_conv.out_channels,
                    old_first_conv.kernel_size,
                    stride=old_first_conv.stride,
                    padding=old_first_conv.padding,
                    dilation=old_first_conv.dilation,
                    groups=old_first_conv.groups,
                    bias=old_first_conv.bias is not None,
                    padding_mode=old_first_conv.padding_mode,
                    device=old_first_conv.weight.device,
                    dtype=old_first_conv.weight.dtype,
                )
                with torch.no_grad():
                    expanded_first_conv.weight[:, : old_first_conv.in_channels].copy_(
                        old_first_conv.weight
                    )
                    expanded_first_conv.weight[:, old_first_conv.in_channels :].zero_()
                    if old_first_conv.bias is not None:
                        expanded_first_conv.bias.copy_(old_first_conv.bias)
                self.score_channels += extra_channels
                self.encoder_film = nn.ModuleList()
                self.decoder_film = nn.ModuleList()
                channels = config.initial_channels
                for _ in range(config.num_scales):
                    self.encoder_film.append(
                        _SpatialFiLM(self.score_channels, channels)
                    )
                    channels += config.growth
                self.bottleneck_film = _SpatialFiLM(
                    self.score_channels, channels
                )
                for _ in range(config.num_scales):
                    channels -= config.growth
                    self.decoder_film.append(
                        _SpatialFiLM(self.score_channels, channels)
                    )
            self.first_conv = expanded_first_conv

        source_channels = self._SOURCE_COUNT * 2 * config.num_subbands
        head_input_channels = config.initial_channels + source_channels
        del self.final_projection
        if not (
            self.magnitude_mask_parameterization
            or self.bounded_polar_complex_mask_parameterization
        ):
            self.target_head = _complex_head(
                head_input_channels, config.initial_channels, config.num_subbands
            )
        if (
            not self.magnitude_mask_parameterization
            and not self.bounded_polar_complex_mask_parameterization
            and not self.unbounded_complex_mask_parameterization
        ):
            self.background_head = _complex_head(
                head_input_channels, config.initial_channels, config.num_subbands
            )
        if (
            self.ownership_parameterization
            or self.magnitude_mask_parameterization
            or self.bounded_polar_complex_mask_parameterization
        ):
            self.ownership_head = _ownership_head(
                head_input_channels, config.initial_channels, config.num_subbands
            )
        if self.bounded_polar_complex_mask_parameterization:
            self.phase_head = _ownership_head(
                head_input_channels, config.initial_channels, config.num_subbands
            )
            nn.init.zeros_(self.phase_head[-1].bias)

    def enable_auxiliary_carrier_conditioning(self) -> None:
        """Expose a pitch-track carrier through a formerly silent source slot.

        The mature occurrence checkpoints were always trained with the three
        compatibility source slots set to zero.  Zeroing the first unused
        slot's encoder and readout weights therefore preserves their exact
        function while opening a short, full-resolution path that fine-tuning
        can use for the hierarchy's final magnitude carrier.
        """

        if self.auxiliary_carrier_conditioning:
            return
        subbands = int(self.config.num_subbands)
        start = 2 * subbands
        stop = 4 * subbands
        with torch.no_grad():
            self.first_conv.weight[:, start:stop].zero_()
            for name in (
                "target_head",
                "background_head",
                "ownership_head",
                "phase_head",
            ):
                head = getattr(self, name, None)
                if head is not None:
                    head[0].weight[:, start:stop].zero_()
        self.auxiliary_carrier_conditioning = True
    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        activation_checkpointing: bool = True,
        ownership_parameterization: bool = False,
        complex_mask_parameterization: bool = False,
        unbounded_complex_mask_parameterization: bool = False,
        magnitude_mask_parameterization: bool = False,
        bounded_polar_complex_mask_parameterization: bool = False,
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        roll_tail_decay_min_ms: float = 0.0,
        roll_tail_decay_max_ms: float = 0.0,
        rt2_pianoroll_embedding: torch.Tensor | None = None,
        relative_octave_decoder_conditioning: bool = False,
        target_relative_frequency_decoder_conditioning: bool = False,
        target_relative_harmonic_position_conditioning: bool = False,
        multi_target_full_score: bool = False,
        multi_occurrence_target_roll: bool = False,
        target_score_input_routing: bool = False,
        full_score_contrast_input: bool = False,
        learned_full_score_rejection_input: bool = False,
        preserve_signed_score_times: bool = False,
    ) -> "OccurrenceJointTFCTDFSeparator":
        return cls(
            dna_residual_tfc_tdf_unet_config(preset),
            activation_checkpointing=activation_checkpointing,
            ownership_parameterization=ownership_parameterization,
            complex_mask_parameterization=complex_mask_parameterization,
            unbounded_complex_mask_parameterization=(
                unbounded_complex_mask_parameterization
            ),
            magnitude_mask_parameterization=magnitude_mask_parameterization,
            bounded_polar_complex_mask_parameterization=(
                bounded_polar_complex_mask_parameterization
            ),
            conditioning_mode=conditioning_mode,
            roll_hold_min_ms=roll_hold_min_ms,
            roll_hold_max_ms=roll_hold_max_ms,
            roll_tail_decay_min_ms=roll_tail_decay_min_ms,
            roll_tail_decay_max_ms=roll_tail_decay_max_ms,
            rt2_pianoroll_embedding=rt2_pianoroll_embedding,
            relative_octave_decoder_conditioning=(
                relative_octave_decoder_conditioning
            ),
            target_relative_frequency_decoder_conditioning=(
                target_relative_frequency_decoder_conditioning
            ),
            target_relative_harmonic_position_conditioning=(
                target_relative_harmonic_position_conditioning
            ),
            multi_target_full_score=multi_target_full_score,
            multi_occurrence_target_roll=multi_occurrence_target_roll,
            target_score_input_routing=target_score_input_routing,
            full_score_contrast_input=full_score_contrast_input,
            learned_full_score_rejection_input=(
                learned_full_score_rejection_input
            ),
            preserve_signed_score_times=preserve_signed_score_times,
        )

    @staticmethod
    def orthogonal_two_source_projection(
        raw_target: torch.Tensor,
        raw_background: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project a source pair onto its exact mixture-consistency plane."""

        if raw_target.shape != raw_background.shape or raw_target.shape != mixture.shape:
            raise ValueError("Raw target, raw background, and mixture must have equal shape")
        shared_correction = 0.5 * (mixture - raw_target - raw_background)
        return (
            raw_target + shared_correction,
            raw_background + shared_correction,
            shared_correction,
        )

    def _harmonic_support(
        self,
        pitches: torch.Tensor,
        active: torch.Tensor,
        frequency_bins: int,
    ) -> torch.Tensor:
        frequency = (
            torch.arange(
                frequency_bins,
                device=pitches.device,
                dtype=torch.float32,
            )
            * float(self.config.sample_rate)
            / float(self.config.n_fft)
        )
        f0 = 440.0 * torch.pow(2.0, (pitches.float() - 69.0) / 12.0)
        harmonics = torch.arange(
            1,
            _HARMONIC_COUNT + 1,
            device=pitches.device,
            dtype=torch.float32,
        )
        centers = f0[..., None] * harmonics
        distance = (frequency[None, None, None, :] - centers[..., None]).abs()
        support = torch.exp(
            -0.5 * (distance / _HARMONIC_BANDWIDTH_HZ).square()
        ).amax(dim=-2)
        return support[:, :, None] * active[..., None].to(support.dtype)

    def _target_relative_harmonic_position_map(
        self,
        *,
        pitch: torch.Tensor,
        active: torch.Tensor,
        frequency_bins: int,
    ) -> torch.Tensor:
        """Encode frequency position in units of the queried fundamental.

        This is conditioning only: it does not multiply, mask, or gate audio.
        The sine/cosine pair identifies signed phase between integer harmonics,
        while the third channel preserves which harmonic order is nearby.
        """

        frequency = (
            torch.arange(
                frequency_bins,
                device=pitch.device,
                dtype=torch.float32,
            )
            * float(self.config.sample_rate)
            / float(self.config.n_fft)
        )
        fundamental = 440.0 * torch.pow(
            2.0, (pitch.float() - 69.0) / 12.0
        )
        ratio = frequency[None] / fundamental[:, None]
        nearest = ratio.round().clamp(1.0, float(_HARMONIC_COUNT))
        valid = (
            (frequency[None] > 0.0)
            & (ratio >= 0.5)
            & (ratio <= float(_HARMONIC_COUNT) + 0.5)
        )
        phase = 2.0 * math.pi * (ratio - nearest)
        harmonic_order = torch.log1p(nearest) / math.log1p(_HARMONIC_COUNT)
        spectral = torch.stack(
            (torch.sin(phase), torch.cos(phase), harmonic_order), dim=1
        )
        spectral = spectral * valid[:, None].to(spectral.dtype)
        return spectral[..., None] * active[:, None, None].to(spectral.dtype)

    def _score_map(
        self,
        *,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        frequency_bins: int,
        frames: int,
        tail_decay_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame_samples = (
            torch.arange(frames, device=pitch.device)[None] * self.config.hop_length
        )
        frame_valid = frame_samples < valid_samples[:, None]
        present = offset > onset
        present_frames = present[:, None] & frame_valid
        if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE:
            # Strict pitch-plus-onset conditioning. The pitch embedding is
            # available throughout the valid crop, while the spatial map
            # contains only the queried pitch at its 25 Hz onset state.
            # Offset is deliberately not read in this branch.
            roll_frame = torch.div(
                frame_samples * 25,
                self.config.sample_rate,
                rounding_mode="floor",
            )
            onset_frame = torch.div(
                onset.long() * 25,
                self.config.sample_rate,
                rounding_mode="floor",
            )
            onset_present = (
                (onset >= 0) & (onset < valid_samples)
            )[:, None] & frame_valid
            onset_state = onset_present & (roll_frame == onset_frame[:, None])
            target = self._harmonic_support(
                pitch[:, None], onset_state[:, None], frequency_bins
            )[:, 0].transpose(1, 2)
            onset_channel = onset_state[:, None, None].expand(
                -1, 1, frequency_bins, -1
            ).to(target.dtype)
            zeros = torch.zeros_like(onset_channel)
            score = torch.cat(
                (target[:, None], onset_channel, zeros, zeros, zeros), dim=1
            )
            return score * frame_valid[:, None, None].to(score.dtype)
        if self.conditioning_mode in {
            "rt2_target_roll",
            _ONSET_ONLY_CONDITIONING_MODE,
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            # Match RT2's 25 Hz target-only piano-roll timing.  The queried
            # pitch has only OFF, HELD, and ONSET states; there is no explicit
            # sustain/release/silence label.  Retain five physical channels so
            # this remains an architecture-matched conditioning ablation.
            roll_frame = torch.div(
                frame_samples * 25,
                self.config.sample_rate,
                rounding_mode="floor",
            )
            onset_frame = torch.div(
                onset.long() * 25,
                self.config.sample_rate,
                rounding_mode="floor",
            )
            offset_frame = torch.div(
                offset.long() * 25 + self.config.sample_rate - 1,
                self.config.sample_rate,
                rounding_mode="floor",
            )
            held = (
                present_frames
                & (roll_frame >= onset_frame[:, None])
                & (roll_frame < offset_frame[:, None])
            )
            onset_state = held & (roll_frame == onset_frame[:, None])
            target = self._harmonic_support(
                pitch[:, None], held[:, None], frequency_bins
            )[:, 0].transpose(1, 2)
            onset_channel = onset_state[:, None, None].expand(
                -1, 1, frequency_bins, -1
            ).to(target.dtype)
            zeros = torch.zeros_like(onset_channel)
            tail_channel = zeros
            if tail_decay_samples is not None:
                decay = tail_decay_samples.long().clamp_min(1)
                after_offset = present_frames & (frame_samples >= offset[:, None])
                distance = (frame_samples - offset[:, None]).clamp_min(0).float()
                tail_weight = torch.exp(-distance / decay[:, None].float())
                tail_weight = tail_weight * after_offset.to(tail_weight.dtype)
                tail_harmonic = self._harmonic_support(
                    pitch[:, None], tail_weight[:, None], frequency_bins
                )[:, 0].transpose(1, 2)
                tail_channel = tail_harmonic[:, None]
            score = torch.cat(
                (target[:, None], onset_channel, tail_channel, zeros, zeros), dim=1
            )
            return score * frame_valid[:, None, None].to(score.dtype)
        onset_samples = round(_ONSET_SECONDS * self.config.sample_rate)
        release_samples = round(_RELEASE_SECONDS * self.config.sample_rate)
        onset_end = torch.minimum(offset, onset + onset_samples)
        release_end = torch.minimum(valid_samples, offset + release_samples)
        onset_region = (
            present_frames
            & (frame_samples >= onset[:, None])
            & (frame_samples < onset_end[:, None])
        )
        sustain = (
            present_frames
            & (frame_samples >= onset_end[:, None])
            & (frame_samples < offset[:, None])
        )
        release = (
            present_frames
            & (frame_samples >= offset[:, None])
            & (frame_samples < release_end[:, None])
        )
        silence = frame_valid & ~(onset_region | sustain | release)
        target_active = present_frames & (frame_samples >= onset[:, None]) & (
            frame_samples < release_end[:, None]
        )
        target = self._harmonic_support(
            pitch[:, None], target_active[:, None], frequency_bins
        )[:, 0].transpose(1, 2)
        lifecycle = torch.stack(
            (onset_region, sustain, release, silence), dim=1
        ).to(target.dtype)
        lifecycle = lifecycle[:, :, None].expand(-1, -1, frequency_bins, -1)
        score = torch.cat((target[:, None], lifecycle), dim=1)
        if self.target_relative_harmonic_position_conditioning:
            score = torch.cat(
                (
                    score,
                    self._target_relative_harmonic_position_map(
                        pitch=pitch,
                        active=target_active,
                        frequency_bins=frequency_bins,
                    ),
                ),
                dim=1,
            )
        return score * frame_valid[:, None, None].to(score.dtype)

    def set_full_score_events(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
        target: torch.Tensor,
        extension_mean: torch.Tensor | None = None,
        extension_scale: torch.Tensor | None = None,
        extension_minimum: torch.Tensor | None = None,
        extension_maximum: torch.Tensor | None = None,
    ) -> None:
        """Set the complete RT2-style score view consumed by the encoder."""

        if self.conditioning_mode not in {
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        } and not self.multi_occurrence_target_roll:
            raise RuntimeError("Full score events require full-roll conditioning")
        if not (
            pitches.shape
            == onsets.shape
            == offsets.shape
            == valid.shape
            == target.shape
        ):
            raise ValueError("Full score tensors must have equal [batch, events] shapes")
        if pitches.ndim != 2 or pitches.shape[1] < 1:
            raise ValueError("Full score tensors must contain at least one event")
        target_counts = (target.bool() & valid.bool()).sum(dim=1)
        expected = (
            target_counts >= 1
            if self.multi_target_full_score or self.multi_occurrence_target_roll
            else target_counts == 1
        )
        if not torch.all(expected):
            requirement = (
                "at least one"
                if self.multi_target_full_score or self.multi_occurrence_target_roll
                else "exactly one"
            )
            raise ValueError(
                f"Each full score must identify {requirement} valid target event"
            )
        self._full_score_events = (pitches, onsets, offsets, valid, target)
        extension_values = (
            extension_mean,
            extension_scale,
            extension_minimum,
            extension_maximum,
        )
        if any(value is not None for value in extension_values):
            if not all(value is not None for value in extension_values):
                raise ValueError("Acoustic extension parameters must be supplied together")
            assert all(value is not None for value in extension_values)
            if any(value.shape != pitches.shape for value in extension_values):
                raise ValueError("Acoustic extension parameters have the wrong shape")
            self._full_score_extension_params = tuple(extension_values)  # type: ignore[arg-type]
        else:
            self._full_score_extension_params = None

    def _condition_full_score_empirical_offsets(
        self,
        valid_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._full_score_events is None or self._full_score_extension_params is None:
            raise RuntimeError("Empirical offset conditioning is unavailable")
        pitches, onsets, offsets, event_valid, target = self._full_score_events
        mean, scale, minimum, maximum = self._full_score_extension_params
        if self.training:
            sampled = mean + torch.randn_like(mean) * scale
            sampled = torch.maximum(sampled, mean - 2.0 * scale)
            sampled = torch.minimum(sampled, mean + 2.0 * scale)
        else:
            sampled = mean
        sampled = torch.minimum(torch.maximum(sampled, minimum), maximum)
        extension = sampled.round().long().clamp_min(0)
        extension = torch.where(
            event_valid.bool() & (offsets > onsets), extension, torch.zeros_like(extension)
        )
        conditioned = torch.minimum(offsets + extension, valid_samples[:, None])
        self._full_score_events = (
            pitches,
            onsets,
            conditioned,
            event_valid,
            target,
        )
        target_extension = (
            extension.masked_fill(~target.bool(), 0).amax(dim=1)
            if self.multi_target_full_score
            else (extension * target.to(extension.dtype)).sum(dim=1)
        )
        return extension, target_extension

    def _condition_full_score_offsets(
        self,
        valid_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._full_score_events is None:
            raise RuntimeError("Full-roll encoder conditioning requires score events")
        pitches, onsets, offsets, event_valid, target = self._full_score_events
        scale = float(self.config.sample_rate) / 1000.0
        low = int(round(self.roll_hold_min_ms * scale))
        high = int(round(self.roll_hold_max_ms * scale))
        if self.training and high > low:
            extension = torch.randint(
                low, high + 1, offsets.shape, device=offsets.device
            )
        else:
            extension = torch.full_like(offsets, (low + high) // 2)
        extension = torch.where(
            event_valid.bool() & (offsets > onsets), extension, torch.zeros_like(extension)
        )
        conditioned = torch.minimum(offsets + extension, valid_samples[:, None])
        self._full_score_events = (
            pitches,
            onsets,
            conditioned,
            event_valid,
            target,
        )
        target_extension = (
            extension.masked_fill(~target.bool(), 0).amax(dim=1)
            if self.multi_target_full_score
            else (extension * target.to(extension.dtype)).sum(dim=1)
        )
        return extension, target_extension

    def _condition_full_score_tail_decay(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Sample a soft, non-silent post-offset uncertainty time constant."""

        if self._full_score_events is None:
            raise RuntimeError("Full-roll encoder conditioning requires score events")
        _, onsets, offsets, event_valid, target = self._full_score_events
        scale = float(self.config.sample_rate) / 1000.0
        low = int(round(self.roll_tail_decay_min_ms * scale))
        high = int(round(self.roll_tail_decay_max_ms * scale))
        if high == 0:
            return None, None
        if self.training and high > low:
            decay = torch.randint(low, high + 1, offsets.shape, device=offsets.device)
        else:
            decay = torch.full_like(offsets, (low + high) // 2)
        decay = torch.where(
            event_valid.bool() & (offsets > onsets), decay, torch.zeros_like(decay)
        )
        target_decay = (
            decay.masked_fill(~target.bool(), 0).amax(dim=1)
            if self.multi_target_full_score
            else (decay * target.to(decay.dtype)).sum(dim=1)
        )
        return decay, target_decay

    def _full_rt2_score_map(
        self,
        *,
        valid_samples: torch.Tensor,
        frequency_bins: int,
        frames: int,
        tail_decay_samples: torch.Tensor | None = None,
        target_only: bool = False,
    ) -> torch.Tensor:
        if self._full_score_events is None:
            raise RuntimeError("Full-roll encoder conditioning requires score events")
        pitches, onsets, offsets, event_valid, target = self._full_score_events
        if target_only:
            event_valid = event_valid.bool() & target.bool()
        frame_samples = (
            torch.arange(frames, device=pitches.device)[None] * self.config.hop_length
        )
        frame_valid = frame_samples < valid_samples[:, None]
        roll_frame = torch.div(
            frame_samples * 25,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        onset_frame = torch.div(
            onsets.long() * 25,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        offset_frame = torch.div(
            offsets.long() * 25 + self.config.sample_rate - 1,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        held = (
            event_valid.bool()[:, :, None]
            & frame_valid[:, None]
            & (roll_frame[:, None] >= onset_frame[:, :, None])
            & (roll_frame[:, None] < offset_frame[:, :, None])
        )
        onset_state = held & (roll_frame[:, None] == onset_frame[:, :, None])
        harmonic = self._harmonic_support(
            pitches.long(), held, frequency_bins
        ).amax(dim=1).transpose(1, 2)
        onset_channel = onset_state.any(dim=1)[:, None, None].expand(
            -1, 1, frequency_bins, -1
        ).to(harmonic.dtype)
        zeros = torch.zeros_like(onset_channel)
        tail_channel = zeros
        if tail_decay_samples is not None:
            decay = tail_decay_samples.long().clamp_min(1)
            after_offset = (
                event_valid.bool()[:, :, None]
                & frame_valid[:, None]
                & (frame_samples[:, None] >= offsets[:, :, None])
            )
            distance = (
                frame_samples[:, None] - offsets[:, :, None]
            ).clamp_min(0).float()
            tail_weight = torch.exp(-distance / decay[:, :, None].float())
            tail_weight = tail_weight * after_offset.to(tail_weight.dtype)
            tail_harmonic = self._harmonic_support(
                pitches.long(), tail_weight, frequency_bins
            ).amax(dim=1).transpose(1, 2)
            tail_channel = tail_harmonic[:, None]
        score = torch.cat(
            (harmonic[:, None], onset_channel, tail_channel, zeros, zeros), dim=1
        )
        return score * frame_valid[:, None, None].to(score.dtype)

    def _native_rt2_sequence(
        self,
        *,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        event_valid: torch.Tensor,
        valid_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RT2's native 25 Hz, 128-pitch categorical embedding.

        RT2 represents every pitch with OFF=0, CONTINUATION=1, and ONSET=2.
        Its regular embedder averages the 128 pitch-channel embeddings.  This
        computes that exact sparse average without materializing the enormous
        ``[batch, frames, 128, 256]`` dense embedding tensor.
        """

        if self.conditioning_mode != _NATIVE_FULL_ROLL_CONDITIONING_MODE:
            raise RuntimeError("Native RT2 sequence requested from another mode")
        batch, events = pitches.shape
        rt2_frames = torch.div(
            valid_samples.long() * 25 + self.config.sample_rate - 1,
            self.config.sample_rate,
            rounding_mode="floor",
        ).clamp_min(1)
        maximum = int(rt2_frames.max().item())
        frame_index = torch.arange(maximum, device=pitches.device)
        frame_valid = frame_index[None] < rt2_frames[:, None]
        onset_frame = torch.div(
            onsets.long() * 25,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        offset_frame = torch.div(
            offsets.long() * 25 + self.config.sample_rate - 1,
            self.config.sample_rate,
            rounding_mode="floor",
        )

        pitch_axis = torch.arange(128, device=pitches.device)
        off = self.rt2_pianoroll_embedding(pitch_axis * 11 + 7).mean(dim=0)
        sequence = off[None, None].expand(batch, maximum, -1).clone()
        # Track the categorical maximum so overlapping/rearticulated notes of
        # one pitch remain one RT2 channel rather than being double counted.
        states = torch.zeros(
            (batch, maximum, 128), dtype=torch.uint8, device=pitches.device
        )
        batch_index = torch.arange(batch, device=pitches.device)
        for event_index in range(events):
            pitch = pitches[:, event_index].long().clamp(0, 127)
            usable = event_valid[:, event_index].bool() & (
                offsets[:, event_index] > onsets[:, event_index]
            )
            held = (
                usable[:, None]
                & frame_valid
                & (frame_index[None] >= onset_frame[:, event_index, None])
                & (frame_index[None] < offset_frame[:, event_index, None])
            )
            onset = held & (
                frame_index[None] == onset_frame[:, event_index, None]
            )
            proposed = torch.where(
                onset,
                torch.full_like(frame_index[None].expand(batch, -1), 2),
                held.long(),
            ).to(torch.uint8)
            previous = states[batch_index[:, None], frame_index[None], pitch[:, None]]
            changed = proposed > previous
            continuation_change = changed & (proposed == 1)
            onset_from_off = changed & (proposed == 2) & (previous == 0)
            onset_from_continuation = changed & (proposed == 2) & (previous == 1)
            off_embedding = self.rt2_pianoroll_embedding(pitch * 11 + 7)
            continuation_embedding = self.rt2_pianoroll_embedding(pitch * 11 + 8)
            onset_embedding = self.rt2_pianoroll_embedding(pitch * 11 + 9)
            sequence = sequence + (
                continuation_change[..., None]
                * (continuation_embedding - off_embedding)[:, None]
                + onset_from_off[..., None]
                * (onset_embedding - off_embedding)[:, None]
                + onset_from_continuation[..., None]
                * (onset_embedding - continuation_embedding)[:, None]
            ) / 128.0
            states[batch_index[:, None], frame_index[None], pitch[:, None]] = torch.maximum(
                previous, proposed
            )
        return sequence * frame_valid[..., None], frame_valid

    def _native_rt2_maps(
        self,
        *,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        event_valid: torch.Tensor,
        valid_samples: torch.Tensor,
        frequency_bins: int,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, valid = self._native_rt2_sequence(
            pitches=pitches,
            onsets=onsets,
            offsets=offsets,
            event_valid=event_valid,
            valid_samples=valid_samples,
        )
        score = self.rt2_score_projection(sequence)
        context = self.rt2_context_projection(sequence)
        score = F.interpolate(
            score.transpose(1, 2), size=frames, mode="nearest"
        )
        context = F.interpolate(
            context.transpose(1, 2), size=frames, mode="nearest"
        )
        score = score[:, :, None].expand(-1, -1, frequency_bins, -1)
        context = context[:, :, None].expand(-1, -1, frequency_bins, -1)
        frame_samples = (
            torch.arange(frames, device=pitches.device)[None]
            * self.config.hop_length
        )
        support = frame_samples < valid_samples[:, None]
        return score * support[:, None, None], context * support[:, None, None]

    def _relative_octave_decoder_map(
        self,
        *,
        target_pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        frequency_bins: int,
        frames: int,
    ) -> torch.Tensor:
        """Expose active octave competitors relative to the decoder query.

        Each channel preserves RT2's OFF=0, CONTINUATION=1, ONSET=2 state for
        one signed relative interval.  The map is decoder-only: the complete
        native RT2 roll remains the sole encoder score condition.
        """

        if not self.relative_octave_decoder_conditioning:
            raise RuntimeError("Relative-octave decoder conditioning is disabled")
        if self._full_score_events is None:
            raise RuntimeError("Relative-octave conditioning requires score events")
        pitches, onsets, offsets, event_valid, _ = self._full_score_events
        frame_samples = (
            torch.arange(frames, device=pitches.device)[None]
            * self.config.hop_length
        )
        frame_valid = frame_samples < valid_samples.long()[:, None]
        roll_frame = torch.div(
            frame_samples * 25,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        onset_frame = torch.div(
            onsets.long() * 25,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        offset_frame = torch.div(
            offsets.long() * 25 + self.config.sample_rate - 1,
            self.config.sample_rate,
            rounding_mode="floor",
        )
        held = (
            event_valid.bool()[:, :, None]
            & frame_valid[:, None]
            & (roll_frame[:, None] >= onset_frame[:, :, None])
            & (roll_frame[:, None] < offset_frame[:, :, None])
        )
        onset = held & (roll_frame[:, None] == onset_frame[:, :, None])
        state = held.to(torch.float32) + onset.to(torch.float32)
        relative = pitches.long() - target_pitch.long()[:, None]
        channels = []
        for interval in _RELATIVE_OCTAVE_INTERVALS:
            selected = relative.eq(interval)[:, :, None]
            channels.append((state * selected).amax(dim=1))
        result = torch.stack(channels, dim=1) / 2.0
        return result[:, :, None].expand(-1, -1, frequency_bins, -1)

    def _target_relative_frequency_decoder_map(
        self,
        *,
        target_pitch: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        valid_samples: torch.Tensor,
        frequency_bins: int,
        frames: int,
    ) -> torch.Tensor:
        """Return continuous target-relative frequency cues without gating audio.

        The channels locate the queried fundamental and its broad harmonic
        neighborhood. They are active only while the queried occurrence is
        held, and they are consumed by zero-initialized decoder FiLM modules.
        No mixture bin is multiplied, masked, clipped, or removed.
        """

        if not self.target_relative_frequency_decoder_conditioning:
            raise RuntimeError(
                "Target-relative frequency decoder conditioning is disabled"
            )
        frequency = (
            torch.arange(
                frequency_bins,
                device=target_pitch.device,
                dtype=torch.float32,
            )
            * float(self.config.sample_rate)
            / float(self.config.n_fft)
        )
        fundamental = 440.0 * torch.pow(
            2.0, (target_pitch.float() - 69.0) / 12.0
        )
        ratio = frequency[None] / fundamental[:, None]
        positive_ratio = ratio.clamp_min(1e-7)
        relative_octaves = torch.log2(positive_ratio).clamp(-4.0, 4.0) / 4.0
        relative_octaves = torch.where(
            frequency[None] > 0,
            relative_octaves,
            torch.full_like(relative_octaves, -1.0),
        )

        cents_from_fundamental = 1200.0 * torch.log2(positive_ratio)
        fundamental_proximity = torch.exp(
            -0.5 * (cents_from_fundamental / 100.0).square()
        )
        nearest_harmonic = ratio.round().clamp_min(1.0)
        cents_from_harmonic = 1200.0 * torch.log2(
            positive_ratio / nearest_harmonic
        )
        harmonic_proximity = torch.exp(
            -0.5 * (cents_from_harmonic / 75.0).square()
        )
        harmonic_proximity = torch.where(
            ratio >= 0.5,
            harmonic_proximity,
            torch.zeros_like(harmonic_proximity),
        )

        spectral = torch.stack(
            (
                relative_octaves,
                fundamental_proximity,
                harmonic_proximity,
            ),
            dim=1,
        )
        frame_samples = (
            torch.arange(frames, device=target_pitch.device)[None]
            * self.config.hop_length
        )
        held = (
            (frame_samples < valid_samples.long()[:, None])
            & (frame_samples >= onset.long()[:, None])
            & (frame_samples < offset.long()[:, None])
        )
        return spectral[..., None] * held[:, None, None].to(spectral.dtype)

    def contract(self, preset: str | None = None) -> dict[str, Any]:
        contract = {
            "kind": (
                _OWNERSHIP_CONTRACT_KIND
                if self.ownership_parameterization
                else (
                    _POLAR_COMPLEX_MASK_CONTRACT_KIND
                    if self.bounded_polar_complex_mask_parameterization
                    else (
                    (
                        _UNBOUNDED_COMPLEX_MASK_CONTRACT_KIND
                        if self.unbounded_complex_mask_parameterization
                        else _COMPLEX_MASK_CONTRACT_KIND
                    )
                    if self.complex_mask_parameterization
                    else (
                        _MAGNITUDE_MASK_CONTRACT_KIND
                        if self.magnitude_mask_parameterization
                        else _CONTRACT_KIND
                    )
                    )
                )
            ),
            "contract_version": (
                9
                if self.multi_target_full_score
                else 8
                if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE
                else 7
                if self.target_relative_harmonic_position_conditioning
                else 6
                if (
                    self.conditioning_mode == _NATIVE_FULL_ROLL_CONDITIONING_MODE
                    and self.target_relative_frequency_decoder_conditioning
                )
                else 5
                if self.conditioning_mode == _NATIVE_FULL_ROLL_CONDITIONING_MODE
                else ((4 if self.roll_tail_decay_max_ms else 3)
                if self.conditioning_mode == _FULL_ROLL_CONDITIONING_MODE
                else (2 if self.conditioning_mode == "rt2_target_roll" else 1)
                )
            ),
            "preset": preset,
            "config": asdict(self.config),
            "architecture": {
                "trunk": "shared_sdx23_tfc_tdf_v3",
                "conditioning_arm": "g2_film",
                "head_layout": "independent_target_and_background_complex_heads",
                "head_hidden_channels": self.config.initial_channels,
                "head_output_channels_each": 2 * self.config.num_subbands,
                "head_final_projection_init": "zeros",
                "activation_checkpointing": self.use_activation_checkpointing,
            },
            "inputs": {
                "audio_sources": ["mixture"],
                "query": (
                    ["pitch", "onset_sample"]
                    if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE
                    else ["pitch", "onset_sample", "offset_sample"]
                ),
                "compatibility_signature": True,
                "ignored_audio_slots": [
                    "dna", "other_dna", "unassigned_residual",
                ],
                "internal_source_slots": "mixture_plus_three_zero_slots",
                "normalization": "valid_mixture_rms_floor_1e-4",
            },
            "conditioning": {
                "type": (
                    "native_rt2_25hz_128pitch_full_roll_encoder_target_roll_decoder"
                    if self.conditioning_mode == _NATIVE_FULL_ROLL_CONDITIONING_MODE
                    else (
                    "rt2_25hz_full_roll_encoder_target_roll_decoder"
                    if self.conditioning_mode == _FULL_ROLL_CONDITIONING_MODE
                    else (
                        "rt2_25hz_onset_only_pitch_plus_multiscale_film"
                        if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE
                        else (
                        "rt2_25hz_target_only_roll_plus_multiscale_film"
                        if self.conditioning_mode == "rt2_target_roll"
                        else "harmonic_lifecycle_input_plus_multiscale_film"
                        )
                    )
                    )
                ),
                "score_channels": list(
                    _RT2_ONSET_ONLY_SCORE_CHANNELS
                    if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE
                    else _RT2_ROLL_SCORE_CHANNELS
                    if self.conditioning_mode in {
                        "rt2_target_roll",
                        _FULL_ROLL_CONDITIONING_MODE,
                        _NATIVE_FULL_ROLL_CONDITIONING_MODE,
                    }
                    else (
                        _SCORE_CHANNELS
                        + (
                            _TARGET_RELATIVE_HARMONIC_POSITION_CHANNELS
                            if self.target_relative_harmonic_position_conditioning
                            else ()
                        )
                    )
                ),
                "pitch_embedding_dim": self.config.pitch_embed_dim,
                "conditioning_channels": self.config.conditioning_channels,
                "harmonic_count": _HARMONIC_COUNT,
                "harmonic_gaussian_bandwidth_hz": _HARMONIC_BANDWIDTH_HZ,
                "onset_region_samples": round(_ONSET_SECONDS * self.config.sample_rate),
                "release_region_samples": round(_RELEASE_SECONDS * self.config.sample_rate),
                "absent_occurrence": (
                    "unsupported_valid_onset_required"
                    if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE
                    else "offset_equals_onset_with_zero_harmonic_activity"
                ),
            },
            "spectra": {
                "stft_center": True,
                "stft_window": "periodic_hann",
                "processed_frequency_bins": self.config.processed_frequency_bins,
                "unprocessed_frequency_policy": "target_zero_background_mixture_passthrough",
                "unsupported_frame_policy": "target_zero_background_mixture_passthrough",
                "raw_parameterization": "two_unbounded_complex_spectra",
                "projection": "orthogonal_equal_correction_per_bin",
                "projection_equation": "s=(mix-raw_target-raw_background)/2",
            },
            "outputs": {
                "primary": "target_waveform_istft",
                "background": "background_waveform_istft",
                "diagnostics": [
                    "raw_target_spectrum",
                    "raw_background_spectrum",
                    "projected_target_spectrum",
                    "projected_background_spectrum",
                    "projected_spectrum_reconstruction",
                    "target",
                    "background",
                    "waveform_reconstruction",
                ],
                "mixture_consistency": (
                    "projected_target_plus_background_equals_mixture_complex_stft"
                ),
            },
            "dna_conditioning": False,
            "residual_prediction": False,
            "parameter_count": self.parameter_count(),
        }

        if self.multi_target_full_score:
            contract["architecture"]["multi_target_full_score"] = True
            contract["architecture"]["multi_target_score_routing"] = {
                "pretrained_input_path": "selected_event_score",
                "encoder_film_context": "complete_score",
                "decoder_film_context": "selected_event_score",
            }
            contract["conditioning"]["target_cardinality"] = (
                "one_or_more_score_events"
            )
        if self.multi_occurrence_target_roll:
            contract["contract_version"] = max(10, int(contract["contract_version"]))
            contract["architecture"]["multi_occurrence_target_roll"] = {
                "semantic_output": "one_exact_midi_pitch_track",
                "event_cardinality": "one_or_more_repeated_occurrences",
                "score_composition": "framewise_union_with_onset_resets",
                "first_to_last_envelope": False,
                "single_occurrence_teacher_parity": True,
            }
            contract["inputs"]["full_score"] = [
                "pitch", "onset_sample", "offset_sample", "valid", "target"
            ]
            contract["conditioning"]["target_cardinality"] = (
                "one_or_more_same_pitch_occurrences"
            )
        if self.auxiliary_carrier_conditioning:
            contract["contract_version"] = max(11, int(contract["contract_version"]))
            contract["architecture"]["auxiliary_carrier_conditioning"] = {
                "input": "final_hierarchy_magnitude_pitch_track",
                "source_slot": "dna_compatibility_slot",
                "routes": ["encoder_input", "direct_readout_input"],
                "migration": "zero_unused_slot_weights_exact_teacher_parity",
                "output_parameterization": "unchanged_direct_complex_target",
            }
            contract["inputs"]["audio_sources"] = [
                "mixture", "final_hierarchy_magnitude_pitch_track"
            ]
            contract["inputs"]["ignored_audio_slots"] = [
                "other_dna", "unassigned_residual"
            ]
            contract["inputs"]["internal_source_slots"] = (
                "mixture_plus_pitch_track_guide_plus_two_zero_slots"
            )
            contract["dna_conditioning"] = True
        if self.target_score_input_routing:
            contract["architecture"]["target_score_input_routing"] = True
        if self.full_score_contrast_input:
            contract["architecture"]["full_score_contrast_input"] = {
                "formula": "2 * selected_event_score - complete_score",
                "purpose": "explicit signed non-target onset and score evidence",
                "added_parameters": 0,
            }
        if self.learned_full_score_rejection_input:
            contract["architecture"]["learned_full_score_rejection_input"] = {
                "formula": "selected + tanh(alpha) * (selected - complete)",
                "alpha_shape": [len(_RT2_ROLL_SCORE_CHANNELS)],
                "initial_alpha": 0.0,
                "pretrained_contract_at_initialization": True,
            }
        if self.preserve_signed_score_times:
            contract["conditioning"]["window_boundary_state"] = (
                "negative_onset_is_continuation_not_onset_at_frame_zero"
            )

        if self.conditioning_mode == _ONSET_ONLY_CONDITIONING_MODE:
            contract["inputs"]["compatibility_signature_ignored_query_fields"] = [
                "offset_sample"
            ]
            contract["conditioning"].update(
                {
                    "states": ["onset"],
                    "pitch_embedding_scope": "all_valid_frames",
                    "offset_exposed_to_network": False,
                    "held_state": False,
                    "explicit_lifecycle_labels": False,
                    "signed_offset_distance": False,
                }
            )

        if self.target_relative_harmonic_position_conditioning:
            contract["architecture"].update(
                {
                    "harmonic_position_condition": (
                        "input_encoder_bottleneck_decoder_multiscale_film"
                    ),
                    "harmonic_position_initialization": "zero_effect_exact_parity",
                }
            )
            contract["conditioning"].update(
                {
                    "target_relative_harmonic_position_channels": list(
                        _TARGET_RELATIVE_HARMONIC_POSITION_CHANNELS
                    ),
                    "target_relative_harmonic_coordinate": (
                        "linear_frequency_hz_divided_by_target_f0_hz"
                    ),
                    "target_relative_harmonic_audio_operation": "none",
                    "target_relative_harmonic_maximum_order": _HARMONIC_COUNT,
                }
            )
            contract["outputs"]["diagnostics"].append(
                "target_relative_harmonic_position_condition"
            )

        if self.conditioning_mode in {
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            contract["architecture"].update(
                {
                    "encoder_condition": "complete_rt2_piano_roll",
                    "bottleneck_condition": "complete_rt2_piano_roll",
                    "decoder_condition": "singular_target_rt2_roll",
                    "skip_condition": "singular_target_rt2_roll_film",
                    "output_skip_condition": "singular_target_rt2_roll_film",
                }
            )
            contract["inputs"]["full_score"] = [
                "pitch",
                "onset_sample",
                "offset_sample",
                "valid",
            ]
            contract["conditioning"].update(
                {
                    "score_channels": [
                        f"native_rt2_embedding_projection_{index}"
                        for index in range(len(_RT2_ROLL_SCORE_CHANNELS))
                    ],
                    "states": ["off", "continuation", "onset"],
                    "explicit_lifecycle_labels": False,
                    "signed_offset_distance": False,
                    "training_positive_hold_ms": [
                        self.roll_hold_min_ms,
                        self.roll_hold_max_ms,
                    ],
                    "evaluation_positive_hold_ms": (
                        self.roll_hold_min_ms + self.roll_hold_max_ms
                    )
                    / 2.0,
                    "hold_scope": (
                        "target_and_all_other_score_events"
                        if self.roll_hold_max_ms
                        else "none"
                    ),
                    "post_offset_uncertainty": (
                        "harmonic_exponential_decay_without_silence_label"
                        if self.roll_tail_decay_max_ms
                        else "none"
                    ),
                    "training_tail_decay_time_constant_ms": [
                        self.roll_tail_decay_min_ms,
                        self.roll_tail_decay_max_ms,
                    ],
                    "evaluation_tail_decay_time_constant_ms": (
                        self.roll_tail_decay_min_ms + self.roll_tail_decay_max_ms
                    )
                    / 2.0,
                }
            )
            if self.relative_octave_decoder_conditioning:
                contract["architecture"].update(
                    {
                        "decoder_condition": (
                            "native_rt2_singular_target_roll_plus_"
                            "relative_octave_competitor_states"
                        ),
                        "skip_condition": (
                            "target_roll_plus_relative_octave_competitor_film"
                        ),
                        "output_skip_condition": (
                            "target_roll_plus_relative_octave_competitor_film"
                        ),
                    }
                )
                contract["conditioning"].update(
                    {
                        "relative_octave_decoder_intervals": list(
                            _RELATIVE_OCTAVE_INTERVALS
                        ),
                        "relative_octave_decoder_states": {
                            "off": 0.0,
                            "continuation": 0.5,
                            "onset": 1.0,
                        },
                        "relative_octave_decoder_scope": "decoder_skip_output_only",
                    }
                )
                contract["outputs"]["diagnostics"].append(
                    "relative_octave_decoder_condition"
                )
            if self.target_relative_frequency_decoder_conditioning:
                decoder_feature_name = (
                    "relative_octave_competitor_states_plus_"
                    "target_relative_frequency_features"
                    if self.relative_octave_decoder_conditioning
                    else "target_relative_frequency_features"
                )
                contract["architecture"].update(
                    {
                        "decoder_condition": (
                            "native_rt2_singular_target_roll_plus_"
                            f"{decoder_feature_name}"
                        ),
                        "skip_condition": (
                            "target_roll_plus_"
                            f"{decoder_feature_name}_film"
                        ),
                        "output_skip_condition": (
                            "target_roll_plus_"
                            f"{decoder_feature_name}_film"
                        ),
                    }
                )
                contract["conditioning"].update(
                    {
                        "target_relative_frequency_decoder_channels": list(
                            _TARGET_RELATIVE_FREQUENCY_CHANNELS
                        ),
                        "target_relative_frequency_decoder_scope": (
                            "decoder_skip_output_only"
                        ),
                        "target_relative_frequency_audio_operation": "none",
                        "target_relative_frequency_time_support": (
                            "queried_occurrence_onset_to_conditioned_offset"
                        ),
                    }
                )
                contract["outputs"]["diagnostics"].append(
                    "target_relative_frequency_decoder_condition"
                )
            for legacy_key in (
                "harmonic_count",
                "harmonic_gaussian_bandwidth_hz",
                "onset_region_samples",
                "release_region_samples",
            ):
                contract["conditioning"].pop(legacy_key, None)

        if self.conditioning_mode == _NATIVE_FULL_ROLL_CONDITIONING_MODE:
            contract["architecture"].update(
                {
                    "encoder_condition": "native_rt2_complete_128_pitch_roll",
                    "bottleneck_condition": "native_rt2_complete_128_pitch_roll",
                    "decoder_condition": (
                        "native_rt2_singular_target_roll_plus_"
                        "relative_octave_competitor_states_plus_"
                        "target_relative_frequency_features"
                        if (
                            self.relative_octave_decoder_conditioning
                            and self.target_relative_frequency_decoder_conditioning
                        )
                        else (
                            "native_rt2_singular_target_roll_plus_"
                            "relative_octave_competitor_states"
                            if self.relative_octave_decoder_conditioning
                            else (
                                "native_rt2_singular_target_roll_plus_"
                                "target_relative_frequency_features"
                                if self.target_relative_frequency_decoder_conditioning
                                else "native_rt2_singular_target_roll"
                            )
                        )
                    ),
                    "target_query_absent_from_encoder": True,
                }
            )
            contract["conditioning"].update(
                {
                    "native_shape": ["batch", "25_hz_frames", 128],
                    "native_state_ids": {"off": 0, "continuation": 1, "onset": 2},
                    "native_token_index": "pitch*11+state+7",
                    "native_channel_reduction": "mean_over_128_pitch_embeddings",
                    "embedding": "frozen_mrt2_small_regular_embedder",
                    "post_offset_uncertainty": "none",
                    "training_tail_decay_time_constant_ms": [0.0, 0.0],
                    "evaluation_tail_decay_time_constant_ms": 0.0,
                    "explicit_onset_sustain_release_silence_channels": False,
                }
            )

        if self.ownership_parameterization:
            contract["architecture"].update(
                {
                    "ownership_head": "sigmoid_magnitude_ratio_plus_free_complex_residuals",
                    "ownership_head_initial_bias": -2.0,
                }
            )
            contract["spectra"]["raw_parameterization"] = (
                "mixture_partition_from_ownership_plus_two_unbounded_complex_residuals"
            )
            contract["outputs"]["diagnostics"].extend(
                ["ownership_logits", "ownership_mask"]
            )
        elif self.bounded_polar_complex_mask_parameterization:
            contract["architecture"].update(
                {
                    "head_layout": "sigmoid_magnitude_plus_bounded_phase_residual",
                    "magnitude_mask_range": [0.0, 1.0],
                    "phase_residual_range_radians": [-math.pi, math.pi],
                    "phase_head_initial_value": 0.0,
                }
            )
            contract["spectra"].update(
                {
                    "raw_parameterization": "magnitude_bounded_polar_complex_mask",
                    "projection": "exact_complement",
                    "projection_equation": "target=M*carrier; background=(1-M)*carrier",
                    "phase_policy": "one_learned_bounded_rotation_at_stage_1",
                }
            )
            contract["outputs"]["diagnostics"].extend(
                [
                    "ownership_logits", "ownership_mask",
                    "phase_logits", "phase_residual", "complex_mask",
                ]
            )
        elif self.complex_mask_parameterization:
            if self.unbounded_complex_mask_parameterization:
                contract["architecture"].update(
                    {
                        "head_layout": "single_unbounded_complex_target_mask",
                        "complex_mask_initial_value": [
                            float(torch.sigmoid(torch.tensor(-2.0))),
                            0.0,
                        ],
                        "complex_mask_component_range": "unbounded",
                        "background_head": False,
                    }
                )
                contract["spectra"]["raw_parameterization"] = (
                    "unbounded_complex_target_mask"
                )
            else:
                contract["architecture"].update(
                    {
                        "head_layout": "single_bounded_complex_target_mask",
                        "complex_mask_initial_value": [0.5, 0.0],
                        "complex_mask_real_range": [-1.0, 2.0],
                        "complex_mask_imaginary_range": [-1.5, 1.5],
                    }
                )
                contract["spectra"]["raw_parameterization"] = (
                    "bounded_complex_target_mask"
                )
            contract["spectra"].update(
                {
                    "projection": "exact_complement",
                    "projection_equation": "target=M*carrier; background=(1-M)*carrier",
                }
            )
            contract["outputs"]["diagnostics"].extend(
                ["complex_mask_logits", "complex_mask"]
            )
        elif self.magnitude_mask_parameterization:
            contract["architecture"].update(
                {
                    "head_layout": "single_sigmoid_real_magnitude_mask",
                    "magnitude_mask_initial_value": float(torch.sigmoid(torch.tensor(-2.0))),
                    "magnitude_mask_range": [0.0, 1.0],
                }
            )
            contract["spectra"].update(
                {
                    "raw_parameterization": "real_nonnegative_magnitude_mask",
                    "projection": "exact_complement",
                    "projection_equation": "target=M*mix; background=(1-M)*mix",
                    "phase_policy": "reuse_input_mixture_phase",
                }
            )
            contract["outputs"]["diagnostics"].extend(
                ["ownership_logits", "ownership_mask"]
            )
        return contract

    @classmethod
    def from_contract(cls, raw: object) -> "OccurrenceJointTFCTDFSeparator":
        return occurrence_joint_tfc_tdf_separator_from_contract(raw)

    def forward(
        self,
        mixture: torch.Tensor,
        dna: torch.Tensor,
        other_dna: torch.Tensor,
        unassigned_residual: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        *,
        output_carrier: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(
            mixture,
            dna,
            other_dna,
            unassigned_residual,
            pitch,
            valid_samples,
            onset,
            offset,
        )
        batch, samples = mixture.shape
        if output_carrier is not None and output_carrier.shape != mixture.shape:
            raise ValueError("output_carrier must match mixture shape")
        sample_positions = torch.arange(samples, device=mixture.device)[None]
        sample_support = sample_positions < valid_samples.long()[:, None]
        masked_mixture = mixture.float() * sample_support
        masked_carrier = (
            masked_mixture
            if output_carrier is None
            else output_carrier.float() * sample_support
        )
        denominator = valid_samples.float().clamp_min(1)[:, None]
        # Clamp power before sqrt.  ``sqrt(0).clamp_min(...)`` has the right
        # forward value but an infinite derivative, which poisons a hierarchy
        # whenever an upstream magnitude mask produces an exactly silent
        # carrier.  The 1e-8 power floor is exactly the prior 1e-4 level floor.
        level = (
            masked_mixture.square().sum(dim=-1, keepdim=True) / denominator
        ).clamp_min(1e-8).sqrt()
        normalized_mixture = masked_mixture / level
        mixture_spectrum = self._stft(normalized_mixture)
        carrier_spectrum = self._stft(masked_carrier / level)
        frequency_bins, frames = mixture_spectrum.shape[-2:]

        full_score_hold_extension = None
        full_score_empirical_extension = None
        target_hold_extension = torch.zeros_like(offset)
        full_score_tail_decay = None
        target_tail_decay = None
        conditioned_offset = offset
        if self.conditioning_mode in {
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            if self._full_score_extension_params is not None:
                full_score_empirical_extension, target_hold_extension = (
                    self._condition_full_score_empirical_offsets(valid_samples)
                )
                conditioned_offset = torch.minimum(
                    offset + target_hold_extension, valid_samples
                )
            elif self.roll_hold_max_ms:
                full_score_hold_extension, target_hold_extension = (
                    self._condition_full_score_offsets(valid_samples)
                )
                conditioned_offset = torch.minimum(
                    offset + target_hold_extension, valid_samples
                )
            if self.conditioning_mode == _FULL_ROLL_CONDITIONING_MODE:
                full_score_tail_decay, target_tail_decay = (
                    self._condition_full_score_tail_decay()
                )

        frame_samples = (
            torch.arange(frames, device=mixture.device)[None] * self.config.hop_length
        )
        frame_support = frame_samples < valid_samples.long()[:, None]
        valid_scale = valid_samples.float().clamp_min(1)[:, None]
        if self.conditioning_mode in {
            "rt2_target_roll",
            _ONSET_ONLY_CONDITIONING_MODE,
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            # Do not leak a signed distance-to-offset boundary around the
            # piano-roll condition. Pitch and temporal state live in the roll.
            zeros = torch.zeros_like(frame_support, dtype=torch.float32)
            frame_context = torch.stack(
                (frame_support.float(), zeros, zeros), dim=-1
            )
        else:
            frame_context = torch.stack(
                (
                    frame_support.float(),
                    ((frame_samples - onset[:, None]) / valid_scale).clamp(-1, 1),
                    ((offset[:, None] - frame_samples) / valid_scale).clamp(-1, 1),
                ),
                dim=-1,
            )
        if self.multi_target_full_score:
            if self._full_score_events is None:
                raise RuntimeError("Multi-target conditioning requires full-score events")
            full_pitches, _, _, full_valid, full_target = self._full_score_events
            selected = full_valid.bool() & full_target.bool()
            selected_float = selected[..., None].to(torch.float32)
            pitch_feature = (
                self.pitch_embedding(full_pitches.long()) * selected_float
            ).sum(dim=1) / selected_float.sum(dim=1).clamp_min(1.0)
            pitch_features = pitch_feature[:, None].expand(-1, frames, -1)
        else:
            pitch_features = self.pitch_embedding(pitch.long())[:, None].expand(
                -1, frames, -1
            )
        conditioning = self.conditioning_projection(
            torch.cat((pitch_features, frame_context), dim=-1)
        ).transpose(1, 2)

        zero_spectrum = torch.zeros_like(mixture_spectrum)
        auxiliary_spectrum = (
            self._stft((dna.float() * sample_support) / level)
            if self.auxiliary_carrier_conditioning
            else zero_spectrum
        )
        spectra = [
            mixture_spectrum,
            auxiliary_spectrum,
            zero_spectrum,
            zero_spectrum,
        ]
        complex_channels = torch.cat(
            [torch.stack((spectrum.real, spectrum.imag), dim=1) for spectrum in spectra],
            dim=1,
        )[..., : self.config.processed_frequency_bins, :]
        source_cws = self._cac_to_cws(complex_channels)
        frequency = source_cws.shape[-2]
        conditioning = conditioning[:, :, None].expand(-1, -1, frequency, -1)
        if self.conditioning_mode == _NATIVE_FULL_ROLL_CONDITIONING_MODE:
            if self._full_score_events is None:
                raise RuntimeError("Native full-roll conditioning requires score events")
            (
                full_pitches,
                full_onsets,
                full_offsets,
                full_valid,
                full_target,
            ) = self._full_score_events
            encoder_score_map, conditioning = self._native_rt2_maps(
                pitches=full_pitches,
                onsets=full_onsets,
                offsets=full_offsets,
                event_valid=full_valid,
                valid_samples=valid_samples,
                frequency_bins=self.config.processed_frequency_bins,
                frames=frames,
            )
            conditioning = conditioning[..., :frequency, :]
            if self.multi_target_full_score:
                target_score_map, _ = self._native_rt2_maps(
                    pitches=full_pitches,
                    onsets=full_onsets,
                    offsets=full_offsets,
                    event_valid=full_valid.bool() & full_target.bool(),
                    valid_samples=valid_samples,
                    frequency_bins=self.config.processed_frequency_bins,
                    frames=frames,
                )
            else:
                target_score_map, _ = self._native_rt2_maps(
                    pitches=pitch[:, None],
                    onsets=onset[:, None],
                    offsets=conditioned_offset[:, None],
                    event_valid=(conditioned_offset > onset)[:, None],
                    valid_samples=valid_samples,
                    frequency_bins=self.config.processed_frequency_bins,
                    frames=frames,
                )
        else:
            target_score_map = (
                self._full_rt2_score_map(
                    valid_samples=valid_samples,
                    frequency_bins=self.config.processed_frequency_bins,
                    frames=frames,
                    target_only=True,
                )
                if self.multi_occurrence_target_roll
                else
                self._full_rt2_score_map(
                    valid_samples=valid_samples,
                    frequency_bins=self.config.processed_frequency_bins,
                    frames=frames,
                    tail_decay_samples=full_score_tail_decay,
                    target_only=True,
                )
                if self.multi_target_full_score
                else self._score_map(
                    pitch=pitch,
                    valid_samples=valid_samples,
                    onset=onset,
                    offset=conditioned_offset,
                    frequency_bins=self.config.processed_frequency_bins,
                    frames=frames,
                    tail_decay_samples=target_tail_decay,
                )
            )
            encoder_score_map = (
                self._full_rt2_score_map(
                valid_samples=valid_samples,
                frequency_bins=self.config.processed_frequency_bins,
                frames=frames,
                tail_decay_samples=full_score_tail_decay,
                )
                if self.conditioning_mode == _FULL_ROLL_CONDITIONING_MODE
                else target_score_map
            )
        decoder_score_map = target_score_map
        relative_octave_map = None
        if self.relative_octave_decoder_conditioning:
            relative_octave_map = self._relative_octave_decoder_map(
                target_pitch=pitch,
                valid_samples=valid_samples,
                frequency_bins=self.config.processed_frequency_bins,
                frames=frames,
            )
            decoder_score_map = torch.cat(
                (target_score_map, relative_octave_map), dim=1
            )
        target_relative_frequency_map = None
        if self.target_relative_frequency_decoder_conditioning:
            target_relative_frequency_map = (
                self._target_relative_frequency_decoder_map(
                    target_pitch=pitch,
                    onset=onset,
                    offset=conditioned_offset,
                    valid_samples=valid_samples,
                    frequency_bins=self.config.processed_frequency_bins,
                    frames=frames,
                )
            )
            decoder_score_map = torch.cat(
                (decoder_score_map, target_relative_frequency_map), dim=1
            )
        # Preserve the mature target-conditioned input contract for event
        # extraction. The complete score remains available to every encoder
        # FiLM rung, while the selected multi-pitch event occupies the early
        # score channels that formerly described one queried note.
        input_score_map = (
            target_score_map
            if self.multi_target_full_score or self.target_score_input_routing
            else encoder_score_map
        )
        if self.full_score_contrast_input:
            if encoder_score_map.shape != target_score_map.shape:
                raise RuntimeError("Full and target score maps must be aligned")
            input_score_map = 2.0 * target_score_map - encoder_score_map
        elif self.learned_full_score_rejection_input:
            if encoder_score_map.shape != target_score_map.shape:
                raise ValueError("Learned rejection requires matched score maps")
            assert self.full_score_rejection_scale is not None
            scale = torch.tanh(self.full_score_rejection_scale)[None, :, None, None]
            input_score_map = target_score_map + scale * (
                target_score_map - encoder_score_map
            )
        score_cws = self._cac_to_cws(input_score_map)
        network_input = torch.cat((source_cws, conditioning, score_cws), dim=1)

        frame_divisor = 2**self.config.num_scales
        padded_frames = math.ceil(frames / frame_divisor) * frame_divisor
        frame_padding = padded_frames - frames
        if frame_padding:
            network_input = F.pad(network_input, (0, frame_padding))
            source_cws = F.pad(source_cws, (0, frame_padding))
            target_score_map = F.pad(target_score_map, (0, frame_padding))
            if (
                self.relative_octave_decoder_conditioning
                or self.target_relative_frequency_decoder_conditioning
            ):
                decoder_score_map = F.pad(
                    decoder_score_map, (0, frame_padding)
                )
            else:
                decoder_score_map = target_score_map
            encoder_score_map = F.pad(encoder_score_map, (0, frame_padding))

        first_conv_out = hidden = self.first_conv(network_input)
        hidden = hidden.transpose(-1, -2)
        encoder_spatial_condition = encoder_score_map.transpose(-1, -2)
        target_spatial_condition = decoder_score_map.transpose(-1, -2)
        encoder_outputs = []
        for index, block in enumerate(self.encoder_blocks):
            hidden = self._run_checkpointed(block["tfc_tdf"], hidden)
            hidden = self.encoder_film[index](hidden, encoder_spatial_condition)
            encoder_outputs.append(hidden)
            hidden = block["downscale"](hidden)
        hidden = self._run_checkpointed(self.bottleneck_block, hidden)
        hidden = self.bottleneck_film(hidden, encoder_spatial_condition)
        for index, block in enumerate(self.decoder_blocks):
            hidden = block["upscale"](hidden)
            skip = encoder_outputs.pop()
            if self.conditioning_mode in {
                _FULL_ROLL_CONDITIONING_MODE,
                _NATIVE_FULL_ROLL_CONDITIONING_MODE,
            }:
                skip = self.skip_film[-(index + 1)](
                    skip, target_spatial_condition
                )
            hidden = torch.cat((hidden, skip), dim=1)
            hidden = self._run_checkpointed(block["tfc_tdf"], hidden)
            hidden = self.decoder_film[index](hidden, target_spatial_condition)
        hidden = hidden.transpose(-1, -2)
        output_skip = first_conv_out
        if self.conditioning_mode in {
            _FULL_ROLL_CONDITIONING_MODE,
            _NATIVE_FULL_ROLL_CONDITIONING_MODE,
        }:
            output_skip = self.output_skip_film(output_skip, decoder_score_map)
        hidden = hidden * output_skip
        head_input = torch.cat((source_cws, hidden), dim=1)
        raw_target = None
        raw_background = None
        target_residual = None
        background_residual = None
        if not (
            self.magnitude_mask_parameterization
            or self.bounded_polar_complex_mask_parameterization
        ):
            raw_target = self._cws_to_cac(self.target_head(head_input))[..., :frames]
            target_residual = torch.complex(
                raw_target[:, 0].float(), raw_target[:, 1].float()
            ) * frame_support[:, None]
        if hasattr(self, "background_head"):
            raw_background = self._cws_to_cac(self.background_head(head_input))[..., :frames]
            background_residual = torch.complex(
                raw_background[:, 0].float(), raw_background[:, 1].float()
            )
        ownership_logits = None
        ownership_mask = None
        complex_mask = None
        phase_logits = None
        phase_residual = None
        if self.magnitude_mask_parameterization:
            ownership_logits = self._cws_to_cac(self.ownership_head(head_input))[
                :, 0, ..., :frames
            ]
            ownership_mask = torch.sigmoid(ownership_logits) * frame_support[:, None]
            processed_mixture = carrier_spectrum[
                ..., : self.config.processed_frequency_bins, :
            ]
            processed_target = ownership_mask * processed_mixture
            processed_background = (1.0 - ownership_mask) * processed_mixture
        elif self.bounded_polar_complex_mask_parameterization:
            ownership_logits = self._cws_to_cac(self.ownership_head(head_input))[
                :, 0, ..., :frames
            ]
            phase_logits = self._cws_to_cac(self.phase_head(head_input))[
                :, 0, ..., :frames
            ]
            ownership_mask = torch.sigmoid(ownership_logits) * frame_support[:, None]
            phase_residual = math.pi * torch.tanh(phase_logits)
            complex_mask = torch.polar(
                ownership_mask.float(), phase_residual.float()
            )
            processed_mixture = carrier_spectrum[
                ..., : self.config.processed_frequency_bins, :
            ]
            processed_target = complex_mask * processed_mixture
            processed_background = (1.0 - complex_mask) * processed_mixture
        elif self.ownership_parameterization:
            assert target_residual is not None and background_residual is not None
            ownership_logits = self._cws_to_cac(self.ownership_head(head_input))[
                :, 0, ..., :frames
            ]
            ownership_mask = torch.sigmoid(ownership_logits) * frame_support[:, None]
            processed_mixture = carrier_spectrum[
                ..., : self.config.processed_frequency_bins, :
            ]
            processed_target = ownership_mask * processed_mixture + target_residual
            processed_background = (
                (1.0 - ownership_mask) * processed_mixture + background_residual
            )
        elif self.complex_mask_parameterization:
            assert raw_target is not None
            if self.unbounded_complex_mask_parameterization:
                complex_mask = torch.complex(
                    float(torch.sigmoid(torch.tensor(-2.0)))
                    + raw_target[:, 0].float(),
                    raw_target[:, 1].float(),
                ) * frame_support[:, None]
            else:
                complex_mask = torch.complex(
                    0.5 + 1.5 * torch.tanh(raw_target[:, 0].float()),
                    1.5 * torch.tanh(raw_target[:, 1].float()),
                ) * frame_support[:, None]
            processed_mixture = carrier_spectrum[
                ..., : self.config.processed_frequency_bins, :
            ]
            processed_target = complex_mask * processed_mixture
            processed_background = (1.0 - complex_mask) * processed_mixture
        else:
            assert target_residual is not None and background_residual is not None
            processed_target = target_residual
            processed_background = background_residual
        processed_background = torch.where(
            frame_support[:, None],
            processed_background,
            carrier_spectrum[..., : self.config.processed_frequency_bins, :],
        )

        raw_target_spectrum = torch.zeros_like(carrier_spectrum)
        raw_background_spectrum = carrier_spectrum.clone()
        raw_target_spectrum[..., : self.config.processed_frequency_bins, :] = (
            processed_target
        )
        raw_background_spectrum[..., : self.config.processed_frequency_bins, :] = (
            processed_background
        )
        projected_target, projected_background, projection_correction = (
            self.orthogonal_two_source_projection(
                raw_target_spectrum, raw_background_spectrum, carrier_spectrum
            )
        )

        window = self._window(mixture.float())
        target = torch.istft(
            projected_target,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            length=samples,
        ) * level
        background = torch.istft(
            projected_background,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            length=samples,
        ) * level
        target = target * sample_support
        background = background * sample_support

        spectrum_scale = level[:, :, None]
        scaled_mixture_spectrum = mixture_spectrum * spectrum_scale
        scaled_carrier_spectrum = carrier_spectrum * spectrum_scale
        scaled_raw_target = raw_target_spectrum * spectrum_scale
        scaled_raw_background = raw_background_spectrum * spectrum_scale
        scaled_projected_target = projected_target * spectrum_scale
        scaled_projected_background = projected_background * spectrum_scale
        waveform_reconstruction = target + background
        diagnostics = {
            "target": target,
            "background": background,
            "masked_mixture": masked_mixture,
            "masked_output_carrier": masked_carrier,
            "waveform_reconstruction": waveform_reconstruction,
            "waveform_reconstruction_error": waveform_reconstruction - masked_carrier,
            "mixture_spectrum": scaled_mixture_spectrum,
            "output_carrier_spectrum": scaled_carrier_spectrum,
            "raw_target_spectrum": scaled_raw_target,
            "raw_background_spectrum": scaled_raw_background,
            "projected_target_spectrum": scaled_projected_target,
            "projected_background_spectrum": scaled_projected_background,
            "projected_spectrum_reconstruction": (
                scaled_projected_target + scaled_projected_background
            ),
            "projection_correction_spectrum": projection_correction * spectrum_scale,
            "target_head_processed_spectrum": processed_target * spectrum_scale,
            "background_head_processed_spectrum": ((
                processed_background
                if raw_background is None
                else torch.complex(raw_background[:, 0].float(), raw_background[:, 1].float())
            ) * spectrum_scale),
            "frame_support": frame_support,
            "sample_support": sample_support,
            "level": level,
            "score_condition": target_score_map[..., :frames],
            "encoder_score_condition": encoder_score_map[..., :frames],
            "conditioning_target_hold_extension_samples": target_hold_extension,
            # Shared full-resolution decoder representation for fixed-stem
            # joint heads. This is before the single-target ownership readout;
            # exposing it prevents the joint model from falling back to an
            # identity-carried teacher waveform.
            "shared_head_input": head_input[..., :frames],
        }
        if relative_octave_map is not None:
            diagnostics["relative_octave_decoder_condition"] = (
                relative_octave_map[..., :frames]
            )
        if target_relative_frequency_map is not None:
            diagnostics["target_relative_frequency_decoder_condition"] = (
                target_relative_frequency_map[..., :frames]
            )
        if self.target_relative_harmonic_position_conditioning:
            diagnostics["target_relative_harmonic_position_condition"] = (
                target_score_map[
                    :, -len(_TARGET_RELATIVE_HARMONIC_POSITION_CHANNELS) :, ..., :frames
                ]
            )
        if full_score_hold_extension is not None:
            diagnostics["conditioning_full_score_hold_extension_samples"] = (
                full_score_hold_extension
            )
        if full_score_empirical_extension is not None:
            diagnostics["conditioning_full_score_empirical_extension_samples"] = (
                full_score_empirical_extension
            )
        if full_score_tail_decay is not None and target_tail_decay is not None:
            diagnostics["conditioning_full_score_tail_decay_samples"] = (
                full_score_tail_decay
            )
            diagnostics["conditioning_target_tail_decay_samples"] = target_tail_decay
        self._full_score_events = None
        self._full_score_extension_params = None
        if ownership_mask is not None and ownership_logits is not None:
            diagnostics.update(
                {
                    "ownership_logits": ownership_logits,
                    "ownership_mask": ownership_mask,
                }
            )
        if complex_mask is not None:
            mask_diagnostics = {
                    "complex_mask_logits": torch.complex(
                        raw_target[:, 0].float(), raw_target[:, 1].float()
                    ),
                    "complex_mask": complex_mask,
                } if raw_target is not None else {"complex_mask": complex_mask}
            if phase_logits is not None and phase_residual is not None:
                mask_diagnostics.update({
                    "phase_logits": phase_logits,
                    "phase_residual": phase_residual,
                })
            diagnostics.update(mask_diagnostics)
        return target, diagnostics


def occurrence_joint_tfc_tdf_separator_from_contract(
    raw: object,
) -> OccurrenceJointTFCTDFSeparator:
    if not isinstance(raw, dict) or raw.get("kind") not in {
        _CONTRACT_KIND,
        _OWNERSHIP_CONTRACT_KIND,
        _COMPLEX_MASK_CONTRACT_KIND,
        _UNBOUNDED_COMPLEX_MASK_CONTRACT_KIND,
        _MAGNITUDE_MASK_CONTRACT_KIND,
        _POLAR_COMPLEX_MASK_CONTRACT_KIND,
    }:
        raise ValueError("Checkpoint lacks a joint occurrence TFC-TDF contract")
    config_raw = raw.get("config")
    if not isinstance(config_raw, dict):
        raise ValueError("Joint occurrence TFC-TDF contract lacks a config")
    try:
        config = DNAResidualTFCTDFUNetConfig(**config_raw)
        activation_checkpointing = raw["architecture"]["activation_checkpointing"]
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid joint occurrence TFC-TDF contract") from error
    if not isinstance(activation_checkpointing, bool):
        raise ValueError("activation_checkpointing must be boolean")
    model = OccurrenceJointTFCTDFSeparator(
        config,
        activation_checkpointing=activation_checkpointing,
        ownership_parameterization=raw.get("kind") == _OWNERSHIP_CONTRACT_KIND,
        complex_mask_parameterization=raw.get("kind") in {
            _COMPLEX_MASK_CONTRACT_KIND,
            _UNBOUNDED_COMPLEX_MASK_CONTRACT_KIND,
        },
        unbounded_complex_mask_parameterization=(
            raw.get("kind") == _UNBOUNDED_COMPLEX_MASK_CONTRACT_KIND
        ),
        magnitude_mask_parameterization=raw.get("kind") == _MAGNITUDE_MASK_CONTRACT_KIND,
        bounded_polar_complex_mask_parameterization=(
            raw.get("kind") == _POLAR_COMPLEX_MASK_CONTRACT_KIND
        ),
        conditioning_mode=(
            _NATIVE_FULL_ROLL_CONDITIONING_MODE
            if raw.get("conditioning", {}).get("type")
            == "native_rt2_25hz_128pitch_full_roll_encoder_target_roll_decoder"
            else (
            _FULL_ROLL_CONDITIONING_MODE
            if raw.get("conditioning", {}).get("type")
            == "rt2_25hz_full_roll_encoder_target_roll_decoder"
            else (
                _ONSET_ONLY_CONDITIONING_MODE
                if raw.get("conditioning", {}).get("type")
                == "rt2_25hz_onset_only_pitch_plus_multiscale_film"
                else (
                "rt2_target_roll"
                if raw.get("conditioning", {}).get("type")
                == "rt2_25hz_target_only_roll_plus_multiscale_film"
                else "harmonic_lifecycle"
                )
            )
            )
        ),
        roll_hold_min_ms=float(
            raw.get("conditioning", {}).get("training_positive_hold_ms", [0.0, 0.0])[0]
        ),
        roll_hold_max_ms=float(
            raw.get("conditioning", {}).get("training_positive_hold_ms", [0.0, 0.0])[1]
        ),
        roll_tail_decay_min_ms=float(
            raw.get("conditioning", {}).get(
                "training_tail_decay_time_constant_ms", [0.0, 0.0]
            )[0]
        ),
        roll_tail_decay_max_ms=float(
            raw.get("conditioning", {}).get(
                "training_tail_decay_time_constant_ms", [0.0, 0.0]
            )[1]
        ),
        relative_octave_decoder_conditioning=bool(
            raw.get("conditioning", {}).get("relative_octave_decoder_intervals")
        ),
        target_relative_frequency_decoder_conditioning=bool(
            raw.get("conditioning", {}).get(
                "target_relative_frequency_decoder_channels"
            )
        ),
        target_relative_harmonic_position_conditioning=bool(
            raw.get("conditioning", {}).get(
                "target_relative_harmonic_position_channels"
            )
        ),
        multi_target_full_score=bool(
            raw.get("architecture", {}).get("multi_target_full_score", False)
        ),
        multi_occurrence_target_roll=bool(
            raw.get("architecture", {}).get(
                "multi_occurrence_target_roll", False
            )
        ),
        target_score_input_routing=bool(
            raw.get("architecture", {}).get("target_score_input_routing", False)
        ),
        full_score_contrast_input=bool(
            raw.get("architecture", {}).get("full_score_contrast_input", False)
        ),
        learned_full_score_rejection_input=bool(
            raw.get("architecture", {}).get(
                "learned_full_score_rejection_input", False
            )
        ),
        preserve_signed_score_times=(
            raw.get("conditioning", {}).get("window_boundary_state")
            == "negative_onset_is_continuation_not_onset_at_frame_zero"
        ),
    )
    if raw.get("architecture", {}).get("auxiliary_carrier_conditioning"):
        model.enable_auxiliary_carrier_conditioning()
    if model.contract(raw.get("preset")) != raw:
        raise ValueError("Joint occurrence TFC-TDF checkpoint contract changed")
    return model


__all__ = [
    "OccurrenceJointTFCTDFSeparator",
    "magnitude_source_ownership_loss",
    "occurrence_joint_tfc_tdf_separator_from_contract",
]
