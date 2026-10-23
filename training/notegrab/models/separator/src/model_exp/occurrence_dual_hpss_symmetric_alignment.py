"""HPSS dual separator with simultaneous symmetric low-resolution alignment."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .occurrence_dual_hpss_separator import OccurrenceDualHPSSSeparator
from notesep.models.pitch_relative_harmonic_gate import (
    gate_applies_to_target_and_prediction,
    pitch_relative_harmonic_gate,
)


class SymmetricCrossAttention2d(nn.Module):
    """Write source features into a target stream without ordering bias."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.q_norm = nn.LayerNorm(channels)
        self.k_norm = nn.LayerNorm(channels)
        self.v_norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)
        # Preserve the exact independent-branch function at initialization.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        valid_time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source.shape != target.shape or source.ndim != 4:
            raise ValueError("aligned source and target must share [B,C,T,F] shape")
        batch, channels, time, frequency = target.shape
        source_tokens = source.flatten(2).transpose(1, 2)
        target_tokens = target.flatten(2).transpose(1, 2)

        def heads(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(
                batch, time * frequency, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = heads(self.q_norm(self.q(target_tokens)))
        key = heads(self.k_norm(self.k(source_tokens)))
        value = heads(self.v_norm(self.v(source_tokens)))
        attention_mask = None
        query_support = None
        if valid_time_steps is not None:
            if valid_time_steps.shape != (batch,):
                raise ValueError("valid_time_steps must have shape [batch]")
            valid_time_steps = valid_time_steps.to(device=target.device).long()
            if torch.any(valid_time_steps < 1) or torch.any(valid_time_steps > time):
                raise ValueError("valid_time_steps must be in [1, time]")
            valid_time = (
                torch.arange(time, device=target.device)[None]
                < valid_time_steps[:, None]
            )
            valid_tokens = valid_time[:, :, None].expand(-1, -1, frequency).reshape(
                batch, time * frequency
            )
            attention_mask = torch.zeros(
                batch,
                1,
                1,
                time * frequency,
                device=target.device,
                dtype=query.dtype,
            ).masked_fill(~valid_tokens[:, None, None, :], float("-inf"))
            query_support = valid_tokens[:, None, :, None]
        message = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        message = message.transpose(1, 2).reshape(batch, time * frequency, channels)
        message = self.proj(message)
        if query_support is not None:
            message = message * query_support[:, 0]
        return message.transpose(1, 2).reshape_as(target)


class OccurrenceDualHPSSSymmetricAlignment(OccurrenceDualHPSSSeparator):
    """Communicating HPSS branches at deep encoder, bottleneck, and decoder."""

    ALIGNMENT_RUNGS = ("deep_encoder", "bottleneck", "deep_decoder")

    def __init__(self, *args: Any, attention_heads: int = 8, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self.config
        deep_channels = config.initial_channels + config.growth * (config.num_scales - 1)
        bottleneck_channels = config.initial_channels + config.growth * config.num_scales
        rung_channels = {
            "deep_encoder": deep_channels,
            "bottleneck": bottleneck_channels,
            "deep_decoder": deep_channels,
        }
        self.harmonic_to_percussive = nn.ModuleDict(
            {
                name: SymmetricCrossAttention2d(channels, attention_heads)
                for name, channels in rung_channels.items()
            }
        )
        self.percussive_to_harmonic = nn.ModuleDict(
            {
                name: SymmetricCrossAttention2d(channels, attention_heads)
                for name, channels in rung_channels.items()
            }
        )
        self.attention_heads = int(attention_heads)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        harmonic_gate_mode: str = "none",
        percussive_parameterization: str = "ownership",
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        attention_heads: int = 8,
        harmonic_parameterization: str = "ownership",
        output_carrier_mode: str = "encoder_input",
        complex_mask_mode: str = "bounded",
    ) -> "OccurrenceDualHPSSSymmetricAlignment":
        return cls(
            preset,
            harmonic_gate_mode,
            percussive_parameterization,
            conditioning_mode,
            roll_hold_min_ms,
            roll_hold_max_ms,
            attention_heads=attention_heads,
            harmonic_parameterization=harmonic_parameterization,
            output_carrier_mode=output_carrier_mode,
            complex_mask_mode=complex_mask_mode,
        )

    def contract(self) -> dict[str, Any]:
        contract = super().contract()
        contract["contract_version"] = max(5, int(contract["contract_version"]))
        contract["architecture"]["layout"] = (
            "symmetric_low_resolution_aligned_hpss_harmonic_and_percussive_branches"
        )
        contract["architecture"]["communication"] = {
            "type": "simultaneous_bidirectional_cross_attention",
            "rungs": list(self.ALIGNMENT_RUNGS),
            "attention_heads": self.attention_heads,
            "direction_parameters": "untied",
            "output_projection_initialization": "zeros",
            "message_fusion": "residual_add",
            "snapshot_policy": "both_directions_read_same_pre_fusion_states",
        }
        contract["parameter_count"] = self.parameter_count()
        initialization = getattr(self, "communication_initialization", None)
        if initialization is not None:
            contract["initialization_checkpoint"] = initialization
        auxiliary = (
            self.harmonic_branch.auxiliary_carrier_conditioning,
            self.percussive_branch.auxiliary_carrier_conditioning,
        )
        repeated = (
            self.harmonic_branch.multi_occurrence_target_roll,
            self.percussive_branch.multi_occurrence_target_roll,
        )
        if len(set(auxiliary)) != 1 or len(set(repeated)) != 1:
            raise RuntimeError("HPSS renderer branches must share one input contract")
        if auxiliary[0] or repeated[0]:
            if not (auxiliary[0] and repeated[0]):
                raise RuntimeError(
                    "Shared pitch-track rendering requires both guide and repeated roll"
                )
            contract["contract_version"] = max(8, int(contract["contract_version"]))
            contract["architecture"]["shared_pitch_track_complex_renderer"] = {
                "weights": "one_shared_renderer_for_every_final_pitch_track",
                "mixture_encoding": "repeated_per_active_track_in_training_v1",
                "guide": "hierarchy_final_magnitude_carrier",
                "score": "all_repeated_occurrences_of_one_exact_midi_pitch",
                "complex_output": "direct_harmonic_plus_percussive_target",
                "leaf_route_dependent_parameters": False,
                "migration": "exact_single_occurrence_teacher_parity",
            }
            contract["inputs"]["pitch_track_guide"] = {
                "harmonic": "shared-mask HPSS harmonic carrier",
                "percussive": "shared-mask HPSS percussive carrier",
            }
            contract["inputs"]["target_roll"] = (
                "complete_repeated_occurrence_roll_for_one_pitch_track"
            )
            contract["dna_conditioning"] = True
        return contract

    def enable_shared_pitch_track_renderer(self) -> None:
        """Open one zero-start carrier path shared by every final pitch track."""

        for branch in (self.harmonic_branch, self.percussive_branch):
            branch.multi_occurrence_target_roll = True
            branch.enable_auxiliary_carrier_conditioning()

    def set_pitch_track_events(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Condition each batch row on every occurrence of one MIDI pitch."""

        if not (
            pitches.shape == onsets.shape == offsets.shape == valid.shape
            and pitches.ndim == 2
        ):
            raise ValueError("Pitch-track score tensors must share [batch,events]")
        for branch in (self.harmonic_branch, self.percussive_branch):
            if not branch.multi_occurrence_target_roll:
                raise RuntimeError("Shared pitch-track rendering is disabled")
            branch.set_full_score_events(
                pitches, onsets, offsets, valid, valid
            )

    def _exchange(
        self,
        rung: str,
        harmonic: torch.Tensor,
        percussive: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both messages read frozen same-depth snapshots. Do not update either
        # stream before calculating the reverse direction.
        harmonic_snapshot = harmonic
        percussive_snapshot = percussive
        time_divisor = 32 if rung == "bottleneck" else 16
        valid_time_steps = torch.div(
            valid_frames.long() + time_divisor - 1,
            time_divisor,
            rounding_mode="floor",
        ).clamp(max=harmonic.shape[-2])
        harmonic_message = self.percussive_to_harmonic[rung](
            percussive_snapshot, harmonic_snapshot, valid_time_steps
        )
        percussive_message = self.harmonic_to_percussive[rung](
            harmonic_snapshot, percussive_snapshot, valid_time_steps
        )
        self._alignment_diagnostics[f"{rung}_percussive_to_harmonic_rms"] = (
            harmonic_message.float().square().mean().sqrt()
        )
        self._alignment_diagnostics[f"{rung}_harmonic_to_percussive_rms"] = (
            percussive_message.float().square().mean().sqrt()
        )
        return harmonic_snapshot + harmonic_message, percussive_snapshot + percussive_message

    def _prepare_harmonic_side_condition(
        self,
        harmonic_state: dict[str, Any],
        harmonic_audio: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
    ) -> torch.Tensor | None:
        """Extension hook for zero-start spatial harmonic conditioning."""

        return None

    def _apply_harmonic_side_condition(
        self,
        rung: str,
        index: int | None,
        values: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """Preserve the historical symmetric model when no side path exists."""

        return values

    def _merge_branch_targets(
        self,
        harmonic_target: torch.Tensor,
        percussive_target: torch.Tensor,
        harmonic_diagnostics: dict[str, torch.Tensor],
        percussive_diagnostics: dict[str, torch.Tensor],
        reference_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extension hook; the mature symmetric contract is an exact sum."""

        return harmonic_target + percussive_target, {}

    @staticmethod
    def _prepare_branch(
        branch: nn.Module,
        mixture: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        output_carrier: torch.Tensor | None = None,
        loop_condition: torch.Tensor | None = None,
        auxiliary_carrier: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | int]:
        if output_carrier is not None and output_carrier.shape != mixture.shape:
            raise ValueError("output_carrier must match branch mixture shape")
        if loop_condition is not None and loop_condition.shape != mixture.shape:
            raise ValueError("loop_condition must match branch mixture shape")
        if auxiliary_carrier is not None and auxiliary_carrier.shape != mixture.shape:
            raise ValueError("auxiliary_carrier must match branch mixture shape")
        if (
            auxiliary_carrier is not None
            and not branch.auxiliary_carrier_conditioning
        ):
            raise ValueError("auxiliary carrier was supplied to a disabled branch")
        zeros = torch.zeros_like(mixture)
        branch._validate_inputs(
            mixture, zeros, zeros, zeros, pitch, valid_samples, onset, offset
        )
        batch, samples = mixture.shape
        sample_positions = torch.arange(samples, device=mixture.device)[None]
        sample_support = sample_positions < valid_samples.long()[:, None]
        masked_mixture = mixture.float() * sample_support
        masked_carrier = (
            masked_mixture
            if output_carrier is None
            else output_carrier.float() * sample_support
        )
        denominator = valid_samples.float().clamp_min(1)[:, None]
        level = (
            masked_mixture.square().sum(dim=-1, keepdim=True) / denominator
        ).sqrt().clamp_min(1e-4)
        mixture_spectrum = branch._stft(masked_mixture / level)
        carrier_spectrum = branch._stft(masked_carrier / level)
        frequency_bins, frames = mixture_spectrum.shape[-2:]
        frame_samples = (
            torch.arange(frames, device=mixture.device)[None] * branch.config.hop_length
        )
        frame_support = frame_samples < valid_samples.long()[:, None]
        if branch.conditioning_mode in {"rt2_target_roll", "rt2_onset_only"}:
            context_zeros = torch.zeros_like(frame_support, dtype=torch.float32)
            frame_context = torch.stack(
                (frame_support.float(), context_zeros, context_zeros), dim=-1
            )
        else:
            valid_scale = valid_samples.float().clamp_min(1)[:, None]
            frame_context = torch.stack(
                (
                    frame_support.float(),
                    ((frame_samples - onset[:, None]) / valid_scale).clamp(-1, 1),
                    ((offset[:, None] - frame_samples) / valid_scale).clamp(-1, 1),
                ),
                dim=-1,
            )
        pitch_features = branch.pitch_embedding(pitch.long())[:, None].expand(
            -1, frames, -1
        )
        conditioning = branch.conditioning_projection(
            torch.cat((pitch_features, frame_context), dim=-1)
        ).transpose(1, 2)
        zero_spectrum = torch.zeros_like(mixture_spectrum)
        auxiliary_spectrum = (
            branch._stft((auxiliary_carrier.float() * sample_support) / level)
            if branch.auxiliary_carrier_conditioning
            and auxiliary_carrier is not None
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
        )[..., : branch.config.processed_frequency_bins, :]
        source_cws = branch._cac_to_cws(complex_channels)
        frequency = source_cws.shape[-2]
        conditioning = conditioning[:, :, None].expand(-1, -1, frequency, -1)
        score_map = (
            branch._full_rt2_score_map(
                valid_samples=valid_samples,
                frequency_bins=branch.config.processed_frequency_bins,
                frames=frames,
                target_only=True,
            )
            if branch.multi_occurrence_target_roll
            else branch._score_map(
                pitch=pitch,
                valid_samples=valid_samples,
                onset=onset,
                offset=offset,
                frequency_bins=branch.config.processed_frequency_bins,
                frames=frames,
            )
        )
        score_cws = branch._cac_to_cws(score_map)
        network_input = torch.cat((source_cws, conditioning, score_cws), dim=1)
        frame_divisor = 2**branch.config.num_scales
        padded_frames = math.ceil(frames / frame_divisor) * frame_divisor
        if padded_frames != frames:
            padding = padded_frames - frames
            network_input = F.pad(network_input, (0, padding))
            source_cws = F.pad(source_cws, (0, padding))
            score_map = F.pad(score_map, (0, padding))
        first_conv_out = branch.first_conv(network_input)
        if loop_condition is not None:
            adapter = getattr(branch, "loop_condition_adapter", None)
            if adapter is None:
                raise ValueError("loop_condition requires a learned loop adapter")
            masked_loop = loop_condition.float() * sample_support
            loop_spectrum = branch._stft(masked_loop / level)
            loop_channels = torch.stack(
                (loop_spectrum.real, loop_spectrum.imag), dim=1
            )[..., : branch.config.processed_frequency_bins, :]
            loop_cws = branch._cac_to_cws(loop_channels)
            if padded_frames != frames:
                loop_cws = F.pad(loop_cws, (0, padded_frames - frames))
            first_conv_out = first_conv_out + adapter(loop_cws)
        return {
            "batch": batch,
            "samples": samples,
            "sample_support": sample_support,
            "masked_mixture": masked_mixture,
            "masked_carrier": masked_carrier,
            "level": level,
            "mixture_spectrum": mixture_spectrum,
            "carrier_spectrum": carrier_spectrum,
            "frequency_bins": frequency_bins,
            "frames": frames,
            "frame_support": frame_support,
            "valid_frames": frame_support.sum(dim=-1),
            "source_cws": source_cws,
            "score_map": score_map,
            "first_conv_out": first_conv_out,
            "loop_condition": loop_condition,
            "auxiliary_carrier": auxiliary_carrier,
            "hidden": first_conv_out.transpose(-1, -2),
            "spatial_condition": score_map.transpose(-1, -2),
            "encoder_outputs": [],
        }

    @staticmethod
    def _complete_branch(
        branch: nn.Module,
        state: dict[str, Any],
        hidden: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frames = int(state["frames"])
        source_cws = state["source_cws"]
        hidden = hidden.transpose(-1, -2) * state["first_conv_out"]
        head_input = torch.cat((source_cws, hidden), dim=1)
        mixture_spectrum = state["mixture_spectrum"]
        carrier_spectrum = state["carrier_spectrum"]
        frame_support = state["frame_support"]

        raw_target = raw_background = target_residual = background_residual = None
        if not branch.magnitude_mask_parameterization:
            raw_target = branch._cws_to_cac(branch.target_head(head_input))[..., :frames]
            target_residual = torch.complex(
                raw_target[:, 0].float(), raw_target[:, 1].float()
            ) * frame_support[:, None]
        if hasattr(branch, "background_head"):
            raw_background = branch._cws_to_cac(branch.background_head(head_input))[..., :frames]
            background_residual = torch.complex(
                raw_background[:, 0].float(), raw_background[:, 1].float()
            )
        ownership_logits = ownership_mask = complex_mask = None
        if branch.magnitude_mask_parameterization:
            ownership_logits = branch._cws_to_cac(branch.ownership_head(head_input))[
                :, 0, ..., :frames
            ]
            ownership_mask = torch.sigmoid(ownership_logits) * frame_support[:, None]
            processed_mixture = carrier_spectrum[
                ..., : branch.config.processed_frequency_bins, :
            ]
            processed_target = ownership_mask * processed_mixture
            processed_background = (1.0 - ownership_mask) * processed_mixture
        elif branch.ownership_parameterization:
            ownership_logits = branch._cws_to_cac(branch.ownership_head(head_input))[
                :, 0, ..., :frames
            ]
            ownership_mask = torch.sigmoid(ownership_logits) * frame_support[:, None]
            processed_mixture = carrier_spectrum[
                ..., : branch.config.processed_frequency_bins, :
            ]
            processed_target = ownership_mask * processed_mixture + target_residual
            processed_background = (
                (1.0 - ownership_mask) * processed_mixture + background_residual
            )
        elif branch.complex_mask_parameterization:
            if branch.unbounded_complex_mask_parameterization:
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
                ..., : branch.config.processed_frequency_bins, :
            ]
            processed_target = complex_mask * processed_mixture
            processed_background = (1.0 - complex_mask) * processed_mixture
        else:
            processed_target = target_residual
            processed_background = background_residual
        processed_background = torch.where(
            frame_support[:, None],
            processed_background,
            carrier_spectrum[..., : branch.config.processed_frequency_bins, :],
        )

        raw_target_spectrum = torch.zeros_like(carrier_spectrum)
        raw_background_spectrum = carrier_spectrum.clone()
        raw_target_spectrum[..., : branch.config.processed_frequency_bins, :] = processed_target
        raw_background_spectrum[..., : branch.config.processed_frequency_bins, :] = processed_background
        projected_target, projected_background, projection_correction = (
            branch.orthogonal_two_source_projection(
                raw_target_spectrum, raw_background_spectrum, carrier_spectrum
            )
        )
        samples = int(state["samples"])
        level = state["level"]
        window = branch._window(mixture.float())
        target = torch.istft(
            projected_target,
            n_fft=branch.config.n_fft,
            hop_length=branch.config.hop_length,
            win_length=branch.config.n_fft,
            window=window,
            center=True,
            length=samples,
        ) * level
        background = torch.istft(
            projected_background,
            n_fft=branch.config.n_fft,
            hop_length=branch.config.hop_length,
            win_length=branch.config.n_fft,
            window=window,
            center=True,
            length=samples,
        ) * level
        sample_support = state["sample_support"]
        target = target * sample_support
        background = background * sample_support
        spectrum_scale = level[:, :, None]
        scaled_target = projected_target * spectrum_scale
        scaled_background = projected_background * spectrum_scale
        diagnostics = {
            "target": target,
            "background": background,
            "masked_mixture": state["masked_mixture"],
            "masked_output_carrier": state["masked_carrier"],
            "waveform_reconstruction": target + background,
            "waveform_reconstruction_error": (
                target + background - state["masked_carrier"]
            ),
            "mixture_spectrum": mixture_spectrum * spectrum_scale,
            "output_carrier_spectrum": carrier_spectrum * spectrum_scale,
            "raw_target_spectrum": raw_target_spectrum * spectrum_scale,
            "raw_background_spectrum": raw_background_spectrum * spectrum_scale,
            "projected_target_spectrum": scaled_target,
            "projected_background_spectrum": scaled_background,
            "projected_spectrum_reconstruction": scaled_target + scaled_background,
            "projection_correction_spectrum": projection_correction * spectrum_scale,
            "target_head_processed_spectrum": processed_target * spectrum_scale,
            "background_head_processed_spectrum": ((
                processed_background
                if raw_background is None
                else torch.complex(
                    raw_background[:, 0].float(), raw_background[:, 1].float()
                )
            ) * spectrum_scale),
            "frame_support": frame_support,
            "sample_support": sample_support,
            "level": level,
            "score_condition": state["score_map"][..., :frames],
        }
        if ownership_mask is not None:
            diagnostics["ownership_logits"] = ownership_logits
            diagnostics["ownership_mask"] = ownership_mask
        if complex_mask is not None:
            diagnostics["complex_mask_logits"] = torch.complex(
                raw_target[:, 0].float(), raw_target[:, 1].float()
            )
            diagnostics["complex_mask"] = complex_mask
        return target, diagnostics

    def forward(
        self,
        harmonic_mixture: torch.Tensor,
        percussive_mixture: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        *,
        input_gate_strength: torch.Tensor | None = None,
        loop_harmonic_condition: torch.Tensor | None = None,
        loop_percussive_condition: torch.Tensor | None = None,
        guide_harmonic: torch.Tensor | None = None,
        guide_percussive: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if harmonic_mixture.shape != percussive_mixture.shape:
            raise ValueError("Harmonic and percussive mixtures must have equal shape")
        self._alignment_diagnostics: dict[str, torch.Tensor] = {}
        conditioned_offset, offset_extension = self._condition_offset(
            onset, offset, valid_samples
        )
        gated_harmonic = self._gate_harmonic_mixture(
            harmonic_mixture, pitch, valid_samples, input_gate_strength
        )
        gated_guide_harmonic = (
            None
            if guide_harmonic is None
            else self._gate_harmonic_mixture(
                guide_harmonic, pitch, valid_samples, input_gate_strength
            )
        )
        harmonic_state = self._prepare_branch(
            self.harmonic_branch,
            gated_harmonic,
            pitch,
            valid_samples,
            onset,
            conditioned_offset,
            (
                harmonic_mixture
                if self.output_carrier_mode == "ungated_hpss"
                else None
            ),
            loop_harmonic_condition,
            gated_guide_harmonic,
        )
        percussive_state = self._prepare_branch(
            self.percussive_branch,
            percussive_mixture,
            pitch,
            valid_samples,
            onset,
            conditioned_offset,
            (
                percussive_mixture
                if self.output_carrier_mode == "ungated_hpss"
                else None
            ),
            loop_percussive_condition,
            guide_percussive,
        )
        harmonic = harmonic_state["hidden"]
        percussive = percussive_state["hidden"]
        harmonic_side_condition = self._prepare_harmonic_side_condition(
            harmonic_state,
            gated_harmonic,
            pitch,
            valid_samples,
        )
        harmonic = self._apply_harmonic_side_condition(
            "input", None, harmonic, harmonic_side_condition
        )
        harmonic_skips: list[torch.Tensor] = harmonic_state["encoder_outputs"]
        percussive_skips: list[torch.Tensor] = percussive_state["encoder_outputs"]

        last_encoder = len(self.harmonic_branch.encoder_blocks) - 1
        for index, (harmonic_block, percussive_block) in enumerate(
            zip(self.harmonic_branch.encoder_blocks, self.percussive_branch.encoder_blocks)
        ):
            harmonic = self.harmonic_branch._run_checkpointed(
                harmonic_block["tfc_tdf"], harmonic
            )
            percussive = self.percussive_branch._run_checkpointed(
                percussive_block["tfc_tdf"], percussive
            )
            harmonic = self.harmonic_branch.encoder_film[index](
                harmonic, harmonic_state["spatial_condition"]
            )
            percussive = self.percussive_branch.encoder_film[index](
                percussive, percussive_state["spatial_condition"]
            )
            if index == last_encoder:
                harmonic, percussive = self._exchange(
                    "deep_encoder",
                    harmonic,
                    percussive,
                    harmonic_state["valid_frames"],
                )
            harmonic_skips.append(harmonic)
            percussive_skips.append(percussive)
            harmonic = harmonic_block["downscale"](harmonic)
            percussive = percussive_block["downscale"](percussive)

        harmonic = self.harmonic_branch._run_checkpointed(
            self.harmonic_branch.bottleneck_block, harmonic
        )
        percussive = self.percussive_branch._run_checkpointed(
            self.percussive_branch.bottleneck_block, percussive
        )
        harmonic = self.harmonic_branch.bottleneck_film(
            harmonic, harmonic_state["spatial_condition"]
        )
        percussive = self.percussive_branch.bottleneck_film(
            percussive, percussive_state["spatial_condition"]
        )
        harmonic, percussive = self._exchange(
            "bottleneck", harmonic, percussive, harmonic_state["valid_frames"]
        )

        for index, (harmonic_block, percussive_block) in enumerate(
            zip(self.harmonic_branch.decoder_blocks, self.percussive_branch.decoder_blocks)
        ):
            harmonic = harmonic_block["upscale"](harmonic)
            percussive = percussive_block["upscale"](percussive)
            harmonic = torch.cat((harmonic, harmonic_skips.pop()), dim=1)
            percussive = torch.cat((percussive, percussive_skips.pop()), dim=1)
            harmonic = self.harmonic_branch._run_checkpointed(
                harmonic_block["tfc_tdf"], harmonic
            )
            percussive = self.percussive_branch._run_checkpointed(
                percussive_block["tfc_tdf"], percussive
            )
            harmonic = self.harmonic_branch.decoder_film[index](
                harmonic, harmonic_state["spatial_condition"]
            )
            percussive = self.percussive_branch.decoder_film[index](
                percussive, percussive_state["spatial_condition"]
            )
            if index == 0:
                harmonic, percussive = self._exchange(
                    "deep_decoder",
                    harmonic,
                    percussive,
                    harmonic_state["valid_frames"],
                )

        harmonic_target, harmonic_diagnostics = self._complete_branch(
            self.harmonic_branch, harmonic_state, harmonic, gated_harmonic
        )
        percussive_target, percussive_diagnostics = self._complete_branch(
            self.percussive_branch, percussive_state, percussive, percussive_mixture
        )
        if gate_applies_to_target_and_prediction(self.harmonic_gate_mode):
            harmonic_target = pitch_relative_harmonic_gate(
                harmonic_target,
                pitch,
                valid_samples,
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                mode=self.harmonic_gate_mode,
            )
        estimate, joint_output_diagnostics = self._merge_branch_targets(
            harmonic_target,
            percussive_target,
            harmonic_diagnostics,
            percussive_diagnostics,
            harmonic_mixture,
        )
        background = (
            harmonic_diagnostics["background"]
            + percussive_diagnostics["background"]
        )
        return estimate, {
            "harmonic": harmonic_target,
            "percussive": percussive_target,
            "background": background,
            "waveform_reconstruction": estimate + background,
            "harmonic_input": gated_harmonic,
            "harmonic_carrier": (
                harmonic_mixture
                if self.output_carrier_mode == "ungated_hpss"
                else gated_harmonic
            ),
            "percussive_carrier": percussive_mixture,
            "loop_harmonic_condition": loop_harmonic_condition,
            "loop_percussive_condition": loop_percussive_condition,
            "guide_harmonic": guide_harmonic,
            "harmonic_side_condition": harmonic_side_condition,
            "guide_percussive": guide_percussive,
            "harmonic_branch": harmonic_diagnostics,
            "percussive_branch": percussive_diagnostics,
            "conditioned_offset_samples": conditioned_offset,
            "conditioning_hold_extension_samples": offset_extension,
            "input_gate_strength": (
                torch.ones_like(pitch, dtype=harmonic_mixture.dtype)
                if input_gate_strength is None
                else input_gate_strength.to(harmonic_mixture.dtype)
            ),
            "alignment": self._alignment_diagnostics,
            "joint_output": joint_output_diagnostics,
        }


__all__ = ["OccurrenceDualHPSSSymmetricAlignment", "SymmetricCrossAttention2d"]
