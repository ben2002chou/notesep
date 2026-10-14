"""Symmetric HPSS evidence trunks with one full-band occurrence output head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .occurrence_dual_hpss_symmetric_alignment import (
    OccurrenceDualHPSSSymmetricAlignment,
)
from .occurrence_dual_hpss_separator import OccurrenceDualHPSSSeparator
from .occurrence_joint_tfc_tdf_separator import _complex_head, _ownership_head


HARMONIC_EVIDENCE_MODES = {"gated_only", "gated_plus_full"}


class OccurrenceDualHPSSMergedOutput(OccurrenceDualHPSSSymmetricAlignment):
    """Keep specialized H/P trunks but form the note once from both streams.

    The incumbent independently renders an H target and a P target, closes each
    against its own carrier, performs two iSTFTs, and sums the waveforms.  This
    model instead concatenates both final feature maps and both carrier feature
    maps, then predicts one ownership field plus one free complex correction
    against the original full-band mixture.  The background is the exact
    complement of that single target; there is no target/background projection
    and no H+P source summation at the output.
    """

    def __init__(
        self,
        *args: Any,
        harmonic_evidence_mode: str = "gated_plus_full",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if harmonic_evidence_mode not in HARMONIC_EVIDENCE_MODES:
            raise ValueError(
                f"Unknown harmonic evidence mode: {harmonic_evidence_mode}"
            )
        self.harmonic_evidence_mode = str(harmonic_evidence_mode)
        if self.harmonic_evidence_mode == "gated_plus_full":
            # The branch already reserves compatibility source slots.  Opening
            # one slot changes no tensor dimensions and supplies ungated H next
            # to the gated primary evidence instead of randomly switching them.
            self.harmonic_branch.enable_auxiliary_carrier_conditioning()

        subbands = int(self.config.num_subbands)
        source_channels_per_branch = 4 * 2 * subbands
        head_input_channels = (
            2 * int(self.config.initial_channels)
            + 2 * source_channels_per_branch
        )
        self.merged_target_head = _complex_head(
            head_input_channels, int(self.config.initial_channels), subbands
        )
        self.merged_ownership_head = _ownership_head(
            head_input_channels, int(self.config.initial_channels), subbands
        )

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
        harmonic_evidence_mode: str = "gated_plus_full",
    ) -> "OccurrenceDualHPSSMergedOutput":
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
            harmonic_evidence_mode=harmonic_evidence_mode,
        )

    def contract(self) -> dict[str, Any]:
        # This model intentionally gives only the harmonic trunk an auxiliary
        # evidence channel.  The symmetric parent's contract guard rejects
        # asymmetric branch inputs because they were previously only used by
        # the shared pitch-track renderer.  Start from the common dual contract
        # and record this model's communication and evidence topology directly.
        contract = OccurrenceDualHPSSSeparator.contract(self)
        contract["contract_version"] = max(9, int(contract["contract_version"]))
        contract["architecture"]["layout"] = (
            "symmetric_low_resolution_aligned_hpss_evidence_trunks_merged_output"
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
        contract["architecture"]["merge"] = "single_fullband_complex_source_head"
        contract["architecture"]["merged_output"] = {
            "type": "ownership_plus_free_complex_correction",
            "input": "concatenated_final_harmonic_and_percussive_features_and_carriers",
            "carrier": "original_fullband_mixture",
            "target_background_projection": False,
            "background": "exact_waveform_complement_mixture_minus_target",
            "branch_terminal_source_heads_used": False,
        }
        contract["architecture"]["harmonic_evidence"] = {
            "mode": self.harmonic_evidence_mode,
            "primary": "pitch_gated_hpss_harmonic_mixture",
            "auxiliary": (
                "ungated_hpss_harmonic_mixture_in_reserved_complex_source_slot"
                if self.harmonic_evidence_mode == "gated_plus_full"
                else None
            ),
        }
        contract["outputs"] = {
            "primary": "single fullband target waveform",
            "background": "mixture minus single fullband target waveform",
            "mixture_consistency": "exact in waveform space by construction",
        }
        contract["parameter_count"] = self.parameter_count()
        return contract

    def _complete_merged(
        self,
        harmonic_state: dict[str, Any],
        percussive_state: dict[str, Any],
        harmonic_hidden: torch.Tensor,
        percussive_hidden: torch.Tensor,
        full_mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frames = int(harmonic_state["frames"])
        if frames != int(percussive_state["frames"]):
            raise ValueError("Harmonic and percussive states have unequal frame counts")
        hidden_h = harmonic_hidden.transpose(-1, -2) * harmonic_state["first_conv_out"]
        hidden_p = percussive_hidden.transpose(-1, -2) * percussive_state["first_conv_out"]
        head_input = torch.cat(
            (
                harmonic_state["source_cws"],
                percussive_state["source_cws"],
                hidden_h,
                hidden_p,
            ),
            dim=1,
        )
        raw = self.harmonic_branch._cws_to_cac(
            self.merged_target_head(head_input)
        )[..., :frames]
        raw_residual = torch.complex(raw[:, 0].float(), raw[:, 1].float())
        ownership_logits = self.harmonic_branch._cws_to_cac(
            self.merged_ownership_head(head_input)
        )[:, 0, ..., :frames]

        samples = int(harmonic_state["samples"])
        valid_samples = harmonic_state["sample_support"].sum(dim=-1).long()
        denominator = valid_samples.float().clamp_min(1)[:, None]
        sample_support = harmonic_state["sample_support"]
        masked_mixture = full_mixture.float() * sample_support
        level = (
            masked_mixture.square().sum(dim=-1, keepdim=True) / denominator
        ).sqrt().clamp_min(1e-4)
        mixture_spectrum = self.harmonic_branch._stft(masked_mixture / level)
        frame_support = harmonic_state["frame_support"]
        processed_bins = int(self.config.processed_frequency_bins)
        processed_mixture = mixture_spectrum[..., :processed_bins, :]
        ownership = torch.sigmoid(ownership_logits) * frame_support[:, None]
        processed_target = ownership * processed_mixture + (
            raw_residual * frame_support[:, None]
        )
        target_spectrum = torch.zeros_like(mixture_spectrum)
        target_spectrum[..., :processed_bins, :] = processed_target
        target = torch.istft(
            target_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self.harmonic_branch._window(full_mixture.float()),
            center=True,
            length=samples,
        ) * level
        target = target * sample_support
        background = (masked_mixture - target) * sample_support
        spectrum_scale = level[:, :, None]
        return target, {
            "target": target,
            "background": background,
            "waveform_reconstruction": target + background,
            "waveform_reconstruction_error": target + background - masked_mixture,
            "mixture_spectrum": mixture_spectrum * spectrum_scale,
            "raw_target_spectrum": target_spectrum * spectrum_scale,
            "target_head_processed_spectrum": processed_target * spectrum_scale,
            "ownership_logits": ownership_logits,
            "ownership_mask": ownership,
            "frame_support": frame_support,
            "sample_support": sample_support,
            "level": level,
        }

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
        harmonic_state = self._prepare_branch(
            self.harmonic_branch,
            gated_harmonic,
            pitch,
            valid_samples,
            onset,
            conditioned_offset,
            auxiliary_carrier=(
                harmonic_mixture
                if self.harmonic_evidence_mode == "gated_plus_full"
                else None
            ),
        )
        percussive_state = self._prepare_branch(
            self.percussive_branch,
            percussive_mixture,
            pitch,
            valid_samples,
            onset,
            conditioned_offset,
        )
        harmonic = harmonic_state["hidden"]
        percussive = percussive_state["hidden"]
        harmonic_skips: list[torch.Tensor] = harmonic_state["encoder_outputs"]
        percussive_skips: list[torch.Tensor] = percussive_state["encoder_outputs"]

        last_encoder = len(self.harmonic_branch.encoder_blocks) - 1
        for index, (harmonic_block, percussive_block) in enumerate(
            zip(
                self.harmonic_branch.encoder_blocks,
                self.percussive_branch.encoder_blocks,
                strict=True,
            )
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
                    "deep_encoder", harmonic, percussive
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
        harmonic, percussive = self._exchange("bottleneck", harmonic, percussive)

        for index, (harmonic_block, percussive_block) in enumerate(
            zip(
                self.harmonic_branch.decoder_blocks,
                self.percussive_branch.decoder_blocks,
                strict=True,
            )
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
                    "deep_decoder", harmonic, percussive
                )

        full_mixture = harmonic_mixture + percussive_mixture
        estimate, merged = self._complete_merged(
            harmonic_state,
            percussive_state,
            harmonic,
            percussive,
            full_mixture,
        )
        return estimate, {
            "merged": merged,
            "background": merged["background"],
            "waveform_reconstruction": merged["waveform_reconstruction"],
            "harmonic_input": gated_harmonic,
            "harmonic_carrier": harmonic_mixture,
            "percussive_carrier": percussive_mixture,
            "conditioned_offset_samples": conditioned_offset,
            "conditioning_hold_extension_samples": offset_extension,
            "input_gate_strength": (
                torch.ones_like(pitch, dtype=harmonic_mixture.dtype)
                if input_gate_strength is None
                else input_gate_strength.to(harmonic_mixture.dtype)
            ),
            "alignment": self._alignment_diagnostics,
        }


__all__ = ["HARMONIC_EVIDENCE_MODES", "OccurrenceDualHPSSMergedOutput"]
