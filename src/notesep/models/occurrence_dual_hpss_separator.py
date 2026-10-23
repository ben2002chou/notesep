"""Two-branch occurrence separator for HPSS harmonic and percussive audio."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dna_residual_tfc_tdf_unet import DNAResidualTFCTDFUNetConfig
from .occurrence_joint_tfc_tdf_separator import OccurrenceJointTFCTDFSeparator
from .pitch_relative_harmonic_gate import (
    SOFT_HARMONIC_COMB_FLOOR,
    SOFT_HARMONIC_COMB_SIGMA_CENTS,
    GATE_MODES,
    gate_applies_to_target_and_prediction,
    pitch_relative_harmonic_gate,
)


_CONTRACT_KIND = "occurrence_dual_hpss_separator_v1"
HARMONIC_PARAMETERIZATIONS = {"ownership", "complex_mask"}
PERCUSSIVE_PARAMETERIZATIONS = {"ownership", "magnitude_mask", "complex_mask"}
CONDITIONING_MODES = {
    "harmonic_lifecycle",
    "rt2_target_roll",
    "rt2_onset_only",
}
OUTPUT_CARRIER_MODES = {"encoder_input", "ungated_hpss"}
COMPLEX_MASK_MODES = {"bounded", "unbounded"}


def _first_contract_difference(expected: object, actual: object, path: str = "contract") -> str:
    """Return one compact structural difference for strict checkpoint errors."""

    if type(expected) is not type(actual):
        return f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: missing={missing}, extra={extra}"
        for key in sorted(expected):
            difference = _first_contract_difference(
                expected[key], actual[key], f"{path}.{key}"
            )
            if difference:
                return difference
        return ""
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return f"{path}: expected length {len(expected)}, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_contract_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference:
                return difference
        return ""
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return ""


def normalized_occurrence_dual_hpss_contract(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize only documented, semantics-preserving legacy contract fields."""

    normalized = dict(raw)
    architecture = dict(normalized.get("architecture", {}))
    architecture.setdefault("harmonic_parameterization", "ownership")
    architecture.setdefault("percussive_parameterization", "ownership")
    # The first magnitude-mask checkpoints correctly identified the
    # percussive branch kind but retained the older ownership-only layout
    # label.  The label was descriptive rather than architectural; normalize
    # it only when the explicit branch parameterizations prove that this is
    # the known mixed legacy contract.
    if (
        architecture.get("layout")
        == "independent_hpss_harmonic_and_percussive_ownership_branches"
        and {
            architecture["harmonic_parameterization"],
            architecture["percussive_parameterization"],
        }
        != {"ownership"}
    ):
        architecture["layout"] = "independent_hpss_harmonic_and_percussive_branches"
    normalized["architecture"] = architecture
    return normalized


class OccurrenceDualHPSSSeparator(nn.Module):
    def __init__(
        self,
        preset: str | None = "30m",
        harmonic_gate_mode: str = "none",
        percussive_parameterization: str = "ownership",
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        *,
        config: DNAResidualTFCTDFUNetConfig | None = None,
        harmonic_parameterization: str = "ownership",
        output_carrier_mode: str = "encoder_input",
        complex_mask_mode: str = "bounded",
    ) -> None:
        super().__init__()
        if (preset is None) == (config is None):
            raise ValueError("exactly one of preset or config is required")
        self.preset = None if preset is None else str(preset)
        if harmonic_gate_mode not in GATE_MODES:
            raise ValueError(f"Unknown harmonic gate mode: {harmonic_gate_mode}")
        self.harmonic_gate_mode = str(harmonic_gate_mode)
        if harmonic_parameterization not in HARMONIC_PARAMETERIZATIONS:
            raise ValueError(
                f"Unknown harmonic parameterization: {harmonic_parameterization}"
            )
        self.harmonic_parameterization = str(harmonic_parameterization)
        if output_carrier_mode not in OUTPUT_CARRIER_MODES:
            raise ValueError(f"Unknown output carrier mode: {output_carrier_mode}")
        self.output_carrier_mode = str(output_carrier_mode)
        if complex_mask_mode not in COMPLEX_MASK_MODES:
            raise ValueError(f"Unknown complex mask mode: {complex_mask_mode}")
        if complex_mask_mode == "unbounded" and "complex_mask" not in {
            harmonic_parameterization,
            percussive_parameterization,
        }:
            raise ValueError("unbounded complex mask mode requires a complex-mask branch")
        self.complex_mask_mode = str(complex_mask_mode)
        if percussive_parameterization not in PERCUSSIVE_PARAMETERIZATIONS:
            raise ValueError(
                f"Unknown percussive parameterization: {percussive_parameterization}"
            )
        self.percussive_parameterization = str(percussive_parameterization)
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")
        if not 0 <= roll_hold_min_ms <= roll_hold_max_ms:
            raise ValueError("roll hold bounds must satisfy 0 <= min <= max")
        if conditioning_mode != "rt2_target_roll" and roll_hold_max_ms != 0:
            raise ValueError("roll hold extension requires rt2_target_roll conditioning")
        self.conditioning_mode = str(conditioning_mode)
        self.roll_hold_min_ms = float(roll_hold_min_ms)
        self.roll_hold_max_ms = float(roll_hold_max_ms)
        def branch_kwargs(parameterization: str) -> dict[str, bool]:
            return {
                "ownership_parameterization": parameterization == "ownership",
                "complex_mask_parameterization": parameterization == "complex_mask",
                "unbounded_complex_mask_parameterization": (
                    parameterization == "complex_mask"
                    and self.complex_mask_mode == "unbounded"
                ),
                "magnitude_mask_parameterization": parameterization == "magnitude_mask",
            }

        harmonic_kwargs = branch_kwargs(self.harmonic_parameterization)
        percussive_kwargs = branch_kwargs(self.percussive_parameterization)
        if config is None:
            assert self.preset is not None
            self.harmonic_branch = OccurrenceJointTFCTDFSeparator.from_preset(
                self.preset,
                conditioning_mode=self.conditioning_mode,
                **harmonic_kwargs,
            )
            self.percussive_branch = OccurrenceJointTFCTDFSeparator.from_preset(
                self.preset, conditioning_mode=self.conditioning_mode, **percussive_kwargs
            )
        else:
            self.harmonic_branch = OccurrenceJointTFCTDFSeparator(
                config,
                conditioning_mode=self.conditioning_mode,
                **harmonic_kwargs,
            )
            self.percussive_branch = OccurrenceJointTFCTDFSeparator(
                config, conditioning_mode=self.conditioning_mode, **percussive_kwargs
            )
        self.config = self.harmonic_branch.config

    @classmethod
    def from_preset(
        cls,
        preset: str,
        harmonic_gate_mode: str = "none",
        percussive_parameterization: str = "ownership",
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        harmonic_parameterization: str = "ownership",
        output_carrier_mode: str = "encoder_input",
        complex_mask_mode: str = "bounded",
    ) -> "OccurrenceDualHPSSSeparator":
        return cls(
            preset,
            harmonic_gate_mode,
            percussive_parameterization,
            conditioning_mode,
            roll_hold_min_ms,
            roll_hold_max_ms,
            harmonic_parameterization=harmonic_parameterization,
            output_carrier_mode=output_carrier_mode,
            complex_mask_mode=complex_mask_mode,
        )

    @classmethod
    def from_config(
        cls,
        config: DNAResidualTFCTDFUNetConfig,
        harmonic_gate_mode: str = "none",
        percussive_parameterization: str = "ownership",
        conditioning_mode: str = "rt2_target_roll",
        roll_hold_min_ms: float = 0.0,
        roll_hold_max_ms: float = 0.0,
        harmonic_parameterization: str = "ownership",
        output_carrier_mode: str = "encoder_input",
        complex_mask_mode: str = "bounded",
    ) -> "OccurrenceDualHPSSSeparator":
        """Build a strict non-preset model for native-rate canaries."""

        return cls(
            None,
            harmonic_gate_mode,
            percussive_parameterization,
            conditioning_mode,
            roll_hold_min_ms,
            roll_hold_max_ms,
            config=config,
            harmonic_parameterization=harmonic_parameterization,
            output_carrier_mode=output_carrier_mode,
            complex_mask_mode=complex_mask_mode,
        )

    def _condition_offset(
        self,
        onset: torch.Tensor,
        offset: torch.Tensor,
        valid_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conditioning_mode != "rt2_target_roll":
            return offset, torch.zeros_like(offset)
        scale = float(self.config.sample_rate) / 1000.0
        low = int(round(self.roll_hold_min_ms * scale))
        high = int(round(self.roll_hold_max_ms * scale))
        if self.training and high > low:
            extension = torch.randint(
                low, high + 1, offset.shape, device=offset.device
            )
        else:
            extension = torch.full_like(offset, (low + high) // 2)
        extension = torch.where(offset > onset, extension, torch.zeros_like(extension))
        conditioned = torch.minimum(offset + extension, valid_samples)
        return conditioned, conditioned - offset

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _gate_harmonic_mixture(
        self,
        harmonic_mixture: torch.Tensor,
        pitch: torch.Tensor,
        valid_samples: torch.Tensor,
        input_gate_strength: torch.Tensor | None,
    ) -> torch.Tensor:
        gated = pitch_relative_harmonic_gate(
            harmonic_mixture,
            pitch,
            valid_samples,
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            mode=self.harmonic_gate_mode,
        )
        if input_gate_strength is None:
            return gated
        if gate_applies_to_target_and_prediction(self.harmonic_gate_mode):
            raise ValueError(
                "input_gate_strength is only valid for input-only harmonic gates"
            )
        if input_gate_strength.shape != harmonic_mixture.shape[:1]:
            raise ValueError("input_gate_strength must have shape [batch]")
        strength = input_gate_strength.to(
            device=harmonic_mixture.device, dtype=harmonic_mixture.dtype
        )
        if not bool(torch.all((strength >= 0) & (strength <= 1))):
            raise ValueError("input_gate_strength must lie in [0, 1]")
        return torch.lerp(harmonic_mixture, gated, strength[:, None])

    def contract(self) -> dict[str, Any]:
        contract = {
            "kind": _CONTRACT_KIND,
            "contract_version": (
                6
                if "complex_mask" in {
                    self.harmonic_parameterization,
                    self.percussive_parameterization,
                }
                else 3
                if self.percussive_parameterization == "magnitude_mask"
                else (1 if self.harmonic_gate_mode == "none" else 2)
            ),
            "preset": self.preset,
            "architecture": {
                "layout": (
                    "independent_hpss_harmonic_and_percussive_ownership_branches"
                    if self.harmonic_parameterization == "ownership"
                    and self.percussive_parameterization == "ownership"
                    else "independent_hpss_harmonic_and_percussive_branches"
                ),
                "merge": "waveform_sum",
                "harmonic_branch": self.harmonic_branch.contract(self.preset),
                "percussive_branch": self.percussive_branch.contract(self.preset),
                "harmonic_parameterization": self.harmonic_parameterization,
                "percussive_parameterization": self.percussive_parameterization,
            },
            "inputs": {
                "harmonic_branch": "HPSS harmonic mixture",
                "percussive_branch": "ordinary margin-1 HPSS percussive mixture",
                "query": (
                    ["pitch", "onset_sample"]
                    if self.conditioning_mode == "rt2_onset_only"
                    else ["pitch", "onset_sample", "offset_sample"]
                ),
            },
            "outputs": {
                "harmonic": (
                    "HPSS harmonic target from a bounded complex ratio mask"
                    if self.harmonic_parameterization == "complex_mask"
                    else "HPSS harmonic target estimate"
                ),
                "percussive": (
                    "ordinary margin-1 HPSS percussive target magnitude mask "
                    "with input phase"
                    if self.percussive_parameterization == "magnitude_mask"
                    else (
                        "ordinary margin-1 HPSS percussive target from a "
                        "bounded complex ratio mask"
                    )
                    if self.percussive_parameterization == "complex_mask"
                    else "ordinary margin-1 HPSS percussive target estimate"
                ),
                "primary": "harmonic plus percussive target estimate",
            },
            "dna_conditioning": False,
            "parameter_count": self.parameter_count(),
        }
        if self.output_carrier_mode != "encoder_input":
            contract["architecture"]["output_carrier_mode"] = self.output_carrier_mode
            contract["inputs"].update(
                {
                    "harmonic_encoder_evidence": "pitch-gated HPSS harmonic mixture",
                    "harmonic_output_carrier": "ungated HPSS harmonic mixture",
                    "percussive_output_carrier": "ungated HPSS percussive mixture",
                }
            )
            contract["outputs"]["background"] = (
                "exact branch-carrier complements summing to original mixture"
            )
        if self.complex_mask_mode != "bounded":
            contract["architecture"]["complex_mask_mode"] = self.complex_mask_mode
            for branch_name, parameterization in (
                ("harmonic", self.harmonic_parameterization),
                ("percussive", self.percussive_parameterization),
            ):
                if parameterization == "complex_mask":
                    contract["outputs"][branch_name] = (
                        f"HPSS {branch_name} target from an unbounded complex ratio mask"
                    )
        if self.harmonic_gate_mode != "none":
            static_output_gate = gate_applies_to_target_and_prediction(
                self.harmonic_gate_mode
            )
            contract["harmonic_gate"] = {
                "mode": self.harmonic_gate_mode,
                "application": (
                    ["mixture_input", "target", "prediction"]
                    if static_output_gate else ["mixture_input"]
                ),
                "below_f0_cutoff_hz": "max(30, f0-60)" if self.harmonic_gate_mode == "below_f0" else None,
                "percussive_component": "unchanged ordinary margin-1 HPSS percussive",
            }
            if self.harmonic_gate_mode == "soft_harmonic_comb_input_only":
                contract["harmonic_gate"].update(
                    {
                        "shape": "nearest_integer_partial_gaussian_in_cents",
                        "sigma_cents": SOFT_HARMONIC_COMB_SIGMA_CENTS,
                        "minimum_gain": SOFT_HARMONIC_COMB_FLOOR,
                        "scope": "harmonic_encoder_input_only",
                    }
                )
            if (
                self.output_carrier_mode == "ungated_hpss"
                and not static_output_gate
            ):
                contract["harmonic_gate"]["application"] = ["encoder_evidence"]
        if self.conditioning_mode == "rt2_target_roll":
            contract["contract_version"] = max(4, int(contract["contract_version"]))
            contract["conditioning"] = {
                "type": "rt2_25hz_target_only_roll",
                "states": ["off", "held", "onset"],
                "explicit_lifecycle_labels": False,
                "signed_offset_distance": False,
                "training_positive_hold_ms": [
                    self.roll_hold_min_ms,
                    self.roll_hold_max_ms,
                ],
                "evaluation_positive_hold_ms": (
                    self.roll_hold_min_ms + self.roll_hold_max_ms
                ) / 2.0,
            }
        elif self.conditioning_mode == "rt2_onset_only":
            contract["contract_version"] = max(7, int(contract["contract_version"]))
            contract["conditioning"] = {
                "type": "rt2_25hz_onset_only_pitch",
                "states": ["onset"],
                "pitch_embedding_scope": "all_valid_frames",
                "offset_exposed_to_network": False,
                "held_state": False,
                "explicit_lifecycle_labels": False,
                "signed_offset_distance": False,
                "training_positive_hold_ms": [0.0, 0.0],
                "evaluation_positive_hold_ms": 0.0,
            }
        return contract

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
        zeros = torch.zeros_like(harmonic_mixture)
        conditioned_offset, offset_extension = self._condition_offset(
            onset, offset, valid_samples
        )
        gated_harmonic_mixture = self._gate_harmonic_mixture(
            harmonic_mixture, pitch, valid_samples, input_gate_strength
        )
        harmonic, harmonic_diagnostics = self.harmonic_branch(
            gated_harmonic_mixture, zeros, zeros, zeros,
            pitch, valid_samples, onset, conditioned_offset,
            output_carrier=(
                harmonic_mixture
                if self.output_carrier_mode == "ungated_hpss"
                else None
            ),
        )
        if gate_applies_to_target_and_prediction(self.harmonic_gate_mode):
            harmonic = pitch_relative_harmonic_gate(
                harmonic,
                pitch,
                valid_samples,
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                mode=self.harmonic_gate_mode,
            )
        percussive, percussive_diagnostics = self.percussive_branch(
            percussive_mixture, zeros, zeros, zeros,
            pitch, valid_samples, onset, conditioned_offset,
            output_carrier=(
                percussive_mixture
                if self.output_carrier_mode == "ungated_hpss"
                else None
            ),
        )
        estimate = harmonic + percussive
        background = (
            harmonic_diagnostics["background"]
            + percussive_diagnostics["background"]
        )
        return estimate, {
            "harmonic": harmonic,
            "percussive": percussive,
            "background": background,
            "waveform_reconstruction": estimate + background,
            "harmonic_input": gated_harmonic_mixture,
            "harmonic_carrier": (
                harmonic_mixture
                if self.output_carrier_mode == "ungated_hpss"
                else gated_harmonic_mixture
            ),
            "percussive_carrier": percussive_mixture,
            "harmonic_branch": harmonic_diagnostics,
            "percussive_branch": percussive_diagnostics,
            "conditioned_offset_samples": conditioned_offset,
            "conditioning_hold_extension_samples": offset_extension,
            "input_gate_strength": (
                torch.ones_like(pitch, dtype=harmonic_mixture.dtype)
                if input_gate_strength is None
                else input_gate_strength.to(harmonic_mixture.dtype)
            ),
        }


def occurrence_dual_hpss_separator_from_contract(raw: object) -> OccurrenceDualHPSSSeparator:
    if not isinstance(raw, dict) or raw.get("kind") != _CONTRACT_KIND:
        raise ValueError("Checkpoint lacks a dual HPSS contract")
    architecture = raw.get("architecture", {})
    communication = architecture.get("communication")
    if raw.get("preset") != "30m" or not isinstance(communication, dict):
        raise ValueError("Checkpoint does not use the released NoteSep architecture")
    if any(
        architecture.get(key) is not None
        for key in ("joint_output", "merged_output", "learned_loop")
    ):
        raise ValueError("Checkpoint does not use the released NoteSep architecture")
    if architecture.get("layout") != (
        "symmetric_low_resolution_aligned_hpss_harmonic_and_percussive_branches"
    ):
        raise ValueError("Checkpoint does not use the released NoteSep architecture")
    preset = raw["preset"]
    gate = raw.get("harmonic_gate") or {}
    mode = str(gate.get("mode", "none"))
    parameterization = str(
        raw.get("architecture", {}).get("percussive_parameterization", "ownership")
    )
    harmonic_parameterization = str(
        raw.get("architecture", {}).get("harmonic_parameterization", "ownership")
    )
    output_carrier_mode = str(
        raw.get("architecture", {}).get("output_carrier_mode", "encoder_input")
    )
    complex_mask_mode = str(
        raw.get("architecture", {}).get("complex_mask_mode", "bounded")
    )
    conditioning = raw.get("conditioning") or {}
    conditioning_mode = (
        "rt2_target_roll"
        if conditioning.get("type") == "rt2_25hz_target_only_roll"
        else (
            "rt2_onset_only"
            if conditioning.get("type") == "rt2_25hz_onset_only_pitch"
            else "harmonic_lifecycle"
        )
    )
    hold = conditioning.get("training_positive_hold_ms", [0.0, 0.0])
    if not isinstance(hold, list) or len(hold) != 2:
        raise ValueError("Dual HPSS contract has invalid roll hold bounds")
    expected_communication = {
        "type": "simultaneous_bidirectional_cross_attention",
        "rungs": ["deep_encoder", "bottleneck", "deep_decoder"],
        "attention_heads": communication.get("attention_heads"),
        "direction_parameters": "untied",
        "output_projection_initialization": "zeros",
        "message_fusion": "residual_add",
        "snapshot_policy": "both_directions_read_same_pre_fusion_states",
    }
    if communication != expected_communication:
        raise ValueError("Checkpoint has an unsupported NoteSep communication layout")
    from .occurrence_dual_hpss_symmetric_alignment import (
        OccurrenceDualHPSSSymmetricAlignment,
    )

    attention_heads = int(communication["attention_heads"])
    model = OccurrenceDualHPSSSymmetricAlignment.from_preset(
        preset,
        mode,
        parameterization,
        conditioning_mode,
        *map(float, hold),
        attention_heads=attention_heads,
        harmonic_parameterization=harmonic_parameterization,
        output_carrier_mode=output_carrier_mode,
        complex_mask_mode=complex_mask_mode,
    )
    # Ownership checkpoints written before the magnitude-mask ablation did not
    # spell out the (then implicit) percussive parameterization.  Normalize
    # that one additive field while preserving strict equality for everything
    # else, so old frozen checkpoints remain loadable without weakening the
    # contract check.
    normalized = normalized_occurrence_dual_hpss_contract(raw)
    reconstructed = model.contract()
    if reconstructed != normalized:
        difference = _first_contract_difference(normalized, reconstructed)
        raise ValueError(f"Dual HPSS checkpoint contract changed: {difference}")
    return model


__all__ = [
    "OccurrenceDualHPSSSeparator",
    "HARMONIC_PARAMETERIZATIONS",
    "PERCUSSIVE_PARAMETERIZATIONS",
    "OUTPUT_CARRIER_MODES",
    "COMPLEX_MASK_MODES",
    "normalized_occurrence_dual_hpss_contract",
    "occurrence_dual_hpss_separator_from_contract",
]
