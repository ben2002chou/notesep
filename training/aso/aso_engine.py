"""Shared loss, evaluation, and checkpoint helpers for the selected ASO recipe."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
import aso_data as scaled
from model_exp.temporal_hybrid_set_ownership import TemporalHybridSetOwnership

# Set by the selected trainer before calls to evaluate or train_loss.
call = None
objective = None
ENERGY_FLOOR = 1e-8
DISTORTION_FLOOR = 1e-4
REGRET_MARGIN_DB = 0.10
MEAN_REGRET_WEIGHT = 2.0
SQUARED_REGRET_WEIGHT = 0.25


def stable_log_si_distortion(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_energy = target.square().sum(-1)
    if bool((target_energy <= 1e-12).any()):
        raise ValueError("silent target passed to SI-regret objective")
    estimate_energy = value.square().sum(-1)
    dot = (value * target).sum(-1)
    rho2 = dot.square() / (
        (estimate_energy + ENERGY_FLOOR * target_energy) * target_energy
    ).clamp_min(1e-20)
    rho2 = rho2.clamp(0.0, 1.0)
    distortion = (1.0 - rho2 + DISTORTION_FLOOR) / (rho2 + DISTORTION_FLOOR)
    return torch.log(distortion)


def supervised_nonsilent(record: dict[str, Any]) -> torch.Tensor:
    target_energy = record["targets_wave"].float().square().sum(-1)
    active = record["query_valid"] & (target_energy > 1e-12)
    if not bool(active.any()):
        raise ValueError("training scene contains no nonsilent target")
    return active

def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def train_loss(model: TemporalHybridSetOwnership, record: dict[str, Any]):
    candidate_spec, diagnostics = call(model, record)
    waveform_risk, risk_diagnostics = objective(candidate_spec, record)
    target_magnitude = record["targets"].abs()
    target_share = target_magnitude / target_magnitude.sum(0, keepdim=True).clamp_min(1e-8)
    energy = record["mixture"].abs().square()
    energy = energy / energy.sum().clamp_min(1e-12)
    ownership = diagnostics["ownership"][0].clamp_min(1e-8)
    ownership_loss = -(target_share * ownership.log()).sum(0).mul(energy).sum()
    smoothness = diagnostics["smoothness_penalty"]
    loss = waveform_risk + ownership_loss + 0.01 * smoothness
    return loss, diagnostics | {
        "waveform_risk": waveform_risk,
        "ownership_loss": ownership_loss,
        "smoothness_loss": smoothness,
        **{f"risk_{key}": value for key, value in risk_diagnostics.items()},
    }

def summarize(events: list[dict[str, Any]], method: str) -> dict[str, Any]:
    result = {
        metric: {
            "median": float(np.median([row[method][metric] for row in events])),
            "mean": float(np.mean([row[method][metric] for row in events])),
        }
        for metric in ("si_sdr_db", "off_e", "gain", "scale_invariant_distortion")
    }
    if "post_offset_energy_ratio" in events[0][method]:
        result["post_offset_energy_ratio"] = {
            "median": float(np.median([row[method]["post_offset_energy_ratio"] for row in events])),
            "mean": float(np.mean([row[method]["post_offset_energy_ratio"] for row in events])),
        }
    return result

def compare(events: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, Any]:
    delta = np.asarray([
        row[candidate]["si_sdr_db"] - row[baseline]["si_sdr_db"]
        for row in events
    ])
    return {
        "median_db": float(np.median(delta)),
        "mean_db": float(np.mean(delta)),
        "wins_gt_0p1db": int((delta > 0.1).sum()),
        "losses_gt_0p1db": int((delta < -0.1).sum()),
        "wins_gt_1db": int((delta > 1.0).sum()),
        "losses_gt_1db": int((delta < -1.0).sum()),
        "wins_gt_3db": int((delta > 3.0).sum()),
        "losses_gt_3db": int((delta < -3.0).sum()),
    }

@torch.no_grad()
def evaluate(
    model: TemporalHybridSetOwnership,
    authorities: scaled.Authorities,
    maximum_mixtures: int,
    parent: TemporalHybridSetOwnership | None = None,
) -> dict[str, Any]:
    model.eval()
    if parent is not None:
        parent.eval()
    events: list[dict[str, Any]] = []
    closure = 0.0
    complex_closure = 0.0
    magnitude_closure_relative = 0.0
    complex_closure_relative = 0.0
    maximum_mean_complex_delta = 0.0
    silent = 0
    for kind in ("fresh", "recorded"):
        count = len(authorities.ids(kind, "validation"))
        if maximum_mixtures:
            count = min(count, maximum_mixtures)
        for index in range(count):
            record = authorities.record(kind, "validation", index, torch.device("cuda"))
            if len(record["raw"]) == 1:
                if (
                    model.config.exact_mixture_additivity
                    and model.config.magnitude_reference == "mixture"
                    and model.config.phase_mode == "mixture"
                ):
                    estimate = record["mixture"][None, None]
                elif model.config.magnitude_reference == "raw_total":
                    mix_mag = record["mixture"].abs()
                    mix_unit = torch.where(
                        mix_mag > model.config.epsilon,
                        record["mixture"] / mix_mag.clamp_min(model.config.epsilon),
                        torch.ones_like(record["mixture"]),
                    )
                    estimate = (record["raw"].abs() * mix_unit[None])[None]
                else:
                    estimate = record["base"][None]
                closure_value = 0.0
                complex_closure_value = 0.0
                magnitude_closure_relative_value = 0.0
                complex_closure_relative_value = 0.0
                mean_complex_delta_value = 0.0
            else:
                estimate, diagnostics = call(model, record)
                closure_value = float(diagnostics["magnitude_closure_max_abs"])
                complex_closure_value = float(
                    diagnostics.get(
                        "complex_closure_max_abs",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(float("nan")),
                    )
                )
                magnitude_closure_relative_value = float(
                    diagnostics.get(
                        "magnitude_closure_max_relative",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(float("nan")),
                    )
                )
                complex_closure_relative_value = float(
                    diagnostics.get(
                        "complex_closure_max_relative",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(float("nan")),
                    )
                )
                mean_complex_delta_value = float(
                    diagnostics.get(
                        "mean_complex_delta",
                        diagnostics["magnitude_closure_max_abs"].new_tensor(0.0),
                    )
                )
            parent_estimate = None
            if parent is not None:
                parent_estimate = (
                    (
                        record["mixture"][None, None]
                        if parent.config.exact_mixture_additivity
                        and parent.config.magnitude_reference == "mixture"
                        and parent.config.phase_mode == "mixture"
                        else record["base"][None]
                    )
                    if len(record["raw"]) == 1
                    else call(parent, record)[0]
                )
            estimate_wave = scaled.istft(estimate[0], record["length"]).float()
            parent_wave = (
                scaled.istft(parent_estimate[0], record["length"]).float()
                if parent_estimate is not None else None
            )
            base_wave = scaled.istft(record["base"], record["length"]).float()
            target_energy = record["targets_wave"].square().sum(-1)
            valid = record["query_valid"] & (target_energy > 1e-12)
            silent += int((record["query_valid"] & ~valid).sum())
            indices = torch.where(valid)[0]
            if not len(indices):
                continue
            raw_metrics = scaled.metrics(
                record["targets_wave"][indices], record["raw_wave"][indices]
            )
            candidate_metrics = scaled.metrics(
                record["targets_wave"][indices], estimate_wave[indices]
            )
            base_metrics = scaled.metrics(
                record["targets_wave"][indices], base_wave[indices]
            )
            parent_metrics = (
                scaled.metrics(record["targets_wave"][indices], parent_wave[indices])
                if parent_wave is not None else None
            )
            method_waves = {
                "candidate": estimate_wave,
                "magnitude": base_wave,
                "raw": record["raw_wave"],
                **({"parent": parent_wave} if parent_wave is not None else {}),
            }
            positions = torch.arange(record["length"], device=record["offsets"].device)[None]
            post_mask = positions >= record["offsets"][:, None]
            target_energy_safe = record["targets_wave"].double().square().sum(-1).clamp_min(1e-12)
            post_ratios = {
                method: (wave.double().square() * post_mask).sum(-1) / target_energy_safe
                for method, wave in method_waves.items()
            }
            for local_note, note in enumerate(indices.tolist()):
                events.append({
                    "dataset_kind": kind,
                    "mixture_id": record["mixture_id"],
                    "note": int(note),
                    **{
                        method: {
                            **{key: float(metric[key][local_note]) for key in metric},
                            "post_offset_energy_ratio": float(post_ratios[method][note]),
                        }
                        for method, metric in {
                            "candidate": candidate_metrics,
                            "magnitude": base_metrics,
                            "raw": raw_metrics,
                            **({"parent": parent_metrics} if parent_metrics is not None else {}),
                        }.items()
                    },
                })
            closure = max(closure, closure_value)
            if np.isfinite(complex_closure_value):
                complex_closure = max(complex_closure, complex_closure_value)
            if np.isfinite(magnitude_closure_relative_value):
                magnitude_closure_relative = max(
                    magnitude_closure_relative, magnitude_closure_relative_value
                )
            if np.isfinite(complex_closure_relative_value):
                complex_closure_relative = max(
                    complex_closure_relative, complex_closure_relative_value
                )
            if np.isfinite(mean_complex_delta_value):
                maximum_mean_complex_delta = max(
                    maximum_mean_complex_delta, mean_complex_delta_value
                )
    if not events:
        raise ValueError("empty development evaluation")
    methods = ("candidate", "magnitude", "raw", "parent") if parent is not None else ("candidate", "magnitude", "raw")
    comparisons = {
        "candidate_minus_magnitude": compare(events, "candidate", "magnitude"),
        "candidate_minus_raw": compare(events, "candidate", "raw"),
    }
    if parent is not None:
        comparisons.update({
            "candidate_minus_parent": compare(events, "candidate", "parent"),
            "parent_minus_magnitude": compare(events, "parent", "magnitude"),
            "parent_minus_raw": compare(events, "parent", "raw"),
        })
    return {
        "coverage": len(events),
        "silent_targets_excluded": silent,
        "summary": {
            method: summarize(events, method)
            for method in methods
        },
        "comparisons": comparisons,
        "maximum_magnitude_closure_abs": closure,
        "maximum_complex_closure_abs": complex_closure,
        "maximum_magnitude_closure_relative": magnitude_closure_relative,
        "maximum_complex_closure_relative": complex_closure_relative,
        "maximum_mean_complex_delta": maximum_mean_complex_delta,
        "events": events,
    }
