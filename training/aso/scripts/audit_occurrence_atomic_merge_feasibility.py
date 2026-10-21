#!/usr/bin/env python3
"""Falsify or support strict acoustic-atom merging before training a model.

The audit uses complete held-out note banks.  At several complex-STFT
resolutions it gives an oracle merger either individual time-frequency bins or
small rectangular tracklets, assigns each whole atom to exactly one note (or
background), and compares that strict partition with the mature occurrence
separator.  If even the one-bin oracle cannot materially beat the separator,
learning a source-agnostic atomizer and a score-conditioned merger is not a
promising next experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_v13_streaming import FrozenV13Contract  # noqa: E402
from model_exp.occurrence_synthetic_note_bank import (  # noqa: E402
    CompleteSyntheticBankReader,
    select_complete_mixture_groups,
)
import scripts.train_dna_residual_v13_streaming as synthetic_corpus  # noqa: E402
from scripts.train_occurrence_joint_note_bank import (  # noqa: E402
    _forward_bank_chunked,
    _synthetic_bank_to_device,
)
from scripts.train_occurrence_score_remix_ssl import (  # noqa: E402
    atomic_json,
    file_sha256,
    load_dual_hpss_checkpoint,
)


SCHEMA = "occurrence_atomic_merge_feasibility_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--grouped-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-groups", type=int, default=8)
    parser.add_argument("--candidate-group-pool", type=int, default=128)
    parser.add_argument("--maximum-true-residual-ratio", type=float, default=1e-3)
    parser.add_argument("--crop-samples", type=int, default=49_152)
    parser.add_argument("--maximum-sources", type=int, default=64)
    parser.add_argument("--query-microbatch", type=int, default=4)
    parser.add_argument("--n-ffts", type=int, nargs="+", default=(256, 512, 1024, 2048))
    parser.add_argument(
        "--atom-tiles",
        type=str,
        nargs="+",
        default=("1x1", "3x4", "7x8"),
        help="frequency_bins x time_frames; 1x1 is the strict oracle ceiling",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def parse_tiles(values: list[str] | tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    output: list[tuple[int, int]] = []
    for value in values:
        pieces = value.lower().split("x")
        if len(pieces) != 2:
            raise ValueError(f"invalid atom tile {value!r}")
        frequency, time = map(int, pieces)
        if frequency < 1 or time < 1:
            raise ValueError("atom tile dimensions must be positive")
        output.append((frequency, time))
    if (1, 1) not in output:
        raise ValueError("the audit requires a 1x1 strict ceiling")
    return tuple(dict.fromkeys(output))


def stft(values: torch.Tensor, n_fft: int) -> torch.Tensor:
    return torch.stft(
        values.float(),
        n_fft=n_fft,
        hop_length=n_fft // 4,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=values.device),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )


def istft(values: torch.Tensor, n_fft: int, length: int) -> torch.Tensor:
    return torch.istft(
        values,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=values.device),
        center=True,
        length=length,
    )


def strict_atom_partition(
    mixture: torch.Tensor,
    targets_with_background: torch.Tensor,
    tile: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign every complete mixture atom to one oracle-selected source."""

    sources, frequency, frames = targets_with_background.shape
    frequency_tile, time_tile = tile
    energy = targets_with_background.abs().square()
    pad_frequency = (-frequency) % frequency_tile
    pad_time = (-frames) % time_tile
    padded = F.pad(energy, (0, pad_time, 0, pad_frequency))
    pooled = F.avg_pool2d(
        padded[:, None],
        kernel_size=(frequency_tile, time_tile),
        stride=(frequency_tile, time_tile),
    )[:, 0]
    winners = pooled.argmax(dim=0)
    winners = winners.repeat_interleave(frequency_tile, dim=0).repeat_interleave(
        time_tile, dim=1
    )[:frequency, :frames]
    assignment = F.one_hot(winners, num_classes=sources).permute(2, 0, 1).bool()
    output = mixture[None] * assignment
    return output, winners


def soft_energy_partition(
    mixture: torch.Tensor, targets_with_background: torch.Tensor
) -> torch.Tensor:
    energy = targets_with_background.abs().square()
    weights = energy / energy.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return weights * mixture[None]


def source_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    pitches: torch.Tensor,
    *,
    target_waveform: torch.Tensor,
    n_fft: int,
) -> dict[str, float]:
    """Energy-stable note metrics; background is deliberately excluded."""

    if estimate.shape != target.shape or estimate.ndim != 3:
        raise ValueError("note spectra must share [J,F,T]")
    error_energy = (estimate - target).abs().square().sum()
    target_energy = target.abs().square().sum().clamp_min(1e-12)
    target_inner = (estimate * target.conj()).real.flatten(1).sum(dim=1)
    per_target_energy = target.abs().square().flatten(1).sum(dim=1).clamp_min(1e-12)
    gains = target_inner / per_target_energy
    orthogonal = estimate - gains[:, None, None] * target
    output_energy = estimate.abs().square().flatten(1).sum(dim=1).clamp_min(1e-12)

    normalized_estimate = estimate.flatten(1) / output_energy.sqrt()[:, None]
    normalized_target = target.flatten(1) / per_target_energy.sqrt()[:, None]
    similarities = (normalized_estimate[:, None] * normalized_target[None].conj()).real.sum(
        dim=-1
    )
    diagonal = similarities.diagonal()
    correct = similarities.argmax(dim=1) == torch.arange(
        estimate.shape[0], device=estimate.device
    )
    competitors = similarities.masked_fill(
        torch.eye(estimate.shape[0], dtype=torch.bool, device=estimate.device),
        -torch.inf,
    ).amax(dim=1)
    pitch_distance = (pitches[:, None] - pitches[None]).abs()
    octave = (pitch_distance == 12) | (pitch_distance == 24)
    has_octave = octave.any(dim=1)
    octave_competitor = similarities.masked_fill(~octave, -torch.inf).amax(dim=1)
    octave_margin = diagonal[has_octave] - octave_competitor[has_octave]

    estimate_waveform = istft(estimate, n_fft, target_waveform.shape[-1])
    waveform_error = (estimate_waveform - target_waveform).square().sum()
    waveform_energy = target_waveform.square().sum().clamp_min(1e-12)
    mixture_floor = 0.03 * mixture.abs().mean()
    logmag = (
        torch.log1p(20.0 * estimate.abs()) - torch.log1p(20.0 * target.abs())
    ).abs().mean()
    active_bin = target.abs() >= mixture_floor

    return {
        "note_complex_nmse": float(error_energy / target_energy),
        "note_waveform_nmse": float(waveform_error / waveform_energy),
        "note_log_magnitude_l1": float(logmag),
        "note_projection_gain": float(gains.mean()),
        "note_orthogonal_ratio": float(orthogonal.abs().square().sum() / target_energy),
        "pitch_assignment_accuracy": float(correct.float().mean()),
        "pitch_assignment_margin": float((diagonal - competitors).mean()),
        "octave_margin": float(octave_margin.mean()) if bool(has_octave.any()) else float("nan"),
        "active_target_bin_fraction": float(active_bin.float().mean()),
        "mixture_sum_error_max_abs": 0.0,
    }


def atom_purity_metrics(targets_with_background: torch.Tensor) -> dict[str, float]:
    energy = targets_with_background.abs().square()
    total = energy.sum(dim=0)
    maximum = energy.amax(dim=0)
    purity = maximum / total.clamp_min(1e-12)
    weight = total / total.sum().clamp_min(1e-12)
    second = energy.topk(k=min(2, energy.shape[0]), dim=0).values
    overlap_ratio = (
        second[1] / second[0].clamp_min(1e-12)
        if second.shape[0] == 2
        else torch.zeros_like(total)
    )
    return {
        "energy_weighted_purity": float((weight * purity).sum()),
        "energy_fraction_purity_ge_0_90": float(weight[purity >= 0.90].sum()),
        "energy_fraction_purity_ge_0_75": float(weight[purity >= 0.75].sum()),
        "energy_fraction_competitor_ge_0_25": float(weight[overlap_ratio >= 0.25].sum()),
    }


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    return {
        method: {
            key: finite_mean([float(row[key]) for row in method_rows])
            for key in method_rows[0]
            if key
            not in {
                "method",
                "mixture_id",
                "pitches",
                "source_count",
                "n_fft",
                "atom_tile",
            }
            and isinstance(method_rows[0][key], (float, int))
        }
        for method, method_rows in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    tiles = parse_tiles(args.atom_tiles)
    if (
        args.validation_groups < 1
        or args.candidate_group_pool < args.validation_groups
        or args.crop_samples < 16_384
        or not 0 <= args.maximum_true_residual_ratio <= 1
    ):
        raise ValueError("invalid audit size")
    if any(n_fft < 64 or n_fft % 4 for n_fft in args.n_ffts):
        raise ValueError("n-ffts must be multiples of four and at least 64")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    frozen = FrozenV13Contract.load(args.frozen_manifest)
    _, validation_ordinals, view_binding = synthetic_corpus.load_synthetic_ordinal_view(
        args.grouped_view, contract=frozen
    )
    if validation_ordinals is None or view_binding is None:
        raise ValueError("grouped view has no held-out split")
    groups = select_complete_mixture_groups(
        frozen,
        validation_ordinals,
        count=args.candidate_group_pool,
        seed=args.seed,
    )
    reader = CompleteSyntheticBankReader(
        frozen,
        fixed_samples=args.crop_samples,
        maximum_sources=args.maximum_sources,
        cache_size=4,
    )
    backbone, initializer_provenance = load_dual_hpss_checkpoint(
        args.initializer, device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    contract = {
        "schema": SCHEMA,
        "purpose": "no-training strict acoustic-atom merge ceiling",
        "initializer": initializer_provenance,
        "initializer_sha256": file_sha256(args.initializer),
        "frozen_manifest": str(Path(args.frozen_manifest).resolve()),
        "grouped_view": view_binding,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "atom_contract": {
            "representation": "complex_STFT_with_original_mixture_phase",
            "strict": "every_atom_is_assigned_wholly_to_one_note_or_background",
            "one_by_one": "most_favorable_strict_partition_in_this_representation",
            "background": "mixture_minus_sum_complete_ground_truth_note_bank",
            "learned_parameters": 0,
        },
        "source": str(Path(__file__).resolve()),
        "source_sha256": file_sha256(Path(__file__).resolve()),
    }
    atomic_json(contract, output / "contract.json")
    rows: list[dict[str, Any]] = []
    rows_path = output / "rows.jsonl"

    try:
        accepted = 0
        for candidate_index, (mixture_id, group_rows) in enumerate(groups):
            try:
                example = reader.load(
                    mixture_id, group_rows, crop_seed=args.seed + candidate_index
                )
            except ValueError as error:
                print(
                    json.dumps(
                        {
                            "candidate": candidate_index + 1,
                            "mixture_id": mixture_id,
                            "skip": str(error),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            if example["true_residual_ratio"] > args.maximum_true_residual_ratio:
                print(
                    json.dumps(
                        {
                            "candidate": candidate_index + 1,
                            "mixture_id": mixture_id,
                            "skip": "incomplete_target_bank",
                            "true_residual_ratio": float(example["true_residual_ratio"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            accepted += 1
            bank = _synthetic_bank_to_device(example, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                teacher_waveform = _forward_bank_chunked(
                    backbone, bank, chunk_size=args.query_microbatch
                ).float()
            mixture_waveform = bank["mixture"][0].float()
            target_waveform = bank["targets"].float()
            pitches = bank["pitch"].long()

            for n_fft in args.n_ffts:
                mixture_spectrum = stft(mixture_waveform, n_fft)
                target_spectrum = stft(target_waveform, n_fft)
                background_target = mixture_spectrum - target_spectrum.sum(dim=0)
                target_bank = torch.cat((target_spectrum, background_target[None]), dim=0)
                purity = atom_purity_metrics(target_bank)
                teacher_spectrum = stft(teacher_waveform, n_fft)
                teacher_background = mixture_spectrum - teacher_spectrum.sum(dim=0)
                teacher = source_metrics(
                    teacher_spectrum,
                    target_spectrum,
                    mixture_spectrum,
                    pitches,
                    target_waveform=target_waveform,
                    n_fft=n_fft,
                )
                teacher["mixture_sum_error_max_abs"] = float(
                    (teacher_spectrum.sum(dim=0) + teacher_background - mixture_spectrum)
                    .abs()
                    .amax()
                )
                common = {
                    "mixture_id": mixture_id,
                    "source_count": int(target_spectrum.shape[0]),
                    "pitches": [int(value) for value in pitches.cpu().tolist()],
                    "n_fft": int(n_fft),
                    "true_residual_ratio": float(example["true_residual_ratio"]),
                    **purity,
                }
                teacher_row = {**common, "method": f"teacher_nfft{n_fft}", **teacher}
                rows.append(teacher_row)

                soft_bank = soft_energy_partition(mixture_spectrum, target_bank)
                soft = source_metrics(
                    soft_bank[:-1],
                    target_spectrum,
                    mixture_spectrum,
                    pitches,
                    target_waveform=target_waveform,
                    n_fft=n_fft,
                )
                soft["mixture_sum_error_max_abs"] = float(
                    (soft_bank.sum(dim=0) - mixture_spectrum).abs().amax()
                )
                rows.append({**common, "method": f"soft_energy_nfft{n_fft}", **soft})

                for tile in tiles:
                    strict_bank, _ = strict_atom_partition(
                        mixture_spectrum, target_bank, tile
                    )
                    strict = source_metrics(
                        strict_bank[:-1],
                        target_spectrum,
                        mixture_spectrum,
                        pitches,
                        target_waveform=target_waveform,
                        n_fft=n_fft,
                    )
                    strict["mixture_sum_error_max_abs"] = float(
                        (strict_bank.sum(dim=0) - mixture_spectrum).abs().amax()
                    )
                    label = f"strict_{tile[0]}x{tile[1]}_nfft{n_fft}"
                    rows.append(
                        {**common, "method": label, "atom_tile": list(tile), **strict}
                    )
            with rows_path.open("a", encoding="utf-8") as handle:
                for row in rows[-(len(args.n_ffts) * (2 + len(tiles))) :]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "group": accepted,
                        "groups": args.validation_groups,
                        "mixture_id": mixture_id,
                        "sources": int(target_waveform.shape[0]),
                        "true_residual_ratio": float(example["true_residual_ratio"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if accepted == args.validation_groups:
                break
    finally:
        reader.close()

    if accepted != args.validation_groups:
        raise RuntimeError(
            f"candidate pool yielded {accepted}/{args.validation_groups} complete banks"
        )

    summary = aggregate(rows)
    candidates: list[dict[str, Any]] = []
    for n_fft in args.n_ffts:
        teacher = summary[f"teacher_nfft{n_fft}"]
        strict = summary[f"strict_1x1_nfft{n_fft}"]
        candidates.append(
            {
                "n_fft": n_fft,
                "teacher_complex_nmse": teacher["note_complex_nmse"],
                "strict_complex_nmse": strict["note_complex_nmse"],
                "relative_complex_improvement": (
                    teacher["note_complex_nmse"] - strict["note_complex_nmse"]
                )
                / max(teacher["note_complex_nmse"], 1e-12),
                "pitch_accuracy_gain": strict["pitch_assignment_accuracy"]
                - teacher["pitch_assignment_accuracy"],
                "octave_margin_gain": strict["octave_margin"] - teacher["octave_margin"],
                "energy_weighted_purity": strict["energy_weighted_purity"],
                "energy_fraction_purity_ge_0_90": strict[
                    "energy_fraction_purity_ge_0_90"
                ],
            }
        )
    best = min(candidates, key=lambda value: value["strict_complex_nmse"])
    decision = {
        "status": "worth_learning_merger" if (
            best["relative_complex_improvement"] >= 0.25
            and best["pitch_accuracy_gain"] >= 0.05
            and best["energy_fraction_purity_ge_0_90"] >= 0.70
        ) else "strict_stft_atoms_not_yet_sufficient",
        "gate": {
            "relative_complex_improvement_min": 0.25,
            "pitch_accuracy_gain_min": 0.05,
            "energy_fraction_purity_ge_0_90_min": 0.70,
        },
        "best_one_bin_resolution": best,
        "interpretation": (
            "This is an oracle ceiling, not model performance. Passing only licenses a learned "
            "atomizer/merger canary; failing closes strict partitioning in these STFT spaces, "
            "not all possible learned invertible atom spaces."
        ),
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "groups": accepted,
        "rows": len(rows),
        "summary": summary,
        "resolution_comparison": candidates,
        "decision": decision,
    }
    atomic_json(result, output / "result.json")
    print("RESULT " + json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
