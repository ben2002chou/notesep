#!/usr/bin/env python3
"""Causal no-training audit of sparse sharing for STFT atom ownership."""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
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
from model_exp.occurrence_atomic_merger import (  # noqa: E402
    OccurrenceAtomicMerger,
    OccurrenceAtomicMergerConfig,
)
import scripts.train_dna_residual_v13_streaming as synthetic_corpus  # noqa: E402
from scripts.audit_occurrence_atomic_merge_feasibility import (  # noqa: E402
    finite_mean,
    istft,
    soft_energy_partition,
    source_metrics,
    strict_atom_partition,
)
from scripts.render_occurrence_stft_atomic_separation_listening import (  # noqa: E402
    shared_gain,
    write_wav,
)
from scripts.train_occurrence_atomic_merger_canary import (  # noqa: E402
    build_record,
    select_examples,
)
from scripts.train_occurrence_score_remix_ssl import (  # noqa: E402
    atomic_json,
    file_sha256,
    load_dual_hpss_checkpoint,
)


SCHEMA = "occurrence_atomic_sparse_allocation_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--listening-cases", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def expand_weights(
    values: torch.Tensor,
    *,
    frequency: int,
    frames: int,
    tile: tuple[int, int],
) -> torch.Tensor:
    return values.repeat_interleave(tile[0], dim=1).repeat_interleave(
        tile[1], dim=2
    )[:, :frequency, :frames]


def sparsemax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Euclidean projection onto the simplex along the source dimension."""

    values = logits / temperature
    sorted_values, _ = values.sort(dim=0, descending=True)
    cumulative = sorted_values.cumsum(dim=0)
    rank = torch.arange(
        1,
        values.shape[0] + 1,
        device=values.device,
        dtype=values.dtype,
    )[:, None, None]
    support = 1.0 + rank * sorted_values > cumulative
    support_size = support.sum(dim=0).clamp_min(1)
    threshold_sum = cumulative.gather(
        0, (support_size - 1)[None]
    )[0]
    threshold = (threshold_sum - 1.0) / support_size.to(values.dtype)
    probabilities = (values - threshold[None]).clamp_min(0.0)
    return probabilities / probabilities.sum(dim=0, keepdim=True).clamp_min(1e-8)


def top2_softmax(logits: torch.Tensor) -> torch.Tensor:
    top_values, top_indices = logits.topk(k=min(2, logits.shape[0]), dim=0)
    top_weights = top_values.softmax(dim=0)
    output = torch.zeros_like(logits)
    output.scatter_(0, top_indices, top_weights)
    return output


def confidence_top2(logits: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
    probabilities = logits.softmax(dim=0)
    top_values, top_indices = probabilities.topk(k=min(2, logits.shape[0]), dim=0)
    top_values = top_values / top_values.sum(dim=0, keepdim=True).clamp_min(1e-8)
    confident = top_values[0] >= threshold
    top_values[0] = torch.where(confident, torch.ones_like(top_values[0]), top_values[0])
    if top_values.shape[0] > 1:
        top_values[1] = torch.where(
            confident, torch.zeros_like(top_values[1]), top_values[1]
        )
    output = torch.zeros_like(logits)
    output.scatter_(0, top_indices, top_values)
    return output


def allocation_methods(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    hard = F.one_hot(
        logits.argmax(dim=0), num_classes=logits.shape[0]
    ).permute(2, 0, 1).to(logits.dtype)
    return {
        "learned_hard": hard,
        "softmax_t1": logits.softmax(dim=0),
        "top2_softmax_t1": top2_softmax(logits),
        "confidence_top2_085": confidence_top2(logits),
        "sparsemax_t1": sparsemax(logits, 1.0),
        "sparsemax_t2": sparsemax(logits, 2.0),
    }


def allocation_statistics(weights: torch.Tensor, mixture: torch.Tensor) -> dict[str, float]:
    energy = mixture.abs().square()
    normalized_energy = energy / energy.sum().clamp_min(1e-12)
    support = (weights > 1e-6).sum(dim=0).float()
    entropy = -(
        weights.clamp_min(1e-12).log() * weights
    ).sum(dim=0)
    return {
        "energy_weighted_support": float((normalized_energy * support).sum()),
        "energy_fraction_shared": float(normalized_energy[support > 1].sum()),
        "energy_weighted_entropy": float((normalized_energy * entropy).sum()),
        "mixture_sum_error_max_abs": float(
            (weights.sum(dim=0) * mixture - mixture).abs().amax()
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    return {
        method: {
            key: finite_mean([float(row[key]) for row in method_rows])
            for key in method_rows[0]
            if key not in {"method", "mixture_id"}
            and isinstance(method_rows[0][key], (float, int))
        }
        for method, method_rows in sorted(grouped.items())
    }


def waveform_nmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (estimate.float() - target.float()).square().sum()
        / target.float().square().sum().clamp_min(1e-12)
    )


def render_listening(output: Path, cases: list[dict[str, Any]]) -> None:
    labels = {
        "target": "isolated ground-truth note",
        "teacher": "gated 30k separator",
        "learned_hard": "hard learned ownership",
        "confidence_top2_085": "share top two only when confidence < 0.85",
        "sparsemax_t1": "sparsemax allocation, T=1",
        "sparsemax_t2": "sparsemax allocation, T=2",
        "oracle_soft": "GROUND-TRUTH-ASSISTED soft oracle",
    }
    order = tuple(labels)
    cards: list[str] = []
    for case_index, case in enumerate(cases):
        prefix = f"case_{case_index:02d}_p{case['pitch']}"
        players = [
            f'<div class="arm"><span>mixture context (separately normalized)</span><audio controls preload="none" src="audio/{prefix}_mixture.wav"></audio></div>'
        ]
        for key in order:
            metric = "" if key == "target" else f" · NMSE {case['metrics'][key]:.3f}"
            css = " oracle" if key == "oracle_soft" else ""
            players.append(
                f'<div class="arm{css}"><span>{html.escape(labels[key] + metric)}</span><audio controls preload="none" src="audio/{prefix}_{key}.wav"></audio></div>'
            )
        cards.append(
            '<section class="card">'
            f"<h2>Pitch {case['pitch']} · octave competitor: {'yes' if case['has_octave_competitor'] else 'no'}</h2>"
            f"<p>{html.escape(case['mixture_id'])}</p>"
            + "".join(players)
            + "</section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sparse collision allocation</title><style>
body{{font-family:system-ui,sans-serif;background:#0d1320;color:#edf3ff;margin:0;padding:24px;line-height:1.45}}main{{max-width:1150px;margin:auto}}
.warning{{background:#422517;border:1px solid #bd6b36;padding:14px;border-radius:10px}}.card{{background:#151e2f;border:1px solid #2d3a52;border-radius:14px;padding:18px;margin:18px 0}}
.card h2{{margin:0}}.card p{{color:#aebbd0}}.arm{{display:grid;grid-template-columns:minmax(330px,1fr) 2fr;gap:14px;align-items:center;padding:9px 0;border-top:1px solid #2a364d}}
.oracle{{color:#ffc18b}}audio{{width:100%}}@media(max-width:760px){{.arm{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Sparse ownership with a collision path</h1>
<p>The saved classifier is unchanged. These arms only change how its logits allocate each complete mixture coefficient. Confident bins remain sparse; collision arms may share an ambiguous coefficient between two notes.</p>
<p class="warning"><strong>Oracle warning:</strong> the soft oracle uses isolated ground-truth notes to calculate allocation weights. It is ceiling evidence, not deployable inference.</p>
{''.join(cards)}
</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.listening_cases < 1:
        raise ValueError("listening-cases must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run = args.atomic_run.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "audio").mkdir()
    contract_path = run / "contract.json"
    checkpoint_path = run / "checkpoints" / "best.pt"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    training = argparse.Namespace(**contract["arguments"])
    device = torch.device(args.device)
    tile = (int(training.atom_frequency_bins), int(training.atom_time_frames))

    frozen = FrozenV13Contract.load(Path(training.frozen_manifest))
    _, validation_ordinals, view_binding = synthetic_corpus.load_synthetic_ordinal_view(
        Path(training.grouped_view), contract=frozen
    )
    if validation_ordinals is None or view_binding is None:
        raise ValueError("atomic run view has no validation split")
    examples = select_examples(
        frozen,
        validation_ordinals,
        count=int(training.validation_groups),
        candidate_pool=int(training.validation_candidate_pool),
        crop_samples=int(training.crop_samples),
        maximum_sources=int(training.maximum_sources),
        maximum_true_residual_ratio=float(training.maximum_true_residual_ratio),
        seed=int(training.seed),
        split="validation_sparse_allocation",
    )
    backbone, initializer_provenance = load_dual_hpss_checkpoint(
        Path(training.initializer), device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = OccurrenceAtomicMergerConfig(**state["config"])
    model = OccurrenceAtomicMerger(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    with torch.inference_mode():
        for example in examples:
            record = build_record(backbone, example, device, training, validation=True)
            logits = model(record["features"].to(device).float())
            mixture = record["mixture_spectrum"].to(device)
            targets = record["target_spectrum"].to(device)
            teacher = record["teacher_spectrum"].to(device)
            target_waveform = record["target_waveform"].to(device)
            pitches = record["pitches"].to(device)
            background = mixture - targets.sum(dim=0)
            target_bank = torch.cat((targets, background[None]), dim=0)
            spectra: dict[str, torch.Tensor] = {"teacher": teacher}
            for method, pooled_weights in allocation_methods(logits).items():
                weights = expand_weights(
                    pooled_weights,
                    frequency=mixture.shape[-2],
                    frames=mixture.shape[-1],
                    tile=tile,
                )
                bank = weights * mixture[None]
                spectra[method] = bank[:-1]
                metrics = source_metrics(
                    bank[:-1],
                    targets,
                    mixture,
                    pitches,
                    target_waveform=target_waveform,
                    n_fft=int(training.n_fft),
                )
                metrics.update(allocation_statistics(weights, mixture))
                rows.append(
                    {"method": method, "mixture_id": record["mixture_id"], **metrics}
                )
            oracle_hard, _ = strict_atom_partition(mixture, target_bank, tile)
            oracle_soft = soft_energy_partition(mixture, target_bank)
            spectra["oracle_hard"] = oracle_hard[:-1]
            spectra["oracle_soft"] = oracle_soft[:-1]
            for method, bank in (("oracle_hard", oracle_hard), ("oracle_soft", oracle_soft)):
                metrics = source_metrics(
                    bank[:-1], targets, mixture, pitches,
                    target_waveform=target_waveform, n_fft=int(training.n_fft),
                )
                metrics["energy_weighted_support"] = float("nan")
                metrics["energy_fraction_shared"] = float("nan")
                metrics["energy_weighted_entropy"] = float("nan")
                rows.append(
                    {"method": method, "mixture_id": record["mixture_id"], **metrics}
                )
            teacher_metrics = source_metrics(
                teacher, targets, mixture, pitches,
                target_waveform=target_waveform, n_fft=int(training.n_fft),
            )
            teacher_metrics.update(
                {
                    "energy_weighted_support": float("nan"),
                    "energy_fraction_shared": float("nan"),
                    "energy_weighted_entropy": float("nan"),
                }
            )
            rows.append(
                {"method": "teacher", "mixture_id": record["mixture_id"], **teacher_metrics}
            )

            length = int(target_waveform.shape[-1])
            waveform = {
                method: istft(spectrum, int(training.n_fft), length).cpu()
                for method, spectrum in spectra.items()
            }
            mixture_waveform = istft(mixture, int(training.n_fft), length).cpu()
            target_waveform_cpu = target_waveform.cpu()
            for source in range(int(target_waveform.shape[0])):
                target = target_waveform_cpu[source]
                rms = float(target.square().mean().sqrt())
                if rms < 1e-5:
                    continue
                distance = (pitches.cpu() - pitches.cpu()[source]).abs()
                has_octave = bool(((distance == 12) | (distance == 24)).any())
                candidates.append(
                    {
                        "mixture_id": record["mixture_id"],
                        "pitch": int(pitches[source]),
                        "has_octave_competitor": has_octave,
                        "rms": rms,
                        "mixture": mixture_waveform,
                        "target": target,
                        "teacher": waveform["teacher"][source],
                        "learned_hard": waveform["learned_hard"][source],
                        "confidence_top2_085": waveform["confidence_top2_085"][source],
                        "sparsemax_t1": waveform["sparsemax_t1"][source],
                        "sparsemax_t2": waveform["sparsemax_t2"][source],
                        "oracle_soft": waveform["oracle_soft"][source],
                    }
                )

    summary = aggregate(rows)
    hard = summary["learned_hard"]
    learned_methods = [
        "softmax_t1", "top2_softmax_t1", "confidence_top2_085",
        "sparsemax_t1", "sparsemax_t2",
    ]
    comparison = {
        method: {
            "complex_nmse_relative_improvement": (
                hard["note_complex_nmse"] - summary[method]["note_complex_nmse"]
            ) / max(hard["note_complex_nmse"], 1e-12),
            "pitch_accuracy_delta": (
                summary[method]["pitch_assignment_accuracy"]
                - hard["pitch_assignment_accuracy"]
            ),
            "octave_margin_delta": summary[method]["octave_margin"] - hard["octave_margin"],
            "energy_fraction_shared": summary[method]["energy_fraction_shared"],
        }
        for method in learned_methods
    }
    passing = [
        method for method in learned_methods
        if comparison[method]["complex_nmse_relative_improvement"] >= 0.05
        and comparison[method]["pitch_accuracy_delta"] >= -0.02
        and comparison[method]["octave_margin_delta"] >= 0.0
        and comparison[method]["energy_fraction_shared"] <= 0.35
    ]
    decision = {
        "status": "sparse_collision_allocation_supported" if passing else "saved_logits_do_not_support_collision_sharing",
        "passing_methods": passing,
        "gate": {
            "complex_nmse_relative_improvement_min": 0.05,
            "pitch_accuracy_delta_min": -0.02,
            "octave_margin_delta_min": 0.0,
            "energy_fraction_shared_max": 0.35,
        },
        "comparison": comparison,
        "interpretation": "No model was retrained; this isolates allocation from representation learning.",
    }

    candidates.sort(
        key=lambda case: (case["has_octave_competitor"], case["rms"]), reverse=True
    )
    selected: list[dict[str, Any]] = []
    per_mixture: dict[str, int] = {}
    for case in candidates:
        mixture_id = str(case["mixture_id"])
        if per_mixture.get(mixture_id, 0) >= 2:
            continue
        selected.append(case)
        per_mixture[mixture_id] = per_mixture.get(mixture_id, 0) + 1
        if len(selected) == args.listening_cases:
            break
    if len(selected) != args.listening_cases:
        raise RuntimeError("not enough audible listening cases")

    audio_keys = (
        "target", "teacher", "learned_hard", "confidence_top2_085",
        "sparsemax_t1", "sparsemax_t2", "oracle_soft",
    )
    manifest_cases: list[dict[str, Any]] = []
    for case_index, case in enumerate(selected):
        prefix = f"case_{case_index:02d}_p{case['pitch']}"
        estimates = [case[key] for key in audio_keys if key != "target"]
        gain = shared_gain(case["target"], estimates)
        mixture_gain = shared_gain(case["mixture"], [])
        write_wav(output / "audio" / f"{prefix}_mixture.wav", case["mixture"], 16_000, mixture_gain)
        metrics: dict[str, float] = {}
        for key in audio_keys:
            write_wav(output / "audio" / f"{prefix}_{key}.wav", case[key], 16_000, gain)
            if key != "target":
                metrics[key] = waveform_nmse(case[key], case["target"])
        case["metrics"] = metrics
        manifest_cases.append(
            {
                "index": case_index,
                "mixture_id": case["mixture_id"],
                "pitch": case["pitch"],
                "has_octave_competitor": case["has_octave_competitor"],
                "metrics": metrics,
            }
        )
    render_listening(output, selected)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "source_run": str(run),
        "source_run_contract_sha256": file_sha256(contract_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(state["step"]),
        "initializer": initializer_provenance,
        "view": view_binding,
        "atom_tile": list(tile),
        "summary": summary,
        "decision": decision,
        "listening_cases": manifest_cases,
        "listening": str(output / "index.html"),
    }
    atomic_json(result, output / "result.json")
    print("RESULT " + json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
