#!/usr/bin/env python3
"""Render audible teacher, learned, and oracle STFT ownership comparisons."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys
import wave
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from model_exp.dna_residual_v13_streaming import FrozenV13Contract  # noqa: E402
from model_exp.occurrence_atomic_merger import (  # noqa: E402
    OccurrenceAtomicMerger,
    OccurrenceAtomicMergerConfig,
)
import scripts.train_dna_residual_v13_streaming as synthetic_corpus  # noqa: E402
from scripts.audit_occurrence_atomic_merge_feasibility import (  # noqa: E402
    istft,
    strict_atom_partition,
)
from scripts.train_occurrence_atomic_merger_canary import (  # noqa: E402
    build_record,
    expand_atom_labels,
    select_examples,
)
from scripts.train_occurrence_score_remix_ssl import (  # noqa: E402
    atomic_json,
    file_sha256,
    load_dual_hpss_checkpoint,
)


SCHEMA = "occurrence_stft_atomic_separation_listening_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--listening-cases", type=int, default=12)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def write_wav(path: Path, values: torch.Tensor, sample_rate: int, gain: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        values.detach().cpu().float().mul(gain).clamp(-1.0, 1.0).numpy() * 32767.0
    ).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def shared_gain(target: torch.Tensor, estimates: list[torch.Tensor]) -> float:
    target = target.float()
    rms = float(target.square().mean().sqrt())
    peak = max(float(value.float().abs().max()) for value in [target, *estimates])
    return min(0.95 / max(peak, 1e-8), 0.126 / max(rms, 1e-8))


def nmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (estimate.float() - target.float()).square().sum()
        / target.float().square().sum().clamp_min(1e-12)
    )


def render_html(output: Path, cases: list[dict[str, Any]], *, atom_tile: tuple[int, int]) -> None:
    arms = (
        ("target", "isolated ground-truth note"),
        ("teacher", "gated 30k separator"),
        ("learned", "learned hard STFT ownership"),
        ("oracle", "GROUND-TRUTH-ASSISTED STFT oracle"),
    )
    cards: list[str] = []
    for index, case in enumerate(cases):
        prefix = f"case_{index:02d}_p{case['pitch']}"
        players = [
            f'<div class="arm"><span>mixture context (separately normalized)</span><audio controls preload="none" src="audio/{prefix}_mixture.wav"></audio></div>'
        ]
        for key, label in arms:
            metric = "" if key == "target" else f" · NMSE {case['metrics'][key]:.3f}"
            players.append(
                f'<div class="arm {key}"><span>{html.escape(label + metric)}</span><audio controls preload="none" src="audio/{prefix}_{key}.wav"></audio></div>'
            )
        octave = "yes" if case["has_octave_competitor"] else "no"
        cards.append(
            '<section class="card">'
            f"<h2>Pitch {case['pitch']} · octave competitor: {octave}</h2>"
            f"<p>{html.escape(case['mixture_id'])}</p>"
            + "".join(players)
            + "</section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STFT atomic separation listening</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0d1320;color:#edf3ff;margin:0;padding:24px;line-height:1.45}}
main{{max-width:1100px;margin:auto}}.warning{{background:#422517;border:1px solid #bd6b36;padding:14px;border-radius:10px}}
.card{{background:#151e2f;border:1px solid #2d3a52;border-radius:14px;padding:18px;margin:18px 0}}.card h2{{margin:0}}.card p{{color:#aebbd0}}
.arm{{display:grid;grid-template-columns:minmax(280px,1fr) 2fr;gap:14px;align-items:center;padding:9px 0;border-top:1px solid #2a364d}}
.oracle{{color:#ffc18b}}audio{{width:100%}}@media(max-width:720px){{.arm{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Does STFT ownership separation “kinda work”?</h1>
<p class="warning"><strong>Critical:</strong> the oracle is not a usable separator. It sees every isolated ground-truth note and uses them to decide which source owns each complete mixture STFT atom. It demonstrates representational headroom only.</p>
<p>The learned arm uses the saved merger checkpoint and only mixture/score/frozen-teacher evidence. Every target/estimate within a case uses one shared gain. Atom tile: {atom_tile[0]} frequency bin × {atom_tile[1]} frame.</p>
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
    result_path = run / "result.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    training = argparse.Namespace(**contract["arguments"])
    device = torch.device(args.device)

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
        split="validation_listening",
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

    candidates: list[dict[str, Any]] = []
    tile = (int(training.atom_frequency_bins), int(training.atom_time_frames))
    with torch.inference_mode():
        for example in examples:
            record = build_record(backbone, example, device, training, validation=True)
            features = record["features"].to(device).float()
            predicted = model(features).argmax(dim=0)
            mixture_spectrum = record["mixture_spectrum"].to(device)
            target_spectrum = record["target_spectrum"].to(device)
            teacher_spectrum = record["teacher_spectrum"].to(device)
            target_background = mixture_spectrum - target_spectrum.sum(dim=0)
            target_bank = torch.cat((target_spectrum, target_background[None]), dim=0)
            assignment = expand_atom_labels(
                predicted,
                sources=target_bank.shape[0],
                frequency=mixture_spectrum.shape[-2],
                frames=mixture_spectrum.shape[-1],
                tile=tile,
            )
            learned_bank = assignment * mixture_spectrum[None]
            oracle_bank, _ = strict_atom_partition(
                mixture_spectrum, target_bank, tile
            )
            length = int(record["target_waveform"].shape[-1])
            mixture_waveform = istft(mixture_spectrum, int(training.n_fft), length).cpu()
            target_waveform = record["target_waveform"].float()
            teacher_waveform = istft(teacher_spectrum, int(training.n_fft), length).cpu()
            learned_waveform = istft(learned_bank[:-1], int(training.n_fft), length).cpu()
            oracle_waveform = istft(oracle_bank[:-1], int(training.n_fft), length).cpu()
            pitches = record["pitches"].long()
            for source in range(int(target_waveform.shape[0])):
                target = target_waveform[source]
                rms = float(target.square().mean().sqrt())
                if rms < 1e-5:
                    continue
                pitch_distance = (pitches - pitches[source]).abs()
                has_octave = bool(((pitch_distance == 12) | (pitch_distance == 24)).any())
                candidates.append(
                    {
                        "mixture_id": record["mixture_id"],
                        "source": source,
                        "pitch": int(pitches[source]),
                        "has_octave_competitor": has_octave,
                        "rms": rms,
                        "mixture": mixture_waveform,
                        "target": target,
                        "teacher": teacher_waveform[source],
                        "learned": learned_waveform[source],
                        "oracle": oracle_waveform[source],
                    }
                )

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
        raise RuntimeError(
            f"could select only {len(selected)}/{args.listening_cases} listening cases"
        )

    manifest_cases: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        prefix = f"case_{index:02d}_p{case['pitch']}"
        estimates = [case["teacher"], case["learned"], case["oracle"]]
        gain = shared_gain(case["target"], estimates)
        mixture_gain = shared_gain(case["mixture"], [])
        write_wav(output / "audio" / f"{prefix}_mixture.wav", case["mixture"], 16_000, mixture_gain)
        for key in ("target", "teacher", "learned", "oracle"):
            write_wav(output / "audio" / f"{prefix}_{key}.wav", case[key], 16_000, gain)
        metrics = {
            key: nmse(case[key], case["target"])
            for key in ("teacher", "learned", "oracle")
        }
        case["metrics"] = metrics
        manifest_cases.append(
            {
                "index": index,
                "mixture_id": case["mixture_id"],
                "source": case["source"],
                "pitch": case["pitch"],
                "has_octave_competitor": case["has_octave_competitor"],
                "target_rms": case["rms"],
                "shared_target_gain": gain,
                "mixture_gain": mixture_gain,
                "metrics": metrics,
            }
        )
    render_html(output, selected, atom_tile=tile)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "source_run": str(run),
        "source_run_result_sha256": file_sha256(result_path),
        "source_run_contract_sha256": file_sha256(contract_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(state["step"]),
        "initializer": initializer_provenance,
        "view": view_binding,
        "model": asdict(config),
        "atom_tile": list(tile),
        "cases": manifest_cases,
    }
    atomic_json(manifest, output / "manifest.json")
    print("RESULT " + json.dumps({"cases": len(selected), "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
