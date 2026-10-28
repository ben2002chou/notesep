"""Generate NoteGrab estimates for ASO training from the prepared audio banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from notesep.assets import PROFILES, sha256, weights_directory  # noqa: E402
from notesep.models.hpss import project_hpss_batch  # noqa: E402
from notesep.models.hpss_separator import (  # noqa: E402
    occurrence_dual_hpss_separator_from_contract,
)
from notesep.training_paths import configure  # noqa: E402
from training.aso.dataset import KIND, recorded_example  # noqa: E402
from training.aso.datasets.recorded import RecordedNaturalScoreSceneDataset  # noqa: E402
from notesep.shards import PackedBlobShardReader, PackedBlobShardWriter  # noqa: E402
from training.aso.datasets.synthetic import FreshSourceCompleteDataset  # noqa: E402
from training.aso.train import (  # noqa: E402
    FRESH_SHA256,
    FRESH_AUTHORITY_SHA256,
    RECORDED_SHA256,
    PROPOSAL_INDEX_SHA256,
)

SCHEMA = "notesep_aso_proposals_v1"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def data_paths(root: Path) -> tuple[Path, Path]:
    """Use the ASO package's relative paths and generated relocation map."""
    root = root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("component") != "aso_training_inputs":
        raise ValueError("Expected the ASO data package")
    configure(root / "path-map.json")
    paths = []
    for key in ("fresh_index", "proposal_index", "recorded_root"):
        path = (root / manifest["training"][key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("ASO input must stay inside its data package")
        paths.append(path)
    fresh, proposal, recorded = paths
    if sha256(fresh) != FRESH_SHA256:
        raise ValueError("ASO synthetic index changed")
    if sha256(fresh.parent / "AUTHORITY.json") != FRESH_AUTHORITY_SHA256:
        raise ValueError("ASO acoustic endpoint metadata changed")
    if sha256(recorded / "FROZEN.json") != RECORDED_SHA256:
        raise ValueError("ASO recorded data definition changed")
    if sha256(proposal) != PROPOSAL_INDEX_SHA256:
        raise ValueError("NoteGrab query index changed")
    return proposal, recorded


@torch.inference_mode()
def infer_notes(
    model: torch.nn.Module, example: dict, device: torch.device
) -> torch.Tensor:
    """Match the paper cache: full crop, octave gate, then float16 storage.

    This uses the fixed ASO crop, rather than the interactive editor's per-note
    crop. Signed boundaries determine the octave gate; model query coordinates
    are clipped to the crop, as in the original estimate generator.
    """
    wave = example["mixture"].to(device)[None]
    samples = wave.shape[-1]
    valid = torch.tensor([samples], device=device)
    harmonic, percussive = project_hpss_batch(
        wave,
        valid,
        sample_rate=int(model.config.sample_rate),
        n_fft=int(model.config.n_fft),
        hop_length=int(model.config.hop_length),
        kernel_size=31,
        power=2.0,
        margin=1.0,
    )
    pitch = example["pitches"].to(device)
    onset, offset = (example[k].to(device) for k in ("onsets", "offsets"))
    duration = (offset - onset).clamp_min(1)
    overlap = (
        torch.minimum(offset[:, None], offset[None, :])
        - torch.maximum(onset[:, None], onset[None, :])
    ).clamp_min(0)
    shorter = torch.minimum(duration[:, None], duration[None, :])
    gate = (
        (
            (pitch[:, None] + 12 == pitch[None, :])
            & (overlap >= 1280)
            & (overlap >= 0.25 * shorter)
        )
        .any(1)
        .float()
    )
    query_onset = onset.clamp(0, samples - 1)
    query_offset = torch.maximum(offset.clamp(1, samples), query_onset + 1)
    output = []
    for start in range(0, len(pitch), 4):
        stop = min(start + 4, len(pitch))
        count = stop - start
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            estimate, _ = model(
                harmonic.repeat(count, 1),
                percussive.repeat(count, 1),
                pitch[start:stop],
                valid.expand(count),
                query_onset[start:stop],
                query_offset[start:stop],
                input_gate_strength=gate[start:stop],
            )
        output.append(estimate.float().cpu())
    estimates = torch.cat(output)
    if (
        estimates.shape != example["targets"].shape
        or not torch.isfinite(estimates).all()
    ):
        raise ValueError("NoteGrab estimate shape or values changed")
    return estimates


def load_separator(checkpoint: Path, device: torch.device):
    """Load released NoteGrab weights or a checkpoint from the training recipe."""
    separator_sha = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if separator_sha not in {pair["separator.pt"] for pair in PROFILES.values()}:
        from training.notegrab_recipe import validate_stage_checkpoint
        validate_stage_checkpoint(payload, "continuation", step=2500)
    model = occurrence_dual_hpss_separator_from_contract(payload["model_contract"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), separator_sha


def planned_units(root: Path, count: int, size: int = 256):
    """Resume completed units and group remaining mixtures into larger units."""
    completed = sorted(root.glob("unit-[0-9][0-9][0-9][0-9][0-9]"))
    position = 0
    for index, path in enumerate(completed):
        if path.name != f"unit-{index:05d}":
            raise ValueError("Completed estimate units are not consecutive")
        marker = json.loads((path / "UNIT_COMPLETE.json").read_text())
        mixtures = int(marker["mixtures"])
        if mixtures < 1 or position + mixtures > count:
            raise ValueError("Completed estimate unit has an invalid size")
        yield index, position, position + mixtures
        position += mixtures
    index = len(completed)
    while position < count:
        stop = min(position + size, count)
        yield index, position, stop
        index += 1
        position = stop


def generate(data_root: Path, checkpoint: Path, output: Path):
    proposal, recorded = data_paths(data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Estimate generation requires a CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, separator_sha = load_separator(checkpoint, device)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": SCHEMA,
        "separator_sha256": separator_sha,
        "fresh_index_sha256": FRESH_SHA256,
        "proposal_index_sha256": PROPOSAL_INDEX_SHA256,
        "recorded_freeze_sha256": RECORDED_SHA256,
        "crop_samples": 49152,
        "maximum_sources": 64,
        "query_microbatch": 4,
        "input_gate_policy": "lower_member_exact_octave",
        "payload_kind": KIND,
    }
    config_path = output / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing estimates were generated with different settings")
    if not config_path.exists():
        if any(output.iterdir()):
            raise FileExistsError("Choose an empty output folder for new estimates")
        write_json(config_path, config)
    identity = sha256(config_path)
    (output / "VALIDATION.json").unlink(missing_ok=True)
    groups = {}
    for kind in ("fresh", "recorded"):
        for split in ("train", "validation"):
            if kind == "fresh":
                # Preserve the original NoteGrab query times; ASO later reads
                # the separate acoustic endpoint index for its own inputs.
                data = FreshSourceCompleteDataset(
                    proposal,
                    split=split,
                    crop_samples=49152,
                    maximum_sources=64,
                    preserve_signed_note_times=True,
                    require_acoustic_offsets=False,
                )
                population = len(data)
                get_example = data.__getitem__
            else:
                data = RecordedNaturalScoreSceneDataset([recorded], split=split)
                population = len(data.scenes)
                get_example = lambda index: recorded_example(data, index)
            count = population
            mixtures = sources = 0
            try:
                units_root = output / kind / split / "tasks/task-000/units"
                for unit_index, start, stop in planned_units(units_root, count):
                    unit = units_root / f"unit-{unit_index:05d}"
                    if unit.exists():
                        marker = json.loads((unit / "UNIT_COMPLETE.json").read_text())
                        if marker["config_sha256"] != identity or marker[
                            "banks_sha256"
                        ] != sha256(unit / "banks.jsonl"):
                            raise ValueError(
                                "Completed estimate unit has different settings"
                            )
                        mixtures += marker["mixtures"]
                        sources += marker["proposals"]
                        continue
                    temporary = unit.with_name(unit.name + ".partial")
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    temporary.mkdir(parents=True)
                    writer = PackedBlobShardWriter(temporary / "packed", prefix="notes")
                    rows = []
                    for index in range(start, stop):
                        example = get_example(index)
                        estimates = infer_notes(model, example, device)
                        identifiers = []
                        for note, estimate in enumerate(estimates):
                            text = f"{SCHEMA}:{kind}:{example['mixture_id']}:{note}:{config['separator_sha256']}"
                            identifier = hashlib.sha256(text.encode()).hexdigest()
                            values = estimate.numpy().astype("<f2")
                            if not np.isfinite(values).all():
                                raise ValueError("Estimate overflows float16 storage")
                            writer.add_bytes(
                                identifier,
                                KIND,
                                values.tobytes(),
                                metadata={"source_index": note, "samples": len(values)},
                            )
                            identifiers.append(identifier)
                        rows.append(
                            {
                                "mixture_id": str(example["mixture_id"]),
                                "proposal_ids": identifiers,
                                "source_parent_id": str(example["source_parent_id"]),
                                "score_parent_id": str(example["score_parent_id"]),
                                "pitches": example["pitches"].tolist(),
                                "onsets": example["onsets"].tolist(),
                                "offsets": example["offsets"].tolist(),
                                "samples": len(example["mixture"]),
                            }
                        )
                        if index == 0 or (index + 1) % 100 == 0 or index + 1 == count:
                            print(f"{kind}/{split}: {index + 1}/{count}", flush=True)
                    writer.finalize({"config_sha256": identity})
                    (temporary / "banks.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in rows)
                    )
                    marker = {
                        "config_sha256": identity,
                        "banks_sha256": sha256(temporary / "banks.jsonl"),
                        "mixtures": len(rows),
                        "proposals": sum(len(r["proposal_ids"]) for r in rows),
                    }
                    write_json(temporary / "UNIT_COMPLETE.json", marker)
                    unit.parent.mkdir(parents=True, exist_ok=True)
                    temporary.replace(unit)
                    mixtures += marker["mixtures"]
                    sources += marker["proposals"]
            finally:
                data.close()
            groups[f"{kind}:{split}"] = {
                "mixtures": mixtures,
                "proposals": sources,
                "population": population,
            }
    validation = config | {
        "status": "passed",
        "test_data_used": False,
        "complete": True,
        "groups": groups,
        "config_sha256": identity,
    }
    # A failed or interrupted validation must never leave a passed marker.
    (output / "VALIDATION.json").unlink(missing_ok=True)
    print("Checking saved note estimates...", flush=True)
    checked = 0
    content = hashlib.sha256()
    for banks in sorted(output.glob("*/*/tasks/task-*/units/unit-*/banks.jsonl")):
        marker = json.loads((banks.parent / "UNIT_COMPLETE.json").read_text())
        if marker["banks_sha256"] != sha256(banks):
            raise ValueError("Estimate metadata changed")
        content.update(marker["banks_sha256"].encode())
        with PackedBlobShardReader(banks.parent / "packed") as reader:
            for line in banks.read_text().splitlines():
                row = json.loads(line)
                for identifier in row["proposal_ids"]:
                    payload = reader.read_bytes(identifier, KIND)
                    checksum = hashlib.sha256(payload).hexdigest()
                    if checksum != reader.record(identifier, KIND).sha256:
                        raise ValueError("Serialized estimate checksum changed")
                    content.update(checksum.encode())
                    values = np.frombuffer(payload, dtype="<f2")
                    if len(values) != row["samples"] or not np.isfinite(values).all():
                        raise ValueError(
                            "Serialized estimate geometry or values changed"
                        )
                    checked += 1
    if checked != sum(group["proposals"] for group in groups.values()):
        raise ValueError("Estimate payload coverage changed")
    validation["payloads_verified"] = checked
    # Bind training resume to the actual estimate values and row geometry.
    validation["content_sha256"] = content.hexdigest()
    write_json(output / "VALIDATION.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=weights_directory() / "separator.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.data_root, args.checkpoint, args.output)
    mixtures = sum(group["mixtures"] for group in result["groups"].values())
    print(
        f"Prepared {mixtures} mixtures and {result['payloads_verified']} note estimates in {args.output}."
    )


if __name__ == "__main__":
    main()
