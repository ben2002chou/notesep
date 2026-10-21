"""Run the NoteGrab training recipe with explicit data and checkpoint paths."""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import validate_notegrab as components


def apply_data_root(options):
    if options.data_root is None:
        return
    if any(
        value is not None
        for value in (
            options.path_map,
            options.packed_metadata_archive,
            options.packed_metadata_manifest,
            options.packed_metadata_manifest_sha256,
        )
    ):
        raise ValueError("Use --data-root or explicit data paths, not both")
    root = options.data_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("component") != "original_notegrab_training_inputs":
        raise ValueError("--data-root needs the original NoteGrab dataset")
    paths = {row["path"]: row for row in manifest["files"]}
    name = "packed-metadata-overlay/manifest.json"
    options.path_map = root / "path-map.json"
    options.packed_metadata_archive = (
        root / "packed-metadata-overlay/payload.zip"
    )
    options.packed_metadata_manifest = root / name
    options.packed_metadata_manifest_sha256 = paths[name]["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--canary-steps",
        type=int,
        help=(
            "Explicit diagnostic stop override; omitted means original "
            "5000-step recipe"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check sources, contracts, parser and checkpoint bytes "
            "without training"
        ),
    )
    parser.add_argument("--packed-metadata-archive", type=Path)
    parser.add_argument("--packed-metadata-manifest", type=Path)
    parser.add_argument("--packed-metadata-manifest-sha256")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Verified NoteGrab download from notesep download-data",
    )
    parser.add_argument(
        "--path-map",
        type=Path,
        help="JSON mapping of historical data roots to local directories",
    )
    options = parser.parse_args()
    apply_data_root(options)
    if options.output.exists() or (
        options.canary_steps is not None
        and not 1 <= options.canary_steps <= 5000
    ):
        raise ValueError(
            "New output and positive bounded canary steps required"
        )
    packed = [
        options.packed_metadata_archive,
        options.packed_metadata_manifest,
        options.packed_metadata_manifest_sha256,
    ]
    if any(value is not None for value in packed) and not all(
        value is not None for value in packed
    ):
        raise ValueError(
            "All packed metadata overlay arguments required together"
        )
    options.output.mkdir(parents=True)
    sys.path.insert(0, str(components.ROOT / "src"))
    from notesep.training_paths import configure, relocated

    configure(options.path_map)
    contract_path = components.ROOT / "training/notegrab/training_contract.json"
    if components.sha(contract_path) != components.TRAINING_CONTRACT_SHA:
        raise ValueError("Training recipe changed")
    contract = json.loads(contract_path.read_text())
    if (
        components.sha(options.initialize_checkpoint)
        != contract["initialization"]["checkpoint_sha256"]
    ):
        raise ValueError(
            "Exact original model-only +250 initialization required"
        )
    os.environ.update(
        DNA_V13_SKIP_INITIAL_MONITOR="1",
        DNA_V13_TORCH_COMPILE=contract["compile_mode"],
        OUTPUT=str((options.output / "training").resolve()),
        SCNS_REQUIRE_ALL_16_RECORDED_FAMILIES="0",
    )
    sys.argv = [
        str(components.__file__),
        "--output",
        str(options.output / "components.json"),
    ]
    if options.path_map is not None:
        sys.argv += ["--path-map", str(options.path_map)]
    components.main()
    import parent_balanced_training as entry

    base = entry.cropped.v2.trackbound.v1.base
    os.environ["SCNS_V3_REQUIRED_SUPPLEMENTAL_FAMILIES"] = ",".join(
        sorted(entry.cropped.v2.SCORE_BOUND_SUPPLEMENTAL_FAMILIES)
    )
    arguments = dict(contract["arguments"])
    overrides = {
        "output_dir": str((options.output / "training").resolve()),
        "wandb_mode": "disabled",
        "wandb_name": "notesep-declared-reconstructed-continuation",
    }
    if options.canary_steps is not None:
        overrides["max_steps"] = options.canary_steps
    arguments.update(overrides)
    for key in (
        "acoustic_offset_index",
        "duration_index",
        "frozen_manifest",
        "priority_index",
        "synthetic_ordinal_view",
    ):
        if arguments.get(key) is not None:
            arguments[key] = relocated(arguments[key])
    arguments["recorded_frozen_root"] = [
        relocated(value) for value in arguments["recorded_frozen_root"]
    ]
    argv = []
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    argv += ["--initialize-checkpoint", str(options.initialize_checkpoint)]
    parsed = base.parse_args(argv)
    if (
        parsed.resume is not None
        or parsed.initialize_checkpoint != options.initialize_checkpoint
    ):
        raise ValueError("Model-only initialization semantics changed")
    plan = {
        "schema": "notesep_reconstructed_notegrab_recipe_v1",
        "status": "prepared",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "launcher_sha256": components.sha(Path(__file__)),
        "training_contract_sha256": components.sha(contract_path),
        "component_validation_sha256": components.sha(
            options.output / "components.json"
        ),
        "initialize_checkpoint_sha256": contract["initialization"][
            "checkpoint_sha256"
        ],
        "argv": argv,
        "declared_overrides": overrides,
        "dry_run": options.dry_run,
        "historical_executed_closure_recovered": False,
        "packed_metadata_transport_sha256": components.sha(
            Path(__file__).with_name("stage_metadata.py")
        ),
        "scope": (
            "Recovered scientific components and saved original recipe; exact "
            "historical source/exposure, full replay and corpus admission "
            "remain separate."
        ),
    }
    (options.output / "recipe.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    if options.dry_run:
        print(
            json.dumps({"status": "passed_preflight", "optimizer_updates": 0}),
            flush=True,
        )
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    overlay = None
    if packed[0] is not None:
        import model_exp.streaming_dataset as streaming
        from stage_metadata import PackedMetadataOverlay

        overlay = PackedMetadataOverlay(
            *packed, options.output / "packed-metadata-overlay"
        )
        overlay.install(streaming)
    try:
        base.main(argv)
        checkpoint = options.output / "training/checkpoints/latest.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(saved["global_step"]) != arguments["max_steps"]:
            raise ValueError("Training did not reach declared endpoint")
        counts = {}

        def finite(value):
            if torch.is_tensor(value):
                if (
                    value.is_floating_point()
                    and not torch.isfinite(value).all()
                ):
                    raise ValueError("Nonfinite trained tensor")
                return 1
            if isinstance(value, dict):
                return sum(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(finite(v) for v in value)
            return 0

        for name in ("model", "optimizer"):
            counts[name] = finite(saved[name])
        report = {
            "status": "passed_training_execution",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "global_step": saved["global_step"],
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": components.sha(checkpoint),
            "finite_tensor_counts": counts,
            "recipe_sha256": components.sha(options.output / "recipe.json"),
            "original_execution_verified": False,
            "training_reproducibility_verified": False,
            "scope": (
                "Actual declared reconstructed training and finite checkpoint "
                "only; independent repeat/resume and historical checkpoint "
                "reproduction remain unverified."
            ),
        }
        (options.output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report), flush=True)
    finally:
        if overlay is not None:
            overlay.close()


if __name__ == "__main__":
    main()
