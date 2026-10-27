"""Run the NoteGrab training recipe with explicit data and checkpoint paths."""

import argparse
import json
import os
import sys
from pathlib import Path

import validate_notegrab as components


def apply_data_root(options):
    root = options.data_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("component") != "original_notegrab_training_inputs":
        raise ValueError("--data-root needs the NoteGrab training package")
    paths = {row["path"]: row for row in manifest["files"]}
    metadata = manifest.get(
        "packed_metadata",
        {
            "archive": "packed-metadata-overlay/payload.zip",
            "manifest": "packed-metadata-overlay/manifest.json",
        },
    )
    name = metadata["manifest"]
    options.path_map = root / "path-map.json"
    options.packed_metadata_archive = root / metadata["archive"]
    options.packed_metadata_manifest = root / name
    options.packed_metadata_manifest_sha256 = paths[name]["sha256"]
    options.packed_blob_paths = {
        name: root / path for name, path in metadata.get("blob_paths", {}).items()
    }


def validate_initialization(path, recipe):
    """Accept the released initializer or a checkpoint from the preceding stage."""
    if recipe == "continuation":
        released = components.load_contract(recipe)["initialization"]["checkpoint_sha256"]
        if components.sha(path) == released:
            return
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    predecessor = "parent" if recipe == "adaptation" else "adaptation"
    components.validate_stage_checkpoint(saved, predecessor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-checkpoint", type=Path)
    parser.add_argument(
        "--resume", action="store_true",
        help="Restore the latest checkpoint and optimizer",
    )
    parser.add_argument(
        "--recipe", choices=("parent", "adaptation", "continuation"), default="continuation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Check sources, contracts, parser and checkpoint bytes without training"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="NoteGrab training package",
    )
    options = parser.parse_args()
    if (
        options.recipe != "parent" and not options.resume
        and options.initialize_checkpoint is None
    ):
        parser.error("Adaptation and continuation need --initialize-checkpoint")
    if options.recipe == "parent" and options.initialize_checkpoint is not None:
        parser.error("Parent training starts from random weights")
    apply_data_root(options)
    if options.output.exists() and not options.resume:
        raise FileExistsError(options.output)
    resume_checkpoint = options.output / "training/checkpoints/latest.pt"
    if options.resume and not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)
    packed = (
        options.packed_metadata_archive,
        options.packed_metadata_manifest,
        options.packed_metadata_manifest_sha256,
    )
    options.output.mkdir(parents=True, exist_ok=options.resume)
    sys.path.insert(0, str(components.ROOT / "src"))
    from notesep.training_paths import configure, relocated

    if options.recipe in {"parent", "adaptation"}:
        mapping = json.loads(options.path_map.read_text())
        root = options.data_root.resolve()
        mapping[f"/note-sep-data/synthetic/views/{options.recipe}.json"] = str(
            root / f"synthetic/sampling-{options.recipe}.json"
        )
        mapping["/note-sep-data/synthetic/render-methods.npz"] = str(
            root / "synthetic/render-methods.npz"
        )
        options.path_map = options.output / "path-map.json"
        options.path_map.write_text(json.dumps(mapping, indent=2) + "\n")
    configure(options.path_map)
    contract = components.load_contract(options.recipe)
    if options.recipe != "parent" and not options.resume:
        validate_initialization(options.initialize_checkpoint, options.recipe)
    os.environ.update(
        DNA_V13_OPTIMIZER="adamw",
        DNA_V13_SKIP_INITIAL_MONITOR="1",
        DNA_V13_TORCH_COMPILE=contract["compile_mode"],
        OUTPUT=str((options.output / "training").resolve()),
    )
    suffix = f"-resume-{os.getpid()}" if options.resume else ""
    sys.argv = [
        str(components.__file__),
        "--output",
        str(options.output / f"components{suffix}.json"),
    ]
    sys.argv += ["--path-map", str(options.path_map), "--recipe", options.recipe]
    components.main()
    if options.recipe == "parent":
        import scripts.train_notegrab_separator as selective
        base = selective.base
    elif options.recipe == "adaptation":
        from event_cluster_data import base
    else:
        import parent_balanced_training as entry
        base = entry.trainer
        os.environ["SCNS_V3_REQUIRED_SUPPLEMENTAL_FAMILIES"] = ",".join(
            sorted(entry.SCORE_BOUND_SUPPLEMENTAL_FAMILIES)
        )
    arguments = dict(contract["arguments"])
    overrides = {
        "output_dir": str((options.output / "training").resolve()),
        "wandb_mode": "disabled",
        "wandb_name": "notesep-notegrab",
    }
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
    if options.resume:
        argv += ["--resume", str(resume_checkpoint)]
    elif options.initialize_checkpoint is not None:
        argv += ["--initialize-checkpoint", str(options.initialize_checkpoint)]
    parsed = base.parse_args(argv)
    expected_init = None if options.resume else options.initialize_checkpoint
    expected_resume = resume_checkpoint if options.resume else None
    if parsed.resume != expected_resume or parsed.initialize_checkpoint != expected_init:
        raise ValueError("Checkpoint initialization or resume semantics changed")
    plan = {
        "initialization_checkpoint": str(options.initialize_checkpoint),
        "arguments": arguments,
        "overrides": overrides,
        "dry_run": options.dry_run,
    }
    (options.output / f"recipe{suffix}.json").write_text(
        json.dumps(plan, indent=2, default=str) + "\n"
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
            *packed,
            options.output / f"packed-metadata-overlay{suffix}",
            blob_paths=options.packed_blob_paths,
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
                if value.is_floating_point() and not torch.isfinite(value).all():
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
            "global_step": saved["global_step"],
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": components.sha(checkpoint),
            "finite_tensors": counts,
        }
        (options.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if overlay is not None:
            overlay.close()


if __name__ == "__main__":
    main()
