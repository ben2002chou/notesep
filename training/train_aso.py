"""Run the portable exact ASO trainer with explicit pinned input identities."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from notesep.training_paths import configure, relocated  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--path-map", type=Path)
    parser.add_argument("--mode", choices=["canary", "screen", "full"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Explicit deterministic Torch/cuDNN/cuBLAS policy; saved runtime "
            "must match on resume"
        ),
    )
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if settings["schema"] != "notesep_aso_training_config_v1":
        raise ValueError("Unknown configuration schema")
    if (
        settings.get("release_successor")
        and (args.mode or settings["mode"]) != "canary"
    ):
        raise ValueError(
            "The regenerated release cache is admitted for the 320-step "
            "canary only; full training admission remains separate"
        )
    source_manifest = json.loads(
        (ROOT / "training/aso/source_manifest.json").read_text()
    )
    source = ROOT / source_manifest["source_root"]
    for name, checksum in source_manifest["source_files_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Training source identity mismatch: " + name)
    for name, checksum in source_manifest["helper_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(
                "Training transport helper identity mismatch: " + name
            )
    prereg = ROOT / settings["preregistration"]
    if (
        hashlib.sha256(prereg.read_bytes()).hexdigest()
        != settings["preregistration_sha256"]
    ):
        raise ValueError("Preregistration identity mismatch")
    configure(args.path_map)
    paths = {
        name: relocated(settings[name])
        for name in ["fresh_index", "recorded_root", "proposal_root"]
    }
    command = [
        sys.executable,
        str(source / "scripts/train_acoustic_ownership.py"),
        "--arm",
        settings["arm"],
        "--mode",
        args.mode or settings["mode"],
        "--seed",
        str(settings["seed"]),
        "--expected-proposal-validation-sha256",
        settings["proposal_validation_sha256"],
        "--scale-preregistration",
        str(prereg),
        "--output",
        str(args.output.resolve()),
    ]
    for name, path in paths.items():
        command.extend(["--" + name.replace("_", "-"), str(path)])
    if args.path_map:
        command.extend(["--path-map", str(args.path_map.resolve())])
    policy_path = args.output.resolve() / "RUNTIME_POLICY.json"
    policy_sidecar = args.output.resolve().with_name(
        args.output.name + ".RUNTIME_POLICY.json"
    )
    if (
        policy_path.exists() or policy_sidecar.exists()
    ) and not args.deterministic:
        raise ValueError(
            "This output was started with --deterministic; retain that "
            "flag on resume"
        )
    if args.deterministic:
        runtime = ROOT / "training/run_aso_deterministic.py"
        entry = Path(command[1])
        command = [
            sys.executable,
            str(runtime),
            "--entry",
            str(entry),
            "--expected-entry-sha256",
            hashlib.sha256(entry.read_bytes()).hexdigest(),
            "--output-policy",
            str(policy_path),
            *command[2:],
        ]
    report = {
        "command": command,
        "preregistration_sha256": settings["preregistration_sha256"],
        "runtime_policy": (
            "deterministic_v2_fixed_order_note_scan"
            if args.deterministic
            else "original_runtime_defaults"
        ),
        "missing_inputs": [
            str(path)
            for path in [
                paths["fresh_index"],
                paths["recorded_root"] / "FROZEN.json",
                paths["proposal_root"] / "VALIDATION.json",
            ]
            if not path.is_file()
        ],
        "scope": (
            "Exact pinned-input training command; a dry run is not dataset "
            "admission or execution evidence."
        ),
    }
    print(json.dumps(report, indent=2), flush=True)
    if report["missing_inputs"]:
        raise FileNotFoundError(
            "Missing ASO inputs: " + ", ".join(report["missing_inputs"])
        )
    if args.dry_run:
        return
    env = os.environ.copy()
    env["PREREG_SHA256"] = settings["preregistration_sha256"]
    env["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
