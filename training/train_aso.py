"""Run the ASO paper training recipe."""

import argparse
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if settings["schema"] != "notesep_aso_training_config_v1":
        raise ValueError("Unknown configuration schema")
    if settings.get("mode") != "full":
        raise ValueError("The release entry point runs the full paper recipe")
    source = ROOT / "training/aso"
    configure(args.path_map)
    paths = {
        name: relocated(settings[name])
        for name in ["fresh_index", "recorded_root", "proposal_root"]
    }
    command = [
        sys.executable,
        str(source / "train.py"),
        "--arm",
        settings["arm"],
        "--seed",
        str(settings["seed"]),
        "--output",
        str(args.output.resolve()),
    ]
    for name, path in paths.items():
        command.extend(["--" + name.replace("_", "-"), str(path)])
    if settings.get("separator_checkpoint"):
        checkpoint = Path(settings["separator_checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        command.extend(["--separator-checkpoint", str(checkpoint)])
    if args.path_map:
        command.extend(["--path-map", str(args.path_map.resolve())])
    policy_path = args.output.resolve() / "RUNTIME_POLICY.json"
    runtime = ROOT / "training/run_aso_training.py"
    entry = Path(command[1])
    command = [
        sys.executable,
        str(runtime),
        "--entry",
        str(entry),
        "--output-policy",
        str(policy_path),
        *command[2:],
    ]
    missing = [
        path
        for path in (
            paths["fresh_index"],
            paths["recorded_root"] / "FROZEN.json",
            paths["proposal_root"] / "VALIDATION.json",
        )
        if not path.is_file()
    ]
    if missing:
        names = ", ".join(map(str, missing))
        raise FileNotFoundError("Missing ASO inputs: " + names)
    if args.dry_run:
        print("ASO inputs are present. Training was not started.", flush=True)
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
