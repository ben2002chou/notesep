"""Train NoteSep from an explicit data configuration.

Run `python train.py --cfg job` to inspect the selected recipe.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent


def required_path(value: str, name: str) -> str:
    if not value or value == "???":
        raise ValueError(f"Set {name} to a local path")
    return str(Path(value).expanduser().resolve())


def run_notegrab(cfg: DictConfig, output: str) -> None:
    """Run one stage or the complete parent -> adaptation -> continuation recipe."""
    data = required_path(cfg.data.root, "data.root")
    full = cfg.training.recipe == "full"
    if full and Path(output).exists() and not cfg.run.resume:
        raise FileExistsError(output)
    if full and cfg.run.dry_run:
        raise ValueError("Inspect the full recipe with --cfg job; dry-run checks one stage")
    stages = ("parent", "adaptation", "continuation") if full else (cfg.training.recipe,)
    initialization = None
    if not full and stages[0] != "parent" and not cfg.run.resume:
        initialization = required_path(cfg.data.initialization, "data.initialization")
    for recipe in stages:
        folder = Path(output) / recipe if full else Path(output)
        steps = {"parent": 30000, "adaptation": 250, "continuation": 5000}[recipe]
        result = folder / "result.json"
        finished = False
        if full and cfg.run.resume and result.is_file():
            saved = json.loads(result.read_text())
            finished = (saved.get("status") == "passed_training_execution"
                        and saved.get("global_step") == steps)
        checkpoint = folder / "training/checkpoints/latest.pt"
        if finished and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not finished:
            command = [
                sys.executable, str(ROOT / "training/train_notegrab.py"),
                "--output", str(folder), "--recipe", recipe, "--data-root", data,
            ]
            if cfg.run.resume and folder.exists():
                command.append("--resume")
            elif initialization is not None:
                command += ["--initialize-checkpoint", initialization]
            if cfg.run.dry_run:
                command.append("--dry-run")
            subprocess.run(command, cwd=ROOT, check=True)
        initialization = str(checkpoint)
    if full:
        selected = Path(output) / "continuation/training/checkpoints/step-000002500.pt"
        if not selected.is_file():
            raise FileNotFoundError(selected)
        (Path(output) / "result.json").write_text(json.dumps({
            "status": "finished", "selected_checkpoint": str(selected),
            "stages": {name: str(Path(output) / name / "result.json") for name in stages},
        }, indent=2) + "\n")


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run = required_path(cfg.run.dir, "run.dir")
    if cfg.stage == "notegrab":
        run_notegrab(cfg, run)
        return
    elif cfg.stage == "aso":
        if cfg.run.resume:
            raise ValueError("run.resume is supported for NoteGrab training")
        data_root = Path(required_path(cfg.data.root, "data.root"))
        manifest = json.loads((data_root / "manifest.json").read_text())
        if manifest.get("component") != "aso_training_inputs":
            raise ValueError("data.root must point to the prepared ASO package")
        inputs = {}
        for name in ("fresh_index", "recorded_root"):
            path = (data_root / manifest["training"][name]).resolve()
            if not path.is_relative_to(data_root):
                raise ValueError("ASO input must stay inside its data package")
            inputs[name] = str(path)
        settings = {
            "schema": "notesep_aso_training_config_v1",
            "arm": "acoustic_multiplicative",
            "mode": "full",
            "seed": int(cfg.training.seed),
            **inputs,
            "proposal_root": required_path(
                cfg.data.proposal_root, "data.proposal_root"
            ),
        }
        if cfg.data.get("separator"):
            settings["separator_checkpoint"] = required_path(
                cfg.data.separator, "data.separator"
            )
        config_path = Path(HydraConfig.get().runtime.output_dir) / "aso-config.json"
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
        command = [
            sys.executable,
            str(ROOT / "training/train_aso.py"),
            "--config",
            str(config_path),
            "--output",
            run,
        ]
        command += ["--path-map", str(data_root / "path-map.json")]
    else:
        raise ValueError(f"Unknown training stage: {cfg.stage}")
    if cfg.run.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
