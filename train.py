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


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run = required_path(cfg.run.dir, "run.dir")
    if cfg.stage == "notegrab":
        command = [
            sys.executable,
            str(ROOT / "training/train_notegrab.py"),
            "--output",
            run,
            "--initialize-checkpoint",
            required_path(cfg.data.initialization, "data.initialization"),
            "--data-root",
            required_path(cfg.data.root, "data.root"),
        ]
    elif cfg.stage == "aso":
        settings = {
            "schema": "notesep_aso_training_config_v1",
            "arm": "acoustic_multiplicative",
            "mode": "full",
            "seed": int(cfg.training.seed),
            "fresh_index": required_path(
                cfg.data.fresh_index, "data.fresh_index"
            ),
            "recorded_root": required_path(
                cfg.data.recorded_root, "data.recorded_root"
            ),
            "proposal_root": required_path(
                cfg.data.proposal_root, "data.proposal_root"
            ),
        }
        config_path = (
            Path(HydraConfig.get().runtime.output_dir) / "aso-config.json"
        )
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
        command = [
            sys.executable,
            str(ROOT / "training/train_aso.py"),
            "--config",
            str(config_path),
            "--output",
            run,
        ]
        if cfg.data.path_map:
            command += [
                "--path-map",
                required_path(cfg.data.path_map, "data.path_map"),
            ]
    else:
        raise ValueError(f"Unknown training stage: {cfg.stage}")
    if cfg.run.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
