"""Train NoteSep from an explicit data configuration.

Run `python train.py --cfg job` to inspect the selected recipe.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent
ASO_PREREG = "training/aso/docs/preregister_acoustic_multiplicative_temporal_aso_full_v1_20260910.md"
ASO_PREREG_SHA = "c689418c734a94ec34d973135154402d376401fb78a720742b65eb53602e6b70"


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
        if cfg.training.canary_steps is not None:
            command += ["--canary-steps", str(cfg.training.canary_steps)]
    elif cfg.stage in ("aso", "aso_canary"):
        settings = {
            "schema": "notesep_aso_training_config_v1",
            "arm": "acoustic_multiplicative",
            "mode": cfg.training.mode,
            "seed": int(cfg.training.seed),
            "initialization": "scratch",
            "regret_reference": "best_raw_or_magnitude",
            "fresh_index": required_path(cfg.data.fresh_index, "data.fresh_index"),
            "recorded_root": required_path(
                cfg.data.recorded_root, "data.recorded_root"
            ),
            "proposal_root": required_path(
                cfg.data.proposal_root, "data.proposal_root"
            ),
            "proposal_validation_sha256": cfg.training.proposal_validation_sha256,
            "preregistration": ASO_PREREG,
            "preregistration_sha256": ASO_PREREG_SHA,
        }
        if cfg.stage == "aso_canary":
            settings["release_successor"] = {
                "admission": "canary only, no benchmark test data"
            }
        config_path = Path(HydraConfig.get().runtime.output_dir) / "aso-config.json"
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
        command = [
            sys.executable,
            str(ROOT / "training/train_aso.py"),
            "--config",
            str(config_path),
            "--mode",
            cfg.training.mode,
            "--output",
            run,
        ]
        if cfg.data.path_map:
            command += ["--path-map", required_path(cfg.data.path_map, "data.path_map")]
        if cfg.training.deterministic:
            command.append("--deterministic")
    else:
        raise ValueError(f"Unknown training stage: {cfg.stage}")
    if cfg.run.dry_run:
        command.append("--dry-run")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
