"""Check that NoteGrab's code matches its saved training recipe."""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/notegrab/data"
MODELS = ROOT / "training/notegrab/models"


def load_contract(recipe):
    """Load each stage's settings; adaptation keeps the parent's model and loss."""
    names = {
        "parent": "parent_contract.json", "adaptation": "adaptation_contract.json",
        "continuation": "training_contract.json",
    }
    folder = ROOT / "training/notegrab"
    settings = json.loads((folder / names[recipe]).read_text())
    if recipe == "adaptation":
        parent = json.loads((folder / names["parent"]).read_text())
        settings = parent | settings
    return settings


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def differences(actual, expected, path=""):
    """List changed or missing settings, including their nested dictionary paths."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        if actual == expected:
            return []
        return [{"path": path, "actual": actual, "expected": expected}]

    changes = []
    for key in sorted(set(actual) | set(expected)):
        # The release recipe omits descriptive model-version labels. Compare
        # every architecture setting, but do not require those omitted labels.
        if key == "contract_version" and key not in expected:
            continue
        location = path + "/" + key
        if key not in actual or key not in expected:
            missing = "actual" if key not in actual else "expected"
            changes.append({"path": location, "missing": missing})
        else:
            changes.extend(differences(actual[key], expected[key], location))
    return changes


def validate_stage_checkpoint(saved, recipe, *, step=None):
    """Check a trained checkpoint against its stage, independent of local paths."""
    if saved.get("schema") != "dna_residual_v13_streaming_checkpoint_v1":
        raise ValueError("Unknown NoteGrab training checkpoint")
    source = load_contract(recipe)
    args = source["arguments"]
    expected = {key: args[key] for key in (
        "batch_size", "fixed_samples", "max_steps", "lr_decay_steps", "seed",
        "learning_rate", "weight_decay",
    )}
    expected.update(optimizer_name="adamw", compile_mode=source["compile_mode"])
    expected_step = args["max_steps"] if step is None else step
    if saved.get("global_step") != expected_step:
        raise ValueError("Checkpoint has not reached the required training step")
    if differences(saved.get("training_contract"), expected):
        raise ValueError("Checkpoint uses a different training recipe")

    def without_paths(value):
        # File identities stay fixed when a dataset moves to another disk.
        if isinstance(value, dict):
            return {key: without_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [without_paths(item) for item in value]
        if isinstance(value, str) and value.startswith("/"):
            return "<data-path>"
        return value

    for name in ("model_contract", "loss_contract", "sampler_contract"):
        if differences(without_paths(saved.get(name)), without_paths(source[name])):
            raise ValueError("Checkpoint differs: " + name)
    binding = saved.get("corpus_binding", {})
    synthetic = binding.get("synthetic", binding)
    identity = load_contract("parent")["corpus_identity"]
    if any(synthetic.get(key) != value for key, value in identity.items()):
        raise ValueError("Checkpoint uses a different synthetic corpus")
    if recipe == "continuation":
        if differences(without_paths(binding.get("recorded")),
                       without_paths(source["corpus_binding"]["recorded"])):
            raise ValueError("Checkpoint uses different recorded training data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path-map", type=Path)
    parser.add_argument(
        "--recipe", choices=("parent", "adaptation", "continuation"), default="continuation"
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(ROOT / "src"))
    from notesep.training_paths import configure, relocated

    configure(args.path_map)
    if args.output is None:
        raise ValueError("--output is required for the full component check")
    contract = load_contract(args.recipe)
    environment_path = ROOT / "training/notegrab/environment.json"
    environment = json.loads(environment_path.read_text())
    if environment["schema"] != "notesep_notegrab_paper_environment_v1":
        raise ValueError("Unknown NoteGrab environment")
    os.environ.update(environment["variables"])
    gate = contract["loss_contract"]["decomposition"]["selective_gate"]
    if sha(relocated(gate["score_index"])) != gate["score_index_sha256"]:
        raise ValueError("Original score index changed")
    os.environ.update(
        OCCURRENCE_SELECTIVE_SOURCE_ROOT=str(MODELS / "gating"),
        OCCURRENCE_FACTORIAL_SOURCE_ROOT=str(MODELS / "separator"),
        OCCURRENCE_FULL_SCORE_INDEX=str(relocated(gate["score_index"])),
        REAL_EXAMPLE_FRACTION=str(contract["arguments"]["real_example_fraction"]),
        MAX_STEPS=str(contract["arguments"]["max_steps"]),
    )
    if args.recipe == "parent":
        # The original 30k run predates the recorded-data sampler wrappers.
        sys.path[:0] = [
            str(MODELS / "gating/src"), str(MODELS / "separator/src"),
            str(MODELS / "separator"),
        ]
        import model_exp  # Load the gating package before the trainer adds its paths.
        import scripts.train_notegrab_separator as selective
        base = selective.base
    else:
        os.environ.update(
            SCNS_MINIMUM_REAL_EXAMPLE_FRACTION=str(contract["arguments"]["real_example_fraction"]),
            SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION=str(contract["arguments"]["real_example_fraction"]),
            SCNS_V3_COMBINED_INSTRUMENT_COVERAGE_AUTHORITY=str(
                relocated(contract["sampler_contract"]["aggregate_16_coverage_authority"])
            ),
        )
        sys.path.insert(0, str(DATA / "scripts"))
        if args.recipe == "adaptation":
            import library_balanced_sampling as entry
            from event_cluster_data import base, selective
            import native_sampling

            index = relocated(contract["native_index"]["path"])
            if sha(index) != contract["native_index"]["sha256"]:
                raise ValueError("Native-library row index changed")
            native_sampling.install(base, index)
        else:
            import parent_balanced_training as entry
            base = entry.trainer
            selective = entry.selective
    # Historical JSON serializes argparse Path values as strings.
    arguments = dict(contract["arguments"])
    for key in [
        "acoustic_offset_index",
        "duration_index",
        "frozen_manifest",
        "output_dir",
        "priority_index",
        "synthetic_ordinal_view",
    ]:
        if arguments.get(key) is not None:
            arguments[key] = relocated(arguments[key])
    arguments["recorded_frozen_root"] = [
        relocated(value) for value in arguments["recorded_frozen_root"]
    ]

    def mapped(value):
        if isinstance(value, dict):
            return {key: mapped(item) for key, item in value.items()}
        if isinstance(value, list):
            return [mapped(item) for item in value]
        if isinstance(value, str) and value.startswith("/"):
            return str(relocated(value))
        return value

    observed = {
        "model_contract": base.ACTIVE_MODEL_CONTRACT,
        "loss_contract": base.active_loss_contract(),
        "sampler_contract": base.sampler_contract(SimpleNamespace(**arguments)),
    }
    # Compare the persisted JSON contract, including list/tuple serialization.
    observed = json.loads(json.dumps(observed, allow_nan=False))
    comparison = {
        name: differences(value, mapped(contract[name]))
        for name, value in observed.items()
    }
    sampler = (
        base.ResumeSafeAudioBudgetBatchSampler if args.recipe in {"parent", "adaptation"}
        else base.DeterministicMixedBatchSampler
    )
    resolved = {
        "sampler": inspect.getsourcefile(sampler),
        "gate": inspect.getsourcefile(selective.lower_member_exact_octave_gate),
        "augmentation": inspect.getsourcefile(selective.apply_occurrence_augmentation),
        "base": base.__file__,
        "selective": selective.__file__,
    }
    component_roots = [DATA.resolve(), MODELS.resolve()]
    for name in resolved.values():
        source = Path(name).resolve()
        if not any(source.is_relative_to(root) for root in component_roots):
            raise ValueError("Runtime dependency outside NoteGrab code: " + name)
    passed = not any(comparison.values())
    report = {
        "status": "passed" if passed else "contract_mismatch",
        "runtime_components": {
            key: str(Path(value).resolve().relative_to(ROOT))
            for key, value in resolved.items()
        },
        "contract_differences": comparison,
        "observed_contracts": observed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "runtime_components": len(resolved),
                "contract_difference_counts": {
                    name: len(items) for name, items in comparison.items()
                },
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
