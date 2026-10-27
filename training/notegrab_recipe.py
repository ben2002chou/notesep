"""Load the NoteGrab recipe and configure its data sources."""

import hashlib
import json
import os
import sys
from pathlib import Path

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



def configure_recipe(recipe, path_map):
    """Set up the data loaders and model used by a training stage."""
    sys.path.insert(0, str(ROOT / "src"))
    from notesep.training_paths import configure, relocated

    configure(path_map)
    contract = load_contract(recipe)
    environment = json.loads((ROOT / "training/notegrab/environment.json").read_text())
    os.environ.update(environment["variables"])
    gate = contract["loss_contract"]["decomposition"]["selective_gate"]
    os.environ.update(
        OCCURRENCE_SELECTIVE_SOURCE_ROOT=str(MODELS / "gating"),
        OCCURRENCE_FACTORIAL_SOURCE_ROOT=str(MODELS / "separator"),
        OCCURRENCE_FULL_SCORE_INDEX=str(relocated(gate["score_index"])),
        REAL_EXAMPLE_FRACTION=str(contract["arguments"]["real_example_fraction"]),
        MAX_STEPS=str(contract["arguments"]["max_steps"]),
    )
    if recipe == "parent":
        sys.path[:0] = [str(MODELS / "gating/src"), str(MODELS / "separator/src"),
                        str(MODELS / "separator")]
        import model_exp  # Load the gate package before the trainer adds its paths.
        import scripts.train_notegrab_separator as selective
        return selective.base

    fraction = str(contract["arguments"]["real_example_fraction"])
    os.environ.update(
        SCNS_MINIMUM_REAL_EXAMPLE_FRACTION=fraction,
        SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION=fraction,
        SCNS_V3_COMBINED_INSTRUMENT_COVERAGE_AUTHORITY=str(
            relocated(contract["sampler_contract"]["aggregate_16_coverage_authority"])
        ),
    )
    sys.path.insert(0, str(DATA / "scripts"))
    if recipe == "adaptation":
        import library_balanced_sampling
        from event_cluster_data import base
        import native_sampling

        native_sampling.install(base, relocated(contract["native_index"]["path"]))
        return base
    import parent_balanced_training
    return parent_balanced_training.trainer
