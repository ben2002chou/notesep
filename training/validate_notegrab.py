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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def differences(actual, expected, path=""):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return [
            item
            for key in sorted(set(actual) | set(expected))
            for item in (
                [
                    {
                        "path": path + "/" + key,
                        "missing": (
                            "actual" if key not in actual else "expected"
                        ),
                    }
                ]
                if key not in actual or key not in expected
                else differences(actual[key], expected[key], path + "/" + key)
            )
        ]
    if actual != expected:
        return [{"path": path, "actual": actual, "expected": expected}]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path-map", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(ROOT / "src"))
    from notesep.training_paths import configure, relocated

    configure(args.path_map)
    contract_path = ROOT / "training/notegrab/training_contract.json"
    if args.output is None:
        raise ValueError("--output is required for the full component check")
    contract = json.loads(contract_path.read_text())
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
        REAL_EXAMPLE_FRACTION=str(
            contract["arguments"]["real_example_fraction"]
        ),
        MAX_STEPS=str(contract["arguments"]["max_steps"]),
        SCNS_MINIMUM_REAL_EXAMPLE_FRACTION=str(
            contract["arguments"]["real_example_fraction"]
        ),
        SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION=str(
            contract["arguments"]["real_example_fraction"]
        ),
        SCNS_V3_COMBINED_INSTRUMENT_COVERAGE_AUTHORITY=str(
            relocated(
                contract["sampler_contract"]["aggregate_16_coverage_authority"]
            )
        ),
    )
    sys.path.insert(0, str(DATA / "scripts"))
    import parent_balanced_training as entry

    base = entry.cropped.v2.trackbound.v1.base
    selective = entry.cropped.v2.trackbound.v1.selective
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
    resolved = {
        "sampler": inspect.getsourcefile(base.DeterministicMixedBatchSampler),
        "gate": inspect.getsourcefile(selective.lower_member_exact_octave_gate),
        "augmentation": inspect.getsourcefile(
            selective.apply_occurrence_augmentation
        ),
        "base": base.__file__,
        "selective": selective.__file__,
    }
    component_roots = [DATA.resolve(), MODELS.resolve()]
    for name in resolved.values():
        source = Path(name).resolve()
        if not any(source.is_relative_to(root) for root in component_roots):
            raise ValueError(
                "Runtime dependency outside NoteGrab code: " + name
            )
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
