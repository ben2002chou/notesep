import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_notegrab import apply_data_root
from validate_notegrab import TRAINING_CONTRACT_SHA


def options(root):
    return SimpleNamespace(
        data_root=root,
        path_map=None,
        packed_metadata_archive=None,
        packed_metadata_manifest=None,
        packed_metadata_manifest_sha256=None,
    )


def test_data_root_resolves_verified_download(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "component": "original_notegrab_training_inputs",
                "files": [
                    {
                        "path": "packed-metadata-overlay/manifest.json",
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )
    configured = options(root)
    apply_data_root(configured)
    assert configured.path_map == root / "path-map.json"
    assert (
        configured.packed_metadata_archive
        == root / "packed-metadata-overlay/payload.zip"
    )
    assert configured.packed_metadata_manifest_sha256 == "a" * 64


def test_data_root_rejects_mixed_explicit_paths(tmp_path):
    configured = options(tmp_path)
    configured.path_map = tmp_path / "paths.json"
    with pytest.raises(ValueError, match="not both"):
        apply_data_root(configured)


def test_notegrab_contract_uses_portable_paths():
    repository = Path(__file__).resolve().parents[1]
    contract = repository / "training/notegrab/training_contract.json"
    payload = json.loads(contract.read_text())

    assert (
        hashlib.sha256(contract.read_bytes()).hexdigest()
        == TRAINING_CONTRACT_SHA
    )

    def values(value):
        if isinstance(value, dict):
            return [item for child in value.values() for item in values(child)]
        if isinstance(value, list):
            return [item for child in value for item in values(child)]
        return [value] if isinstance(value, str) else []

    paths = [value for value in values(payload) if value.startswith("/")]
    assert paths
    assert all(
        value.startswith(
            ("/note-sep-data/", "/note-sep-checkpoints/", "/note-sep-output/")
        )
        for value in paths
    )
