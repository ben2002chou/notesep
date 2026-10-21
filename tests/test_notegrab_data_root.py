import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_notegrab import apply_data_root


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
    (root / "manifest.json").write_text(json.dumps({
        "component": "original_notegrab_training_inputs",
        "files": [{
            "path": "packed-metadata-overlay/manifest.json",
            "sha256": "a" * 64,
        }],
    }))
    configured = options(root)
    apply_data_root(configured)
    assert configured.path_map == root / "path-map.json"
    assert configured.packed_metadata_archive == root / "packed-metadata-overlay/payload.zip"
    assert configured.packed_metadata_manifest_sha256 == "a" * 64


def test_data_root_rejects_mixed_explicit_paths(tmp_path):
    configured = options(tmp_path)
    configured.path_map = tmp_path / "paths.json"
    with pytest.raises(ValueError, match="not both"):
        apply_data_root(configured)
