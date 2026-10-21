import hashlib
import json

import pytest

from notesep.training_data import dataset_files, verify, write_path_map
from notesep.cli import main as notesep_main


def test_dataset_manifest_verifies_bytes_and_rejects_changes(tmp_path):
    data = tmp_path / "data"
    (data / "notes").mkdir(parents=True)
    path = data / "notes/one.bin"
    path.write_bytes(b"one note")
    manifest = {
        "schema": "notesep_dataset_files_v1",
        "files": [
            {
                "path": "notes/one.bin",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    assert verify(manifest, data) == 1
    path.write_bytes(b"two note")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(manifest, data)


def test_dataset_manifest_rejects_escape_and_duplicates():
    row = {"path": "../outside", "bytes": 0, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Invalid dataset path"):
        dataset_files({"schema": "notesep_dataset_files_v1", "files": [row]})
    row["path"] = "same.bin"
    with pytest.raises(ValueError, match="Invalid dataset path"):
        dataset_files({"schema": "notesep_dataset_files_v1", "files": [row, row]})


def test_dataset_verification_rejects_symlinked_parent(tmp_path):
    output = tmp_path / "download"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "one.bin").write_bytes(b"one")
    (output / "notes").symlink_to(outside, target_is_directory=True)
    manifest = {
        "schema": "notesep_dataset_files_v1",
        "files": [
            {
                "path": "notes/one.bin",
                "bytes": 3,
                "sha256": hashlib.sha256(b"one").hexdigest(),
            }
        ],
    }
    with pytest.raises(ValueError, match="Missing or changed"):
        verify(manifest, output)


def test_verified_dataset_writes_local_training_path_map(tmp_path):
    output = tmp_path / "download"
    (output / "audio").mkdir(parents=True)
    payload = output / "audio/one.bin"
    payload.write_bytes(b"one")
    manifest = {
        "schema": "notesep_dataset_files_v1",
        "files": [
            {
                "path": "audio/one.bin",
                "bytes": 3,
                "sha256": hashlib.sha256(b"one").hexdigest(),
            }
        ],
        "path_map": {"/old/audio": "audio"},
    }
    assert verify(manifest, output) == 1
    path_map = write_path_map(manifest, output)
    assert path_map is not None
    assert '"/old/audio":' in path_map.read_text()
    assert str(output / "audio") in path_map.read_text()


def test_dataset_path_map_rejects_escape(tmp_path):
    output = tmp_path / "download"
    output.mkdir()
    manifest = {
        "schema": "notesep_dataset_files_v1",
        "files": [{"path": "one.bin", "bytes": 0, "sha256": "0" * 64}],
        "path_map": {"/old/audio": "../outside"},
    }
    with pytest.raises(ValueError, match="Invalid dataset root"):
        write_path_map(manifest, output)


def test_download_data_command_verifies_local_manifest(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.bin").write_bytes(b"sample")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "notesep_dataset_files_v1",
        "files": [{
            "path": "sample.bin",
            "bytes": 6,
            "sha256": hashlib.sha256(b"sample").hexdigest(),
        }],
    }))
    notesep_main(["download-data", "--manifest", str(manifest), "--output", str(data)])
    assert "Verified 1 dataset files" in capsys.readouterr().out
