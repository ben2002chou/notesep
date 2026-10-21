import hashlib
import json
import sqlite3
import zipfile

from notesep.training_paths import configure
from training.stage_metadata import PackedMetadataOverlay


def test_archived_metadata_reads_relocated_audio_shard(tmp_path):
    local = tmp_path / "data/item"
    local.mkdir(parents=True)
    (local / "audio.bin").write_bytes(b"audio")
    sqlite = tmp_path / "manifest.sqlite"
    with sqlite3.connect(sqlite) as db:
        db.execute("CREATE TABLE assets (shard TEXT)")
        db.execute("INSERT INTO assets VALUES ('audio.bin')")
    complete = b"{}\n"
    members = {"item/manifest.sqlite": sqlite.read_bytes(), "item/COMPLETE.json": complete}
    archive = tmp_path / "metadata.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    manifest = tmp_path / "metadata.json"
    manifest.write_text(json.dumps({"roots": ["/original"], "files": [
        {"path": "/original/" + name, "member": name,
         "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in members.items()]}))
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({"/original": str(tmp_path / "data")}))
    try:
        configure(mapping)
        overlay = PackedMetadataOverlay(archive, manifest,
                                        hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                        tmp_path / "overlay")
        staged = overlay.resolve("/original/item")
        assert (staged / "audio.bin").resolve() == local / "audio.bin"
        assert overlay.resolve(local) == staged
        overlay.close()
    finally:
        configure()
