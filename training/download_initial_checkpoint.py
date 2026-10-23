"""Download NoteGrab's step-250 training checkpoint."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPOSITORY = "ben2002chou/notesep-release-candidate"
REVISION = "2841f0a58a0f541f8f27c9cd5bb39c2bed544f81"
FILES = {
    "0441537c.pt": (
        "0441537c15c41764575122d3749278018c1191185f0b8000c0f339c35aa36623",
        693169646,
    ),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output, cache_dir=None):
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="notesep-training-", dir=output.parent
    ) as temp:
        staged = Path(temp) / "checkpoint"
        staged.mkdir()
        for name, (checksum, size) in FILES.items():
            source = Path(
                hf_hub_download(
                    REPOSITORY,
                    "training-lineage/" + name,
                    revision=REVISION,
                    cache_dir=cache_dir,
                )
            )
            if source.stat().st_size != size or sha(source) != checksum:
                raise ValueError(
                    "Training checkpoint does not match the release: " + name
                )
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged / name)
            if sha(staged / name) != checksum:
                raise ValueError("Downloaded checkpoint changed while copying: " + name)
        receipt = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "files": {
                name: {"sha256": checksum, "size_bytes": size}
                for name, (checksum, size) in FILES.items()
            },
        }
        (staged / "DOWNLOAD.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    return {**receipt, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
