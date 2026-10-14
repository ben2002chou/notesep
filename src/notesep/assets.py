"""Pinned checkpoint retrieval with byte-level verification."""
import hashlib
import os
from pathlib import Path
import shutil

PROFILES = {
    'paper-table1': {
        'separator.pt': '94f122e5335a60f6a2992f71454d3e561f3482017b2fe8d33abaf4dd3e286820',
        'aso.pt': 'f6183cfcb02b32d5e29d4dc37deb3345b83dbac6095483348f177f1104853f06',
    },
}
CHECKPOINTS = PROFILES['paper-table1']
PAPER_REPOSITORY = 'ben2002chou/notesep'
PAPER_REVISION = '93d333118c49b407f37a226321a0896259599091'


def model_profile(profile=None):
    selected = profile or os.environ.get('NOTESEP_PROFILE', 'paper-table1')
    if selected not in PROFILES:
        raise ValueError(f"Unknown model profile: {selected}")
    return selected


def checkpoint_hashes(profile=None):
    return dict(PROFILES[model_profile(profile)])


def weights_directory(profile=None):
    selected = model_profile(profile)
    default = Path.home() / '.cache/notesep/weights' / selected
    return Path(os.environ.get("NOTESEP_WEIGHTS", default))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory=None, *, profile=None):
    root = Path(directory) if directory else weights_directory(profile)
    result = {}
    for name, expected in checkpoint_hashes(profile).items():
        actual = sha256(root / name)
        if actual != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {root / name}")
        result[name] = actual
    return result


def download(directory=None, *, profile=None):
    """Download and verify the pinned paper checkpoint pair."""
    from huggingface_hub import hf_hub_download
    selected = model_profile(profile)
    root = Path(directory) if directory else weights_directory(selected)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in checkpoint_hashes(selected).items():
        target = root / name
        if target.is_file() and sha256(target) == expected:
            continue
        cached = Path(hf_hub_download(PAPER_REPOSITORY, f'paper-table1/{name}',
                                     repo_type='model', revision=PAPER_REVISION))
        if sha256(cached) != expected:
            raise ValueError(f"Downloaded checkpoint checksum mismatch: {name}")
        temporary = root / (name + ".partial")
        shutil.copyfile(cached, temporary)
        temporary.replace(target)
    return verify(root, profile=selected)
