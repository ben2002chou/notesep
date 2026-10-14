"""Pinned checkpoint retrieval with byte-level verification."""
import hashlib
import os
from pathlib import Path
import shutil

REPOSITORY = "ben2002chou/notesep-studio"
REVISION = "935406ef321179b758f2f3e34ef8940db2902246"
PREVIEW_CHECKPOINTS = {
    "separator.pt": "a3535f3ad99a297967dd18a4c4b8a9a8f7d47f3c6b05f72c0e2449d8c4be2d68",
    "aso.pt": "ff89599cef65e5eb98ff061372d8362e0752d2060d167aaf553b24c29577ae2a",
}
PROFILES = {
    'studio-preview': PREVIEW_CHECKPOINTS,
    'paper-table1': {
        'separator.pt': '94f122e5335a60f6a2992f71454d3e561f3482017b2fe8d33abaf4dd3e286820',
        'aso.pt': 'f6183cfcb02b32d5e29d4dc37deb3345b83dbac6095483348f177f1104853f06',
    },
}
CHECKPOINTS = PROFILES['paper-table1']
PAPER_REPOSITORY = 'ben2002chou/notesep-release-candidate'
PAPER_REVISION = 'a1bc08d81c1af82dd86369295cb242869e872ac4'


def model_profile(profile=None):
    selected = profile or os.environ.get('NOTESEP_PROFILE', 'paper-table1')
    if selected not in PROFILES:
        raise ValueError(f"Unknown model profile: {selected}")
    return selected


def checkpoint_hashes(profile=None):
    return dict(PROFILES[model_profile(profile)])


def weights_directory(profile=None):
    selected = model_profile(profile)
    default = Path.home() / '.cache/notesep/weights'
    if selected != 'studio-preview':
        default = default / selected
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
    """Requires author access while both model repositories remain private."""
    from huggingface_hub import hf_hub_download
    selected = model_profile(profile)
    root = Path(directory) if directory else weights_directory(selected)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in checkpoint_hashes(selected).items():
        target = root / name
        if target.is_file() and sha256(target) == expected:
            continue
        if selected == 'paper-table1':
            cached = Path(hf_hub_download(PAPER_REPOSITORY, f'paper-table1/{name}',
                                         repo_type='model', revision=PAPER_REVISION))
        else:
            cached = Path(hf_hub_download(REPOSITORY, f"weights/{name}",
                                         repo_type="space", revision=REVISION))
        if sha256(cached) != expected:
            raise ValueError(f"Downloaded checkpoint checksum mismatch: {name}")
        temporary = root / (name + ".partial")
        shutil.copyfile(cached, temporary)
        temporary.replace(target)
    return verify(root, profile=selected)
