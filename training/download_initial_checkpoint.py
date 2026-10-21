"""Download the pinned original NoteGrab training parents with full-file checks."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
from huggingface_hub import hf_hub_download

REPOSITORY = 'ben2002chou/notesep-release-candidate'
REVISION = '2841f0a58a0f541f8f27c9cd5bb39c2bed544f81'
FILES = {
    'f60f028a.pt': ('f60f028a12e1dec8fef94da64da1230d63b2d6f709ca6dcb49614d09cec7eec5', 693176430),
    '0441537c.pt': ('0441537c15c41764575122d3749278018c1191185f0b8000c0f339c35aa36623', 693169646),
    'views/30k-parent.json': ('0c10443ebc6efb7aebe8ca50c93506dac1c46e8477aa25f5c911519f578a744d', 1157935),
    'views/250-sampler.json': ('fef76c34a63fe4cc71027ef0129ec516dec067f88d3834f3b5f4688dd886a895', 5523965),
    'views/selected-continuation.json': ('f0923fde28f3d4406b8e152fb731b13ad8c1f17231bcb1307df0d7f05a5ebd5d', 3858160),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def download(output, cache_dir=None):
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='notesep-training-', dir=output.parent) as temp:
        staged = Path(temp) / 'lineage'
        staged.mkdir()
        for name, (checksum, size) in FILES.items():
            source = Path(hf_hub_download(REPOSITORY, 'training-lineage/' + name,
                                         revision=REVISION, cache_dir=cache_dir))
            if source.stat().st_size != size or sha(source) != checksum:
                raise ValueError('Original training checkpoint identity differs: ' + name)
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged / name)
            if sha(staged / name) != checksum:
                raise ValueError('Materialized training checkpoint differs: ' + name)
        receipt = {'status': 'passed', 'family': 'paper-table1-notegrab-lineage', 'repository': REPOSITORY, 'revision': REVISION,
            'files_sha256': {name: value[0] for name, value in FILES.items()},
            'scope': 'Original training parent and stage-view bytes. No claim of historical executed source closure, full training replay or corpus admission.'}
        (staged / 'DOWNLOAD.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    return {**receipt, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.cache_dir), indent=2))


if __name__ == '__main__':
    main()
