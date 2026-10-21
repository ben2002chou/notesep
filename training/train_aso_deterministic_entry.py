"""Apply an explicit, resume-bound deterministic policy to the unchanged trainer."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def bind_policy(path, policy):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != policy:
            raise ValueError('Deterministic training runtime differs from the saved policy')
        return
    if any((path.parent / name).exists() for name in ('CONTRACT.json', 'RESUME.pt')):
        raise ValueError('Existing training output has no deterministic policy; use a new output directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(policy, indent=2, sort_keys=True) + '\n')


def policy_sidecar(path):
    """Keep the fresh output absent until the unmodified trainer creates it."""
    output = Path(path).parent
    return output.with_name(output.name + '.RUNTIME_POLICY.json')


def bind_training_policy(path, policy):
    path = Path(path)
    sidecar = policy_sidecar(path)
    if path.exists():
        bind_policy(path, policy)
    elif not sidecar.exists() and any((path.parent / name).exists()
                                     for name in ('CONTRACT.json', 'RESUME.pt')):
        raise ValueError('Existing training output has no deterministic policy; use a new output directory')
    bind_policy(sidecar, policy)
    return sidecar


def run_bound_entry(entry, remaining, path, policy):
    path = Path(path)
    indices = [i for i, value in enumerate(remaining) if value == '--output']
    if len(indices) != 1 or indices[0] + 1 >= len(remaining) or \
            Path(remaining[indices[0] + 1]).resolve() != path.parent.resolve():
        raise ValueError('Trainer output must match the runtime-policy output')
    if path.parent.exists() and not (path.parent / 'RESUME.pt').exists():
        raise FileExistsError(path.parent)
    bind_training_policy(path, policy)
    previous_argv = sys.argv
    try:
        sys.argv = [str(entry), *remaining]
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        sys.argv = previous_argv
        # The sidecar is authoritative before any optimizer update, including
        # abrupt termination. Mirror it inside a trainer-created output after
        # return/failure; never create the fresh output ahead of the trainer.
        if path.parent.is_dir():
            if path.exists():
                bind_policy(path, policy)
            else:
                with path.open('x') as stream:
                    stream.write(json.dumps(policy, indent=2, sort_keys=True) + '\n')


def parse_runtime_arguments(argv=None):
    # Downstream --output must not abbreviate our --output-policy option.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--expected-entry-sha256', required=True)
    parser.add_argument('--output-policy', type=Path, required=True)
    parser.add_argument('--check-runtime-only', action='store_true', help='Bind/check runtime metadata without executing any training')
    return parser.parse_known_args(argv)


def main():
    args, remaining = parse_runtime_arguments()
    entry = args.entry.resolve()
    if hashlib.sha256(entry.read_bytes()).hexdigest() != args.expected_entry_sha256:
        raise ValueError('Pinned original training entry changed')
    # Must precede Torch import and CUDA initialization. The original trainer
    # separately retains its fixed seeds and disables matmul/cuDNN TF32.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('The original ASO trainer requires CUDA')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from aso_deterministic_scan import note_axis_scan_policy
    scan_source = Path(__file__).with_name('aso_deterministic_scan.py')
    drivers = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
        '--format=csv,noheader'], text=True, timeout=30).strip().splitlines()
    driver_versions = sorted(set(line.strip() for line in drivers))
    if len(driver_versions) != 1:
        raise RuntimeError('Could not identify a single installed NVIDIA driver version')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'soundfile')}
    policy = {'schema': 'notesep_aso_deterministic_runtime_v2',
        'entry_sha256': args.expected_entry_sha256,
        'runtime_entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'torch_git_version': torch.version.git_version,
        'note_axis_scan': 'fixed_order_float32_forward_and_reverse_derivative_v1',
        'note_axis_scan_sha256': hashlib.sha256(scan_source.read_bytes()).hexdigest(),
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'nvidia_driver_version': driver_versions[0], 'packages': versions,
        'torch_cpu_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': True, 'warn_only': False,
        'cudnn_deterministic': True, 'cudnn_benchmark': False,
        'matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
        'cublas_workspace_config': ':4096:8'}
    if args.check_runtime_only:
        bound = bind_training_policy(args.output_policy, policy)
        print(json.dumps({'status': 'runtime_policy_checked', 'policy': policy,
                          'policy_path': str(bound)}), flush=True)
        return
    with note_axis_scan_policy():
        run_bound_entry(entry, remaining, args.output_policy, policy)


if __name__ == '__main__':
    main()
