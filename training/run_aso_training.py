"""Run ASO with the fixed numerical settings used by the paper recipe."""

import argparse
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


def note_axis_scan_policy(torch):
    """Use fixed-order accumulation for the ASO note-axis cumulative sum."""

    def ordered_scan(values, dim, reverse=False):
        if values.shape[dim] == 0:
            return values.clone()
        columns = values.unbind(dim)
        accumulator = torch.zeros_like(columns[0])
        results = []
        for column in reversed(columns) if reverse else columns:
            accumulator = accumulator + column
            results.append(accumulator)
        if reverse:
            results.reverse()
        return torch.stack(results, dim=dim)

    class FixedOrderScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, dim):
            ctx.dim = dim
            return ordered_scan(values, dim)

        @staticmethod
        def backward(ctx, gradient):
            return ordered_scan(gradient, ctx.dim, reverse=True), None

    @contextmanager
    def apply_policy():
        original = torch.Tensor.cumsum

        def scoped(values, dim, *, dtype=None):
            eligible = (
                values.is_cuda
                and values.dtype == torch.float32
                and values.ndim == 4
                and dim in (1, -3)
                and values.shape[1] > 0
                and values.numel() != values.shape[1]
                and dtype in (None, torch.float32)
            )
            if eligible:
                if (
                    not torch.are_deterministic_algorithms_enabled()
                    or torch.is_deterministic_algorithms_warn_only_enabled()
                ):
                    raise RuntimeError(
                        "Fixed note-axis scan requires strict deterministic mode"
                    )
                return FixedOrderScan.apply(values, 1)
            return original(values, dim, dtype=dtype)

        torch.Tensor.cumsum = scoped
        try:
            yield
        finally:
            torch.Tensor.cumsum = original

    return apply_policy()


def bind_policy(path, policy):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != policy:
            raise ValueError(
                "Deterministic training runtime differs from the saved policy"
            )
        return
    if any((path.parent / name).exists() for name in ("CONTRACT.json", "RESUME.pt")):
        raise ValueError(
            "Existing output has no matching deterministic policy. "
            "Choose a new output directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(policy, indent=2, sort_keys=True) + "\n")


def policy_sidecar(path):
    """Keep the fresh output absent until the unmodified trainer creates it."""
    output = Path(path).parent
    return output.with_name(output.name + ".RUNTIME_POLICY.json")


def bind_training_policy(path, policy):
    path = Path(path)
    sidecar = policy_sidecar(path)
    if path.exists():
        bind_policy(path, policy)
    elif not sidecar.exists() and any(
        (path.parent / name).exists() for name in ("CONTRACT.json", "RESUME.pt")
    ):
        raise ValueError(
            "Existing output has no matching deterministic policy. "
            "Choose a new output directory."
        )
    bind_policy(sidecar, policy)
    return sidecar


def run_bound_entry(entry, remaining, path, policy):
    path = Path(path)
    indices = [i for i, value in enumerate(remaining) if value == "--output"]
    if (
        len(indices) != 1
        or indices[0] + 1 >= len(remaining)
        or Path(remaining[indices[0] + 1]).resolve() != path.parent.resolve()
    ):
        raise ValueError("Trainer output must match the runtime-policy output")
    if path.parent.exists() and not (path.parent / "RESUME.pt").exists():
        raise FileExistsError(path.parent)
    bind_training_policy(path, policy)
    previous_argv = sys.argv
    try:
        sys.argv = [str(entry), *remaining]
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.argv = previous_argv
        # The sidecar is authoritative before any optimizer update, including
        # abrupt termination. Mirror it inside a trainer-created output after
        # return/failure; never create the fresh output ahead of the trainer.
        if path.parent.is_dir():
            if path.exists():
                bind_policy(path, policy)
            else:
                with path.open("x") as stream:
                    stream.write(json.dumps(policy, indent=2, sort_keys=True) + "\n")


def parse_runtime_arguments(argv=None):
    # Downstream --output must not abbreviate our --output-policy option.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--output-policy", type=Path, required=True)
    return parser.parse_known_args(argv)


def main():
    args, remaining = parse_runtime_arguments()
    entry = args.entry.resolve()
    # Must precede Torch import and CUDA initialization. The original trainer
    # separately retains its fixed seeds and disables matmul/cuDNN TF32.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The original ASO trainer requires CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    drivers = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        )
        .strip()
        .splitlines()
    )
    driver_versions = sorted(set(line.strip() for line in drivers))
    if len(driver_versions) != 1:
        raise RuntimeError(
            "Could not identify a single installed NVIDIA driver version"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "soundfile")
    }
    policy = {
        "schema": "notesep_aso_deterministic_runtime_v2",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "note_axis_scan": ("fixed_order_float32_forward_and_reverse_derivative_v1"),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_driver_version": driver_versions[0],
        "packages": versions,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    with note_axis_scan_policy(torch):
        run_bound_entry(entry, remaining, args.output_policy, policy)


if __name__ == "__main__":
    main()
