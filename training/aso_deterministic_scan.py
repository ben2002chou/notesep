"""Explicit fixed-order note-axis scan for the Torch2.3 training adapter.

The original CUDA outer-axis kernel accumulates each row sequentially. This
adapter expresses that order with ordinary tensor additions, including the
reverse scan in its derivative, while keeping strict deterministic mode on.
It is a new declared runtime implementation, not historical executable identity.
"""
from contextlib import contextmanager

import torch


def _ordered_scan(values, dim, reverse=False):
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


class _FixedOrderScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, dim):
        ctx.dim = dim
        return _ordered_scan(values, dim)

    @staticmethod
    def backward(ctx, gradient):
        return _ordered_scan(gradient, ctx.dim, reverse=True), None


def fixed_order_cumsum(values, dim):
    return _FixedOrderScan.apply(values, dim)


@contextmanager
def note_axis_scan_policy():
    """Override only CUDA float32 [B,K,F,T] Tensor.cumsum along K.

    Other scans retain the original operation and strict rejection policy.
    No global determinism flag, dtype, model source or optimizer is changed.
    """
    original = torch.Tensor.cumsum

    def scoped(values, dim, *, dtype=None):
        eligible = (values.is_cuda and values.dtype == torch.float32
            and values.ndim == 4 and dim in (1, -3) and values.shape[1] > 0
            and values.numel() != values.shape[1] and dtype in (None, torch.float32))
        if eligible:
            if (not torch.are_deterministic_algorithms_enabled()
                    or torch.is_deterministic_algorithms_warn_only_enabled()):
                raise RuntimeError('Fixed note-axis scan requires strict deterministic mode')
            return fixed_order_cumsum(values, 1)
        return original(values, dim, dtype=dtype)

    torch.Tensor.cumsum = scoped
    try:
        yield
    finally:
        torch.Tensor.cumsum = original
