"""Finite, prefix-matched recorded-data schedules for bounded weight tests.

This module does not alter existing frozen experiments. A successor must bind
its optimizer horizon and recorded share explicitly in its run contract.
"""
from fractions import Fraction


def recorded_schedule(optimizer_steps: int, recorded_share: str) -> list[tuple[int, int]]:
    """Return (one-based optimizer step, K), preserving a shared K prefix.

    Share is a fraction of optimizer updates, not a fraction of audio seconds
    or individual notes. Each 125 recorded updates contains K2/K3/K4 counts
    78/24/23. Partial final cycles preserve exactly the same prefix.
    """
    if optimizer_steps <= 0:
        raise ValueError('optimizer_steps must be positive')
    share = Fraction(recorded_share)
    if not 0 <= share < 1:
        raise ValueError('recorded share must be in [0,1)')
    target, used, cycle = {2: 78, 3: 24, 4: 23}, {2: 0, 3: 0, 4: 0}, []
    for ordinal in range(125):
        k = max(target, key=lambda k: ((ordinal+1)*target[k]-125*used[k], -k))
        cycle.append(k)
        used[k] += 1
    assert used == target
    output = []
    for step in range(1, optimizer_steps+1):
        if (step*share).__floor__() > ((step-1)*share).__floor__():
            output.append((step, cycle[len(output) % len(cycle)]))
    return output
