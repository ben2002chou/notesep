#!/usr/bin/env python3
"""Old-contract target crops with uniform global physical-source sampling."""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

import target_cropped_training as cropped


class ParentBalancedTargetCroppedSampler(cropped.TargetCroppedLibraryBalancedSampler):
    def __init__(self, base_sampler: Any, **kwargs: Any) -> None:
        super().__init__(base_sampler, **kwargs)
        by_parent: dict[str, list[Any]] = defaultdict(list)
        for row in self.rows:
            by_parent[row.source_parent].append(row)
        self.rows_by_parent = {
            parent: tuple(sorted(rows, key=lambda row: row.row_index))
            for parent, rows in sorted(by_parent.items())
        }
        self.parents = tuple(self.rows_by_parent)
        self.parent_order = tuple(sorted(
            self.parents,
            key=lambda parent: (self._rank(f"parent-order:{parent}"), parent),
        ))

    def _select_rows(self, count: int, *, step: int) -> list[Any]:
        del step
        selected = []
        first_slot = int(self.cumulative_score_bound_rows)
        for local_slot in range(count):
            absolute_slot = first_slot + local_slot
            cycle, position = divmod(absolute_slot, len(self.parents))
            parent = self.parent_order[position]
            candidates = self.rows_by_parent[parent]
            ordered = tuple(sorted(
                candidates,
                key=lambda row: (
                    self._rank(f"row-cycle:{cycle}:{parent}:{row.row_index}"),
                    row.row_index,
                ),
            ))
            selected.append(ordered[cycle % len(ordered)])
        return selected


def sampler_contract(args: Any) -> dict[str, Any]:
    return cropped.sampler_contract(args) | {
        "strategy": "scns_v3_parent_balanced_target_cropped_v1",
        "source_balance": "resume_stable_uniform_physical_source_parent",
        "recorded_input_geometry": "historical_variable_window_midi_anchor_acoustic_release_4000_2560_min16384_q4096",
    }


cropped.v2.trackbound.v1.base.DeterministicMixedBatchSampler = ParentBalancedTargetCroppedSampler
cropped.v2.trackbound.v1.base.sampler_contract = sampler_contract


if __name__ == "__main__":
    os.environ.setdefault("SCNS_REQUIRE_ALL_16_RECORDED_FAMILIES", "0")
    cropped.v2.trackbound.v1.base.main()
