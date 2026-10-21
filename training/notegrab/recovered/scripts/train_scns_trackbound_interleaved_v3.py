#!/usr/bin/env python3
"""SCNS v3 data adapter: track-bound score scenes with neutral library sampling.

The separation model, losses, score conditioning, optimizer, and scheduler are
the frozen selective symmetric implementation.  This overlay only changes the
recorded-data selection contract: recorded rows replace ordinary rows within a
normal mixed batch, and an eligible source *library* is selected before a
physical sound identity.  It therefore does not make a recorded-only phase or
turn weak classes into a replay curriculum.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from torch.utils.data import Sampler

# Importing v1 installs its complete scene loader/collate/telemetry overlay on
# the frozen selective trainer.  We replace only its within-batch sampler.
import train_scns_eventcluster_interleaved_v1 as v1


@dataclass(frozen=True)
class LibraryRowRef:
    row_index: int
    samples: int
    k: int
    instrument: str
    source_library: str
    source_parent: str


class SourceLibraryThenSoundSampler(Sampler[list[Any]]):
    """Sample eligible libraries uniformly before choosing a sound identity."""

    def __init__(self, base_sampler: Iterable[list[Any]], *, recorded_window_samples: Any,
                 real_fraction: float, fixed_samples: int, seed: int, start_step: int) -> None:
        del recorded_window_samples
        if v1._TRAIN_SCENES is None:
            raise RuntimeError("track-bound SCNS v3 scenes were not loaded")
        minimum = float(os.environ.get("SCNS_MINIMUM_REAL_EXAMPLE_FRACTION", "0.20"))
        maximum = float(os.environ.get("SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION", str(1.0 / 3.0)))
        if not minimum <= float(real_fraction) <= maximum + 1e-9:
            raise ValueError(
                "REAL_EXAMPLE_FRACTION outside declared bounds: "
                f"fraction={real_fraction}, minimum={minimum}, maximum={maximum}"
            )
        self.base_sampler = base_sampler
        self.rows = self._rows()
        self.fraction = float(real_fraction)
        self.fixed_samples = int(fixed_samples)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.output = Path(os.environ.get("OUTPUT", ".")).expanduser().resolve()
        self.library_use: Counter[str] = Counter()
        self.parent_use: Counter[str] = Counter()

    def _rows(self) -> list[LibraryRowRef]:
        rows: list[LibraryRowRef] = []
        assert v1._TRAIN_SCENES is not None
        for scene in v1._TRAIN_SCENES.scenes:
            record = v1._TRAIN_SCENES.readers[scene.root_index].record(scene.scene_id, scene.record_kind)
            notes = list(record.metadata.get("notes", ()))
            if len(notes) != scene.k:
                raise ValueError(f"note metadata mismatch: {scene.scene_id}")
            for note_index, note in enumerate(notes):
                library = str(note.get("source_library", "")).strip()
                parent = str(note.get("source_parent_id", "")).strip()
                if not library or not parent:
                    raise ValueError(
                        "SCNS v3 requires source_library and source_parent_id on every materialized note"
                    )
                rows.append(LibraryRowRef(
                    scene.first_row + note_index, scene.samples, scene.k,
                    v1.canonical_instrument(str(note["recorded_instrument"])), library, parent,
                ))
        if not rows:
            raise RuntimeError("no track-bound training rows")
        return rows

    def __len__(self) -> int:
        return len(self.base_sampler)  # type: ignore[arg-type]

    def _rank(self, value: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{self.seed}:{value}".encode()).digest()[:8], "little")

    def _count(self, step: int, batch_size: int) -> int:
        exact = self.fraction * batch_size
        lower = math.floor(exact)
        return lower + int(self._rank(f"round:{step}") / 2**64 < exact - lower)

    def _choose(self, eligible: list[LibraryRowRef], *, step: int, slot: int) -> LibraryRowRef:
        by_library: dict[str, list[LibraryRowRef]] = defaultdict(list)
        for row in eligible:
            by_library[row.source_library].append(row)
        # Library selection is independent of how many clips that library has.
        # The stable tie break gives a reproducible, approximately uniform
        # rotation across libraries; it is not an instrument-loss weighting.
        library = min(by_library, key=lambda key: (
            self.library_use[key], self._rank(f"library:{step}:{slot}:{key}"), key
        ))
        row = min(by_library[library], key=lambda value: (
            self.parent_use[value.source_parent],
            self._rank(f"sound:{step}:{slot}:{value.row_index}"), value.row_index,
        ))
        self.library_use[library] += 1
        self.parent_use[row.source_parent] += 1
        return row

    def __iter__(self) -> Iterator[list[Any]]:
        for step, batch in enumerate(self.base_sampler, start=self.start_step):
            eligible_slots = [
                slot for slot, synthetic_index in enumerate(batch)
                if any(row.samples <= (int(synthetic_index[1]) if isinstance(synthetic_index, tuple) else self.fixed_samples)
                       for row in self.rows)
            ]
            requested = self._count(step, len(batch))
            slots = set(sorted(eligible_slots, key=lambda slot: self._rank(f"slot:{step}:{slot}"))[:min(requested, len(eligible_slots))])
            selected: list[LibraryRowRef] = []
            used: set[int] = set()
            output: list[Any] = []
            for slot, synthetic_index in enumerate(batch):
                if slot not in slots:
                    output.append(("synthetic", synthetic_index))
                    continue
                window = int(synthetic_index[1]) if isinstance(synthetic_index, tuple) else self.fixed_samples
                eligible = [row for row in self.rows if row.samples <= window and row.row_index not in used]
                if not eligible:
                    raise RuntimeError(f"no SCNS v3 row fits batch length={window}, step={step}")
                row = self._choose(eligible, step=step, slot=slot)
                used.add(row.row_index)
                selected.append(row)
                output.append(("scene", row.row_index))
            authority = self.output / "trackbound_v3_within_batch_yields.jsonl"
            authority.parent.mkdir(parents=True, exist_ok=True)
            with authority.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "schema": "scns_trackbound_v3_within_batch_yield_v1",
                    "candidate_optimizer_step": step + 1,
                    "requested_recorded_fraction": self.fraction,
                    "batch_rows": len(output), "recorded_rows": len(selected),
                    "requested_recorded_rows": requested, "recorded_eligible_slots": len(eligible_slots),
                    "recorded_source_libraries": sorted({row.source_library for row in selected}),
                    "recorded_source_parents": sorted({row.source_parent for row in selected}),
                    "recorded_instruments": sorted({row.instrument for row in selected}),
                }, sort_keys=True) + "\n")
            yield output


_prior_contract = v1.base.sampler_contract


def sampler_contract(args: Any) -> dict[str, Any]:
    return _prior_contract(args) | {
        "strategy": "trackbound_scene_rows_library_then_sound_v3",
        "recorded_scene_contract": "source_complete_track_bound_slakh_windows",
        "source_balance": "eligible_source_library_then_physical_sound_identity",
        "recorded_only_optimizer_steps": False,
    }


v1.base.DeterministicMixedBatchSampler = SourceLibraryThenSoundSampler
v1.base.sampler_contract = sampler_contract


if __name__ == "__main__":
    v1.base.main()
