"""Balance recorded source libraries while keeping instrument coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from fractions import Fraction
from typing import Any, Iterable, Iterator

import track_bound_sampling as trackbound


FAMILIES = {
    "piano", "acoustic_guitar", "electric_guitar", "bass_guitar", "harp",
    "violin", "viola", "cello", "double_bass", "flute", "clarinet", "oboe",
    "saxophone", "trumpet", "french_horn", "trombone",
}
SCORE_BOUND_SUPPLEMENTAL_FAMILIES = FAMILIES - {"harp", "viola"}


def canonical_instrument(value: str) -> str:
    name = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    aliases = {
        "contrabass": "double_bass", "double bass": "double_bass",
        "acoustic bass": "double_bass", "electric bass": "bass_guitar",
        "bass guitar": "bass_guitar", "acoustic guitar": "acoustic_guitar",
        "classical guitar": "acoustic_guitar", "electric guitar": "electric_guitar",
        "alto sax": "saxophone", "alto saxophone": "saxophone",
        "tenor saxophone": "saxophone", "clarinet in bb": "clarinet",
        "trumpet in c": "trumpet", "french horn": "french_horn",
    }
    return aliases.get(name, name.replace(" ", "_"))


def validate_combined_coverage_or_supplemental_roster(present: set[str]) -> None:
    """Check aggregate coverage and the required recorded instruments."""
    authority_value = os.environ.get("SCNS_V3_COMBINED_INSTRUMENT_COVERAGE_AUTHORITY", "")
    if not authority_value:
        raise ValueError(
            "disabling supplemental all-16 requires SCNS_V3_COMBINED_INSTRUMENT_COVERAGE_AUTHORITY"
        )
    authority = json.loads(Path(authority_value).expanduser().resolve().read_text())
    if authority.get("status") != "passed" or set(authority.get("aggregate_instruments", ())) != FAMILIES:
        raise ValueError("combined instrument coverage authority is not passed exact-all-16")
    declared = {
        canonical_instrument(value) for value in
        os.environ.get("SCNS_V3_REQUIRED_SUPPLEMENTAL_FAMILIES", "").split(",") if value
    }
    if declared != SCORE_BOUND_SUPPLEMENTAL_FAMILIES:
        raise ValueError("Required supplemental instrument families do not match")
    missing = declared - present
    if missing:
        raise ValueError("Loaded supplemental data is missing: " + ", ".join(sorted(missing)))


class LibraryBalancedInterleavedSampler(trackbound.SourceLibraryThenSoundSampler):
    """Uniformly rotate libraries, then least-used physical sources."""

    def __init__(self, base_sampler: Iterable[list[Any]], **kwargs: Any) -> None:
        super().__init__(base_sampler, **kwargs)
        validate_combined_coverage_or_supplemental_roster(
            {canonical_instrument(row.instrument) for row in self.rows}
        )
        self.cumulative_rows, self.cumulative_score_bound_rows = self._restore_committed_totals()

    def _restore_committed_totals(self) -> tuple[int, int]:
        """Recover cumulative allocation on resume, ignoring uncommitted yields."""
        authority = self.output / "scns_v3_library_balanced_yields.jsonl"
        if self.start_step <= 0 or not authority.exists():
            return 0, 0
        rows = score_bound = 0
        latest_by_step: dict[int, dict[str, Any]] = {}
        for line in authority.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            step = int(event["candidate_optimizer_step"])
            if step <= self.start_step:
                latest_by_step[step] = event
        for event in latest_by_step.values():
            rows += int(event["batch_rows_after_budget"])
            score_bound += int(event["realized_score_bound_rows"])
        return rows, score_bound

    @staticmethod
    def _requested_samples(index: Any, fallback: int) -> int:
        return int(index[1]) if isinstance(index, tuple) and len(index) >= 2 else fallback

    def _select_rows(self, count: int, *, step: int) -> list[Any]:
        selected = []
        used: set[int] = set()
        for slot in range(count):
            eligible = [row for row in self.rows if row.row_index not in used]
            if not eligible:
                raise RuntimeError(f"not enough distinct SCNS-v3 rows for step={step}")
            row = self._choose(eligible, step=step, slot=slot)
            used.add(row.row_index)
            selected.append(row)
        return selected

    def __iter__(self) -> Iterator[list[Any]]:
        audio_budget = int(os.environ.get("AUDIO_BUDGET_SAMPLES", "524288"))
        fraction = Fraction(str(self.fraction)).limit_denominator(100)
        for local_step, batch in enumerate(self.base_sampler):
            step = self.start_step + local_step
            if not batch:
                continue
            library_before, parent_before = self.library_use.copy(), self.parent_use.copy()
            synthetic_promoted = max(
                self._requested_samples(index, self.fixed_samples) for index in batch
            )
            synthetic_rows = max(1, min(len(batch), audio_budget // synthetic_promoted))

            # Find a small fixed point between selected real-row duration and
            # affordable batch cardinality.  The allocation target is
            # cumulative, so a rare cardinality-1/2 synthetic batch can stay
            # synthetic-only and its fractional debt is repaid later, capped
            # at 1/3 real rows in every individual batch.
            candidate_counts = {0}
            count = min(
                synthetic_rows // 3,
                max(0, round(float(
                    fraction * (self.cumulative_rows + synthetic_rows)
                    - self.cumulative_score_bound_rows
                ))),
            )
            for _ in range(8):
                candidate_counts.add(count)
                self.library_use, self.parent_use = library_before.copy(), parent_before.copy()
                preview = self._select_rows(count, step=step) if count else []
                promoted_preview = max(
                    synthetic_promoted,
                    max((row.samples for row in preview), default=0),
                )
                rows_preview = max(1, min(len(batch), audio_budget // promoted_preview))
                next_count = min(
                    rows_preview // 3,
                    max(0, round(float(
                        fraction * (self.cumulative_rows + rows_preview)
                        - self.cumulative_score_bound_rows
                    ))),
                )
                if next_count == count:
                    break
                count = next_count

            feasible: list[tuple[Fraction, int, int, int, list[Any]]] = []
            for candidate_count in sorted(candidate_counts):
                self.library_use, self.parent_use = library_before.copy(), parent_before.copy()
                preview = self._select_rows(candidate_count, step=step) if candidate_count else []
                promoted_preview = max(
                    synthetic_promoted,
                    max((row.samples for row in preview), default=0),
                )
                rows_preview = max(1, min(len(batch), audio_budget // promoted_preview))
                if candidate_count > rows_preview // 3:
                    continue
                cumulative_error = abs(
                    Fraction(self.cumulative_score_bound_rows + candidate_count, 1)
                    - fraction * (self.cumulative_rows + rows_preview)
                )
                feasible.append((
                    cumulative_error, -rows_preview, candidate_count,
                    promoted_preview, preview,
                ))
            if not feasible:
                raise RuntimeError("no duration-safe SCNS-v3 cumulative allocation exists")
            _, neg_rows, requested, promoted, selected = min(feasible, key=lambda item: item[:3])
            maximum_rows = -neg_rows
            # Commit only the chosen selection's balancing counters.
            self.library_use, self.parent_use = library_before.copy(), parent_before.copy()
            selected = self._select_rows(requested, step=step) if requested else []
            chosen_batch = list(batch[:maximum_rows])
            slots = sorted(
                range(len(chosen_batch)), key=lambda slot: self._rank(f"slot:{step}:{slot}")
            )[:len(selected)]
            by_slot = dict(zip(slots, selected))
            output = []
            for slot, synthetic_index in enumerate(chosen_batch):
                if slot in by_slot:
                    output.append(("scene", by_slot[slot].row_index, promoted))
                else:
                    # Preserve the original synthetic crop.  Padding belongs
                    # at the composite boundary, not in V13's crop API.
                    output.append(("synthetic", synthetic_index, promoted))
            cumulative_rows_after = self.cumulative_rows + len(output)
            cumulative_score_bound_after = self.cumulative_score_bound_rows + len(selected)
            authority = self.output / "scns_v3_library_balanced_yields.jsonl"
            authority.parent.mkdir(parents=True, exist_ok=True)
            with authority.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "schema": "scns_v3_library_balanced_promoted_batch_yield_v2",
                    "candidate_optimizer_step": step + 1,
                    "requested_score_bound_rows": requested,
                    "realized_score_bound_rows": len(selected),
                    "batch_rows_before_budget": len(batch),
                    "batch_rows_after_budget": len(output),
                    "promoted_window_samples": promoted,
                    "audio_budget_samples": audio_budget,
                    "realized_score_bound_fraction": len(selected) / len(output),
                    "configured_score_bound_fraction": self.fraction,
                    "cumulative_batch_rows": cumulative_rows_after,
                    "cumulative_score_bound_rows": cumulative_score_bound_after,
                    "cumulative_score_bound_fraction": (
                        cumulative_score_bound_after / cumulative_rows_after
                    ),
                    "cumulative_fraction_error_rows": float(
                        Fraction(cumulative_score_bound_after, 1)
                        - fraction * cumulative_rows_after
                    ),
                    "recorded_source_libraries": sorted({row.source_library for row in selected}),
                    "recorded_source_parents": sorted({row.source_parent for row in selected}),
                    "recorded_instruments": sorted({row.instrument for row in selected}),
                    "recorded_k": [row.k for row in selected],
                }, sort_keys=True) + "\n")
            self.cumulative_rows = cumulative_rows_after
            self.cumulative_score_bound_rows = cumulative_score_bound_after
            yield output


def sampler_contract(args: Any) -> dict[str, Any]:
    return trackbound.sampler_contract(args) | {
        "strategy": "scns_v3_library_then_physical_source_cumulative_promoted_batch_v4",
        "allocation_unit": "model_rows_within_batch_not_optimizer_steps",
        "source_balance": "least_used_source_library_then_least_used_physical_source",
        "evaluation_instrument_reweighting": "none",
        "polyphonic_contract": "source_complete_real_Slakh_score_mixtures",
        "duration_contract": "choose_score_bound_rows_then_promote_and_pad_batch_without_tail_truncation",
        "audio_budget_contract": "reduce_cardinality_without_truncation; cumulative_20pct_with_per_batch_cap_1over3",
        "aggregate_16_coverage_authority": os.environ.get(
            "SCNS_V3_COMBINED_INSTRUMENT_COVERAGE_AUTHORITY", ""
        ),
    }


def validate_mix_args(args: Any) -> None:
    roots = [Path(value).expanduser().resolve() for value in getattr(args, "recorded_frozen_root", [])]
    fraction = float(getattr(args, "real_example_fraction", 0))
    minimum = float(os.environ.get("SCNS_MINIMUM_REAL_EXAMPLE_FRACTION", "0.20"))
    maximum = float(os.environ.get("SCNS_MAXIMUM_REAL_EXAMPLE_FRACTION", str(1.0 / 3.0)))
    if (fraction > 0 and not roots) or not minimum <= fraction <= maximum + 1e-9:
        raise ValueError(
            "SCNS-v3 score-bound fraction outside declared bounds: "
            f"fraction={fraction}, minimum={minimum}, maximum={maximum}"
        )


trackbound.v1.base.DeterministicMixedBatchSampler = LibraryBalancedInterleavedSampler
trackbound.v1.base.sampler_contract = sampler_contract
trackbound.v1.base.validate_recorded_mix_args = validate_mix_args
