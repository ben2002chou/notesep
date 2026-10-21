"""Read complete residual examples for the NoteGrab training recipe."""

from pathlib import Path
from typing import Any

import numpy as np

from model_exp.dna_residual_corpus import FrozenDNAResidualPriorCatalog

SAMPLE_RATE = 16_000


def target_complete_window(
    metadata: dict[str, Any], total: int
) -> tuple[int, int]:
    onset = int(metadata["onset_sample"])
    offset = int(metadata["offset_sample"])
    start = max(0, onset - int(0.25 * SAMPLE_RATE))
    stop = min(total, offset + int(0.5 * SAMPLE_RATE))
    if not start <= onset < offset <= stop:
        raise ValueError("target note is not completely retained")
    return start, stop


def load_frozen_residual_rows(
    root: Path, *, split: str = "validation"
) -> list[dict[str, Any]]:
    """Normalize the frozen residual corpus to the NoteGrab training format."""

    rows = []
    with FrozenDNAResidualPriorCatalog(
        root.expanduser().resolve(), split=split
    ) as catalog:
        for record in catalog.iter_records():
            metadata = dict(record.metadata)
            samples = int(metadata["samples"])
            dna, _ = catalog.load_priors(record, include_rt2=False)
            mixture, target = catalog.load_self_contained_raw_slice(
                record, start=0, count=samples
            )
            start_step = int(metadata["start_step"])
            end_step = int(metadata["end_step"])
            step_span = end_step - start_step
            if step_span < 1 or samples % step_span:
                raise ValueError(
                    "recorded residual timeline does not map exactly to samples"
                )
            samples_per_step = samples // step_span
            onset = (
                int(metadata["onset_step"]) - start_step
            ) * samples_per_step
            offset = (
                int(metadata["offset_step"]) - start_step
            ) * samples_per_step
            difficulty = dict(metadata.get("difficulty", {}))
            peak_competitors = int(difficulty.get("peak_competitors", 0))
            staggered_competitors = int(
                difficulty.get("staggered_competitors", 0)
            )
            normalized_metadata = {
                **metadata,
                "mixture_id": str(metadata.get("sample_id", record.asset_id)),
                "score_parent_id": str(metadata["source_parent_id"]),
                "matched_score_family": str(
                    metadata.get("source_instrument", "recorded")
                ),
                "velocity": int(metadata.get("velocity", -1)),
                "onset_sample": onset,
                "offset_sample": offset,
                "target_notes": 1,
                "difficulty": {
                    **difficulty,
                    "peak_competitors": peak_competitors,
                    "staggered_competitors": staggered_competitors,
                },
                "context_notes": peak_competitors + 1,
                "release_context_notes": staggered_competitors,
                "dna_rms": (
                    rms := float(
                        np.sqrt(np.mean(np.square(dna, dtype=np.float64)))
                    )
                ),
            }
            target_rms = float(metadata["target_rms"])
            normalized_metadata["dna_to_target_rms_ratio"] = rms / max(
                target_rms, 1e-12
            )
            start, stop = target_complete_window(normalized_metadata, samples)
            rows.append(
                {
                    "sample_id": record.asset_id,
                    "sound_id": (
                        f"{metadata['source_dataset']}:"
                        f"{metadata.get('source_instrument', 'unknown')}"
                    ),
                    "source_id": str(metadata["source_dataset"]),
                    "score_parent_id": str(metadata["source_parent_id"]),
                    "metadata": normalized_metadata,
                    "mixture": np.ascontiguousarray(mixture[start:stop]),
                    "target": np.ascontiguousarray(target[start:stop]),
                    "dna": np.ascontiguousarray(dna[start:stop]),
                    "valid_samples": stop - start,
                    "pitch": int(metadata["pitch"]),
                }
            )
    if not rows:
        raise ValueError(f"frozen residual root is empty: {root}")
    return rows
