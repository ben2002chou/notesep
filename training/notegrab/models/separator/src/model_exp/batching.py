"""Assemble variable-length note examples into model batches."""

from __future__ import annotations

from typing import Any

import torch


def collate_v13(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "mixture",
        "target",
        "dna",
        "other_dna",
        "unassigned_residual",
        "pitch",
        "valid_samples",
        "note_onset_samples",
        "note_offset_samples",
        "ordinal",
        "hard_case",
    )
    maximum = max(int(row["mixture"].shape[-1]) for row in batch)
    waveform_keys = {
        "mixture",
        "target",
        "dna",
        "other_dna",
        "unassigned_residual",
    }
    output: dict[str, Any] = {}
    for key in keys:
        if key == "note_onset_samples":
            values = [
                row.get(key, torch.zeros_like(row["valid_samples"]))
                for row in batch
            ]
        elif key == "note_offset_samples":
            values = [row.get(key, row["valid_samples"]) for row in batch]
        else:
            values = [row[key] for row in batch]
        if key in waveform_keys:
            padded = [
                torch.nn.functional.pad(value, (0, maximum - value.shape[-1]))
                for value in values
            ]
            output[key] = torch.stack(padded)
        else:
            output[key] = torch.stack(values)
    for key in ("pair_id", "sound_family", "source_group", "stratum"):
        output[key] = [str(row[key]) for row in batch]
    if all("mixture_id" in row for row in batch):
        output["mixture_id"] = [str(row["mixture_id"]) for row in batch]
    return output
