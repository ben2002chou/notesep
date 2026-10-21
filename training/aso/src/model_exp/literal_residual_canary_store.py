"""Random-access loader for the packed heterogeneous residual canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .packed_blob_shards import PackedBlobShardReader


class PackedLiteralResidualCanaryCatalog:
    def __init__(self, packed_root: str | Path) -> None:
        self.reader = PackedBlobShardReader(packed_root)
        self._mixture_cache: dict[str, np.ndarray] = {}
        metadata = self.reader.metadata()
        if metadata.get("schema") not in {
            "literal_residual_heterogeneous_canary_v1",
            "literal_residual_musical_canary_v1",
            "literal_residual_musical_canary_v2",
            "literal_residual_production_pack_unit_v1",
        }:
            raise ValueError("Packed root is not a supported residual canary")
        self.sample_ids = self.reader.list_sample_ids("target_f32")
        self.context_sum_ids = set(
            self.reader.list_sample_ids("dna_context_sum_f32")
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def close(self) -> None:
        self._mixture_cache.clear()
        self.reader.close()

    def __enter__(self) -> "PackedLiteralResidualCanaryCatalog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def load(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[int(index)]
        target_record = self.reader.record(sample_id, "target_f32")
        metadata = dict(target_record.metadata)
        dna_record = self.reader.record(sample_id, "dna_f32")
        if dna_record.metadata != metadata:
            raise ValueError(f"Target/DNA metadata mismatch for {sample_id}")
        mixture_id = str(metadata["mixture_id"])
        mixture_record = self.reader.record(mixture_id, "mixture_f32")
        mixture_metadata = dict(mixture_record.metadata)
        for key in ("target_notes",):
            if key not in metadata and key in mixture_metadata:
                metadata[key] = mixture_metadata[key]
            elif key in metadata and key in mixture_metadata:
                if metadata[key] != mixture_metadata[key]:
                    raise ValueError(
                        f"Pair/mixture {key} mismatch for {sample_id}"
                    )
        mixture = self._mixture_cache.get(mixture_id)
        if mixture is None:
            mixture = self.reader.read_array_slice(
                mixture_id, "mixture_f32", dtype="<f4"
            )
            self._mixture_cache[mixture_id] = mixture
        dna_context_sum = None
        if mixture_id in self.context_sum_ids:
            context_record = self.reader.record(mixture_id, "dna_context_sum_f32")
            if context_record.metadata != mixture_record.metadata:
                raise ValueError(f"Mixture/context metadata mismatch for {mixture_id}")
            dna_context_sum = self.reader.read_array_slice(
                mixture_id, "dna_context_sum_f32", dtype="<f4"
            )
            if len(dna_context_sum) != len(mixture):
                raise ValueError(f"Mixture/context length mismatch for {mixture_id}")
        start = int(metadata.get("window_start_sample", 0))
        stop = int(metadata.get("window_end_sample", len(mixture)))
        if not 0 <= start < stop <= len(mixture):
            raise ValueError(f"Invalid mixture window for {sample_id}")
        target = self.reader.read_array_slice(sample_id, "target_f32", dtype="<f4")
        dna = self.reader.read_array_slice(sample_id, "dna_f32", dtype="<f4")
        if len(target) != stop - start or len(dna) != len(target):
            raise ValueError(f"Packed pair length mismatch for {sample_id}")
        return {
            "sample_id": sample_id,
            "mixture": np.ascontiguousarray(mixture[start:stop]),
            "target": target,
            "dna": dna,
            "dna_context_sum": (
                None
                if dna_context_sum is None
                else np.ascontiguousarray(dna_context_sum[start:stop])
            ),
            "pitch": int(metadata["pitch"]),
            "metadata": metadata,
        }
