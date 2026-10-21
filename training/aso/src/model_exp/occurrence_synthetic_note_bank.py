"""Mixture-disjoint complete synthetic note banks for occurrence training."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import sqlite3
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import torch

from .dna_residual_v13_streaming import FrozenV13Contract
from .packed_blob_shards import PackedBlobShardReader


def _chunks(values: list[Any], size: int = 800) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def select_complete_mixture_groups(
    contract: FrozenV13Contract,
    ordinals: np.ndarray,
    *,
    count: int,
    seed: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Select deterministic complete mixture groups from a frozen ordinal split."""
    if count < 1 or ordinals.ndim != 1 or len(ordinals) < 1:
        raise ValueError("group selection needs a positive count and ordinal vector")
    rng = np.random.default_rng(seed)
    candidates = np.asarray(ordinals, dtype=np.int64).copy()
    rng.shuffle(candidates)
    selected: list[str] = []
    seen: set[str] = set()
    uri = f"file:{quote(str(contract.split_paths['train']))}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        for chunk in _chunks(candidates.tolist()):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT ordinal, mixture_id FROM pairs WHERE ordinal IN ({placeholders})",
                list(map(int, chunk)),
            ).fetchall()
            by_ordinal = {int(ordinal): str(mixture_id) for ordinal, mixture_id in rows}
            for ordinal in chunk:
                mixture_id = by_ordinal.get(int(ordinal))
                if mixture_id is None:
                    raise ValueError(f"ordinal is absent from frozen index: {ordinal}")
                if mixture_id not in seen:
                    seen.add(mixture_id)
                    selected.append(mixture_id)
                    if len(selected) == count:
                        break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f"requested {count} groups but found {len(selected)}")
        allowed = set(map(int, ordinals.tolist()))
        groups: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
        for chunk in _chunks(selected):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT ordinal, pair_id, mixture_id, packed_root
                FROM pairs WHERE mixture_id IN ({placeholders})
                ORDER BY mixture_id, ordinal
                """,
                chunk,
            ).fetchall()
            for ordinal, pair_id, mixture_id, packed_root in rows:
                if int(ordinal) not in allowed:
                    raise ValueError(
                        f"mixture-disjoint view is incomplete for mixture {mixture_id}"
                    )
                groups[str(mixture_id)].append(
                    {
                        "ordinal": int(ordinal),
                        "pair_id": str(pair_id),
                        "packed_root": str(packed_root),
                    }
                )
    if any(not rows for rows in groups.values()):
        raise ValueError("selected mixture did not resolve to occurrence rows")
    return [(mixture_id, groups[mixture_id]) for mixture_id in selected]


class CompleteSyntheticBankReader:
    """Read one deterministic crop containing every overlapping synthetic stem."""

    def __init__(
        self,
        contract: FrozenV13Contract,
        *,
        fixed_samples: int,
        maximum_sources: int,
        support_mode: str = "scored",
        cache_size: int = 8,
    ) -> None:
        self.contract = contract
        self.fixed_samples = int(fixed_samples)
        self.maximum_sources = int(maximum_sources)
        self.support_mode = str(support_mode)
        self.cache_size = int(cache_size)
        self.readers: OrderedDict[Path, PackedBlobShardReader] = OrderedDict()
        if (
            self.fixed_samples < 16_384
            or self.maximum_sources < 1
            or self.support_mode not in {"scored", "acoustic"}
        ):
            raise ValueError("invalid synthetic bank geometry")

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def _reader(self, packed_root: str) -> PackedBlobShardReader:
        root = (self.contract.root / packed_root).resolve()
        if root not in self.contract.packed_roots:
            raise ValueError(f"unfrozen packed root: {root}")
        reader = self.readers.get(root)
        if reader is None:
            reader = PackedBlobShardReader(root, require_complete=True)
            self.readers[root] = reader
            while len(self.readers) > self.cache_size:
                _, stale = self.readers.popitem(last=False)
                stale.close()
        else:
            self.readers.move_to_end(root)
        return reader

    @staticmethod
    def _stable_integer(text: str) -> int:
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")

    def load(
        self,
        mixture_id: str,
        rows: list[dict[str, Any]],
        *,
        crop_seed: int,
    ) -> dict[str, Any]:
        enriched: list[tuple[dict[str, Any], PackedBlobShardReader, dict[str, Any]]] = []
        total: int | None = None
        sample_rate: int | None = None
        for row in rows:
            reader = self._reader(str(row["packed_root"]))
            metadata = reader.record(str(row["pair_id"]), "target_f32").metadata
            if str(metadata.get("mixture_id")) != mixture_id:
                raise ValueError("pair/mixture identity mismatch")
            observed_total = int(metadata.get("samples", -1))
            observed_rate = int(metadata.get("sample_rate", -1))
            total = observed_total if total is None else total
            sample_rate = observed_rate if sample_rate is None else sample_rate
            if total != observed_total or sample_rate != observed_rate:
                raise ValueError("group occurrence geometry mismatch")
            enriched.append((row, reader, metadata))
        assert total is not None and sample_rate is not None
        if total < 1 or sample_rate < 1:
            raise ValueError("invalid packed group geometry")

        anchor_index = self._stable_integer(f"{mixture_id}:{crop_seed}") % len(enriched)
        anchor = enriched[anchor_index][2]
        midpoint = (int(anchor["onset_sample"]) + int(anchor["offset_sample"])) // 2
        count = min(total, self.fixed_samples)
        start = max(0, min(midpoint - count // 2, total - count))
        stop = start + count
        scored = [
            item
            for item in enriched
            if int(item[2]["onset_sample"]) < stop
            and int(item[2]["offset_sample"]) > start
        ]
        if not scored:
            raise ValueError("synthetic bank crop contains no scored occurrence")

        mixture_reader = enriched[0][1]
        mixture = mixture_reader.read_array_slice(
            mixture_id, "mixture_f32", dtype="<f4", start=start, count=count
        )
        # A score offset is not an acoustic offset: reverberant and instrument
        # release tails can remain audible well after note-off.  Joint-bank
        # training must retain those stems or their known energy is mislabeled
        # as an unexplained residual.  The legacy scored mode is preserved for
        # callers whose target contract intentionally ends at note-off.
        candidates = enriched if self.support_mode == "acoustic" else scored
        loaded: list[
            tuple[dict[str, Any], dict[str, Any], np.ndarray, bool]
        ] = []
        scored_pair_ids = {str(item[0]["pair_id"]) for item in scored}
        for row, reader, metadata in candidates:
            target = reader.read_array_slice(
                str(row["pair_id"]),
                "target_f32",
                dtype="<f4",
                start=start,
                count=count,
            )
            if self.support_mode == "acoustic" and not np.any(target != 0.0):
                continue
            loaded.append(
                (
                    row,
                    metadata,
                    target,
                    str(row["pair_id"]) not in scored_pair_ids,
                )
            )
        if len(loaded) > self.maximum_sources:
            raise ValueError("synthetic bank exceeds maximum source capacity")
        loaded.sort(key=lambda item: int(item[0]["ordinal"]))
        if not loaded:
            raise ValueError("synthetic bank crop contains no acoustic source")

        targets: list[np.ndarray] = []
        pitches: list[int] = []
        onsets: list[int] = []
        offsets: list[int] = []
        pair_ids: list[str] = []
        tail_only: list[bool] = []
        for row, metadata, target, is_tail_only in loaded:
            targets.append(target)
            pitches.append(int(metadata["pitch"]))
            onset = max(0, min(count, int(metadata["onset_sample"]) - start))
            offset = max(0, min(count, int(metadata["offset_sample"]) - start))
            onsets.append(min(onset, offset))
            offsets.append(offset)
            pair_ids.append(str(row["pair_id"]))
            tail_only.append(is_tail_only)
        arrays = [mixture, *targets]
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("non-finite grouped waveform")
        mixture_tensor = torch.from_numpy(np.asarray(mixture, dtype=np.float32))
        target_tensor = torch.from_numpy(np.stack(targets).astype(np.float32))
        true_residual = mixture_tensor - target_tensor.sum(dim=0)
        ratio = float(
            true_residual.square().mean().sqrt()
            / mixture_tensor.square().mean().sqrt().clamp_min(1e-7)
        )
        return {
            "mixture_id": mixture_id,
            "pair_ids": pair_ids,
            "mixture": mixture_tensor,
            "targets": target_tensor,
            "pitches": torch.tensor(pitches, dtype=torch.long),
            "onsets": torch.tensor(onsets, dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "tail_only": torch.tensor(tail_only, dtype=torch.bool),
            "valid_samples": count,
            "sample_rate": sample_rate,
            "crop_start": start,
            "true_residual_ratio": ratio,
            "support_mode": self.support_mode,
        }


__all__ = ["CompleteSyntheticBankReader", "select_complete_mixture_groups"]
