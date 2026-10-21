"""Frozen full-score conditioning adapter for occurrence separation."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable

import numpy as np
import torch

from .streaming_dataset import V13StreamingResidualDataset


class FullScoreLookup:
    """Worker-local immutable lookup over the frozen score authority."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._pid = -1
        self._connection: sqlite3.Connection | None = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _db(self) -> sqlite3.Connection:
        if self._pid != os.getpid():
            self.close()
            self._connection = sqlite3.connect(
                f"file:{self.path}?mode=ro&immutable=1", uri=True
            )
            self._pid = os.getpid()
            metadata = dict(
                self._connection.execute("SELECT key, value FROM metadata")
            )
            if metadata.get("schema") != "tfc_score_condition_index_v1":
                raise ValueError("Full-score index has the wrong schema")
        assert self._connection is not None
        return self._connection

    def get(self, excerpt_id: str) -> dict[str, Any]:
        cached = self._cache.get(excerpt_id)
        if cached is not None:
            self._cache.move_to_end(excerpt_id)
            return cached
        row = (
            self._db()
            .execute(
                "SELECT row_json FROM excerpts WHERE excerpt_id = ?",
                (excerpt_id,),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(
                f"Score excerpt is absent from full-score index: {excerpt_id}"
            )
        value = json.loads(str(row[0]))
        self._cache[excerpt_id] = value
        while len(self._cache) > 128:
            self._cache.popitem(last=False)
        return value

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._cache.clear()
        self._pid = -1


class AcousticExtensionLookup:
    """Compact KNN release-extension estimator with family fallback."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        with np.load(self.path, allow_pickle=False) as payload:
            if (
                str(np.asarray(payload["schema"]).item())
                != "acoustic_extension_lookup_v1"
            ):
                raise ValueError("Unexpected acoustic extension lookup schema")
            self.sound_ids = [value.decode() for value in payload["sound_ids"]]
            self.sound_offsets = payload["sound_offsets"].astype(np.int64)
            self.pitch = payload["pitch"].astype(np.float32)
            self.velocity = payload["velocity"].astype(np.float32)
            self.duration = payload["duration_samples"].astype(np.float32)
            self.release = payload["release_samples"].astype(np.float32)
            self.scale = payload["sound_robust_scale_samples"].astype(
                np.float32
            )
            family_names = [str(value) for value in payload["family_names"]]
            family_stats = payload[
                "family_stats_mean_std_p01_median_p99"
            ].astype(np.float32)
        self.sound_index = {
            value: index for index, value in enumerate(self.sound_ids)
        }
        self.family_stats = dict(zip(family_names, family_stats, strict=True))

    def estimate(
        self,
        sound_id: str,
        family: str,
        pitch: int,
        velocity: int,
        duration_samples: int,
    ) -> tuple[float, float, float, float]:
        index = self.sound_index.get(sound_id)
        if index is None:
            mean, std, low, median_value, high = self.family_stats[family]
            return float(median_value), float(std), float(low), float(high)
        begin = int(self.sound_offsets[index])
        end = int(self.sound_offsets[index + 1])
        distance = (
            ((self.pitch[begin:end] - pitch) / 12.0) ** 2
            + ((self.velocity[begin:end] - velocity) / 32.0) ** 2
            + (
                (
                    np.log1p(self.duration[begin:end])
                    - np.log1p(float(duration_samples))
                )
                / 2.0
            )
            ** 2
        )
        count = min(8, len(distance))
        neighbors = np.argpartition(distance, count - 1)[:count]
        values = self.release[begin:end]
        center = float(np.median(values[neighbors]))
        return (
            center,
            float(self.scale[index]),
            float(np.quantile(values, 0.01)),
            float(np.quantile(values, 0.99)),
        )

    def estimate_family(self, family: str) -> tuple[float, float, float, float]:
        """Return the held-out-sound fallback distribution for one family."""

        mean, std, low, median_value, high = self.family_stats[family]
        return float(median_value), float(std), float(low), float(high)


def _clipped_event(
    pitch: int, onset: int, offset: int, samples: int
) -> dict[str, int] | None:
    onset = max(0, int(onset))
    offset = min(int(samples), int(offset))
    if onset >= offset or not 0 <= int(pitch) < 128:
        return None
    return {
        "pitch": int(pitch),
        "onset_sample": onset,
        "offset_sample": offset,
    }


class FullScoreSyntheticDataset(V13StreamingResidualDataset):
    """Attach every score event overlapping the sampled occurrence window."""

    score_index: Path | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.score_index is None:
            raise ValueError("FullScoreSyntheticDataset.score_index is unset")
        self._full_score_lookup = FullScoreLookup(self.score_index)

    def __getitem__(self, ordinal: int | tuple[int, int]) -> dict[str, Any]:
        output = super().__getitem__(ordinal)
        raw_ordinal = int(ordinal[0] if isinstance(ordinal, tuple) else ordinal)
        row = (
            self._db()
            .execute(
                "SELECT pair_id, packed_root FROM pairs WHERE ordinal = ?",
                (raw_ordinal,),
            )
            .fetchone()
        )
        if row is None:
            raise IndexError(raw_ordinal)
        pair_id, packed_root = str(row[0]), Path(str(row[1])).resolve()
        metadata = (
            self._reader(packed_root).record(pair_id, "target_f32").metadata
        )
        score = self._full_score_lookup.get(str(metadata["excerpt_id"]))
        notes = list(score["absolute_notes"])
        target_index = int(metadata["target_event_index"])
        if not 0 <= target_index < len(notes):
            raise ValueError(
                "Target event is outside the frozen complete score"
            )
        target_note = notes[target_index]
        transposition = int(output["pitch"]) - int(target_note["pitch"])
        events = []
        target_onset = int(output["note_onset_samples"])
        target_offset = int(output["note_offset_samples"])
        target_score_onset = float(target_note["onset_sec"])
        for index, note in enumerate(notes):
            if index == target_index:
                event = _clipped_event(
                    int(output["pitch"]),
                    target_onset,
                    target_offset,
                    int(output["valid_samples"]),
                )
            else:
                # The rendered metadata is authoritative for the target's
                # cropped location. Express every other score event relative
                # to that anchor instead of reconstructing a legacy expanded
                # timeline, whose boundary convention varied across builders.
                event = _clipped_event(
                    int(note["pitch"]) + transposition,
                    target_onset
                    + round(
                        (float(note["onset_sec"]) - target_score_onset) * 16_000
                    ),
                    target_onset
                    + round(
                        (float(note["offset_sec"]) - target_score_onset)
                        * 16_000
                    ),
                    int(output["valid_samples"]),
                )
            if event is not None:
                event["is_target"] = int(index == target_index)
                event["velocity"] = int(note.get("velocity", 64))
                event["duration_samples"] = max(
                    1,
                    round(
                        (float(note["offset_sec"]) - float(note["onset_sec"]))
                        * 16_000
                    ),
                )
                events.append(event)
        if not events:
            raise ValueError(
                "Full score has no event overlapping the sampled window"
            )
        output["full_score_events"] = events
        output["sound_id"] = str(metadata["sound_id"])
        return output

    def close(self) -> None:
        super().close()
        self._full_score_lookup.close()


def collate_full_score(
    rows: list[dict[str, Any]],
    base_collate: Callable[[list[dict[str, Any]]], dict[str, Any]],
    extension_lookup: AcousticExtensionLookup | None = None,
    family_dropout_probability: float = 0.0,
) -> dict[str, Any]:
    """Pad the complete score to the authority's maximum of 64 events."""

    if not 0.0 <= family_dropout_probability <= 1.0:
        raise ValueError("family dropout probability must lie in [0, 1]")
    output = base_collate(rows)
    maximum = min(64, max(len(row["full_score_events"]) for row in rows))
    shape = (len(rows), maximum)
    pitches = torch.zeros(shape, dtype=torch.long)
    onsets = torch.zeros(shape, dtype=torch.long)
    offsets = torch.zeros(shape, dtype=torch.long)
    valid = torch.zeros(shape, dtype=torch.bool)
    target = torch.zeros(shape, dtype=torch.bool)
    extension_mean = torch.zeros(shape, dtype=torch.float32)
    extension_scale = torch.zeros(shape, dtype=torch.float32)
    extension_minimum = torch.zeros(shape, dtype=torch.float32)
    extension_maximum = torch.zeros(shape, dtype=torch.float32)
    family_fallback = torch.zeros(len(rows), dtype=torch.bool)
    for batch_index, row in enumerate(rows):
        events = list(row["full_score_events"])
        identity_fraction = int.from_bytes(
            hashlib.sha256(str(row["sound_id"]).encode()).digest()[:8], "big"
        ) / float(2**64)
        use_family_fallback = bool(
            extension_lookup is not None
            and identity_fraction < family_dropout_probability
        )
        family_fallback[batch_index] = use_family_fallback
        if len(events) > 64:
            raise ValueError("Full score exceeds the frozen 64-event authority")
        for event_index, event in enumerate(events):
            pitches[batch_index, event_index] = int(event["pitch"])
            onsets[batch_index, event_index] = int(event["onset_sample"])
            offsets[batch_index, event_index] = int(event["offset_sample"])
            valid[batch_index, event_index] = True
            target[batch_index, event_index] = bool(event["is_target"])
            if extension_lookup is not None:
                if use_family_fallback:
                    mean, scale, minimum, maximum = (
                        extension_lookup.estimate_family(
                            str(row["sound_family"])
                        )
                    )
                else:
                    mean, scale, minimum, maximum = extension_lookup.estimate(
                        str(row["sound_id"]),
                        str(row["sound_family"]),
                        int(event["pitch"]),
                        int(event["velocity"]),
                        int(event["duration_samples"]),
                    )
                extension_mean[batch_index, event_index] = mean
                extension_scale[batch_index, event_index] = scale
                extension_minimum[batch_index, event_index] = minimum
                extension_maximum[batch_index, event_index] = maximum
    output.update(
        full_score_pitches=pitches,
        full_score_onsets=onsets,
        full_score_offsets=offsets,
        full_score_valid=valid,
        full_score_target=target,
    )
    if extension_lookup is not None:
        output.update(
            full_score_extension_mean=extension_mean,
            full_score_extension_scale=extension_scale,
            full_score_extension_minimum=extension_minimum,
            full_score_extension_maximum=extension_maximum,
            full_score_family_fallback=family_fallback,
        )
    return output
