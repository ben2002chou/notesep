"""Deterministic batch samplers for reproducible training and resume."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from typing import Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler


class ResumeSafeBatchSampler(Sampler[list[int]]):
    """Resume-safe uniform or opt-in family-temperature batch sampling."""

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        start_step: int,
        stop_step: int,
        seed: int,
        family_temperature_alpha: float = 1.0,
        family_ordinals: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        self.family_temperature_alpha = float(family_temperature_alpha)
        if min(self.dataset_size, self.batch_size) < 1:
            raise ValueError("Dataset and batch sizes must be positive")
        if not 0 <= self.start_step <= self.stop_step:
            raise ValueError("Invalid resume step interval")
        if not 0.0 <= self.family_temperature_alpha <= 1.0:
            raise ValueError("family temperature alpha must be in [0, 1]")
        self._family_ordinals: tuple[np.ndarray, ...] = ()
        self._family_names: tuple[str, ...] = ()
        self._family_quotas: tuple[int, ...] = ()
        self._family_cumulative: tuple[int, ...] = ()
        if self.family_temperature_alpha != 1.0:
            if not family_ordinals:
                raise ValueError(
                    "Family ordinals are required when alpha != 1.0"
                )
            normalized = {
                str(family): np.asarray(ordinals, dtype=np.int32)
                for family, ordinals in family_ordinals.items()
            }
            if any(len(ordinals) < 1 for ordinals in normalized.values()):
                raise ValueError("Family ordinal arrays must be non-empty")
            seen = np.zeros(self.dataset_size, dtype=np.bool_)
            observed = 0
            for ordinals in normalized.values():
                if (
                    int(ordinals[0]) < 0
                    or int(ordinals[-1]) >= self.dataset_size
                    or np.any(ordinals[1:] <= ordinals[:-1])
                    or np.any(seen[ordinals])
                ):
                    raise ValueError(
                        "Family ordinals must partition the dataset exactly"
                    )
                seen[ordinals] = True
                observed += len(ordinals)
            if observed != self.dataset_size or not bool(np.all(seen)):
                raise ValueError(
                    "Family ordinals must partition the dataset exactly"
                )
            self._family_names = tuple(sorted(normalized))
            self._family_ordinals = tuple(
                normalized[family] for family in self._family_names
            )
            self._family_quotas = self._temperature_quotas(
                tuple(map(len, self._family_ordinals)),
                self.family_temperature_alpha,
            )
            cumulative = np.cumsum(self._family_quotas, dtype=np.int64)
            self._family_cumulative = tuple(map(int, cumulative))

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _permutation(self, epoch: int) -> tuple[int, int]:
        digest = hashlib.sha256(f"{self.seed}:{epoch}".encode()).digest()
        multiplier = max(
            1, int.from_bytes(digest[:8], "little") % self.dataset_size
        )
        while math.gcd(multiplier, self.dataset_size) != 1:
            multiplier = (multiplier + 1) % self.dataset_size or 1
        offset = int.from_bytes(digest[8:16], "little") % self.dataset_size
        return multiplier, offset

    def _keyed_permutation(self, size: int, key: str) -> tuple[int, int]:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        multiplier = max(1, int.from_bytes(digest[:8], "little") % size)
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size or 1
        offset = int.from_bytes(digest[8:16], "little") % size
        return multiplier, offset

    @staticmethod
    def _temperature_quotas(
        counts: tuple[int, ...], alpha: float
    ) -> tuple[int, ...]:
        total = sum(counts)
        weights = np.power(np.asarray(counts, dtype=np.float64), alpha)
        exact = weights * (total / float(weights.sum()))
        quotas = np.floor(exact).astype(np.int64)
        remainder = total - int(quotas.sum())
        order = sorted(
            range(len(counts)),
            key=lambda index: (-(exact[index] - quotas[index]), index),
        )
        for index in order[:remainder]:
            quotas[index] += 1
        if int(quotas.sum()) != total or np.any(quotas < 1):
            raise ValueError("Invalid family-temperature quotas")
        return tuple(map(int, quotas))

    def _family_sample(self, cursor: int) -> int:
        epoch, position = divmod(cursor, self.dataset_size)
        multiplier, offset = self._keyed_permutation(
            self.dataset_size, f"family-slots:{epoch}"
        )
        slot = (multiplier * position + offset) % self.dataset_size
        family_index = bisect_right(self._family_cumulative, slot)
        previous = (
            0
            if family_index == 0
            else self._family_cumulative[family_index - 1]
        )
        local_slot = slot - previous
        ordinals = self._family_ordinals[family_index]
        cycle, local_position = divmod(local_slot, len(ordinals))
        inner_multiplier, inner_offset = self._keyed_permutation(
            len(ordinals),
            f"family:{self._family_names[family_index]}:{epoch}:{cycle}",
        )
        return int(
            ordinals[
                (inner_multiplier * local_position + inner_offset)
                % len(ordinals)
            ]
        )

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.stop_step):
            batch = []
            for cursor in range(
                step * self.batch_size, (step + 1) * self.batch_size
            ):
                if self.family_temperature_alpha == 1.0:
                    epoch, position = divmod(cursor, self.dataset_size)
                    multiplier, offset = self._permutation(epoch)
                    batch.append(
                        (multiplier * position + offset) % self.dataset_size
                    )
                else:
                    batch.append(self._family_sample(cursor))
            yield batch


class ResumeSafeAudioBudgetBatchSampler(Sampler[list[tuple[int, int]]]):
    """Stateless duration-bucket sampling with a fixed padded-audio budget."""

    def __init__(
        self,
        window_samples: Sequence[int] | np.ndarray,
        *,
        audio_budget_samples: int,
        maximum_batch_size: int,
        start_step: int,
        stop_step: int,
        seed: int,
        family_temperature_alpha: float = 1.0,
        family_ordinals: Mapping[str, np.ndarray] | None = None,
        priority_ordinals: Sequence[int] | np.ndarray | None = None,
        priority_fraction: float = 0.0,
        priority_family_temperature_alpha: float | None = None,
    ) -> None:
        windows = np.asarray(window_samples, dtype=np.int64)
        if windows.ndim != 1 or len(windows) < 1 or np.any(windows < 1024):
            raise ValueError(
                "window_samples must be a nonempty vector of valid lengths"
            )
        self.window_samples = windows
        self.audio_budget_samples = int(audio_budget_samples)
        self.maximum_batch_size = int(maximum_batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        self.family_temperature_alpha = float(family_temperature_alpha)
        self.priority_fraction = float(priority_fraction)
        self.priority_family_temperature_alpha = (
            self.family_temperature_alpha
            if priority_family_temperature_alpha is None
            else float(priority_family_temperature_alpha)
        )
        if min(self.audio_budget_samples, self.maximum_batch_size) < 1:
            raise ValueError("Dynamic batch limits must be positive")
        if not 0 <= self.start_step <= self.stop_step:
            raise ValueError("Invalid resume step interval")
        if not 0.0 <= self.family_temperature_alpha <= 1.0:
            raise ValueError("family temperature alpha must be in [0, 1]")
        if not 0.0 <= self.priority_fraction <= 1.0:
            raise ValueError("priority fraction must be in [0, 1]")
        if not 0.0 <= self.priority_family_temperature_alpha <= 1.0:
            raise ValueError(
                "priority family temperature alpha must be in [0, 1]"
            )

        priority = np.zeros(len(windows), dtype=np.bool_)
        if priority_ordinals is not None:
            values = np.asarray(priority_ordinals, dtype=np.int64)
            if values.ndim != 1 or len(values) < 1:
                raise ValueError("priority ordinals must be a nonempty vector")
            if int(values.min()) < 0 or int(values.max()) >= len(windows):
                raise ValueError("priority ordinal is out of range")
            if len(np.unique(values)) != len(values):
                raise ValueError("priority ordinals must be unique")
            priority[values] = True
        if self.priority_fraction > 0.0 and not bool(priority.any()):
            raise ValueError(
                "positive priority fraction requires priority ordinals"
            )
        if self.priority_fraction < 1.0 and bool(priority.all()):
            raise ValueError(
                "priority fraction below one requires non-priority rows"
            )
        if self.priority_fraction == 0.0:
            priority[:] = False

        if family_ordinals is None:
            family_ordinals = {"all": np.arange(len(windows), dtype=np.int32)}
        covered = np.zeros(len(windows), dtype=np.bool_)
        cells: dict[int, dict[bool, list[tuple[str, np.ndarray]]]] = {}
        for family, ordinals_value in sorted(family_ordinals.items()):
            ordinals = np.asarray(ordinals_value, dtype=np.int32)
            if len(ordinals) < 1 or np.any(covered[ordinals]):
                raise ValueError(
                    "Family ordinals must partition dynamic windows"
                )
            covered[ordinals] = True
            family_priority = priority[ordinals]
            for is_priority in np.unique(family_priority):
                group = ordinals[family_priority == is_priority]
                group_windows = windows[group]
                for bucket in np.unique(group_windows):
                    selected = group[group_windows == bucket]
                    cells.setdefault(int(bucket), {}).setdefault(
                        bool(is_priority), []
                    ).append((str(family), selected))
        if not bool(np.all(covered)):
            raise ValueError("Family ordinals must cover dynamic windows")
        group_fractions = {
            False: 1.0 - self.priority_fraction,
            True: self.priority_fraction,
        }
        self._buckets = tuple(sorted(cells))
        if not self._buckets:
            raise ValueError("priority policy leaves no sampleable rows")
        self._cells = {
            bucket: {group: tuple(values) for group, values in groups.items()}
            for bucket, groups in cells.items()
        }
        self._cell_cumulative: dict[tuple[bool, int], np.ndarray] = {}
        group_count_totals: dict[bool, int] = {}
        self._bucket_sizes: dict[int, int] = {}
        self._group_counts: dict[tuple[bool, int], int] = {}
        for bucket in self._buckets:
            total = 0
            for group, group_cells in self._cells[bucket].items():
                alpha = (
                    self.priority_family_temperature_alpha
                    if group
                    else self.family_temperature_alpha
                )
                weights = np.power(
                    np.asarray(
                        [len(values) for _, values in group_cells],
                        dtype=np.float64,
                    ),
                    alpha,
                )
                key = (group, bucket)
                self._cell_cumulative[key] = np.cumsum(weights)
                count = sum(len(values) for _, values in group_cells)
                self._group_counts[key] = count
                group_count_totals[group] = (
                    group_count_totals.get(group, 0) + count
                )
                total += count
            self._bucket_sizes[bucket] = min(
                self.maximum_batch_size,
                max(1, self.audio_budget_samples // bucket),
                total,
            )
        bucket_weights = []
        self._priority_fraction_by_bucket: dict[int, float] = {}
        for bucket in self._buckets:
            masses = {}
            for group in (False, True):
                key = (group, bucket)
                masses[group] = (
                    group_fractions[group]
                    * self._group_counts.get(key, 0)
                    / max(group_count_totals.get(group, 0), 1)
                )
            mass = masses[False] + masses[True]
            if mass <= 0.0:
                raise ValueError(
                    "duration bucket has no requested sampling mass"
                )
            self._priority_fraction_by_bucket[bucket] = masses[True] / mass
            # Use the full mixed-bucket capacity. This preserves the padded
            # audio budget instead of fragmenting rare durations by priority.
            bucket_weights.append(mass / self._bucket_sizes[bucket])
        self._bucket_cumulative = np.cumsum(
            np.asarray(bucket_weights, dtype=np.float64)
        )

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def _uniform(self, key: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "little") / float(2**64)

    def _permuted_position(self, size: int, key: str, position: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        multiplier = max(1, int.from_bytes(digest[:8], "little") % size)
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size or 1
        offset = int.from_bytes(digest[8:16], "little") % size
        return (multiplier * int(position) + offset) % size

    @staticmethod
    def _weighted_index(cumulative: np.ndarray, unit: float) -> int:
        return min(
            len(cumulative) - 1,
            int(np.searchsorted(cumulative, unit * cumulative[-1])),
        )

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for step in range(self.start_step, self.stop_step):
            bucket_index = self._weighted_index(
                self._bucket_cumulative, self._uniform(f"bucket:{step}")
            )
            bucket = self._buckets[bucket_index]
            batch_size = self._bucket_sizes[bucket]
            desired_priority = (
                self._priority_fraction_by_bucket[bucket] * batch_size
            )
            priority_count = int(math.floor(desired_priority))
            if (
                self._uniform(f"priority-round:{step}")
                < desired_priority - priority_count
            ):
                priority_count += 1
            priority_available = self._group_counts.get((True, bucket), 0)
            ordinary_available = self._group_counts.get((False, bucket), 0)
            priority_count = min(priority_count, priority_available)
            priority_count = max(
                priority_count, batch_size - ordinary_available
            )
            group_slots = [True] * priority_count + [False] * (
                batch_size - priority_count
            )
            if len(group_slots) > 1:
                group_slots = [
                    group_slots[index]
                    for index in sorted(
                        range(len(group_slots)),
                        key=lambda index: self._uniform(
                            f"group-order:{step}:{index}"
                        ),
                    )
                ]
            batch: list[tuple[int, int]] = []
            family_positions: dict[tuple[bool, int], int] = {}
            used_ordinals: set[int] = set()
            for slot, is_priority in enumerate(group_slots):
                key = (is_priority, bucket)
                cells = self._cells[bucket][is_priority]
                cumulative = self._cell_cumulative[key]
                cell_index = self._weighted_index(
                    cumulative,
                    self._uniform(f"family:{int(is_priority)}:{step}:{slot}"),
                )
                family, ordinals = cells[cell_index]
                position_key = (is_priority, cell_index)
                local_position = family_positions.get(position_key, 0)
                if local_position >= len(ordinals):
                    available = [
                        index
                        for index, (_, values) in enumerate(cells)
                        if family_positions.get((is_priority, index), 0)
                        < len(values)
                    ]
                    cell_index = available[
                        int(
                            self._uniform(
                                f"fallback:{int(is_priority)}:{step}:{slot}"
                            )
                            * len(available)
                        )
                        % len(available)
                    ]
                    family, ordinals = cells[cell_index]
                    position_key = (is_priority, cell_index)
                    local_position = family_positions.get(position_key, 0)
                family_positions[position_key] = local_position + 1
                position = self._permuted_position(
                    len(ordinals),
                    f"ordinal:{int(is_priority)}:{family}:{bucket}:{step}",
                    local_position,
                )
                ordinal = int(ordinals[position])
                if ordinal in used_ordinals:
                    raise RuntimeError(
                        "Dynamic sampler produced a duplicate ordinal"
                    )
                used_ordinals.add(ordinal)
                batch.append((ordinal, bucket))
            yield batch
