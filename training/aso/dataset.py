"""Data access and sampling for the selected NoteSep ASO training recipe."""

from __future__ import annotations
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from training.aso.datasets.synthetic import FreshSourceCompleteDataset
from notesep.shards import PackedBlobShardReader
from training.aso.datasets.recorded import RecordedNaturalScoreSceneDataset

NFFT, HOP, SR = 2048, 512, 16000
SEED = 20260907
KIND = "best3750_anchor_f16"
FRESH_SHA = "ba5eec95ee976b71832368804c0bb863d733f5342da063d40fab3685e1103544"
RECORDED_SHA = "24c30535b171b19a6ce06b29d886aece2f1cd65d7828f4f719c928df5ea534c7"
TEST_SHA = "aa83ce707e8118b87c865d0ec5a4752eaf80a79b03842cb36cb5fa6726c8e"


def stft(y):
    return torch.stft(
        y,
        NFFT,
        HOP,
        window=torch.hann_window(NFFT, device=y.device),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )


def istft(y, length):
    return torch.istft(
        y,
        NFFT,
        HOP,
        window=torch.hann_window(NFFT, device=y.device),
        center=True,
        length=length,
    )


def metrics(target, estimate):
    t, e = target.double(), estimate.double()
    energy = t.square().sum(-1)
    if bool((energy <= 1e-12).any()):
        raise ValueError("silent target")
    gain = (e * t).sum(-1) / energy
    off = (e - gain[:, None] * t).square().sum(-1) / energy
    return dict(
        si_sdr_db=10
        * torch.log10(gain.square().clamp_min(1e-30) / off.clamp_min(1e-30)),
        off_e=off,
        gain=gain,
        scale_invariant_distortion=off / gain.square().clamp_min(1e-30),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forbid_test_path(path: Path) -> None:
    resolved = str(path.expanduser().resolve()).lower()
    forbidden = ("scns_eval_v2", "scns-eval-v2", TEST_SHA)
    if any(token in resolved for token in forbidden):
        raise ValueError(f"sealed SCNS-Eval-v2 path forbidden in training: {path}")


class ProposalCache:
    def __init__(self, root: Path, maximum_readers: int = 24) -> None:
        self.root = root.resolve()
        forbid_test_path(self.root)
        validation = json.loads((self.root / "VALIDATION.json").read_text())
        if (
            validation.get("status") != "passed"
            or validation.get("test_data_used") is not False
        ):
            raise ValueError("raw cache lacks independent train/dev validation")
        if validation.get("payload_kind") != KIND:
            raise ValueError("raw cache payload changed")
        self.validation_sha256 = sha256(self.root / "VALIDATION.json")
        self.entries: dict[tuple[str, str, str], dict[str, Any]] = {}
        for kind in ("fresh", "recorded"):
            for split in ("train", "validation"):
                for banks_path in sorted(
                    (self.root / kind / split / "tasks").glob(
                        "task-*/units/unit-*/banks.jsonl"
                    )
                ):
                    packed = banks_path.parent / "packed"
                    for line in banks_path.read_text().splitlines():
                        row = json.loads(line)
                        key = (kind, split, str(row["mixture_id"]))
                        if key in self.entries:
                            raise ValueError(f"duplicate cached mixture: {key}")
                        self.entries[key] = row | {"packed": packed}
        self.maximum_readers = maximum_readers
        self.readers: OrderedDict[Path, PackedBlobShardReader] = OrderedDict()

    def _reader(self, path: Path) -> PackedBlobShardReader:
        reader = self.readers.get(path)
        if reader is None:
            reader = PackedBlobShardReader(path, require_complete=True)
            self.readers[path] = reader
            while len(self.readers) > self.maximum_readers:
                _, stale = self.readers.popitem(last=False)
                stale.close()
        else:
            self.readers.move_to_end(path)
        return reader

    def get(
        self, kind: str, split: str, mixture_id: str, sources: int, samples: int
    ) -> torch.Tensor:
        row = self.entries[(kind, split, str(mixture_id))]
        identifiers = list(row["proposal_ids"])
        if len(identifiers) != sources:
            raise ValueError("raw proposal/source count mismatch")
        reader = self._reader(row["packed"])
        output = []
        for identifier in identifiers:
            value = np.frombuffer(
                reader.read_bytes(identifier, KIND), dtype="<f2"
            ).astype(np.float32)
            if value.size != samples or not np.isfinite(value).all():
                raise ValueError("raw proposal sample geometry changed")
            output.append(torch.from_numpy(value))
        return torch.stack(output)

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()


class Authorities:
    def __init__(
        self, fresh_index: Path, recorded_root: Path, cache: ProposalCache
    ) -> None:
        for path in (fresh_index, recorded_root, cache.root):
            forbid_test_path(path)
        if (
            sha256(fresh_index) != FRESH_SHA
            or sha256(recorded_root / "FROZEN.json") != RECORDED_SHA
        ):
            raise ValueError("training/development authority hash mismatch")
        self.cache = cache
        self.fresh = {
            split: FreshSourceCompleteDataset(
                fresh_index, split=split, crop_samples=49152, maximum_sources=64
            )
            for split in ("train", "validation")
        }
        self.recorded = {
            split: RecordedNaturalScoreSceneDataset([recorded_root], split=split)
            for split in ("train", "validation")
        }

    def ids(self, kind: str, split: str) -> list[str]:
        if kind == "fresh":
            return list(self.fresh[split].mixture_ids)
        return [scene.scene_id for scene in self.recorded[split].scenes]

    def record(
        self, kind: str, split: str, index: int, device: torch.device
    ) -> dict[str, Any]:
        if kind == "fresh":
            value = self.fresh[split][index]
            query_valid = value.get("query_valid_symbolic", value["query_valid"]).bool()
            mixture_id = str(value["mixture_id"])
        else:
            value = recorded_example(self.recorded[split], index)
            query_valid = torch.ones(len(value["targets"]), dtype=torch.bool)
            mixture_id = str(value["mixture_id"])
        mixture_wave = value["mixture"].float().to(device)
        targets_wave = value["targets"].float().to(device)
        raw_wave = self.cache.get(
            kind, split, mixture_id, len(targets_wave), len(mixture_wave)
        ).to(device)
        mixture_spec = stft(mixture_wave)
        raw_spec = stft(raw_wave)
        target_spec = stft(targets_wave)
        total = raw_spec.abs().sum(0, keepdim=True)
        base_share = torch.where(
            total > 1e-20,
            raw_spec.abs() / total.clamp_min(1e-20),
            torch.zeros_like(raw_spec.abs()),
        )
        return {
            "kind": kind,
            "split": split,
            "mixture_id": mixture_id,
            "source_parent_id": str(value["source_parent_id"]),
            "score_parent_id": str(value["score_parent_id"]),
            "length": len(mixture_wave),
            "mixture_wave": mixture_wave,
            "targets_wave": targets_wave,
            "raw_wave": raw_wave,
            "mixture": mixture_spec,
            "targets": target_spec,
            "raw": raw_spec,
            "base": base_share * mixture_spec[None],
            "pitches": value["pitches"].long().to(device),
            "onsets": value["onsets"].long().to(device),
            "offsets": value["offsets"].long().to(device),
            "symbolic_offsets": value.get("symbolic_offsets", value["offsets"])
            .long()
            .to(device),
            "query_valid": query_valid.to(device),
        }

    def close(self) -> None:
        for dataset in (*self.fresh.values(), *self.recorded.values()):
            dataset.close()
        self.cache.close()


def hashed_order(values: list[str], seed: int) -> list[int]:
    return sorted(
        range(len(values)),
        key=lambda i: hashlib.sha256(f"{seed}:{values[i]}".encode()).digest(),
    )


def recorded_balanced_order(
    authorities: Authorities, split: str, seed: int
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, mixture_id in enumerate(authorities.ids("recorded", split)):
        row = authorities.cache.entries[("recorded", split, mixture_id)]
        groups[str(row["source_parent_id"])].append(index)
    for parent, indices in groups.items():
        indices.sort(
            key=lambda i: hashlib.sha256(f"{seed}:{parent}:{i}".encode()).digest()
        )
    parents = sorted(
        groups, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest()
    )
    output = []
    while any(groups.values()):
        for parent in parents:
            if groups[parent]:
                output.append(groups[parent].pop())
    return output


def recorded_example(
    dataset: RecordedNaturalScoreSceneDataset,
    scene_index: int,
) -> dict[str, Any]:
    indices = dataset.scene_batch_indices()[scene_index]
    rows = [dataset[index] for index in indices]
    lengths = {int(row["valid_samples"]) for row in rows}
    scene_ids = {str(row["scene_id"]) for row in rows}
    if len(lengths) != 1 or len(scene_ids) != 1:
        raise ValueError("recorded scene lost aligned geometry")
    mixtures = torch.stack([row["mixture"] for row in rows])
    if not torch.equal(mixtures, mixtures[0:1].expand_as(mixtures)):
        raise ValueError("recorded note rows do not share a mixture")
    targets = torch.stack([row["target"] for row in rows])
    mixture = mixtures[0]
    if float((targets.sum(0) - mixture).abs().max()) > 1e-6:
        raise ValueError("recorded source-complete scene lost closure")
    return {
        "mixture_id": rows[0]["scene_id"],
        "source_parent_id": rows[0]["source_parent_id"],
        "score_parent_id": rows[0]["score_parent_id"],
        "mixture": mixture,
        "targets": targets,
        "pitches": torch.stack([row["pitch"] for row in rows]),
        "onsets": torch.stack([row["note_onset_samples"] for row in rows]),
        "offsets": torch.stack([row["note_offset_samples"] for row in rows]),
        "symbolic_offsets": torch.stack([row["midi_offset_samples"] for row in rows]),
        "valid_samples": len(mixture),
    }


def make_schedule(authorities: Authorities) -> tuple[list[tuple[str, int]], list[str]]:
    fresh_ids = authorities.ids("fresh", "train")
    ordered = hashed_order(fresh_ids, SEED)
    singleton_ids = [
        fresh_ids[index]
        for index in ordered
        if len(
            authorities.cache.entries[("fresh", "train", fresh_ids[index])][
                "proposal_ids"
            ]
        )
        == 1
    ]
    # A one-note crop has no ownership ambiguity: its sole feasible simplex
    # allocation is the unchanged magnitude partition.  Such crops provide no
    # gradient to a joint allocator and are recorded as identity cases rather
    # than fed through a model whose contract begins at K=2.
    full = [
        index
        for index in ordered
        if len(
            authorities.cache.entries[("fresh", "train", fresh_ids[index])][
                "proposal_ids"
            ]
        )
        >= 2
    ]
    fresh = full
    recorded_base = recorded_balanced_order(authorities, "train", SEED)
    recorded = []
    cycle = 0
    while len(recorded) < len(fresh) // 4:
        order = list(recorded_base)
        # Retain round-robin source-parent balance while changing only the
        # deterministic starting phase on repeated passes.
        if order:
            shift = int.from_bytes(
                hashlib.sha256(f"{SEED}:{cycle}".encode()).digest()[:8], "little"
            ) % len(order)
            order = order[shift:] + order[:shift]
        recorded.extend(order)
        cycle += 1
    schedule = []
    r = 0
    for position, index in enumerate(fresh, 1):
        schedule.append(("fresh", index))
        if position % 4 == 0:
            schedule.append(("recorded", recorded[r]))
            r += 1
    return schedule, singleton_ids
