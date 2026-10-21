#!/usr/bin/env python3
"""Stream scaled ASO training from frozen train/development sources only."""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from evaluate_adaptive_anchor_blend_v1 import istft, metrics, stft  # noqa: E402
from cache_best3750_raw_aso_proposals_v1 import recorded_example  # noqa: E402
from model_exp.adaptive_set_ownership import AdaptiveSetOwnership  # noqa: E402
from model_exp.adaptive_sparse_ownership import AdaptiveSparseOwnershipConfig  # noqa: E402
from model_exp.score_context_set_ownership import (  # noqa: E402
    ActiveMaskedScoreAdapterOwnership,
    FrozenCoreScoreContextSetOwnership,
    ScoreContextSetOwnership,
    initialize_frozen_core_from_aso,
    initialize_from_aso,
)
from model_exp.fresh_source_complete_data import FreshSourceCompleteDataset  # noqa: E402
from model_exp.packed_blob_shards import PackedBlobShardReader  # noqa: E402
from model_exp.recorded_natural_score_scene_data import RecordedNaturalScoreSceneDataset  # noqa: E402

SEED = 20260907
KIND = "best3750_anchor_f16"
SCHEMA = "aso_scaled_source_diverse_training_v1"
CHECKPOINT_SHA = "3e8a81d98c76306caf99486c556597f4614dc303c8012a36678b4679e33e7496"
SELECTED_ASO_SHA = "8d41713af8174254c77adafa8d8e225d7de48b0cfda3ada074e97d30c3284ac5"
SCORE_CONTEXT_PREREG_SHA = "0874aae967490e2bc542f35d3a3762b0b9384378ec2b7c973e518fa905a44652"
FROZEN_CORE_PREREG_SHA = "9d1fcb9f0ba1468adc0357dfa2e420dbd2e5640b9999c83cd37176edbf00ce54"
FROZEN_ADAPTER_PREREG_SHA = "7e552261f8d28b60171d643811ffca0d7d16e5d2ef32aac4d3499eefb00ea3cb"
ACTIVE_MASKED_ADAPTER_PREREG_SHA = "1063b0638692c385af696b1c4ce9f21dc79a32cde93b3f990e610c1879e5cf28"
FRESH_SHA = "ba5eec95ee976b71832368804c0bb863d733f5342da063d40fab3685e1103544"
RECORDED_SHA = "24c30535b171b19a6ce06b29d886aece2f1cd65d7828f4f719c928df5ea534c7"
TEST_SHA = "aa83ce707e8118b87c865d0ec5a4752e52eaf80a79b03842cb36cb5fa6726c8e"
STOP = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


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
        if validation.get("status") != "passed" or validation.get("test_data_used") is not False:
            raise ValueError("raw cache lacks independent train/dev validation")
        if validation.get("payload_kind") != KIND:
            raise ValueError("raw cache payload changed")
        self.validation_sha256 = sha256(self.root / "VALIDATION.json")
        self.entries: dict[tuple[str, str, str], dict[str, Any]] = {}
        for kind in ("fresh", "recorded"):
            for split in ("train", "validation"):
                for banks_path in sorted((self.root / kind / split / "tasks").glob("task-*/units/unit-*/banks.jsonl")):
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

    def get(self, kind: str, split: str, mixture_id: str, sources: int, samples: int) -> torch.Tensor:
        row = self.entries[(kind, split, str(mixture_id))]
        identifiers = list(row["proposal_ids"])
        if len(identifiers) != sources:
            raise ValueError("raw proposal/source count mismatch")
        reader = self._reader(row["packed"])
        output = []
        for identifier in identifiers:
            value = np.frombuffer(reader.read_bytes(identifier, KIND), dtype="<f2").astype(np.float32)
            if value.size != samples or not np.isfinite(value).all():
                raise ValueError("raw proposal sample geometry changed")
            output.append(torch.from_numpy(value))
        return torch.stack(output)

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()


class Authorities:
    def __init__(self, fresh_index: Path, recorded_root: Path, cache: ProposalCache) -> None:
        for path in (fresh_index, recorded_root, cache.root):
            forbid_test_path(path)
        if sha256(fresh_index) != FRESH_SHA or sha256(recorded_root / "FROZEN.json") != RECORDED_SHA:
            raise ValueError("training/development authority hash mismatch")
        self.cache = cache
        self.fresh = {
            split: FreshSourceCompleteDataset(fresh_index, split=split, crop_samples=49152, maximum_sources=64)
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

    def record(self, kind: str, split: str, index: int, device: torch.device) -> dict[str, Any]:
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
        raw_wave = self.cache.get(kind, split, mixture_id, len(targets_wave), len(mixture_wave)).to(device)
        mixture_spec = stft(mixture_wave)
        raw_spec = stft(raw_wave)
        target_spec = stft(targets_wave)
        total = raw_spec.abs().sum(0, keepdim=True)
        base_share = torch.where(total > 1e-20, raw_spec.abs() / total.clamp_min(1e-20), torch.zeros_like(raw_spec.abs()))
        return {
            "kind": kind, "split": split, "mixture_id": mixture_id,
            "source_parent_id": str(value["source_parent_id"]),
            "score_parent_id": str(value["score_parent_id"]),
            "length": len(mixture_wave), "mixture_wave": mixture_wave,
            "targets_wave": targets_wave, "raw_wave": raw_wave,
            "mixture": mixture_spec, "targets": target_spec, "raw": raw_spec,
            "base": base_share * mixture_spec[None],
            "pitches": value["pitches"].long().to(device),
            "onsets": value["onsets"].long().to(device),
            "offsets": value["offsets"].long().to(device),
            "symbolic_offsets": value.get(
                "symbolic_offsets", value["offsets"]
            ).long().to(device),
            "query_valid": query_valid.to(device),
        }

    def close(self) -> None:
        for dataset in (*self.fresh.values(), *self.recorded.values()):
            dataset.close()
        self.cache.close()


def call(model: AdaptiveSetOwnership, record: dict[str, Any], mode: str = "adaptive"):
    return model(record["mixture"][None], record["base"][None], record["raw"][None],
                 record["pitches"][None], record["onsets"][None], record["offsets"][None],
                 record["offsets"][None], confidence_mode=mode)


def note_error(estimate: torch.Tensor, record: dict[str, Any]) -> torch.Tensor:
    return (estimate - record["targets"]).abs().square().sum((-2, -1)) / record["targets"].abs().square().sum((-2, -1)).clamp_min(1e-12)


def train_loss(model: AdaptiveSetOwnership, record: dict[str, Any]):
    estimate, diagnostics = call(model, record)
    estimate = estimate[0]
    errors = note_error(estimate, record)
    baseline = note_error(record["base"], record).detach()
    valid = record["query_valid"]
    if not bool(valid.any()):
        raise ValueError("training scene has no valid supervised query")
    selected = errors[valid]
    regret = torch.relu(errors[valid] - baseline[valid])
    target_magnitude = record["targets"].abs()
    target_share = target_magnitude / target_magnitude.sum(0, keepdim=True).clamp_min(1e-8)
    weight = record["mixture"].abs().square()
    weight = weight / weight.sum().clamp_min(1e-12)
    ownership = -(target_share * diagnostics["ownership"][0].clamp_min(1e-8).log()).sum(0).mul(weight).sum()
    loss = selected.mean() + 2 * regret.mean() + .25 * selected.max() + ownership + .01 * diagnostics["mean_abs_correction"]
    return loss, errors, regret, ownership, diagnostics


def hashed_order(values: list[str], seed: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: hashlib.sha256(f"{seed}:{values[i]}".encode()).digest())


def recorded_balanced_order(authorities: Authorities, split: str, seed: int) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, mixture_id in enumerate(authorities.ids("recorded", split)):
        row = authorities.cache.entries[("recorded", split, mixture_id)]
        groups[str(row["source_parent_id"])].append(index)
    for parent, indices in groups.items():
        indices.sort(key=lambda i: hashlib.sha256(f"{seed}:{parent}:{i}".encode()).digest())
    parents = sorted(groups, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest())
    output = []
    while any(groups.values()):
        for parent in parents:
            if groups[parent]:
                output.append(groups[parent].pop())
    return output


def permute_record(record: dict[str, Any], order: torch.Tensor) -> dict[str, Any]:
    output = dict(record)
    for key in (
        "targets_wave", "raw_wave", "targets", "raw", "base", "pitches",
        "onsets", "offsets", "symbolic_offsets", "query_valid",
    ):
        output[key] = record[key][order]
    return output


def make_schedule(authorities: Authorities, arm: str) -> tuple[list[tuple[str, int]], list[str]]:
    fresh_ids = authorities.ids("fresh", "train")
    ordered = hashed_order(fresh_ids, SEED)
    singleton_ids = [
        fresh_ids[index] for index in ordered
        if len(authorities.cache.entries[("fresh", "train", fresh_ids[index])]["proposal_ids"]) == 1
    ]
    # A one-note crop has no ownership ambiguity: its sole feasible simplex
    # allocation is the unchanged magnitude partition.  Such crops provide no
    # gradient to a joint allocator and are recorded as identity cases rather
    # than fed through a model whose contract begins at K=2.
    full = [
        index for index in ordered
        if len(authorities.cache.entries[("fresh", "train", fresh_ids[index])]["proposal_ids"]) >= 2
    ]
    if arm == "current_data25":
        subset = sorted(full[: len(full) // 4])
        rng = random.Random(SEED)
        rng.shuffle(subset)
        fresh = [subset[i % len(subset)] for i in range(len(full))]
    else:
        fresh = full
    recorded_base = recorded_balanced_order(authorities, "train", SEED)
    recorded = []
    cycle = 0
    while len(recorded) < len(fresh) // 4:
        order = list(recorded_base)
        # Retain round-robin source-parent balance while changing only the
        # deterministic starting phase on repeated passes.
        if order:
            shift = int.from_bytes(hashlib.sha256(f"{SEED}:{cycle}".encode()).digest()[:8], "little") % len(order)
            order = order[shift:] + order[:shift]
        recorded.extend(order)
        cycle += 1
    schedule = []
    r = 0
    for position, index in enumerate(fresh, 1):
        schedule.append(("fresh", index))
        if position % 4 == 0:
            schedule.append(("recorded", recorded[r])); r += 1
    return schedule, singleton_ids


@torch.no_grad()
def evaluate(model: AdaptiveSetOwnership, authorities: Authorities, maximum: int = 0) -> dict[str, Any]:
    model.eval()
    events = []
    closure = 0.0
    for kind in ("fresh", "recorded"):
        count = len(authorities.ids(kind, "validation"))
        if maximum:
            count = min(count, maximum)
        for index in range(count):
            record = authorities.record(kind, "validation", index, torch.device("cuda"))
            estimate, diagnostics = call(model, record)
            estimate = estimate[0]
            wave = istft(estimate, record["length"]).float()
            base_wave = istft(record["base"], record["length"]).float()
            candidate_metrics = metrics(record["targets_wave"], wave)
            base_metrics = metrics(record["targets_wave"], base_wave)
            for note in torch.where(record["query_valid"])[0].tolist():
                events.append({"dataset_kind": kind, "mixture_id": record["mixture_id"], "note": note,
                               "candidate": {key: float(value[note]) for key, value in candidate_metrics.items()},
                               "magnitude": {key: float(value[note]) for key, value in base_metrics.items()}})
            closure = max(closure, float(diagnostics["sum_error_max_abs"]))
    if not events:
        raise ValueError("empty development evaluation")
    summary = {}
    for method in ("candidate", "magnitude"):
        summary[method] = {metric: {"median": float(np.median([e[method][metric] for e in events])),
                                    "mean": float(np.mean([e[method][metric] for e in events]))}
                           for metric in ("si_sdr_db", "off_e", "gain", "scale_invariant_distortion")}
    delta = np.asarray([e["candidate"]["si_sdr_db"] - e["magnitude"]["si_sdr_db"] for e in events])
    return {"coverage": len(events), "events": events, "summary": summary,
            "candidate_minus_magnitude": {"median_db": float(np.median(delta)), "mean_db": float(np.mean(delta)),
                "losses_gt_0p1db": int((delta < -.1).sum()), "losses_gt_3db": int((delta < -3).sum()),
                "wins_gt_0p1db": int((delta > .1).sum())},
            "maximum_spectral_closure_abs": closure}


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-index", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("current_data25", "small_full", "current_full", "large_full", "score_context_full", "score_context_frozen_core", "frozen_core_score_adapter", "active_masked_score_adapter"), required=True)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA compute required")
    for path in (args.fresh_index, args.recorded_root, args.proposal_root, args.output):
        forbid_test_path(path)
    for path in (args.warm_start, args.preregister):
        if path is not None:
            forbid_test_path(path)
    if args.output.exists() and not (args.output / "RESUME.pt").exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGUSR1, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cache = ProposalCache(args.proposal_root)
    authorities = Authorities(args.fresh_index, args.recorded_root, cache)
    config_by_arm = {"current_data25": (24, 2), "small_full": (16, 1),
                     "current_full": (24, 2), "large_full": (48, 3),
                     "score_context_full": (16, 1),
                     "score_context_frozen_core": (16, 1),
                     "frozen_core_score_adapter": (16, 1),
                     "active_masked_score_adapter": (16, 1)}
    channels, blocks = config_by_arm[args.arm]
    config = AdaptiveSparseOwnershipConfig(hidden_channels=channels, local_blocks=blocks)
    initialization = None
    score_context_arm = args.arm in {
        "score_context_full", "score_context_frozen_core",
        "frozen_core_score_adapter", "active_masked_score_adapter",
    }
    if score_context_arm:
        if args.warm_start is None or sha256(args.warm_start) != SELECTED_ASO_SHA:
            raise ValueError("score-context arm requires selected ASO warm start")
        expected_preregister = {
            "score_context_full": SCORE_CONTEXT_PREREG_SHA,
            "score_context_frozen_core": FROZEN_CORE_PREREG_SHA,
            "frozen_core_score_adapter": FROZEN_ADAPTER_PREREG_SHA,
            "active_masked_score_adapter": ACTIVE_MASKED_ADAPTER_PREREG_SHA,
        }[args.arm]
        if args.preregister is None or sha256(args.preregister) != expected_preregister:
            raise ValueError("score-context preregistration mismatch")
        source = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if args.arm in {"frozen_core_score_adapter", "active_masked_score_adapter"}:
            model_type = (
                ActiveMaskedScoreAdapterOwnership
                if args.arm == "active_masked_score_adapter"
                else FrozenCoreScoreContextSetOwnership
            )
            model = model_type(config).cuda()
            initialization = initialize_frozen_core_from_aso(model, source["model"])
        else:
            model = ScoreContextSetOwnership(config).cuda()
            initialization = initialize_from_aso(model, source["model"])
    else:
        if args.warm_start is not None:
            raise ValueError("warm start is exclusive to score-context arm")
        if args.preregister is not None:
            raise ValueError("preregister argument is exclusive to score-context arm")
        model = AdaptiveSetOwnership(config).cuda()
    frozen_input_prefix = None
    if args.arm == "score_context_frozen_core":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.input.weight.requires_grad_(True)
        frozen_input_prefix = model.input.weight[:, : model.base_feature_channels].detach().clone()
        gradient_mask = torch.zeros_like(model.input.weight)
        gradient_mask[:, model.base_feature_channels :] = 1
        model.input.weight.register_hook(lambda gradient: gradient * gradient_mask)
        optimizer = torch.optim.AdamW([model.input.weight], lr=3e-4, weight_decay=1e-4)
    elif args.arm in {"frozen_core_score_adapter", "active_masked_score_adapter"}:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.score_input.weight.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [model.score_input.weight], lr=3e-4, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule, singleton_ids = make_schedule(authorities, args.arm)
    canary_records = None
    if args.mode == "canary":
        schedule = [("fresh", 0), ("recorded", 0)] * 100
        canary_records = {
            kind: authorities.record(kind, "train", 0, torch.device("cuda"))
            for kind in ("fresh", "recorded")
        }
    contract = {"schema": SCHEMA, "arm": args.arm, "mode": args.mode, "seed": SEED,
                "config": asdict(config), "steps": len(schedule), "test_data_used": False,
                "singleton_identity_rule": {
                    "behavior": "K=1 has unique ownership and is omitted from optimization as an exact no-op",
                    "fresh_train_mixtures": len(singleton_ids),
                    "mixture_ids": singleton_ids,
                },
                "test_authority_forbidden_sha256": TEST_SHA,
                "proposal_validation_sha256": cache.validation_sha256,
                "fresh_index_sha256": FRESH_SHA, "recorded_frozen_sha256": RECORDED_SHA,
                "best3750_checkpoint_sha256": CHECKPOINT_SHA,
                "selected_aso_warm_start_sha256": SELECTED_ASO_SHA if score_context_arm else None,
                "preregister_sha256": (
                    FROZEN_CORE_PREREG_SHA if args.arm == "score_context_frozen_core"
                    else ACTIVE_MASKED_ADAPTER_PREREG_SHA if args.arm == "active_masked_score_adapter"
                    else FROZEN_ADAPTER_PREREG_SHA if args.arm == "frozen_core_score_adapter"
                    else SCORE_CONTEXT_PREREG_SHA if args.arm == "score_context_full" else None
                ),
                "trainable_parameter_contract": (
                    "only input.weight[:,11:16,:,:] (720 scalars); all selected-ASO parameters bit-exact"
                    if args.arm == "score_context_frozen_core"
                    else "only score_input.weight (720 scalars); all selected-ASO parameters bit-exact"
                    if args.arm in {"frozen_core_score_adapter", "active_masked_score_adapter"}
                    else "all model parameters"
                ),
                "initialization": initialization,
                "trainer_sha256": sha256(Path(__file__)),
                "model_sha256": sha256(ROOT / "src/model_exp/score_context_set_ownership.py") if score_context_arm else sha256(ROOT / "src/model_exp/adaptive_set_ownership.py"),
                "base_model_sha256": sha256(ROOT / "src/model_exp/adaptive_sparse_ownership.py")}
    contract_path = args.output / "CONTRACT.json"
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    if contract_path.exists() and contract_path.read_text() != encoded + "\n":
        raise ValueError("resume contract mismatch")
    if not contract_path.exists():
        contract_path.write_text(encoded + "\n")
    start_step = 0; history = []; development_checkpoints = []
    saved_canary = None
    resume = args.output / "RESUME.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]); history = list(state["history"])
        development_checkpoints = list(state.get("development_checkpoints", []))
        saved_canary = state.get("canary_initial")
    started = time.monotonic()
    canary_initial_error = None
    canary_initial_parity = None
    canary_permutation_error = None
    canary_owner_parity = None
    if canary_records is not None and start_step == 0:
        with torch.no_grad():
            initial_errors = []
            canary_initial_parity = 0.0
            canary_permutation_error = 0.0
            reference = None
            if score_context_arm:
                reference = AdaptiveSetOwnership(config).cuda()
                reference.load_state_dict(source["model"], strict=True)
                reference.eval()
                canary_owner_parity = 0.0
            for record in canary_records.values():
                estimate = call(model, record)[0][0]
                if reference is not None:
                    reference_estimate = call(reference, record)[0][0]
                    canary_owner_parity = max(
                        canary_owner_parity, float((estimate - reference_estimate).abs().max())
                    )
                zero_control = call(model, record, "zero")[0][0]
                initial_errors.extend(note_error(estimate, record)[record["query_valid"]].cpu().tolist())
                canary_initial_parity = max(canary_initial_parity, float((zero_control - record["base"]).abs().max()))
                order = torch.arange(len(record["pitches"]), device="cuda").flip(0)
                permuted = call(model, permute_record(record, order))[0][0]
                restored = permuted[torch.argsort(order)]
                canary_permutation_error = max(canary_permutation_error, float((estimate - restored).abs().max()))
            canary_initial_error = float(np.mean(initial_errors))
            del reference
    elif saved_canary is not None:
        canary_initial_error = float(saved_canary["error"])
        canary_initial_parity = float(saved_canary["parity"])
        canary_permutation_error = float(saved_canary["permutation"])
        canary_owner_parity = saved_canary.get("owner_parity")
    try:
        for step in range(start_step, len(schedule)):
            kind, index = schedule[step]
            record = (canary_records[kind] if canary_records is not None
                      else authorities.record(kind, "train", index, torch.device("cuda")))
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, errors, regret, ownership, diagnostics = train_loss(model, record)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite loss")
            loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient)):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            if frozen_input_prefix is not None:
                # AdamW applies decoupled decay to an entire parameter tensor.
                # Restore the frozen slice after every step so only the five
                # new score-channel slices can change, including on resume.
                with torch.no_grad():
                    model.input.weight[:, : model.base_feature_channels].copy_(frozen_input_prefix)
            done = step + 1
            if done == 1 or done % 100 == 0 or done == len(schedule):
                row = {"step": done, "kind": kind, "loss": float(loss.detach()),
                       "complex_error": float(errors[record["query_valid"]].mean().detach()),
                       "regret": float(regret.mean().detach()), "ownership": float(ownership.detach()),
                       "gradient_norm": float(gradient.detach()),
                       "confidence": float(diagnostics["mean_confidence"].detach())}
                history.append(row); print(json.dumps(row), flush=True)
            checkpoint_due = done in {len(schedule) // 4, len(schedule) // 2, 3 * len(schedule) // 4, len(schedule)}
            if args.mode == "full" and done == 3 * len(schedule) // 4:
                curve = evaluate(model, authorities)
                development_checkpoints.append({"step": done, "coverage": curve["coverage"],
                    "summary": curve["summary"], "candidate_minus_magnitude": curve["candidate_minus_magnitude"],
                    "maximum_spectral_closure_abs": curve["maximum_spectral_closure_abs"]})
            if checkpoint_due or STOP:
                payload = {"schema": SCHEMA, "contract": contract, "step": done,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                           "development_checkpoints": development_checkpoints,
                           "canary_initial": None if canary_records is None else {
                               "error": canary_initial_error, "parity": canary_initial_parity,
                               "permutation": canary_permutation_error,
                               "owner_parity": canary_owner_parity}}
                atomic_torch(args.output / f"checkpoint-step-{done:06d}.pt", payload)
                atomic_torch(resume, payload)
            if STOP:
                return 75
        development = evaluate(model, authorities, maximum=2 if args.mode == "canary" else 0)
        canary = None
        status = "completed_pending_independent_validation"
        if canary_records is not None:
            final_errors = []
            with torch.no_grad():
                for record in canary_records.values():
                    final_errors.extend(note_error(call(model, record)[0][0], record)[record["query_valid"]].cpu().tolist())
            final_error = float(np.mean(final_errors))
            reduction = 1.0 - final_error / max(float(canary_initial_error), 1e-12)
            gates = {"training_error_reduction_10pct": reduction >= .10,
                     "confidence_zero_magnitude_parity_1e_minus6": float(canary_initial_parity) <= 1e-6,
                     "note_permutation_equivariance_1e_minus4": float(canary_permutation_error) <= 1e-4,
                     "spectral_closure_1e_minus4": development["maximum_spectral_closure_abs"] <= 1e-4,
                     "finite_development_metrics": all(np.isfinite(e["candidate"]["si_sdr_db"]) for e in development["events"])}
            if score_context_arm:
                gates["warm_start_output_parity_1e_minus6"] = float(canary_owner_parity) <= 1e-6
            if args.arm == "score_context_frozen_core":
                current = model.state_dict()
                source_state = source["model"]
                frozen_exact = all(
                    torch.equal(current[key].detach().cpu(), value)
                    for key, value in source_state.items() if key != "input.weight"
                ) and torch.equal(
                    current["input.weight"][:, : model.base_feature_channels].detach().cpu(),
                    source_state["input.weight"],
                )
                score_changed = bool(
                    torch.count_nonzero(
                        current["input.weight"][:, model.base_feature_channels :]
                    ).item()
                )
                gates["frozen_core_bit_exact"] = frozen_exact
                gates["score_input_weights_changed"] = score_changed
            if args.arm in {"frozen_core_score_adapter", "active_masked_score_adapter"}:
                current = model.state_dict()
                source_state = source["model"]
                gates["frozen_core_bit_exact"] = all(
                    torch.equal(current[key].detach().cpu(), value)
                    for key, value in source_state.items()
                )
                gates["score_input_weights_changed"] = bool(
                    torch.count_nonzero(current["score_input.weight"]).item()
                )
            canary = {"initial_error": canary_initial_error, "final_error": final_error,
                      "relative_reduction": reduction, "initial_parity_max_abs": canary_initial_parity,
                      "permutation_error_max_abs": canary_permutation_error,
                      "warm_start_output_parity_max_abs": canary_owner_parity, "gates": gates}
            status = "passed_canary" if all(gates.values()) else "failed_canary"
        result = {"schema": SCHEMA, "status": status,
                  "role": "training/development only; sealed SCNS-Eval-v2 untouched",
                  "arm": args.arm, "mode": args.mode, "steps": len(schedule),
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                  "contract": contract, "development": development,
                  "development_checkpoints": development_checkpoints, "history": history,
                  "runtime_seconds": time.monotonic() - started, "job_id": os.environ["SLURM_JOB_ID"],
                  "device": torch.cuda.get_device_name(), "test_data_used": False,
                  "checkpoint": str(args.output / f"checkpoint-step-{len(schedule):06d}.pt")}
        if args.arm == "score_context_frozen_core":
            current = model.state_dict()
            source_state = source["model"]
            result["frozen_core_audit"] = {
                "all_downstream_tensors_bit_exact": all(
                    torch.equal(current[key].detach().cpu(), value)
                    for key, value in source_state.items() if key != "input.weight"
                ),
                "original_input_slices_bit_exact": torch.equal(
                    current["input.weight"][:, : model.base_feature_channels].detach().cpu(),
                    source_state["input.weight"],
                ),
                "changed_score_input_weights": int(torch.count_nonzero(
                    current["input.weight"][:, model.base_feature_channels :]
                ).item()),
                "score_input_weights_total": int(
                    current["input.weight"][:, model.base_feature_channels :].numel()
                ),
            }
        if args.arm in {"frozen_core_score_adapter", "active_masked_score_adapter"}:
            current = model.state_dict()
            source_state = source["model"]
            result["frozen_core_audit"] = {
                "all_base_tensors_bit_exact": all(
                    torch.equal(current[key].detach().cpu(), value)
                    for key, value in source_state.items()
                ),
                "changed_score_adapter_weights": int(torch.count_nonzero(
                    current["score_input.weight"]
                ).item()),
                "score_adapter_weights_total": int(
                    current["score_input.weight"].numel()
                ),
            }
        if canary is not None:
            result["canary"] = canary
        result["checkpoint_sha256"] = sha256(Path(result["checkpoint"]))
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"], "arm": args.arm, "parameters": result["parameters"],
                          "development": development["summary"], "comparison": development["candidate_minus_magnitude"]}, indent=2))
        return 0 if status != "failed_canary" else 2
    finally:
        authorities.close()


if __name__ == "__main__":
    raise SystemExit(main())
