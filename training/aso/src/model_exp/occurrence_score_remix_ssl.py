"""Pure helpers for score-conditioned self-remixing of occurrence separators.

The audio target is produced by a frozen or deliberately slow EMA teacher.
Only score metadata from the real recording is used; no real-domain source stem
enters training.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Iterable

import torch
from torch import nn


def restore_teacher_state(
    teacher: nn.Module, state: Mapping[str, Any]
) -> bool:
    """Restore an EMA SSL teacher from a resumable adaptation state.

    Older frozen-teacher states intentionally have no ``teacher`` entry and
    remain valid; returning ``False`` keeps the initialized teacher in that
    case.  The retention teacher is never restored here because it is always
    the immutable initializer authority.
    """

    teacher_state = state.get("teacher")
    if teacher_state is None:
        return False
    if not isinstance(teacher_state, Mapping):
        raise ValueError("resume teacher state must be a state-dict mapping")
    teacher.load_state_dict(teacher_state, strict=True)
    return True


def batch_derangement(size: int, *, generator: torch.Generator) -> torch.Tensor:
    """Return a deterministic random cyclic derangement on CPU."""

    if size < 2:
        raise ValueError("self-remixing requires at least two examples")
    shift = int(torch.randint(1, size, (), generator=generator).item())
    return (torch.arange(size, dtype=torch.long) + shift) % size


def collision_free_partner_indices(
    active_pitch_mask: torch.Tensor,
    query_pitch: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Choose a score-certified bidirectional remix partner for every row.

    A partner ``j`` is allowed for anchor ``i`` only when the complete scored
    crop ``j`` contains no occurrence of query ``i`` and crop ``i`` contains
    no occurrence of query ``j``. Partners may be reused; this avoids silently
    accepting a colliding pair merely because a perfect batch permutation does
    not exist. ``-1`` denotes a row with no valid partner in the current batch.
    """

    if active_pitch_mask.ndim != 2 or active_pitch_mask.shape[-1] != 128:
        raise ValueError("active_pitch_mask must have shape [B, 128]")
    batch = active_pitch_mask.shape[0]
    if query_pitch.shape != (batch,):
        raise ValueError("query_pitch must have shape [B]")
    if batch < 2:
        raise ValueError("collision-free remixing requires at least two examples")
    mask = active_pitch_mask.detach().to(device="cpu", dtype=torch.bool)
    pitch = query_pitch.detach().to(device="cpu", dtype=torch.long)
    if torch.any((pitch < 0) | (pitch >= 128)):
        raise ValueError("query pitches must be MIDI values in [0, 127]")

    partners = torch.full((batch,), -1, dtype=torch.long)
    for anchor in range(batch):
        candidates = [
            other
            for other in range(batch)
            if other != anchor
            and not bool(mask[other, pitch[anchor]])
            and not bool(mask[anchor, pitch[other]])
        ]
        if candidates:
            selected = int(
                torch.randint(len(candidates), (), generator=generator).item()
            )
            partners[anchor] = int(candidates[selected])
    return partners


def collision_free_triplet_indices(
    active_pitch_mask: torch.Tensor,
    query_pitch: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Choose two score-certified, mutually non-colliding partners per row.

    This supports a 1-to-3 real-mixture invariance curriculum without stems:
    when row ``i`` is queried, neither added recording contains its score
    pitch, and none of the three recordings contains either other query
    pitch.  A row of ``(-1, -1)`` has no valid triplet in this batch. Partners
    may be reused by different anchors, which keeps sampling scalable at small
    batch sizes instead of weakening the no-same-note guarantee.
    """

    if active_pitch_mask.ndim != 2 or active_pitch_mask.shape[-1] != 128:
        raise ValueError("active_pitch_mask must have shape [B, 128]")
    batch = active_pitch_mask.shape[0]
    if query_pitch.shape != (batch,):
        raise ValueError("query_pitch must have shape [B]")
    if batch < 3:
        raise ValueError("three-way remixing requires at least three examples")
    mask = active_pitch_mask.detach().to(device="cpu", dtype=torch.bool)
    pitch = query_pitch.detach().to(device="cpu", dtype=torch.long)
    if torch.any((pitch < 0) | (pitch >= 128)):
        raise ValueError("query pitches must be MIDI values in [0, 127]")

    partners = torch.full((batch, 2), -1, dtype=torch.long)
    for anchor in range(batch):
        candidates = [
            other
            for other in range(batch)
            if other != anchor
            and not bool(mask[other, pitch[anchor]])
            and not bool(mask[anchor, pitch[other]])
        ]
        valid_pairs = [
            (left, right)
            for position, left in enumerate(candidates)
            for right in candidates[position + 1 :]
            if not bool(mask[left, pitch[right]])
            and not bool(mask[right, pitch[left]])
        ]
        if valid_pairs:
            selected = valid_pairs[
                int(torch.randint(len(valid_pairs), (), generator=generator).item())
            ]
            partners[anchor] = torch.tensor(selected, dtype=torch.long)
    return partners


def select_collision_free_triplet_anchors(
    triplets: torch.Tensor,
    *,
    maximum_anchors: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a reproducible bounded subset of valid three-way anchors.

    This permits a larger score-only proposal pool while retaining a fixed
    student-gradient batch.  It never accepts a row without two certified
    collision-free partners.
    """

    if triplets.ndim != 2 or triplets.shape[1] != 2:
        raise ValueError("triplets must have shape [B, 2]")
    if maximum_anchors < 0:
        raise ValueError("maximum_anchors must be nonnegative")
    valid = torch.nonzero(
        torch.all(triplets >= 0, dim=1), as_tuple=False
    ).flatten()
    if maximum_anchors and valid.numel() > maximum_anchors:
        order = torch.randperm(valid.numel(), generator=generator)
        valid = valid.index_select(0, order[:maximum_anchors]).sort().values
    return valid


def score_complete_mixture_closure(
    note_estimates: torch.Tensor,
    note_to_mixture: torch.Tensor,
    mixture: torch.Tensor,
    valid_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum score-query estimates and return masked per-mixture closure error.

    ``note_to_mixture`` assigns each queried score occurrence to its real
    recording.  There is deliberately no learned residual: all energy in an
    eligible sparse crop must be allocated among its score-present note
    queries.  Callers should pair this constraint with pseudo-target/replay
    losses so a trivial zero allocation cannot satisfy the whole objective.
    """

    if note_estimates.ndim != 2 or mixture.ndim != 2:
        raise ValueError("note_estimates and mixture must have shape [N, T]/[B, T]")
    batch, samples = mixture.shape
    if note_estimates.shape[1] != samples:
        raise ValueError("estimate and mixture sample counts differ")
    if note_to_mixture.ndim != 1 or note_to_mixture.shape[0] != note_estimates.shape[0]:
        raise ValueError("note_to_mixture must have one row per estimate")
    if valid_samples.shape != (batch,):
        raise ValueError("valid_samples must have shape [B]")
    if torch.any((note_to_mixture < 0) | (note_to_mixture >= batch)):
        raise ValueError("note_to_mixture contains an invalid mixture index")
    summed = torch.zeros_like(mixture)
    summed.index_add_(0, note_to_mixture.to(device=mixture.device), note_estimates)
    sample_index = torch.arange(samples, device=mixture.device)[None]
    mask = sample_index < valid_samples.to(device=mixture.device)[:, None]
    error = (summed - mixture).abs().mul(mask).sum(dim=-1)
    error = error / valid_samples.to(device=mixture.device).clamp_min(1)
    return summed, error


def select_overlapping_competing_score_events(
    score_events: Iterable[Iterable[tuple[int, int, int]]],
    query_pitch: torch.Tensor,
    query_onset: torch.Tensor,
    query_offset: torch.Tensor,
) -> tuple[tuple[int, int, int] | None, ...]:
    """Choose a scored competing event for output-purity SSL.

    A note estimate that has bundled another simultaneously sounding pitch
    should not allow that other score query to be recovered from the estimate
    alone.  This selector makes that property testable without a real stem:
    it chooses the maximally overlapping *different-pitch* score event in the
    originating crop.  ``None`` means the query has no such competition and
    is simply excluded from the purity term; it is never turned into an
    arbitrary negative or quality-filtered example.
    """

    if not (
        query_pitch.ndim == query_onset.ndim == query_offset.ndim == 1
        and query_pitch.shape == query_onset.shape == query_offset.shape
    ):
        raise ValueError("query pitch/onset/offset must be aligned vectors")
    rows = tuple(tuple(events) for events in score_events)
    if len(rows) != query_pitch.numel():
        raise ValueError("score events must have one row per query")
    selected: list[tuple[int, int, int] | None] = []
    for events, pitch, onset, offset in zip(
        rows,
        query_pitch.detach().cpu().tolist(),
        query_onset.detach().cpu().tolist(),
        query_offset.detach().cpu().tolist(),
        strict=True,
    ):
        candidates: list[tuple[int, int, int, int]] = []
        for event_pitch, event_onset, event_offset in events:
            overlap = max(0, min(int(offset), int(event_offset)) - max(int(onset), int(event_onset)))
            if int(event_pitch) != int(pitch) and overlap:
                # Largest true overlap first; stable metadata tie breaks make
                # resumes independent of Python/container ordering.
                candidates.append((-overlap, int(event_pitch), int(event_onset), int(event_offset)))
        if not candidates:
            selected.append(None)
            continue
        _, event_pitch, event_onset, event_offset = min(candidates)
        selected.append((event_pitch, event_onset, event_offset))
    return tuple(selected)


def score_event_groups_temporally_collide(
    anchor_events: Iterable[tuple[int, int, int]],
    interferer_events: Iterable[tuple[int, int, int]],
) -> bool:
    """Return whether two score groups contain an inseparable query collision.

    For score-conditioned mixture-of-mixtures training, the model receives no
    recording identity beyond pitch and event bounds.  Two events with the
    same pitch and overlapping support would therefore be the same query but
    have different recording-group ownership, an impossible target.  Exact
    score geometry—not an audio heuristic—excludes only those cases.
    """

    by_pitch: dict[int, list[tuple[int, int]]] = {}
    for pitch, onset, offset in interferer_events:
        if int(onset) < int(offset):
            by_pitch.setdefault(int(pitch), []).append((int(onset), int(offset)))
    for pitch, onset, offset in anchor_events:
        for other_onset, other_offset in by_pitch.get(int(pitch), ()):
            if max(int(onset), other_onset) < min(int(offset), other_offset):
                return True
    return False


def select_score_group_pair_positions(
    eligible: torch.Tensor,
    *,
    number_of_pairs: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Select a fixed gradient pair batch, prioritizing valid group targets."""

    if eligible.ndim != 1 or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a boolean vector")
    if not 1 <= number_of_pairs <= eligible.numel():
        raise ValueError("number_of_pairs must fit the proposal pool")
    valid = torch.nonzero(eligible, as_tuple=False).flatten()
    selected: list[int] = []
    if valid.numel():
        order = torch.randperm(valid.numel(), generator=generator)
        selected.extend(valid.index_select(0, order).tolist()[:number_of_pairs])
    if len(selected) < number_of_pairs:
        remaining = [
            index for index in range(eligible.numel()) if index not in selected
        ]
        order = torch.randperm(len(remaining), generator=generator).tolist()
        selected.extend(
            remaining[index]
            for index in order[: number_of_pairs - len(selected)]
        )
    return torch.tensor(sorted(selected), dtype=torch.long)


def stable_unit_interval(*parts: object) -> float:
    """Map an identity tuple to a stable value in [0, 1)."""

    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(1 << 64)


def note_crop_bounds(
    *,
    onset_sample: int,
    offset_sample: int,
    total_samples: int,
    crop_samples: int,
    lead_samples: int,
    identity: object,
) -> tuple[int, int, int]:
    """Choose a fixed crop containing the complete queried score event."""

    onset = int(onset_sample)
    offset = int(offset_sample)
    total = int(total_samples)
    crop = min(int(crop_samples), total)
    if not 0 <= onset < offset <= total:
        raise ValueError("invalid note bounds")
    if offset - onset > crop:
        raise ValueError("queried note is longer than the crop")
    minimum = max(0, offset - crop)
    maximum = min(onset, total - crop)
    preferred = onset - int(lead_samples)
    if minimum > maximum:
        raise ValueError("no fixed crop can contain the queried note")
    preferred = max(minimum, min(preferred, maximum))
    slack = maximum - minimum
    if slack:
        # A small deterministic jitter prevents a crop-position shortcut while
        # retaining the requested leading context whenever possible.
        radius = min(slack, max(1, int(lead_samples) // 4))
        jitter = round((stable_unit_interval(identity) * 2.0 - 1.0) * radius)
        preferred = max(minimum, min(preferred + jitter, maximum))
    return preferred, onset - preferred, offset - preferred


def note_fits_timeline(
    *,
    onset_sec: float,
    offset_sec: float,
    duration_sec: float,
    crop_samples: int,
    sample_rate: int = 16_000,
) -> bool:
    """Return whether rounded score bounds fit the declared audio timeline."""

    total = max(int(crop_samples), int(round(float(duration_sec) * sample_rate)))
    onset = int(round(float(onset_sec) * sample_rate))
    offset = int(round(float(offset_sec) * sample_rate))
    return 0 <= onset < offset <= total


def score_absent_pitch(
    active_pitches: Iterable[int], *, target_pitch: int, identity: object
) -> int:
    """Select a reproducible nearby pitch absent from the scored crop."""

    occupied = {int(value) for value in active_pitches}
    candidates = [
        int(target_pitch) + delta
        for delta in (1, -1, 2, -2, 12, -12, 7, -7, 24, -24)
        if 0 <= int(target_pitch) + delta < 128
        and int(target_pitch) + delta not in occupied
    ]
    if not candidates:
        candidates = [pitch for pitch in range(128) if pitch not in occupied]
    if not candidates:
        raise ValueError("all MIDI pitches are occupied")
    index = min(
        len(candidates) - 1,
        int(math.floor(stable_unit_interval(identity, "absent") * len(candidates))),
    )
    return int(candidates[index])


def rt2_absent_bounds(onset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode an RT2 OFF query: equal onset/offset means zero held activity."""

    return onset, onset


def per_example_pre_onset_rms(
    estimate: torch.Tensor,
    onset: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    guard_samples: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RMS before each score onset and the available prefix length.

    A positive guard excludes onset-adjacent samples, where aligned-score and
    acoustic onset timing can differ slightly.  This is a score-only negative
    constraint: it never asserts an audio target after the offset, so note
    tails remain governed by remix removal/completeness rather than a heuristic
    truncation loss.
    """

    if estimate.ndim != 2:
        raise ValueError("estimate must have shape [B, T]")
    batch, samples = estimate.shape
    if onset.shape != (batch,) or valid_samples.shape != (batch,):
        raise ValueError("onset and valid_samples must have shape [B]")
    if guard_samples < 0:
        raise ValueError("guard_samples must be nonnegative")
    prefix = torch.minimum(
        (onset.to(dtype=torch.long) - int(guard_samples)).clamp_min(0),
        valid_samples.to(dtype=torch.long).clamp(min=0, max=samples),
    )
    sample_index = torch.arange(samples, device=estimate.device)[None, :]
    mask = sample_index < prefix[:, None].to(device=estimate.device)
    energy = (estimate.square() * mask.to(dtype=estimate.dtype)).sum(dim=-1)
    rms = torch.sqrt(energy / prefix.to(device=estimate.device, dtype=estimate.dtype).clamp_min(1))
    return rms, prefix


def self_remix(
    mixture: torch.Tensor,
    teacher_target: torch.Tensor,
    permutation: torch.Tensor,
    *,
    residual_scale: torch.Tensor,
    peak_limit: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Swap teacher residuals and apply one mixture-consistent gain per row.

    Returns ``(remix, scaled_target, common_gain)``.  The same gain is applied
    to both sources, preserving the exact identity
    ``remix = scaled_target + scaled_permuted_residual``.
    """

    if mixture.ndim != 2 or teacher_target.shape != mixture.shape:
        raise ValueError("mixture and teacher_target must have shape [B, T]")
    batch = mixture.shape[0]
    if permutation.shape != (batch,):
        raise ValueError("permutation must have shape [B]")
    if residual_scale.shape not in {(batch,), (batch, 1)}:
        raise ValueError("residual_scale must have shape [B] or [B, 1]")
    permutation = permutation.to(device=mixture.device, dtype=torch.long)
    if torch.any(permutation == torch.arange(batch, device=mixture.device)):
        raise ValueError("permutation is not a derangement")
    scale = residual_scale.to(device=mixture.device, dtype=mixture.dtype).reshape(batch, 1)
    teacher_residual = mixture - teacher_target
    remixed_residual = teacher_residual.index_select(0, permutation) * scale
    raw = teacher_target + remixed_residual
    peak = raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    gain = torch.minimum(torch.ones_like(peak), float(peak_limit) / peak)
    return raw * gain, teacher_target * gain, gain


def real_mixture_invariance_remix(
    mixture: torch.Tensor,
    teacher_target: torch.Tensor,
    permutation: torch.Tensor,
    *,
    interferer_scale: torch.Tensor,
    peak_limit: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a complete second real mixture and retain the first score query.

    Unlike residual self-remixing, this transformation does not decompose or
    filter either real recording. The frozen teacher target from the first
    view is only a stop-gradient invariance anchor.
    """

    if mixture.ndim != 2 or teacher_target.shape != mixture.shape:
        raise ValueError("mixture and teacher_target must have shape [B, T]")
    batch = mixture.shape[0]
    if permutation.shape != (batch,):
        raise ValueError("permutation must have shape [B]")
    if interferer_scale.shape not in {(batch,), (batch, 1)}:
        raise ValueError("interferer_scale must have shape [B] or [B, 1]")
    permutation = permutation.to(device=mixture.device, dtype=torch.long)
    if torch.any(permutation == torch.arange(batch, device=mixture.device)):
        raise ValueError("permutation is not a derangement")
    scale = interferer_scale.to(
        device=mixture.device, dtype=mixture.dtype
    ).reshape(batch, 1)
    raw = mixture + mixture.index_select(0, permutation) * scale
    peak = raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    gain = torch.minimum(torch.ones_like(peak), float(peak_limit) / peak)
    return raw * gain, teacher_target * gain, gain


def threeway_real_mixture_invariance_remix(
    mixture: torch.Tensor,
    teacher_target: torch.Tensor,
    partners: torch.Tensor,
    *,
    interferer_scales: torch.Tensor,
    anchor_rows: torch.Tensor | None = None,
    peak_limit: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add two complete real recordings while keeping the anchor query fixed.

    ``partners[i]`` identifies two score-collision-free recordings chosen by
    :func:`collision_free_triplet_indices`; ``interferer_scales[i]`` has their
    respective gains. The only pseudo-target is the anchor teacher output,
    scaled by the single common limiter applied to the complete three-way
    waveform. This deliberately introduces no estimated partner source.
    """

    if mixture.ndim != 2 or teacher_target.shape != mixture.shape:
        raise ValueError("mixture and teacher_target must have shape [B, T]")
    batch = mixture.shape[0]
    if anchor_rows is None:
        anchor_rows = torch.arange(batch, device=mixture.device)
    anchor_rows = anchor_rows.to(device=mixture.device, dtype=torch.long)
    anchors = int(anchor_rows.numel())
    if anchor_rows.ndim != 1 or anchors < 1:
        raise ValueError("anchor_rows must select at least one batch row")
    if torch.any((anchor_rows < 0) | (anchor_rows >= batch)):
        raise ValueError("anchor rows must be valid batch rows")
    if partners.shape != (anchors, 2):
        raise ValueError("partners must have shape [A, 2] for selected anchors")
    if interferer_scales.shape not in {(anchors, 2), (anchors, 2, 1)}:
        raise ValueError("interferer_scales must have shape [A, 2] or [A, 2, 1]")
    partners = partners.to(device=mixture.device, dtype=torch.long)
    if torch.any((partners < 0) | (partners >= batch)):
        raise ValueError("partner indices must be valid batch rows")
    if torch.any(partners[:, 0] == partners[:, 1]):
        raise ValueError("three-way partners must be distinct")
    if torch.any(partners == anchor_rows[:, None]):
        raise ValueError("a row cannot remix with itself")
    scale = interferer_scales.to(device=mixture.device, dtype=mixture.dtype).reshape(
        anchors, 2, 1
    )
    partner_audio = mixture.index_select(0, partners.reshape(-1)).reshape(
        anchors, 2, mixture.shape[-1]
    )
    raw = mixture.index_select(0, anchor_rows) + (partner_audio * scale).sum(dim=1)
    peak = raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    gain = torch.minimum(torch.ones_like(peak), float(peak_limit) / peak)
    return raw * gain, teacher_target.index_select(0, anchor_rows) * gain, gain


def symmetric_real_mixture_remix(
    mixture: torch.Tensor,
    teacher_target: torch.Tensor,
    partner: torch.Tensor,
    *,
    interferer_scale: torch.Tensor,
    peak_limit: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer two real crops and retain a pseudo-target for both score queries.

    The caller is responsible for score collision checks. For each row ``i``
    and partner ``j``, this returns the shared mixture

    ``g * (x_i + scale_i * x_j)``

    together with teacher anchors ``g*T(x_i,q_i)`` and
    ``g*scale_i*T(x_j,q_j)``. Both targets therefore use exactly the same gain
    and alignment as the mixed waveform.
    """

    if mixture.ndim != 2 or teacher_target.shape != mixture.shape:
        raise ValueError("mixture and teacher_target must have shape [B, T]")
    batch = mixture.shape[0]
    if partner.shape != (batch,):
        raise ValueError("partner must have shape [B]")
    if interferer_scale.shape not in {(batch,), (batch, 1)}:
        raise ValueError("interferer_scale must have shape [B] or [B, 1]")
    partner = partner.to(device=mixture.device, dtype=torch.long)
    if torch.any((partner < 0) | (partner >= batch)):
        raise ValueError("partner indices must be valid batch rows")
    if torch.any(partner == torch.arange(batch, device=mixture.device)):
        raise ValueError("a row cannot remix with itself")
    scale = interferer_scale.to(
        device=mixture.device, dtype=mixture.dtype
    ).reshape(batch, 1)
    raw = mixture + mixture.index_select(0, partner) * scale
    peak = raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    gain = torch.minimum(torch.ones_like(peak), float(peak_limit) / peak)
    first_target = teacher_target * gain
    second_target = teacher_target.index_select(0, partner) * scale * gain
    return raw * gain, first_target, second_target, gain


def masked_waveform_l1(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid_samples: torch.Tensor,
) -> torch.Tensor:
    """Mean per-example L1 over valid audio only."""

    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate and target must have shape [B, T]")
    support = torch.arange(estimate.shape[-1], device=estimate.device)[None] < (
        valid_samples.long()[:, None]
    )
    return per_example_masked_waveform_l1(
        estimate, target, valid_samples
    ).mean()


def per_example_masked_waveform_l1(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid_samples: torch.Tensor,
) -> torch.Tensor:
    """Per-example L1 over valid audio only."""

    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate and target must have shape [B, T]")
    support = torch.arange(estimate.shape[-1], device=estimate.device)[None] < (
        valid_samples.long()[:, None]
    )
    numerator = ((estimate.float() - target.float()).abs() * support).sum(dim=-1)
    return numerator / valid_samples.float().clamp_min(1)


def masked_rms(values: torch.Tensor, valid_samples: torch.Tensor) -> torch.Tensor:
    support = torch.arange(values.shape[-1], device=values.device)[None] < (
        valid_samples.long()[:, None]
    )
    energy = (values.float().square() * support).sum(dim=-1)
    return (energy / valid_samples.float().clamp_min(1)).sqrt()


def relative_rms_interferer_scale(
    anchor_rms: torch.Tensor,
    interferer_rms: torch.Tensor,
    desired_relative_rms: torch.Tensor,
    *,
    minimum_rms: float = 1e-4,
    maximum_scale: float = 16.0,
) -> torch.Tensor:
    """Convert a desired relative level into a recording-level-invariant gain."""

    if not (
        anchor_rms.shape == interferer_rms.shape == desired_relative_rms.shape
    ):
        raise ValueError("anchor, interferer, and desired RMS tensors must align")
    if minimum_rms <= 0.0 or maximum_scale <= 0.0:
        raise ValueError("RMS floor and maximum scale must be positive")
    scale = (
        desired_relative_rms
        * anchor_rms.float().clamp_min(minimum_rms)
        / interferer_rms.float().clamp_min(minimum_rms)
    )
    return scale.clamp(max=maximum_scale).to(desired_relative_rms.dtype)


def per_example_projection_gain(
    estimate: torch.Tensor,
    target: torch.Tensor,
    valid_samples: torch.Tensor,
) -> torch.Tensor:
    """Least-squares gain of each estimate along its stop-gradient target."""

    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate and target must have shape [B, T]")
    if valid_samples.shape != (estimate.shape[0],):
        raise ValueError("valid_samples must have shape [B]")
    support = torch.arange(estimate.shape[-1], device=estimate.device)[None] < (
        valid_samples.long()[:, None]
    )
    reference = target.float() * support
    numerator = (estimate.float() * reference).sum(dim=-1)
    denominator = reference.square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def harmonic_energy_concentration(
    values: torch.Tensor,
    pitch: torch.Tensor,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 2048,
    bandwidth_hz: float = 30.0,
    maximum_harmonics: int = 60,
) -> torch.Tensor:
    """Fraction of STFT power near integer harmonics of each query pitch."""

    if values.ndim != 2 or pitch.shape != (values.shape[0],):
        raise ValueError("values must be [B,T] and pitch must be [B]")
    window = torch.hann_window(n_fft, device=values.device, dtype=torch.float32)
    spectrum = torch.stft(
        values.float(),
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spectrum.abs().square().mean(dim=-1)
    frequencies = torch.fft.rfftfreq(
        n_fft, d=1.0 / float(sample_rate), device=values.device
    )
    fundamental = 440.0 * torch.pow(
        torch.tensor(2.0, device=values.device),
        (pitch.float() - 69.0) / 12.0,
    )
    harmonics = torch.arange(
        1, maximum_harmonics + 1, device=values.device, dtype=torch.float32
    )
    centers = fundamental[:, None] * harmonics[None]
    distance = (frequencies[None, None] - centers[:, :, None]).abs()
    mask = (distance <= float(bandwidth_hz)).any(dim=1)
    mask = mask & (frequencies[None] > 0)
    numerator = (power * mask).sum(dim=-1)
    return numerator / power.sum(dim=-1).clamp_min(1e-12)


def confidence_filter(
    mixture: torch.Tensor,
    teacher_target: torch.Tensor,
    second_view_target: torch.Tensor,
    pitch: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    maximum_relative_disagreement: float,
    minimum_harmonic_concentration: float,
    minimum_target_to_mixture_rms: float,
    maximum_target_to_mixture_rms: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select pseudo-targets that are stable, pitched, and energy-plausible."""

    target_rms = masked_rms(teacher_target, valid_samples).clamp_min(1e-5)
    mixture_rms = masked_rms(mixture, valid_samples).clamp_min(1e-5)
    disagreement = masked_rms(
        second_view_target - teacher_target, valid_samples
    ) / target_rms
    concentration = harmonic_energy_concentration(teacher_target, pitch)
    energy_ratio = target_rms / mixture_rms
    accepted = (
        (disagreement <= float(maximum_relative_disagreement))
        & (concentration >= float(minimum_harmonic_concentration))
        & (energy_ratio >= float(minimum_target_to_mixture_rms))
        & (energy_ratio <= float(maximum_target_to_mixture_rms))
    )
    return accepted, {
        "relative_disagreement": disagreement,
        "harmonic_concentration": concentration,
        "target_to_mixture_rms": energy_ratio,
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduce per-example losses, requiring at least one positive weight."""

    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must have equal [B] shapes")
    denominator = weights.float().sum()
    if not bool(denominator > 0):
        raise ValueError("weighted_mean requires a positive weight")
    return (values.float() * weights.float()).sum() / denominator


def trainable_parameter_names(
    model: torch.nn.Module, scope: str
) -> tuple[list[str], int]:
    """Freeze the model and expose the deliberately narrow adaptation scope."""

    scopes = {
        "heads": (".target_head.", ".background_head.", ".ownership_head."),
        "ownership": (".ownership_head.",),
        "heads_and_film": (
            ".target_head.",
            ".background_head.",
            ".ownership_head.",
            ".encoder_film.",
            ".decoder_film.",
            ".bottleneck_film.",
        ),
    }
    if scope == "full":
        names = []
        count = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(True)
            names.append(name)
            count += parameter.numel()
        if not names:
            raise ValueError("model has no trainable parameters")
        return names, int(count)
    if scope not in scopes:
        raise ValueError(f"unknown trainable scope: {scope}")
    needles = scopes[scope]
    names: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(needle in f".{name}." for needle in needles))
        if parameter.requires_grad:
            names.append(name)
            count += parameter.numel()
    if not names:
        raise ValueError(f"trainable scope {scope!r} selected no parameters")
    return names, int(count)
