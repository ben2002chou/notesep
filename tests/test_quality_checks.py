import numpy as np
import pytest

from notesep.quality_checks import separated_coverage


RATE = 16000


def test_unclaimed_audible_lead_in_is_flagged():
    time = np.arange(4 * RATE) / RATE
    source = 0.02 * np.sin(2 * np.pi * 440 * time).astype(np.float32)
    bank = np.stack([source / 2, source / 2])
    bank[:, :RATE] = 0
    report = separated_coverage(source, bank)
    assert report["uncoveredRegions"] == [{"start": 0.0, "end": 1.0}]
    assert 24 < report["errorEnergyPct"] < 26


def test_quiet_gap_and_small_reconstruction_error_are_not_flagged():
    time = np.arange(4 * RATE) / RATE
    source = (0.02 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    source[:RATE] = 0
    bank = np.stack([source * 0.48, source * 0.48])
    report = separated_coverage(source, bank)
    assert report["uncoveredRegions"] == []
    assert report["errorEnergyPct"] < 1


def test_short_error_burst_does_not_raise_note_warning():
    source = np.full(2 * RATE, 0.01, np.float32)
    bank = np.stack([source / 2, source / 2])
    bank[:, :RATE // 10] = 0
    assert separated_coverage(source, bank)["uncoveredRegions"] == []


def test_rejects_mismatched_or_nonfinite_banks():
    with pytest.raises(ValueError):
        separated_coverage(np.zeros(10), np.zeros((2, 9)))
    with pytest.raises(ValueError):
        separated_coverage(np.zeros(10), np.full((2, 10), np.nan))
