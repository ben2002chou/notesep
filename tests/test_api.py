"""Protect audible edits, tail preservation, and portable project round trips."""
import numpy as np
import pytest

from notesep import NoteBank


def bank():
    # Deliberately retain energy outside score bounds; no-edit playback must
    # preserve it, while muting one note must remove only that note's waveform.
    waves = np.zeros((2, 32000), dtype=np.float32)
    waves[0, 0] = .02
    waves[0, 31999] = .1
    waves[1, 8000:9000] = .2
    notes = [{'id': 'a', 'pitch': 60, 'start': .1, 'end': .5},
             {'id': 'b', 'pitch': 64, 'start': .5, 'end': .8}]
    return NoteBank(notes, waves, raw_stems=waves.copy())


def test_direct_sum_preserves_full_aligned_tails_and_mutes_only_selected_note():
    value = bank()
    np.testing.assert_array_equal(value.render(), value.stems.sum(axis=0))
    np.testing.assert_array_equal(value.render({'a': {'mute': True}}), value.stems[1])
    np.testing.assert_allclose(value.render({'b': {'gainDb': -6.020599913}}),
                               value.stems[0] + value.stems[1] * .5, rtol=1e-6)


def test_project_moves_without_original_audio_or_model_and_detects_tampering(tmp_path):
    import shutil
    original = bank()
    path = original.export(tmp_path / 'before')
    shutil.move(path.parent, tmp_path / 'after')
    restored = NoteBank.load(tmp_path / 'after/project.json')
    np.testing.assert_array_equal(restored.render(), original.render())
    audio = next((tmp_path / 'after').glob('note-*.wav'))
    with audio.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        NoteBank.load(tmp_path / 'after/project.json')


def test_delayed_note_extends_timeline_and_negative_shift_clips_at_zero():
    value = bank()
    shifted = value.render({'b': {'delta': 2}})
    assert len(shifted) > value.stems.shape[1]
    assert np.count_nonzero(shifted[40000:41000]) == 1000
    negative = value.render({'b': {'delta': -.75}})
    np.testing.assert_array_equal(negative, value.stems[0])


@pytest.mark.parametrize('edits', [
    {'unknown': {'mute': True}}, {'a': {'pitchShift': float('nan')}},
    {'a': {'stretch': 0}}, {'a': {'mute': 'yes'}}, {'a': {'gainDb': 100}},
])
def test_bad_edits_are_rejected(edits):
    with pytest.raises(ValueError):
        bank().render(edits)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match='unique'):
        NoteBank([{'id': 'same'}, {'id': 'same'}], np.zeros((2, 100), np.float32))
