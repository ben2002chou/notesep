import zipfile
import numpy as np

from notesep import NoteBank
from notesep.project_archive import write_project_archive


def test_download_archive_reloads_exact_audio_and_keeps_input(tmp_path):
    (tmp_path / 'input.wav').write_bytes(b'original recording')
    stem = np.zeros((1, 32000), np.float32)
    stem[0, -1] = 1.25  # The export must preserve levels above PCM clipping.
    bank = NoteBank([{'id': 'x', 'pitch': 60, 'start': .2, 'end': .5}], stem,
                    raw_stems=stem.copy())
    path = write_project_archive(tmp_path, '0123456789ab', bank)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert archive.read('input.wav') == b'original recording'
        archive.extractall(tmp_path / 'downloaded')
    loaded = NoteBank.load(tmp_path / 'downloaded/project.json')
    np.testing.assert_array_equal(loaded.render(), bank.render())
    assert not (tmp_path / 'project-0123456789ab').exists()


def test_full_recording_window_keeps_daw_placement_and_origin(tmp_path):
    stem=np.zeros((1,32000),np.float32)
    bank=NoteBank([{'id':'x','pitch':60,'start':.2,'end':.5}],stem,
                  provenance={'window_start_seconds':14.25,'source_duration_seconds':22,
                              'source_recording_sha256':'original-source-hash'})
    path=write_project_archive(tmp_path,'123456789abc',bank)
    with zipfile.ZipFile(path) as archive:
        assert '14.250000 seconds' in archive.read('IMPORT.txt').decode()
        archive.extractall(tmp_path/'download')
    loaded=NoteBank.load(tmp_path/'download/project.json')
    assert loaded.provenance==bank.provenance
