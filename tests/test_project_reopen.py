from pathlib import Path
import io,json,stat,zipfile
import numpy as np
import soundfile as sf
import pytest
from notesep import NoteBank
from notesep.project_archive import write_project_archive,read_project_archive


def project(tmp_path,with_input=True):
    folder=tmp_path/'source';folder.mkdir()
    stems=np.zeros((2,32000),dtype=np.float32);stems[0,200:400]=1.25;stems[1,500:700]=-.3
    bank=NoteBank([{'id':'one','pitch':60,'start':.01,'end':.2},{'id':'two','pitch':64,'start':.03,'end':.2}],stems,
        raw_stems=stems*.9,provenance={'window_start_seconds':14,'source_duration_seconds':22,'apply_aso':True})
    if with_input:sf.write(folder/'input.wav',stems.sum(0),16000,subtype='FLOAT')
    return write_project_archive(folder,'0123456789ab',bank).read_bytes(),bank


def rewrite(payload,transform):
    result=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as source,zipfile.ZipFile(result,'w') as dest:
        for info in source.infolist():
            body=source.read(info);info,body=transform(info,body);dest.writestr(info,body)
    return result.getvalue()


def test_reopen_keeps_full_stems_raw_input_and_daw_origin(tmp_path):
    payload,original=project(tmp_path);loaded,edits,has_input=read_project_archive(payload,tmp_path/'opened')
    assert has_input and edits=={} and loaded.provenance==original.provenance
    np.testing.assert_array_equal(loaded.stems,original.stems);np.testing.assert_array_equal(loaded.raw_stems,original.raw_stems)
    np.testing.assert_array_equal(sf.read(tmp_path/'opened/input.wav',dtype='float32')[0],original.stems.sum(0))


def test_python_project_without_original_can_reopen_stems(tmp_path):
    payload,original=project(tmp_path,False);loaded,_,has_input=read_project_archive(payload,tmp_path/'opened')
    assert not has_input;np.testing.assert_array_equal(loaded.stems,original.stems)


@pytest.mark.parametrize('corruption',['traversal','symlink','input','stem','ids','geometry','extra'])
def test_corrupt_or_unsafe_project_rejected(tmp_path,corruption):
    payload,_=project(tmp_path)
    def transform(info,body):
        if corruption=='traversal' and info.filename=='input.wav':info.filename='../outside.wav'
        if corruption=='symlink' and info.filename=='input.wav':info.external_attr=(stat.S_IFLNK|0o777)<<16
        if corruption in ['input','stem'] and info.filename==('input.wav' if corruption=='input' else 'note-0000-midi-60.wav'):body=body[:-1]+bytes([body[-1]^1])
        if corruption in ['ids','geometry'] and info.filename=='project.json':
            data=json.loads(body)
            if corruption=='ids':data['files'].reverse()
            else:data['samples']=500000
            body=json.dumps(data).encode()
        if corruption=='extra' and info.filename=='IMPORT.txt':info.filename='unexpected.html'
        return info,body
    with pytest.raises(ValueError):read_project_archive(rewrite(payload,transform),tmp_path/'opened')
    assert not (tmp_path/'outside.wav').exists()


def test_duplicate_members_rejected(tmp_path):
    payload,_=project(tmp_path);buffer=io.BytesIO(payload)
    with zipfile.ZipFile(buffer,'a') as archive:
        with pytest.warns(UserWarning):archive.writestr('input.wav',b'duplicate')
    with pytest.raises(ValueError,match='duplicate'):read_project_archive(buffer.getvalue(),tmp_path/'opened')
