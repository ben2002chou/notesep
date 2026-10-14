import json
import numpy as np
import soundfile as sf
from notesep import NoteBank
from notesep.cli import main


def test_saved_controls_round_trip_without_changing_base_estimates(tmp_path):
    stems=np.ones((1,32000),np.float32)*1.25
    controls={'note':{'gainDb':-6,'mute':False}}
    bank=NoteBank([{'id':'note','pitch':60,'start':.1,'end':.4}],stems,saved_edits=controls)
    project=bank.export(tmp_path/'project');loaded=NoteBank.load(project)
    assert loaded.saved_edits==controls
    np.testing.assert_array_equal(loaded.stems,stems)
    np.testing.assert_array_equal(loaded.render(),bank.render())
    output=tmp_path/'edited.wav';main(['render',str(project),'--output',str(output)])
    np.testing.assert_array_equal(sf.read(output,dtype='float32')[0],loaded.render(controls))


def test_explicit_cli_controls_override_saved_controls(tmp_path):
    stems=np.ones((1,32000),np.float32)
    bank=NoteBank([{'id':'note','pitch':60,'start':.1,'end':.4}],stems,saved_edits={'note':{'mute':True}})
    project=bank.export(tmp_path/'project');edits=tmp_path/'edits.json';edits.write_text(json.dumps({'note':{'gainDb':0}}))
    output=tmp_path/'edited.wav';main(['render',str(project),'--edits',str(edits),'--output',str(output)])
    np.testing.assert_array_equal(sf.read(output,dtype='float32')[0],stems[0])
