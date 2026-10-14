import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {correctNote,scoreTimeValue,scoreValues} from '../src/notesep/studio_static/score-edit.mjs';

for (const [name,start,end] of [
  ['one audio sample',1+1/16000,1+2/16000],
  ['five milliseconds',.123456,.128456],
  ['twenty-five milliseconds',1.123456,1.148456],
]) {
  test(`a pitch-only correction preserves ${name} timing`,()=>{
    const note={id:'original',pitch:60,start,end};
    const values=scoreValues(61,Number(scoreTimeValue(start)),Number(scoreTimeValue(end)),2);
    const corrected=correctNote(note,values);
    assert.equal(corrected.id,note.id);
    assert.equal(corrected.pitch,61);
    assert.equal(corrected.start,start);
    assert.equal(corrected.end,end);
  });
}

test('the editor uses lossless values for both score time fields',()=>{
  const source=readFileSync(new URL('../src/notesep/studio_static/note-editor.js',import.meta.url),'utf8');
  assert.match(source,/scoreFields\.start\.value=n\?scoreTimeValue\(visualStart\(n\)\)/);
  assert.match(source,/scoreFields\.end\.value=n\?scoreTimeValue\(visualEnd\(n\)\)/);
});

test('zero-duration and sub-sample notes remain invalid',()=>{
  assert.throws(()=>scoreValues(60,.1,.1,2));
  assert.throws(()=>scoreValues(60,.1,.10001,2));
});
