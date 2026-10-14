export function scoreValues(pitch,start,end,duration){
  if(!Number.isInteger(pitch)||pitch<21||pitch>108)throw new Error('Use a MIDI pitch from 21 to 108.');
  if(![start,end,duration].every(Number.isFinite)||start<0||end>duration||end-start<.04-1e-9)
    throw new Error('Keep the note inside the window and at least 40 ms long.');
  return {pitch,start,end};
}

export function correctNote(note,values){
  return {...note,...values,pitchShift:0,delta:0,stretch:1,gainDb:0};
}
