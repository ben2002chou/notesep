export function scoreValues(pitch,start,end,duration){
  if(!Number.isInteger(pitch)||pitch<21||pitch>108)throw new Error('Use a MIDI pitch from 21 to 108.');
  if(![start,end,duration].every(Number.isFinite)||start<0||end>duration||
      Math.round(end*16000)<=Math.round(start*16000))
    throw new Error('Keep the note inside the window with a positive duration at 16 kHz.');
  return {pitch,start,end};
}

export function correctNote(note,values){
  return {...note,...values,pitchShift:0,delta:0,stretch:1,gainDb:0};
}
