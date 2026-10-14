export function audioEditValues(pitchShift,timingMs,stretch,gainDb){
  if(![pitchShift,timingMs,stretch,gainDb].every(Number.isFinite))throw new Error('Enter finite numbers for each audio edit.');
  if(!Number.isInteger(pitchShift)||pitchShift < -12||pitchShift > 12)throw new Error('Use a pitch shift from −12 to +12 semitones.');
  if(Math.abs(timingMs)>1000)throw new Error('Use a timing shift from −1000 to +1000 ms.');
  if(stretch<.5||stretch>2)throw new Error('Use a length factor from 0.5 to 2.');
  if(gainDb < -24||gainDb > 12)throw new Error('Use a volume change from −24 to +12 dB.');
  return {pitchShift,delta:timingMs/1000,stretch,gainDb};
}
