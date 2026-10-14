// Generated Studio audio is PCM16 or float32 WAV. Decode those samples directly
// so spectrogram preparation does not depend on the browser media decoder.
export function readWav(bytes) {
  const view=new DataView(bytes);
  const word=offset=>String.fromCharCode(...new Uint8Array(bytes,offset,4));
  if(view.byteLength<44||word(0)!=='RIFF'||word(8)!=='WAVE')throw new Error('Expected a WAV recording');
  let format=null,data=null;
  for(let offset=12;offset+8<=view.byteLength;){
    const size=view.getUint32(offset+4,true),start=offset+8;
    if(start+size>view.byteLength)throw new Error('Truncated WAV chunk');
    const kind=word(offset);
    if(kind==='fmt '){
      if(size<16)throw new Error('Invalid WAV format');
      format={kind:view.getUint16(start,true),channels:view.getUint16(start+2,true),
        sampleRate:view.getUint32(start+4,true),align:view.getUint16(start+12,true),
        bits:view.getUint16(start+14,true)};
    }
    if(kind==='data')data={start,size};
    offset=start+size+(size%2);
  }
  if(!format||!data||![1,3].includes(format.kind)||format.channels<1||format.channels>8
     ||format.sampleRate<8000||format.sampleRate>192000
     ||!(format.kind===1&&format.bits===16||format.kind===3&&format.bits===32)
     ||format.align!==format.channels*(format.bits/8)||data.size%format.align)
    throw new Error('Unsupported WAV geometry');
  const frames=data.size/format.align;
  const channels=Array.from({length:format.channels},()=>new Float32Array(frames));
  for(let frame=0;frame<frames;frame++)for(let channel=0;channel<format.channels;channel++){
    const offset=data.start+frame*format.align+channel*format.bits/8;
    const sample=format.kind===3?view.getFloat32(offset,true):view.getInt16(offset,true)/32768;
    if(!Number.isFinite(sample))throw new Error('Non-finite WAV audio');
    channels[channel][frame]=sample;
  }
  return {sampleRate:format.sampleRate,channels,frames};
}

export function wavBuffer(bytes,context){
  const data=readWav(bytes),buffer=context.createBuffer(data.channels.length,data.frames,data.sampleRate);
  data.channels.forEach((samples,index)=>buffer.copyToChannel(samples,index));
  return buffer;
}
