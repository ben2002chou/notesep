import {transformNote} from './vendor/note-audio-transform.mjs';
import {floatWav, saveBlob} from './audio-export.mjs';
import {wavBuffer} from './wav-decode.mjs';
import {scoreValues,correctNote} from './score-edit.mjs';
import {audioEditValues} from './audio-edit-values.mjs';

const $=selector=>document.querySelector(selector);
const els={piece:$('#piece'),play:$('#play'),solo:$('#solo'),compare:$('#compare'),reset:$('#reset'),grid:$('#roll-grid'),shell:$('#roll-shell'),viewport:$('#roll-viewport'),drawMode:$('#draw-mode'),drawHint:$('#draw-hint'),scoreToolRow:$('#score-tool-row'),intro:$('.editor-intro'),footHint:$('.editor-foot p:last-child'),spectrogram:$('#spectrogram'),spectrogramStatus:$('#spectrogram-status'),spectrogramViewport:$('#spectrogram-viewport'),spectrogramContent:$('#spectrogram-content'),spectrogramNotes:$('#spectrogram-notes'),spectrogramPlayhead:$('#spectrogram-playhead'),frequencyLabels:$('#spectrogram-frequency-labels'),labels:$('#pitch-labels'),ruler:$('#ruler'),playhead:$('#roll-playhead'),clock:$('#clock'),title:$('#selected-title'),detail:$('#selected-detail'),status:$('#status'),mode:$('#playback-label')};
const inputs={pitch:$('#pitch'),timing:$('#timing'),length:$('#length'),volume:$('#volume')};
const scoreFields={row:$('#score-note-fields'),pitch:$('#score-midi-pitch'),start:$('#score-start'),end:$('#score-end'),apply:$('#apply-score-note'),add:$('#add-score-note')};
const audioFields={row:$('#audio-note-fields'),pitch:$('#audio-pitch-shift'),timing:$('#audio-timing-shift'),length:$('#audio-length-factor'),gain:$('#audio-gain-db'),apply:$('#apply-audio-edit')};
const muteControl=$('#audio-mute');
const values={pitch:$('#pitch-value'),timing:$('#timing-value'),length:$('#length-value'),volume:$('#volume-value')};
const notes=['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B'];
const color=['#92c8fa','#7facfd','#9d9df0','#be9cec','#dd98ca','#f09b9a','#f3bd80','#e6d776','#aad986','#7dcfbc','#75c7d7','#87b8f4'];
const SPEC_HEIGHT=220,MIN_FREQUENCY=27.5,MAX_FREQUENCY=8000;
const model={manifest:null,example:null,notes:[],selected:null,ctx:null,decodeCtx:null,inputBuffer:null,spectrogramData:null,spectrogramImage:null,spectrogramPromise:null,spectrogramToken:0,buffers:new Map(),transformed:new Map(),loadToken:0,playbackToken:0,playing:false,solo:false,soloEnd:0,original:false,playStart:0,seek:0,nodes:[],timer:0,audition:null,rowHeight:8,pitchMin:48,pitchMax:84,width:900,suppressClickUntil:0,drawMode:false};
const clamp=(v,min,max)=>Math.min(max,Math.max(min,v));
const pitchName=p=>`${notes[p%12]}${Math.floor(p/12)-1}`;
const qualityLabels={weak_pitch_evidence:'weak pitch evidence',displaced_attack:'possible onset mismatch',same_pitch_overlap:'overlapping same-pitch notes',weak_target_pitch:'low target-pitch clarity',near_silent_output:'very quiet estimate',other_score_pitch_dominates:'another concurrent score pitch dominates this estimate',model_disagreement:'not confirmed by the second transcription model'};
const qualityText=flags=>(flags||[]).map(flag=>qualityLabels[flag]||flag).join(', ');
const gainFromDb=db=>10**(db/20);
const changed=n=>n.pitchShift!==0||n.delta!==0||n.stretch!==1||n.gainDb!==0||n.mute===true;
const formatTime=t=>t.toFixed(2);
const methodName=()=>model.example?.method||'NoteSep';
const assetUrl=p=>new URL(p,import.meta.url).href;
function status(message){els.status.textContent=message;}
function emit(kind,details={}){window.parent?.postMessage({type:'notesep-editor-'+kind,...details},location.origin);}
function emitScore(){if(model.example?.scoreOnly)window.parent?.postMessage({type:'notesep-editor-score-change',notes:model.notes.map(n=>({id:n.id,pitch:n.pitch+n.pitchShift,start:visualStart(n),end:visualEnd(n)}))},location.origin);}
function emitOutputEdit(noteIds){if(model.example&&!model.example.scoreOnly&&noteIds.length)emit('audio-edit',{noteIds:[...new Set(noteIds.map(String))]});}
function setDrawMode(enabled){
  model.drawMode=Boolean(enabled&&model.example?.scoreOnly);
  els.drawMode.setAttribute('aria-pressed',String(model.drawMode));
  els.viewport.classList.toggle('draw-mode',model.drawMode);
  els.drawHint.textContent=model.drawMode?'Click for a short note; drag right for a longer one. Click an existing note to erase it. Press B or Escape to leave Draw Mode.':'Turn on Draw to add notes. With Draw off, drag a note to move it or its right edge to resize it.';
  window.parent?.postMessage({type:'notesep-editor-draw-mode',enabled:model.drawMode},location.origin);
  if(model.drawMode)status('Draw Mode on. Click or drag on an empty piano-roll row to add a note.');
  return model.drawMode;
}
if('ResizeObserver' in window)new ResizeObserver(()=>{
  window.parent?.postMessage({type:'notesep-editor-height',height:Math.ceil($('#editor').getBoundingClientRect().height)},location.origin);
}).observe($('#editor'));

function noteState(n){return {id:String(n.id??n.event),pitch:n.pitch,start:n.start,end:n.end,path:n.outputs?.aso?.audio,qualityFlags:Array.isArray(n.qualityFlags)?n.qualityFlags:[],delta:0,pitchShift:0,stretch:1,gainDb:0,mute:false};}
function visualStart(n){return n.start+n.delta;}
function visualEnd(n){return visualStart(n)+(n.end-n.start)*n.stretch;}
function notesChanged(){return model.notes.some(changed);}
function timelineDuration(){
  const duration=model.example?.duration||0;
  return Math.max(duration,...model.notes.filter(n=>n.pitchShift!==0||n.delta!==0||n.stretch!==1).map(n=>{
    const left=Math.max(0,n.start-.10),right=Math.min(duration,n.end+.45);
    return left+n.delta+(right-left)*n.stretch;
  }));
}

async function audioContext(){
  if(!model.ctx)model.ctx=new (window.AudioContext||window.webkitAudioContext)();
  if(model.ctx.state!=='running')await model.ctx.resume();
  return model.ctx;
}
function decodeContext(){
  if(!model.decodeCtx)model.decodeCtx=new OfflineAudioContext(1,1,16000);
  return model.decodeCtx;
}
async function loadBuffer(note){
  if(!note.path)throw new Error(`No ${methodName()} estimate for ${pitchName(note.pitch)}.`);
  if(!model.buffers.has(note.path)){
    model.buffers.set(note.path,fetch(assetUrl(note.path)).then(async response=>{
      if(!response.ok)throw new Error(`Audio request failed (${response.status}).`);
      return wavBuffer(await response.arrayBuffer(),decodeContext());
    }).catch(error=>{model.buffers.delete(note.path);throw error;}));
  }
  return model.buffers.get(note.path);
}
async function preload(){
  const token=++model.loadToken;
  if(model.example?.scoreOnly){els.play.disabled=true;els.solo.disabled=true;els.compare.disabled=true;status('Adjust the score, then run NoteGrab for one note or NoteSep for several notes.');return false;}
  els.compare.disabled=false;
  status(`Loading the ${methodName()} estimates…`);
  const results=await Promise.allSettled(model.notes.map(loadBuffer));
  if(token!==model.loadToken)return;
  results.forEach((result,i)=>{if(result.status==='fulfilled')model.notes[i].waveform=waveformPeaks(model.notes[i],result.value);});
  renderNotes();
  const failures=results.filter(x=>x.status==='rejected');
  els.play.disabled=failures.length>0;
  els.solo.disabled=!model.selected||failures.length>0;
  status(failures.length?`${failures.length} note files could not load. Check the connection and retry by changing recordings.`:`${model.notes.length} separated notes ready. Press Play edit or choose a note.`);
  return failures.length===0;
}

function stop(keepTime=true){
  model.playbackToken++;
  if(model.playing&&keepTime)model.seek=currentTime();
  model.playing=false;
  cancelAnimationFrame(model.timer);
  for(const node of model.nodes){
    try{if(node.stop)node.stop();}catch{}
    try{node.disconnect();}catch{}
  }
  model.nodes=[];
  model.audition=null;
  model.previewOriginal=false;
  els.play.querySelector('span').textContent=model.example?.scoreOnly?'Preview note':'Play edit';
  renderClock();
  emit('stop');
}
function currentTime(){return model.playing?clamp(model.seek+model.ctx.currentTime-model.playStart,0,timelineDuration()):model.seek;}
function renderClock(){
  const t=currentTime(),duration=timelineDuration();
  els.clock.textContent=`${formatTime(t)} / ${formatTime(duration)} s`;
  els.playhead.style.left=`${t/duration*model.width}px`;
  els.spectrogramPlayhead.style.left=`${t/duration*model.width}px`;
  if(model.playing){
    if(t>=(model.solo?model.soloEnd:duration)-.015){
      const completed=model.audition?.qualifying?model.audition:null;
      stop(false);model.seek=model.solo&&model.selected?clamp(visualStart(model.selected)-.08,0,duration):0;
      if(completed)emit('audition-complete',{mode:completed.mode,noteId:completed.noteId});
      renderClock();return;
    }
    model.timer=requestAnimationFrame(renderClock);
  }
}

function cropBounds(note,buffer){
  // A small pre-onset and release margin belongs to the separated note. No
  // mixture subtraction, residual, or synthesized replacement is involved.
  return {start:clamp(note.start-.10,0,buffer.duration),end:clamp(note.end+.45,0,buffer.duration)};
}
function soloMonitorGain(note,buffer){
  // A fixed listening boost based on the unedited estimate preserves the
  // user's volume edits. It never changes the direct-sum playback or WAVs.
  const samples=buffer.getChannelData(0),rate=buffer.sampleRate;
  const first=clamp(Math.floor((note.start-.05)*rate),0,samples.length);
  const last=clamp(Math.ceil((note.end+.2)*rate),first+1,samples.length);
  let peak=0,energy=0;
  for(let i=first;i<last;i++){const v=samples[i];peak=Math.max(peak,Math.abs(v));energy+=v*v;}
  const rms=Math.sqrt(energy/(last-first));
  return Math.max(1,Math.min(.08/Math.max(rms,1e-8),.8/Math.max(peak,1e-8),32));
}
async function scheduleNote(note,buffer,ctx,startAt,seek,original,solo,nodes=model.nodes){
  const edit=original?{...note,delta:0,pitchShift:0,stretch:1,gainDb:0,mute:false}:note;
  if(edit.mute&&!solo)return;
  const noteGain=ctx.createGain();noteGain.gain.value=gainFromDb(edit.gainDb)*(solo?soloMonitorGain(note,buffer):1);noteGain.connect(ctx.destination);nodes.push(noteGain);
  if(edit.pitchShift===0&&edit.delta===0&&edit.stretch===1){
    // The stored WAV is already aligned to the complete excerpt. Native sources
    // preserve its attacks and tails and are summed directly in the destination.
    const source=ctx.createBufferSource();source.buffer=buffer;source.connect(noteGain);
    source.start(startAt,seek);nodes.push(source);return;
  }
  const bounds=cropBounds(note,buffer);
  const outputStart=bounds.start+edit.delta;
  const outputEnd=outputStart+(bounds.end-bounds.start)*edit.stretch;
  if(outputEnd<=seek)return;
  const visibleStart=Math.max(outputStart,seek);
  const cropStart=Math.floor(bounds.start*buffer.sampleRate),cropEnd=Math.ceil(bounds.end*buffer.sampleRate);
  const key=`${note.path}:${cropStart}:${cropEnd}:${edit.pitchShift}:${edit.stretch}`;
  if(!model.transformed.has(key)){
    const sourceSamples=buffer.getChannelData(0).slice(cropStart,cropEnd);
    const samples=transformNote(sourceSamples,buffer.sampleRate,edit.pitchShift,edit.stretch);
    const prepared=ctx.createBuffer(1,samples.length,buffer.sampleRate);
    prepared.copyToChannel(samples,0);
    model.transformed.set(key,prepared);
  }
  const source=ctx.createBufferSource();source.buffer=model.transformed.get(key);source.connect(noteGain);
  source.start(startAt+visibleStart-seek,visibleStart-outputStart);
  nodes.push(source);
}
async function start(solo=false,fromSeek=false){
  if(!model.example||!model.notes.length)return;
  if(model.example.scoreOnly){await previewOriginalNote(model.selected);return;}
  if(solo&&!model.selected)return;
  stop(false);
  const token=++model.playbackToken;
  emit('play');
  try{
    const ctx=await audioContext();
    const selected=solo?[model.selected]:model.notes;
    if(solo&&!fromSeek)model.seek=clamp(visualStart(model.selected)-0.08,0,timelineDuration());
    if(model.seek>=timelineDuration()-.02)model.seek=0;
    const buffers=await Promise.all(selected.map(loadBuffer));
    if(token!==model.playbackToken)return;
    const startAt=ctx.currentTime+.12;
    const jobs=selected.map((n,i)=>scheduleNote(n,buffers[i],ctx,startAt,model.seek,model.original&&!solo,solo));
    await Promise.all(jobs);
    if(token!==model.playbackToken)return;
    model.playStart=startAt;model.playing=true;model.solo=solo;
    model.audition={qualifying:!fromSeek&&(solo||model.seek<=0.02),mode:solo?'note':'sum',noteId:solo?String(model.selected.id):null};
    model.soloEnd=solo?clamp(visualEnd(model.selected)+.45,0,timelineDuration()):timelineDuration();
    els.play.querySelector('span').textContent='Pause';
    els.mode.textContent=solo?`${pitchName(model.selected.pitch+model.selected.pitchShift)} · isolated note`:(model.original?`${methodName()} · original sum`:`${methodName()} · edited sum`);
    renderClock();
    status(solo?'Playing the selected separated note at solo listening level.':'Playing the direct sum of the separated notes.');
  }catch(error){stop(false);status(`Playback unavailable: ${error.message}`);console.error(error);}
}
async function previewOriginalNote(note){
  if(!note||!model.inputBuffer)return;
  stop(false);
  const token=++model.playbackToken;
  emit('play');
  try{
    const ctx=await audioContext();
    if(token!==model.playbackToken)return;
    const start=Math.max(0,visualStart(note)-.12);
    const end=Math.min(model.inputBuffer.duration,visualEnd(note)+.2);
    if(end-start<.02)return;
    model.seek=start;model.playStart=ctx.currentTime+.06;model.soloEnd=end;model.solo=true;model.playing=true;model.previewOriginal=true;
    const source=ctx.createBufferSource();source.buffer=model.inputBuffer;source.connect(ctx.destination);
    source.start(model.playStart,start,end-start);model.nodes.push(source);
    els.mode.textContent=`Original mixture · ${pitchName(note.pitch+note.pitchShift)} preview`;
    els.play.querySelector('span').textContent='Pause';
    status(`Auditioning the original mixture around ${pitchName(note.pitch+note.pitchShift)}.`);
    renderClock();
  }catch(error){stop(false);status(`Original preview unavailable: ${error.message}`);console.error(error);}
}
function seekTo(time){
  const wasPlaying=model.playing,solo=model.solo;
  stop(false);model.seek=clamp(time,0,timelineDuration());renderClock();
  if(wasPlaying)start(solo,true);
}
function updateControls(){
  const n=model.selected;
  els.play.querySelector('span').textContent=model.playing?'Pause':model.example?.scoreOnly?'Preview note':'Play edit';
  audioFields.row.hidden=!model.example||Boolean(model.example.scoreOnly);
  audioFields.apply.disabled=!n||audioFields.row.hidden;
  for(const field of [audioFields.pitch,audioFields.timing,audioFields.length,audioFields.gain])field.disabled=!n;
  audioFields.pitch.value=n?n.pitchShift:0;
  audioFields.timing.value=n?n.delta*1000:0;
  audioFields.length.value=n?n.stretch:1;
  audioFields.gain.value=n?n.gainDb:0;
  muteControl.disabled=!n||audioFields.row.hidden;muteControl.checked=n?.mute===true;
  scoreFields.row.hidden=!model.example?.scoreOnly;
  scoreFields.apply.disabled=!n;
  scoreFields.add.disabled=!model.example?.scoreOnly;
  scoreFields.pitch.value=n?n.pitch+n.pitchShift:60;
  scoreFields.start.value=n?visualStart(n).toFixed(4):0;
  scoreFields.end.value=n?visualEnd(n).toFixed(4):Math.min(.25,model.example?.duration||.25);
  if(model.example?.scoreOnly){els.play.disabled=!n;els.play.querySelector('span').textContent=model.playing?'Pause':'Preview note';els.solo.disabled=true;}
  els.title.textContent=n?`${pitchName(n.pitch+n.pitchShift)} · ${formatTime(visualStart(n))} s`:'Choose a note above';
  els.detail.textContent=model.example?.scoreOnly?(n?`Score query ${pitchName(n.pitch)}, ${formatTime(n.start)}–${formatTime(n.end)} s${n.qualityFlags?.length?` · review: ${qualityText(n.qualityFlags)}`:''}`:'Add or transcribe notes to make an aligned score.'):(n?`Original ${pitchName(n.pitch)}, ${formatTime(n.start)}–${formatTime(n.end)} s · ${methodName()} estimate${n.qualityFlags?.length?` · review: ${qualityText(n.qualityFlags)}`:''}`:`The audio comes from a ${methodName()} estimate.`);
  els.solo.disabled=Boolean(model.example?.scoreOnly)||!n||els.play.disabled;
  for(const [key,input] of Object.entries(inputs)){input.disabled=!n;input.value=n?({pitch:n.pitchShift,timing:Math.round(n.delta*1000),length:n.stretch,volume:n.gainDb})[key]:({pitch:0,timing:0,length:1,volume:0})[key];}
  values.pitch.textContent=`${Number(inputs.pitch.value)>0?'+':''}${inputs.pitch.value} st`;
  values.timing.textContent=`${Number(inputs.timing.value)>0?'+':''}${inputs.timing.value} ms`;
  values.length.textContent=`${Number(inputs.length.value).toFixed(2)}×`;
  values.volume.textContent=`${Number(inputs.volume.value)>0?'+':''}${inputs.volume.value} dB`;
  els.reset.disabled=!notesChanged();
}
function selectNote(n){model.selected=n;renderNotes();updateControls();}
function rowTop(pitch){return (model.pitchMax-pitch)*model.rowHeight;}
function xFor(time){return time/timelineDuration()*model.width;}
function timeFor(x){return x/model.width*timelineDuration();}
function waveformPath(n){
  const values=n.waveform||Array(28).fill(.35),points=values.length;
  const top=values.map((v,i)=>`${i/(points-1)*100},${50-v*44}`).join(' L ');
  const bottom=values.map((_,i)=>`${(points-1-i)/(points-1)*100},${50+values[points-1-i]*44}`).join(' L ');
  return `M ${top} L ${bottom} Z`;
}
function waveformPeaks(note,buffer){
  const data=buffer.getChannelData(0),rate=buffer.sampleRate,points=28;
  const from=clamp(Math.floor(note.start*rate),0,data.length),to=clamp(Math.ceil((note.end+.2)*rate),from+1,data.length);
  const peaks=Array.from({length:points},(_,i)=>{
    const first=from+Math.floor(i/points*(to-from)),last=from+Math.floor((i+1)/points*(to-from));
    let peak=0;for(let j=first;j<last;j++)peak=Math.max(peak,Math.abs(data[j]));return peak;
  });
  const max=Math.max(...peaks,1e-6);
  return peaks.map(v=>.12+.88*Math.sqrt(v/max));
}
const fftSize=2048,fftHop=256,fftBins=fftSize/2+1;
const windowFunction=Float32Array.from({length:fftSize},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(fftSize-1)));
function fft(real,imag){
  const n=real.length;
  for(let i=1,j=0;i<n;i++){
    let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
    if(i<j){[real[i],real[j]]=[real[j],real[i]];[imag[i],imag[j]]=[imag[j],imag[i]];}
  }
  for(let len=2;len<=n;len<<=1){
    const angle=-2*Math.PI/len,stepReal=Math.cos(angle),stepImag=Math.sin(angle),half=len>>1;
    for(let base=0;base<n;base+=len){
      let wr=1,wi=0;
      for(let j=0;j<half;j++){
        const even=base+j,odd=even+half,tr=wr*real[odd]-wi*imag[odd],ti=wr*imag[odd]+wi*real[odd];
        real[odd]=real[even]-tr;imag[odd]=imag[even]-ti;real[even]+=tr;imag[even]+=ti;
        const next=wr*stepReal-wi*stepImag;wi=wr*stepImag+wi*stepReal;wr=next;
      }
    }
  }
}
async function computeSpectrogram(buffer,token){
  const samples=buffer.getChannelData(0),rate=buffer.sampleRate,frames=Math.ceil(samples.length/fftHop);
  const db=new Float32Array(frames*fftBins),real=new Float32Array(fftSize),imag=new Float32Array(fftSize);
  for(let frame=0;frame<frames;frame++){
    real.fill(0);imag.fill(0);const center=frame*fftHop;
    for(let i=0;i<fftSize;i++){const sample=center+i-(fftSize>>1);if(sample>=0&&sample<samples.length)real[i]=samples[sample]*windowFunction[i];}
    fft(real,imag);const offset=frame*fftBins;
    for(let bin=0;bin<fftBins;bin++)db[offset+bin]=20*Math.log10(Math.hypot(real[bin],imag[bin])/fftSize+1e-8);
    if(frame%24===23){if(token!==model.spectrogramToken)return null;await new Promise(resolve=>setTimeout(resolve,0));}
  }
  const sorted=db.slice();sorted.sort();
  return {db,frames,bins:fftBins,rate,reference:sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.995))]};
}
function palette(value){
  const stops=[[5,8,20],[44,12,91],[133,30,133],[220,67,104],[250,145,37],[255,231,116]],x=clamp(value,0,1)*(stops.length-1),lo=Math.floor(x),mix=x-lo;
  return stops[lo].map((v,i)=>Math.round(v+(stops[Math.min(lo+1,stops.length-1)][i]-v)*mix));
}
function renderSpectrogram(){
  const canvas=els.spectrogram,width=Math.max(1,model.spectrogramData?.frames||model.width);
  if(canvas.width!==width)canvas.width=width;if(canvas.height!==SPEC_HEIGHT)canvas.height=SPEC_HEIGHT;
  canvas.style.width=`${model.width}px`;els.spectrogramContent.style.width=`${model.width}px`;
  const context=canvas.getContext('2d');context.clearRect(0,0,canvas.width,canvas.height);
  renderFrequencyLabels();
  if(!model.spectrogramData){renderSpectrogramNotes();return;}
  const {db,frames,bins,rate,reference}=model.spectrogramData;
  const imageKey=`${frames}:${SPEC_HEIGHT}`;
  if(model.spectrogramImage?.key!==imageKey){
    const energy=new Float32Array(frames*SPEC_HEIGHT);energy.fill(-120);
    for(let y=0;y<SPEC_HEIGHT;y++){
      const frequency=MAX_FREQUENCY*Math.pow(MIN_FREQUENCY/MAX_FREQUENCY,y/(SPEC_HEIGHT-1));
      const center=clamp(Math.round(frequency*fftSize/rate),1,bins-1),first=Math.max(1,center-2),last=Math.min(bins-1,center+2);
      for(let frame=0;frame<frames;frame++){
        let best=-120;const offset=frame*bins;
        for(let bin=first;bin<=last;bin++)best=Math.max(best,db[offset+bin]);
        energy[y*frames+frame]=best;
      }
    }
    const imageCanvas=document.createElement('canvas');imageCanvas.width=frames;imageCanvas.height=SPEC_HEIGHT;
    const image=imageCanvas.getContext('2d').createImageData(frames,SPEC_HEIGHT),pixels=image.data;
    for(let y=0;y<SPEC_HEIGHT;y++)for(let frame=0;frame<frames;frame++){
      const value=clamp((energy[y*frames+frame]-(reference-58))/58,0,1),alpha=value<.12?0:Math.round(66+133*value);
      const rgb=palette(value),index=(y*frames+frame)*4;
      pixels[index]=rgb[0];pixels[index+1]=rgb[1];pixels[index+2]=rgb[2];pixels[index+3]=alpha;
    }
    imageCanvas.getContext('2d').putImageData(image,0,0);model.spectrogramImage={key:imageKey,canvas:imageCanvas};
  }
  context.imageSmoothingEnabled=false;context.drawImage(model.spectrogramImage.canvas,0,0,canvas.width,SPEC_HEIGHT);
  renderSpectrogramNotes();
}
function frequencyY(frequency){
  const value=clamp(frequency,MIN_FREQUENCY,MAX_FREQUENCY);
  return Math.log(MAX_FREQUENCY/value)/Math.log(MAX_FREQUENCY/MIN_FREQUENCY)*SPEC_HEIGHT;
}
function renderFrequencyLabels(){
  const ticks=[[8000,'8 kHz'],[4000,'4 kHz'],[2000,'2 kHz'],[1000,'1 kHz'],[500,'500'],[250,'250'],[125,'125'],[62.5,'62']];
  els.frequencyLabels.replaceChildren();
  for(const [frequency,labelText] of ticks){
    const label=document.createElement('span');label.textContent=labelText;label.style.top=`${frequencyY(frequency)}px`;els.frequencyLabels.append(label);
  }
}
function renderSpectrogramNotes(){
  els.spectrogramNotes.replaceChildren();
  for(const note of model.notes){
    const pitch=note.pitch+note.pitchShift,frequency=440*2**((pitch-69)/12),mark=document.createElement('button');
    mark.type='button';mark.className='spectrogram-note'+(note===model.selected?' selected':'')+(note.qualityFlags?.length?' needs-review':'');
    mark.style.cssText=`--note-color:${color[pitch%12]};left:${xFor(visualStart(note))}px;top:${frequencyY(frequency)-2.5}px;width:${Math.max(4,xFor(visualEnd(note)-visualStart(note)))}px`;
    mark.title=`${pitchName(pitch)} · ${formatTime(visualStart(note))}–${formatTime(visualEnd(note))} s${note.qualityFlags?.length?' · audio check: '+qualityText(note.qualityFlags):''}`;
    mark.setAttribute('aria-label',`Select ${mark.title}`);
    mark.addEventListener('click',event=>{event.stopPropagation();selectNote(note);start(true);});
    els.spectrogramNotes.append(mark);
  }
}
function selectFromSpectrogram(event){
  const rect=els.spectrogramViewport.getBoundingClientRect();
  const x=event.clientX-rect.left+els.spectrogramViewport.scrollLeft;
  const y=clamp(event.clientY-rect.top,0,SPEC_HEIGHT);
  const time=clamp(x/model.width*timelineDuration(),0,timelineDuration());
  const frequency=MAX_FREQUENCY*Math.pow(MIN_FREQUENCY/MAX_FREQUENCY,y/SPEC_HEIGHT);
  const pitch=69+12*Math.log2(frequency/440);
  const active=model.notes.filter(note=>time>=visualStart(note)-.05&&time<=visualEnd(note)+.05);
  const nearest=active.sort((a,b)=>Math.abs(a.pitch+a.pitchShift-pitch)-Math.abs(b.pitch+b.pitchShift-pitch))[0];
  if(nearest){selectNote(nearest);start(true);return;}
  const next=model.notes.map(note=>({note,distance:time<visualStart(note)?visualStart(note)-time:time>visualEnd(note)?time-visualEnd(note):0})).sort((a,b)=>a.distance-b.distance)[0];
  if(next&&next.distance<=.25&&Math.abs(next.note.pitch+next.note.pitchShift-pitch)<=8){selectNote(next.note);start(true);return;}
  seekTo(time);
}
async function loadSpectrogram(path){
  const token=++model.spectrogramToken;model.inputBuffer=null;model.spectrogramData=null;model.spectrogramImage=null;renderSpectrogram();
  els.spectrogramStatus.textContent=path?'Loading original spectrogram…':'Upload audio to see its spectrogram here.';
  if(!path)return;
  try{
    const response=await fetch(assetUrl(path));if(!response.ok)throw new Error(`Audio request failed (${response.status}).`);
    const buffer=wavBuffer(await response.arrayBuffer(),decodeContext());if(token!==model.spectrogramToken)return;
    model.inputBuffer=buffer;model.spectrogramPromise=computeSpectrogram(buffer,token);model.spectrogramData=await model.spectrogramPromise;model.spectrogramPromise=null;model.spectrogramImage=null;
    if(token===model.spectrogramToken){renderSpectrogram();els.spectrogramStatus.textContent='Colored bars mark score-note fundamentals and timing; click one to inspect or audition it.';}
  }catch(error){if(token===model.spectrogramToken){model.spectrogramPromise=null;els.spectrogramStatus.textContent='Could not render the spectrogram. Use the original audio player to check notes.';console.warn('Original spectrogram unavailable:',error);}}
}
function renderNotes(){
  els.grid.querySelectorAll('.note').forEach(el=>el.remove());
  for(const n of model.notes){
    const el=document.createElement('button');el.type='button';el.className='note'+(n===model.selected?' selected':'')+(n.qualityFlags?.length?' needs-review':'');
    const visualPitch=n.pitch+n.pitchShift;
    el.style.cssText=`--note-color:${color[visualPitch%12]};top:${rowTop(visualPitch)+model.rowHeight*.06}px;left:${xFor(visualStart(n))}px;width:${Math.max(7,xFor(visualEnd(n)-visualStart(n)))}px`;
    if(n.mute)el.style.opacity='.35';
    el.title=`${pitchName(visualPitch)}, ${formatTime(visualStart(n))} s.${n.mute?' Muted in the edited sum.':''}${n.qualityFlags?.length?' Audio check: '+qualityText(n.qualityFlags)+'.':''} Drag to move; drag right edge to stretch.`;
    el.setAttribute('aria-label',el.title);
    el.innerHTML=`<svg class="blob" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="${waveformPath(n)}"/></svg>`;
    el.addEventListener('click',event=>{event.stopPropagation();if(performance.now()<model.suppressClickUntil)return;selectNote(n);start(true);});
    el.addEventListener('pointerdown',event=>beginDrag(event,n,el));
    els.grid.append(el);
  }
  renderSpectrogramNotes();
}
function beginDrag(event,n,el){
  if(event.button!==0)return;
  if(model.drawMode&&model.example?.scoreOnly){
    event.preventDefault();event.stopPropagation();
    model.notes=model.notes.filter(note=>note!==n);if(model.selected===n)model.selected=null;
    model.suppressClickUntil=performance.now()+250;
    renderGrid();updateControls();emitScore();status(`${pitchName(n.pitch)} removed. Draw another note, or press B to leave Draw Mode.`);return;
  }
  event.preventDefault();model.selected=n;
  els.grid.querySelectorAll('.note').forEach(button=>button.classList.toggle('selected',button===el));
  updateControls();
  const rect=el.getBoundingClientRect(),edge=event.clientX>rect.right-15;
  const initial={x:event.clientX,y:event.clientY,delta:n.delta,pitch:n.pitchShift,stretch:n.stretch};
  let moved=false;
  el.setPointerCapture(event.pointerId);
  const move=e=>{
    if(Math.hypot(e.clientX-initial.x,e.clientY-initial.y)<4&&!moved)return;
    moved=true;
    const scale=model.width/els.grid.getBoundingClientRect().width;
    if(edge){
      const seconds=timeFor((e.clientX-initial.x)*scale);
      n.stretch=clamp(Math.round((initial.stretch+seconds/(n.end-n.start))*50)/50,.5,2);
    }else{
      const seconds=timeFor((e.clientX-initial.x)*scale);
      n.delta=clamp(Math.round((initial.delta+seconds)*100)/100,-1,1);
      n.pitchShift=clamp(initial.pitch-Math.round((e.clientY-initial.y)/model.rowHeight),-12,12);
    }
    const visualPitch=n.pitch+n.pitchShift;
    el.style.top=`${rowTop(visualPitch)+model.rowHeight*.06}px`;
    el.style.left=`${xFor(visualStart(n))}px`;
    el.style.width=`${Math.max(7,xFor(visualEnd(n)-visualStart(n)))}px`;
    el.style.setProperty('--note-color',color[visualPitch%12]);
    updateControls();
  };
  const end=()=>{el.removeEventListener('pointermove',move);el.removeEventListener('pointerup',end);el.removeEventListener('pointercancel',end);model.suppressClickUntil=performance.now()+200;if(moved){renderGrid();if(model.example?.scoreOnly)emitScore();else emitOutputEdit([n.id]);if(model.playing){const t=currentTime();seekTo(t);}}else start(true);};
  el.addEventListener('pointermove',move);el.addEventListener('pointerup',end);el.addEventListener('pointercancel',end);
}
function renderGrid(){
  const pitchMargin=model.example?.scoreOnly?12:2;
  model.pitchMin=model.notes.length?Math.max(21,Math.min(...model.notes.map(n=>n.pitch+n.pitchShift))-pitchMargin):48;
  model.pitchMax=model.notes.length?Math.min(108,Math.max(...model.notes.map(n=>n.pitch+n.pitchShift))+pitchMargin):84;
  model.rowHeight=clamp(326/(model.pitchMax-model.pitchMin+1),7,13);
  const height=(model.pitchMax-model.pitchMin+1)*model.rowHeight;
  els.shell.style.height=`${height}px`;
  els.grid.style.height=`${height}px`;
  els.grid.style.setProperty('--row-height',`${model.rowHeight}px`);
  const duration=timelineDuration();
  model.width=Math.max(860,Math.round(duration*140),Math.ceil(els.viewport.clientWidth));
  els.grid.style.width=`${model.width}px`;
  els.ruler.style.width=`${model.width}px`;
  renderSpectrogram();
  els.labels.replaceChildren();els.ruler.replaceChildren();
  for(let pitch=model.pitchMin;pitch<=model.pitchMax;pitch++)if(pitch%12===0){
    const label=document.createElement('span');label.className='pitch-label';label.textContent=pitchName(pitch);label.style.top=`${rowTop(pitch)+model.rowHeight/2}px`;els.labels.append(label);
  }
  for(let time=0;time<=duration+.001;time+=duration>4?1:.25){
    const x=xFor(time),line=document.createElement('i');line.className='beat-line'+(Math.abs(time-Math.round(time))<.001?' major':'');line.style.left=`${x}px`;els.grid.append(line);
    const tick=document.createElement('span');tick.className='ruler-tick';tick.textContent=`${Number(time.toFixed(2))} s`;tick.style.left=`${x}px`;els.ruler.append(tick);
  }
  renderNotes();renderClock();
}
async function loadExample(id){
  stop(false);model.seek=0;model.original=false;model.buffers.clear();model.transformed.clear();
  model.example=model.manifest.examples.find(e=>e.id===id);
  model.notes=model.example.notes.filter(n=>model.example.scoreOnly||n.outputs?.aso?.audio).map(noteState);
  model.selected=null;els.play.disabled=true;els.compare.querySelector('span').textContent='Unedited sum';
  els.scoreToolRow.hidden=!model.example.scoreOnly;setDrawMode(false);
  els.intro.textContent=model.example.scoreOnly?'Check the draft against the spectrogram. Draw missing notes, or move and resize existing notes. Then run the model above this editor.':'Drag a note in time or pitch. Pull its right edge to stretch it. Hear the edited notes together, or isolate one.';
  els.footHint.textContent=model.example.scoreOnly?'Score changes become new audio when you run NoteGrab or NoteSep. Preview note plays the original recording around its score time.':'Solo notes get a listening boost; the edit mix keeps its original levels. Audio edits do not re-run separation.';
  els.mode.textContent=model.example.scoreOnly?'Score · awaiting separation':`${methodName()} · edited sum`;
  renderGrid();updateControls();loadSpectrogram(model.example.inputAudio);return preload();
}

window.noteSepEditor={
  async renderedWav(){
    if(!model.example||model.example.scoreOnly)throw new Error('Separate notes before exporting audio.');
    stop(false);
    const buffers=await Promise.all(model.notes.map(loadBuffer));
    const rate=16000;
    let duration=model.example.duration;
    model.notes.forEach((note,index)=>{
      if(note.mute)return;
      if(!changed(note))return;
      const bounds=cropBounds(note,buffers[index]);
      duration=Math.max(duration,bounds.start+note.delta+(bounds.end-bounds.start)*note.stretch);
    });
    const context=new OfflineAudioContext(1,Math.ceil(duration*rate),rate);
    // Playback pauses and visibility changes must not stop/disconnect an
    // offline export that is still rendering, especially in WebKit.
    const exportNodes=[];
    await Promise.all(model.notes.map((note,index)=>scheduleNote(note,buffers[index],context,0,0,false,false,exportNodes)));
    const rendered=await context.startRendering();
    return floatWav(rendered.getChannelData(0),rate);
  },
  async downloadEditedWav(){
    const bytes=await this.renderedWav();
    saveBlob(new Blob([bytes],{type:'audio/wav'}),'notesep-edited.wav');
    status('Downloaded the direct sum of edited note estimates.');
  },
  editSettings(){
    return Object.fromEntries(model.notes.map(n=>[String(n.id),{pitchShift:n.pitchShift,delta:n.delta,stretch:n.stretch,gainDb:n.gainDb,mute:n.mute===true}]));
  },
  applyEditSettings(settings={}){
    if(!settings||typeof settings!=='object'||Array.isArray(settings))throw new Error('Edit settings must be a note-ID object.');
    const ids=new Set(model.notes.map(note=>String(note.id)));
    if(Object.keys(settings).some(id=>!ids.has(id)))throw new Error('Edit settings contain notes outside this project.');
    const validated=new Map();
    for(const [id,value] of Object.entries(settings)){
      if(!value||typeof value!=='object'||Object.keys(value).some(key=>!['pitchShift','delta','stretch','gainDb','mute'].includes(key)))throw new Error('Invalid saved audio-edit fields.');
      if(value.mute!==undefined&&typeof value.mute!=='boolean')throw new Error('Saved mute must be boolean.');
      const edit=audioEditValues(value.pitchShift??0,(value.delta??0)*1000,value.stretch??1,value.gainDb??0);
      validated.set(id,{...edit,mute:value.mute??false});
    }
    stop(false);
    for(const note of model.notes)Object.assign(note,validated.get(String(note.id))||{pitchShift:0,delta:0,stretch:1,gainDb:0,mute:false});
    renderGrid();updateControls();emitOutputEdit([...validated.keys()]);
  },
  downloadEditSettings(){
    saveBlob(new Blob([JSON.stringify(this.editSettings(),null,2)+'\n'],{type:'application/json'}),'notesep-edits.json');
    status('Edit settings downloaded. Use them with the note-stem project in Python.');
  },
  downloadScore(){
    saveBlob(new Blob([JSON.stringify(this.scoreNotes(),null,2)+'\n'],{type:'application/json'}),'notesep-score.json');
  },
  async loadSession(example){
    if(!example||!Array.isArray(example.notes)||!Number.isFinite(example.duration)||example.duration<=0)throw new Error('Invalid editor session');
    await this.ready;
    model.manifest.examples=model.manifest.examples.filter(item=>item.id!==example.id);
    model.manifest.examples.push(example);
    let option=Array.from(els.piece.options).find(item=>item.value===example.id);
    if(!option){option=document.createElement('option');option.value=example.id;els.piece.append(option);}
    option.textContent=example.title||'Uploaded recording';els.piece.value=example.id;
    return await loadExample(example.id);
  },
  sourceScoreNotes(){return model.notes.map(n=>({id:n.id,pitch:n.pitch,start:n.start,end:n.end}));},
  scoreNotes(){return model.notes.map(n=>({id:n.id,pitch:n.pitch+n.pitchShift,start:visualStart(n),end:visualEnd(n)}));},
  setQualityChecks(checks=[]){
    const byId=new Map((checks||[]).map(item=>[String(item.id),item.flags||[]]));
    for(const note of model.notes)note.qualityFlags=byId.get(String(note.id))||[];
    renderNotes();
  },
  setDrawMode,
  addNote(pitch=60,start=0,end=0.5){
    if(!model.example?.scoreOnly)throw new Error('Switch to score editing before adding notes.');
    const values=scoreValues(pitch,start,end,model.example.duration);
    const note={id:crypto.randomUUID(),...values,delta:0,pitchShift:0,stretch:1,gainDb:0};
    model.notes.push(note);model.selected=note;renderGrid();updateControls();emitScore();return note.id;
  },
  correctSelected(pitch,start,end){
    if(!model.example?.scoreOnly||!model.selected)throw new Error('Select a score note to correct.');
    const values=scoreValues(pitch,start,end,model.example.duration);
    const corrected=correctNote(model.selected,values);
    model.notes=model.notes.map(note=>note===model.selected?corrected:note);model.selected=corrected;
    renderGrid();updateControls();emitScore();
  },
  removeSelected(){
    if(!model.example?.scoreOnly||!model.selected)return false;
    model.notes=model.notes.filter(note=>note!==model.selected);model.selected=null;renderGrid();updateControls();emitScore();return true;
  },
  pause(){stop();},
};
window.noteSepEditor.ready=new Promise(resolve=>{window.noteSepEditor.readyResolve=resolve;});
muteControl.addEventListener('change',()=>{
  const note=model.selected;if(!note||model.example?.scoreOnly)return;
  note.mute=muteControl.checked;renderNotes();emitOutputEdit([note.id]);
  if(model.playing){const time=currentTime();seekTo(time);}
});

for(const [key,input] of Object.entries(inputs))input.addEventListener('input',()=>{
  const n=model.selected;if(!n)return;
  if(key==='pitch')n.pitchShift=Number(input.value);
  if(key==='timing')n.delta=Number(input.value)/1000;
  if(key==='length')n.stretch=Number(input.value);
  if(key==='volume')n.gainDb=Number(input.value);
  if(key==='pitch'||key==='timing'||key==='length')renderGrid();else renderNotes();updateControls();
  if(model.example?.scoreOnly)emitScore();else emitOutputEdit([n.id]);
  if(model.playing){const t=currentTime();seekTo(t);}
});
audioFields.apply.addEventListener('click',()=>{
  try{
    const note=model.selected;
    if(!note||model.example?.scoreOnly)throw new Error('Select a separated note first.');
    const edits=audioEditValues(Number(audioFields.pitch.value),Number(audioFields.timing.value),Number(audioFields.length.value),Number(audioFields.gain.value));
    const time=currentTime(),playing=model.playing;
    Object.assign(note,edits);renderGrid();updateControls();emitOutputEdit([note.id]);
    if(playing)seekTo(time);
    status('Audio edit applied. Play the edit or download the edited WAV.');
  }catch(error){status(error.message);}
});
els.play.addEventListener('click',()=>model.playing?stop():start(false));
els.solo.addEventListener('click',()=>start(true));
els.compare.addEventListener('click',()=>{
  model.original=!model.original;
  els.compare.querySelector('span').textContent=model.original?'Edited sum':'Unedited sum';
  els.mode.textContent=model.original?`${methodName()} · original sum`:`${methodName()} · edited sum`;
  if(model.playing){const t=currentTime();seekTo(t);}
});
els.reset.addEventListener('click',()=>{
  const edited=model.notes.filter(changed).map(n=>n.id);
  model.notes=model.example.notes.filter(n=>model.example.scoreOnly||n.outputs?.aso?.audio).map(noteState);
  model.selected=null;renderNotes();updateControls();
  if(model.example?.scoreOnly)emitScore();else emitOutputEdit(edited);
  if(model.playing){const t=currentTime();seekTo(t);}
});
els.piece.addEventListener('change',()=>loadExample(els.piece.value));
els.viewport.addEventListener('scroll',()=>{
  $('#ruler-viewport').scrollLeft=els.viewport.scrollLeft;
  els.spectrogramViewport.scrollLeft=els.viewport.scrollLeft;
});
els.spectrogramViewport.addEventListener('scroll',()=>{
  if(els.viewport.scrollLeft!==els.spectrogramViewport.scrollLeft)els.viewport.scrollLeft=els.spectrogramViewport.scrollLeft;
  $('#ruler-viewport').scrollLeft=els.spectrogramViewport.scrollLeft;
});
els.spectrogramViewport.addEventListener('click',event=>{if(!event.target.closest('.spectrogram-note'))selectFromSpectrogram(event);});
els.ruler.addEventListener('click',event=>seekTo(timeFor(event.clientX-els.ruler.getBoundingClientRect().left)));
els.grid.addEventListener('click',event=>{if(!model.drawMode&&event.target===els.grid)seekTo(timeFor(event.offsetX));});
els.drawMode.addEventListener('click',()=>setDrawMode(!model.drawMode));
els.grid.addEventListener('pointerdown',event=>{
  if(!model.drawMode||!model.example?.scoreOnly||event.button!==0||event.target!==els.grid)return;
  event.preventDefault();
  const bounds=els.grid.getBoundingClientRect();
  const x=clamp(event.clientX-bounds.left,0,model.width);
  const y=clamp(event.clientY-bounds.top,0,bounds.height-1);
  const pitch=clamp(model.pitchMax-Math.floor(y/model.rowHeight),model.pitchMin,model.pitchMax);
  const onset=clamp(Math.round(timeFor(x)*100)/100,0,model.example.duration-.01);
  const note={id:crypto.randomUUID(),pitch,start:onset,end:Math.min(model.example.duration,onset+.25),delta:0,pitchShift:0,stretch:1,gainDb:0};
  model.notes.push(note);model.selected=note;renderNotes();updateControls();
  els.grid.setPointerCapture(event.pointerId);
  const move=e=>{
    const seconds=Math.round(timeFor(clamp(e.clientX-bounds.left,0,model.width))*100)/100;
    if(Math.abs(e.clientX-event.clientX)<4)return;
    note.start=clamp(Math.min(onset,seconds),0,model.example.duration-.01);
    note.end=clamp(Math.max(onset,seconds),note.start+.01,model.example.duration);
    renderNotes();updateControls();
  };
  const end=()=>{
    els.grid.removeEventListener('pointermove',move);els.grid.removeEventListener('pointerup',end);els.grid.removeEventListener('pointercancel',end);
    model.suppressClickUntil=performance.now()+250;
    renderGrid();updateControls();emitScore();
    status(`${pitchName(note.pitch)} added at ${formatTime(note.start)} s. Draw another, or press B to move and resize notes.`);
  };
  els.grid.addEventListener('pointermove',move);els.grid.addEventListener('pointerup',end);els.grid.addEventListener('pointercancel',end);
});
function enteredScore(){return [Number(scoreFields.pitch.value),Number(scoreFields.start.value),Number(scoreFields.end.value)];}
scoreFields.apply.addEventListener('click',()=>{try{window.noteSepEditor.correctSelected(...enteredScore());status('Score note corrected. Review the score, then rerun separation.');}catch(error){status(error.message);}});
scoreFields.add.addEventListener('click',()=>{try{window.noteSepEditor.addNote(...enteredScore());status('Score note added. Review coverage before separation.');}catch(error){status(error.message);}});
window.addEventListener('message',event=>{if(event.origin===location.origin&&event.data?.type==='notesep-editor-pause')stop();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('keydown',event=>{
  if(event.target.matches('input,select,textarea,[contenteditable="true"]'))return;
  if(model.example?.scoreOnly&&(event.key==='b'||event.key==='B'||event.key==='Escape')){event.preventDefault();setDrawMode(event.key==='Escape'?false:!model.drawMode);return;}
  if(event.target.matches('button'))return;
  if(event.code==='Space'){event.preventDefault();model.playing?stop():start(false);}
  if(event.key==='ArrowLeft'||event.key==='ArrowRight'){
    event.preventDefault();const ordered=[...model.notes].sort((a,b)=>visualStart(a)-visualStart(b)||a.pitch-b.pitch);
    const index=ordered.indexOf(model.selected);selectNote(ordered[clamp(index+(event.key==='ArrowRight'?1:-1),0,ordered.length-1)]);start(true);
  }
});

try{
  const response=await fetch(assetUrl('assets/manifest.json'));
  if(!response.ok)throw new Error(`Manifest request failed (${response.status}).`);
  model.manifest=await response.json();
  for(const e of model.manifest.examples){const option=document.createElement('option');option.value=e.id;option.textContent=e.title;els.piece.append(option);}
  const wanted=new URLSearchParams(location.search).get('piece');
  els.piece.value=model.manifest.examples.some(e=>e.id===wanted)?wanted:model.manifest.examples[0].id;
  await loadExample(els.piece.value);
  window.noteSepEditor.readyResolve();
}catch(error){status(`Editor unavailable: ${error.message}`);console.error(error);}
