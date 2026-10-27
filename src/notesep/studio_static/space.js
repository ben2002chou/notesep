import {client} from './vendor/gradio-client.mjs';
import {saveBlob} from './audio-export.mjs';

const $=selector=>document.querySelector(selector);
const pitchClasses=['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B'];
const pitchName=pitch=>`${pitchClasses[pitch%12]}${Math.floor(pitch/12)-1}`;
const ui={audio:$('#audio-file'),midi:$('#midi-file'),provider:$('#transcription-provider'),load:$('#load'),loadLabel:$('#load .load-label'),loadStatus:$('#load-status'),progress:$('#task-progress'),progressLabel:$('#task-progress-label'),progressElapsed:$('#task-progress-elapsed'),progressBar:$('#task-progress-bar'),sourcePreviewRow:$('#source-preview-row'),sourcePreview:$('#source-preview'),windowStart:$('#window-start'),windowSeconds:$('#window-seconds'),usePlaybackPosition:$('#use-playback-position'),recordingLabel:$('#recording-label'),scoreSubtitle:$('#score-subtitle'),transcribe:$('#transcribe'),alternateTranscriber:$('#alternate-transcriber'),crossCheck:$('#cross-check'),crossCheckResult:$('#cross-check-result'),crossCheckSummary:$('#cross-check-summary'),crossCheckCaveat:$('#cross-check-caveat'),crossCheckCandidates:$('#cross-check-candidates'),reviewGate:$('#score-review-gate'),reviewed:$('#score-reviewed'),outputReviewGate:$('#output-review-gate'),outputReviewStatus:$('#output-review-status'),outputReviewed:$('#output-reviewed'),instrumentHintChoice:$('#instrument-hint-choice'),instrumentHint:$('#instrument-hint'),transcriptionPrivacy:$('#transcription-privacy'),partChoice:$('#part-choice'),partFilter:$('#part-filter'),add:$('#add-note'),remove:$('#delete-note'),editScore:$('#edit-score'),separate:$('#separate'),separateLabel:$('#separate .separate-label'),separationHint:$('#separation-hint'),frame:$('#editor-frame'),original:$('#original-audio'),recording:$('#recording-row'),scoreCard:$('.score-card'),resultCard:$('.result-card'),status:$('#status'),name:$('#audio-name'),count:$('#note-count')};
const state={session:null,duration:0,sourceDuration:0,windowStart:0,score:[],scoreReviewed:false,outputReview:{active:false,audioReady:false,sumHeard:false,requiredNoteIds:[],heardNoteIds:new Set()},audioUrl:null,selectedAudio:null,allNotes:[],currentPart:'all',defaultPart:'all',partScores:new Map(),allPartScore:null,transcriptionDone:false,transcriptionProvider:'',crossCheck:null,crossCheckStale:false,separatedCheck:null,reconstructionCheck:null,busy:false,action:null,client:null,qualityWarning:'',audioCheck:null};
const exampleButton=$("#try-example"),alignedExampleButton=$("#try-example-score");
const nativeUI={row:$('#native-controls'),start:$('#native-recording-start'),send:$('#native-send')};
const nativeBackend=window.__JUCE__?.backend;
const projectUI={file:$('#project-file'),settings:$('#project-settings-file'),open:$('#restore-project'),jointRow:$('#project-joint-choice'),joint:$('#project-joint')};
const exportsUI={score:$("#download-score"),edit:$("#download-edit"),stems:$("#download-stems"),settings:$("#download-edit-settings")};
let config={provider:'YourMT3+',instrumentHint:false,ready:true,message:''};
let sourcePreviewUrl=null;
let progressTicker=null,progressStarted=0,progressStep='';
const report=(message,isError=false)=>{ui.status.textContent=message;ui.status.style.color=isError?'#ffaaaa':'#bfd9ee';};
const uploadReport=(message,isError=false)=>{ui.loadStatus.textContent=message;ui.loadStatus.classList.toggle('error',isError);};
function renderProgress(){
  const elapsed=Math.floor((Date.now()-progressStarted)/1000);
  ui.progressElapsed.textContent=elapsed?`${elapsed} s elapsed`:'';
}
function showProgress(step){
  progressStep=step;progressStarted=Date.now();
  ui.progressLabel.textContent=step;
  ui.progressBar.setAttribute('aria-valuetext',`${step} in progress`);
  ui.progress.hidden=false;renderProgress();
  if(!progressTicker)progressTicker=setInterval(renderProgress,1000);
}
function hideProgress(){
  if(progressTicker)clearInterval(progressTicker);
  progressTicker=null;ui.progress.hidden=true;
}
const transcriptionFailureMessage=error=>{
  const message=String(error?.message||error||'Unknown transcription error');
  const provider=selectedProvider();
  const alternate=provider==='muscriptor_public'?'YourMT3+ · public':'MuScriptor · public';
  if(/quota|rate.?limit|could not reserve.*\d+\s*s|not enough.*gpu/i.test(message)){
    const retry=message.match(/Try again in\s+([^\.\r\n]+)/i)?.[1]?.trim();
    return `This transcription service is temporarily at its usage limit${retry?` (try again in ${retry})`:''}. Your audio and spectrogram are still loaded; choose ${alternate} above and click “Try a new draft,” or add notes/import MIDI.`;
  }
  return `Automatic transcription did not return a draft${message?`: ${message}`:''}. Your audio and spectrogram are still loaded; choose ${alternate} above and click “Try a new draft,” or add notes/import MIDI.`;
};
const separationFailureMessage=error=>{
  const message=String(error?.message||error||'Unknown inference error');
  if(/quota|rate.?limit|could not reserve.*\d+\s*s|not enough.*gpu|usage limit/i.test(message)){
    const retry=message.match(/Try again in\s+([^\.\r\n]+)/i)?.[1]?.trim();
    return `The Space GPU is temporarily at its usage limit${retry?` (try again in ${retry})`:''}. Your recording and score edits are still loaded. Try fewer notes or a shorter window, or retry after the quota resets.`;
  }
  return `NoteSep could not finish${message?`: ${message}`:''}. Your recording and current score are still loaded; check the score and try again.`;
};
const update=()=>{
  nativeBackend?.emitEvent('notesepBusy',{busy:state.busy});
  projectUI.open.disabled=state.busy||!projectUI.file.files?.length;
  projectUI.joint.disabled=state.busy;
  ui.load.disabled=state.busy;
  exampleButton.disabled=state.busy;
  alignedExampleButton.disabled=state.busy;
  ui.frame.inert=state.busy;
  ui.loadLabel.textContent=state.action==='load'?'Loading & drafting…':(ui.midi.files?.length?'Load aligned MIDI':(state.sourceDuration>12?'Load selected window':(!state.selectedAudio&&!ui.audio.files?.length?'Choose a recording first':'Load & draft score')));
  ui.load.querySelector('span:last-child').textContent=state.action==='load'?'…':'→';
  ui.transcribe.disabled=state.busy||!state.session;
  ui.alternateTranscriber.disabled=state.busy||!state.session;
  ui.crossCheck.hidden=!state.session||!state.transcriptionDone;
  ui.crossCheck.disabled=state.busy||!state.session||!state.transcriptionDone||Boolean(state.crossCheck&&!state.crossCheckStale);
  ui.reviewGate.hidden=!state.session||!state.score.length||!ui.editScore.hidden;
  ui.reviewed.checked=state.scoreReviewed;
  ui.reviewed.disabled=state.busy||!state.session;
  const outputReview=state.outputReview;
  ui.outputReviewGate.hidden=!outputReview.active;
  ui.outputReviewed.checked=outputReview.active&&outputReview.reviewed===true;
  const pendingNotes=outputReview.requiredNoteIds.filter(id=>!outputReview.heardNoteIds.has(id));
  ui.outputReviewed.disabled=state.busy||!outputReview.audioReady||!outputReview.sumHeard||pendingNotes.length>0;
  ui.outputReviewStatus.textContent=!outputReview.audioReady?'Waiting for every separated note file to load…':!outputReview.sumHeard?'Move the playhead to 0 s, then play the full separated sum through to the end.':pendingNotes.length?`Now audition ${pendingNotes.length} outlined note${pendingNotes.length===1?'':'s'} to the end.`:'The complete sum and every flagged note have been played through.';
  if(state.transcriptionDone){
    const source=state.transcriptionProvider||selectedProvider();
    const alternate=source.startsWith('muscriptor')?'YourMT3+':'MuScriptor';
    ui.crossCheck.textContent=state.crossCheck&&!state.crossCheckStale?'✓ Cross-check complete':state.crossCheckStale?`↻ Recheck with ${alternate}`:`↔ Cross-check with ${alternate}`;
  }
  exportsUI.score.disabled=state.busy||(!state.session&&!state.outputReview.active)||!state.score.length;
  nativeUI.send.disabled=state.busy||!state.outputReview.active||!state.outputReview.audioReady;
  exportsUI.settings.disabled=state.busy||!state.outputReview.active||!state.outputReview.audioReady;
  exportsUI.stems.hidden=!exportsUI.stems.hasAttribute('href')||!state.outputReview.active;
  exportsUI.edit.disabled=state.busy||!state.outputReview.active||!state.outputReview.audioReady;
  ui.add.disabled=state.busy||!state.session;
  ui.remove.disabled=state.busy||!state.session||!ui.editScore.hidden;
  ui.add.setAttribute('aria-pressed',String(ui.add.dataset.drawing==='true'));
  const inRange=state.score.length>=1&&state.score.length<=96;
  ui.separate.disabled=state.busy||!state.session||!inRange||!state.scoreReviewed;
  ui.separateLabel.textContent=state.score.length===1||partialPartSelection()||state.forceNoteGrab?'Run NoteGrab':'Run NoteSep';
  ui.separate.title=state.busy?'Wait for the current task to finish':!state.session?'Load a recording first':!state.score.length?'Draw or import a note first':state.score.length>96?'Select a part or shorten the edit window to at most 96 notes':!state.scoreReviewed?'Review score coverage and confirm the check above':'';
  if(!state.session&&!state.outputReview.active)ui.count.textContent='No score loaded';
  else if(state.score.length===0)ui.count.textContent='No notes yet · add notes or try another draft';
  else if(state.score.length>96)ui.count.textContent=`${state.score.length} notes · select at most 96`;
  else if(state.score.length===1)ui.count.textContent='1 note · ready for NoteGrab';
  else ui.count.textContent=`${state.score.length} notes in selected part`;
  if(state.session&&ui.editScore.hidden)ui.scoreSubtitle.textContent=state.score.length>=2?`${state.score.length} notes in this draft. Check for omissions, fix pitch or timing errors, then confirm the review to run NoteSep.`:state.score.length===1?'Check this note’s pitch and timing, confirm the review, then run NoteGrab.':'Draw or import a note in the piano roll, then review it before extraction.';
  ui.separationHint.textContent=state.qualityWarning||(!state.session?'Load a recording to begin.':state.busy?'Wait for the current task to finish.':!state.score.length?'No notes are selected. Draw a note, import MIDI, or try another transcription draft.':state.score.length>96?'This edit window has more than 96 notes. Select one instrument part or a shorter window.':!state.scoreReviewed?'Check the score for missing notes and pitch/timing errors, then confirm the review above.':partialPartSelection()?'Review complete. Extract the selected part with NoteGrab. Choose All detected parts for joint NoteSep separation.':state.score.length===1?'One note is ready for NoteGrab extraction. Add another for joint NoteSep separation.':'Review complete. Run NoteSep to create estimates. After separation, audio edits play locally; corrected score queries require another run.');
  ui.resultCard.classList.toggle('audio-review',Boolean(state.audioCheck?.reviewCount));
};
const api=async()=>{await new Promise(resolve=>{if(ui.frame.contentWindow?.noteSepEditor)resolve();else ui.frame.addEventListener('load',resolve,{once:true});});const editor=ui.frame.contentWindow.noteSepEditor;await editor.ready;return editor;};
async function call(name,args){
  if(!state.client)state.client=await client(location.origin);
  const response=await state.client.predict('/'+name,args);
  return Array.isArray(response.data)?response.data[0]:response.data;
}
async function encoded(file,maxBytes=config.fileLimits?.audioBytes??128_000_000,label='Audio file'){
  if(file.size>maxBytes)throw new Error(`${label} exceeds ${(maxBytes/1000000).toFixed(0)} MB.`);
  // Native FileReader avoids building a second full binary string in JS.
  return await new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.onerror=()=>reject(new Error(`${label} could not be read.`));
    reader.onload=()=>{
      const data=reader.result;
      if(typeof data!=='string'||!data.includes(';base64,'))return reject(new Error(`${label} could not be encoded.`));
      resolve(data.slice(data.indexOf(',')+1));
    };
    reader.readAsDataURL(file);
  });
}
function setWindowControls(duration,reset=false){
  if(!Number.isFinite(duration)||duration<=0)return;
  state.sourceDuration=duration;
  const current=Number(ui.windowSeconds.value);
  const seconds=reset||!Number.isFinite(current)?Math.min(12,duration):Math.min(12,duration,Math.max(1.1,current));
  ui.windowSeconds.value=seconds.toFixed(1);
  const maxStart=Math.max(0,duration-seconds);
  ui.windowStart.max=maxStart.toFixed(1);
  if(reset)ui.windowStart.value='0';
  else ui.windowStart.value=Math.min(maxStart,Math.max(0,Number(ui.windowStart.value)||0)).toFixed(1);
}
async function checkAudioDuration(file){
  const url=URL.createObjectURL(file);
  const audio=new Audio();
  audio.preload='metadata';
  try{
    const duration=await new Promise(resolve=>{
      const finish=value=>{clearTimeout(timer);audio.removeAttribute('src');audio.load();resolve(value);};
      const timer=setTimeout(()=>finish(null),4000);
      audio.addEventListener('loadedmetadata',()=>finish(audio.duration),{once:true});
      audio.addEventListener('error',()=>finish(null),{once:true});
      audio.src=url;
    });
    if(duration!==null&&Number.isFinite(duration)){
      if(duration>300)throw new Error(`This recording is ${duration.toFixed(1)} s long. The current preview supports recordings up to 5 minutes.`);
      if(duration<1.1)throw new Error(`This recording is ${duration.toFixed(1)} s long. Use a recording at least 1.1 s long.`);
      setWindowControls(duration);
      return duration;
    }
  }finally{URL.revokeObjectURL(url);}
}
async function scoreEditor(notes,title='Correct this score',audioCheck=null){
  state.audioCheck=audioCheck;state.qualityWarning=audioCheck?.summary||'';
  const editor=await api();
  const mergedChecks=new Map();
  for(const check of [audioCheck,state.crossCheckStale?null:state.crossCheck,state.separatedCheck])for(const item of check?.notes||[]){
    const id=String(item.id),entry=mergedChecks.get(id)||{id,flags:[]};
    entry.flags=[...new Set([...entry.flags,...(item.flags||[])])];mergedChecks.set(id,entry);
  }
  const checks=mergedChecks;
  await editor.loadSession({id:'score-'+state.session,title,duration:state.duration,inputAudio:state.audioUrl,notes:notes.map((n,i)=>({event:i,...n,qualityFlags:checks.get(String(n.id))?.flags||[]})),audioCheck,scoreOnly:true});
  state.score=editor.scoreNotes();update();
}
function guideScore(){
  update();
  ui.scoreCard.scrollIntoView({behavior:'auto',block:'start'});
}
function selectedProvider(){return ui.provider.value||config.defaultProvider||'yourmt3';}
function offerAlternateProvider(show){
  const alternate=selectedProvider()==='yourmt3'?'muscriptor_public':'yourmt3';
  const label=alternate==='yourmt3'?'YourMT3+ · public':'MuScriptor · public';
  ui.alternateTranscriber.dataset.provider=alternate;
  ui.alternateTranscriber.textContent=`Try ${label}`;
  ui.alternateTranscriber.hidden=!show;
}
function renderCrossCheck(check,stale=false){
  ui.crossCheckResult.hidden=!check;
  ui.crossCheckCandidates.replaceChildren();
  if(!check)return;
  ui.crossCheckSummary.textContent=stale?'This cross-check is for the previous score; rerun it after editing.':(check.summary||'Cross-check completed.');
  ui.crossCheckCaveat.textContent=stale?'The list below is retained as a reminder, but note edits may change these disagreements.': 'This compares note pitch and onset only. Agreement can still miss notes both models overlook.';
  const items=[...(check.primaryOnly||[]).map(note=>({side:'Current draft only',note})),
               ...(check.independentOnly||[]).map(note=>({side:'Other model only · possible omission',note}))];
  for(const {side,note} of items.slice(0,12)){
    const item=document.createElement('li');
    item.append(document.createTextNode(`${side}: ${note.pitchName} at ${Number(note.start).toFixed(2)} s`));
    if(side.startsWith('Other model only')&&!stale){
      const add=document.createElement('button');add.type='button';add.textContent='Add for review';
      add.addEventListener('click',()=>busy(async()=>{
        let editor=await api();
        if(!ui.editScore.hidden){
          state.score=editor.scoreNotes();
          await scoreEditor(state.score,'Correct score · then reseparate',state.audioCheck);
          ui.editScore.hidden=true;
          editor=await api();
        }
        editor.addNote(note.pitch,note.start,note.end);
        state.score=editor.scoreNotes();
        guideScore();
        report(`Added a possible ${pitchName(note.pitch)} at ${Number(note.start).toFixed(2)} s. Check it against the spectrogram before rerunning NoteSep.`);
      }));
      item.append(' ',add);
    }
    ui.crossCheckCandidates.append(item);
  }
  if(items.length>12){const item=document.createElement('li');item.textContent=`${items.length-12} more disagreements; review the full score against the spectrogram.`;ui.crossCheckCandidates.append(item);}
}
function reconstructionSummary(check){
  if(!check||!Number.isFinite(Number(check.errorEnergyPct)))return '';
  const regions=Array.isArray(check.uncoveredRegions)?check.uncoveredRegions:[];
  const spans=regions.map(region=>`${Number(region.start).toFixed(1)}–${Number(region.end).toFixed(1)} s`);
  const mismatch=spans.length?`; long mismatch ${spans.join(', ')}`:'; no long mismatch windows';
  return `Sum check: ${Number(check.errorEnergyPct).toFixed(1)}% waveform-residual energy${mismatch}.`;
}
function updateTranscriptionPrivacy(){
  const provider=selectedProvider();
  const isMuScriptor=provider==='muscriptor_public';
  ui.instrumentHintChoice.hidden=!isMuScriptor||Boolean(ui.midi.files?.length);
  if(provider==='muscriptor_public')ui.transcriptionPrivacy.innerHTML='Only the selected edit window is sent to the <a href="https://muscriptor.kyutai.org/" target="_blank" rel="noopener noreferrer">public MuScriptor service</a>; its FAQ says uploaded audio is not kept afterwards. If you choose cross-check, the same window is also sent to public YourMT3+.';
  else ui.transcriptionPrivacy.innerHTML='Only the selected edit window is sent to the public <a href="https://huggingface.co/spaces/mimbres/YourMT3" target="_blank" rel="noopener noreferrer">YourMT3+ Space</a>. If you choose cross-check, the same window is also sent to public MuScriptor.';
}
function partialPartSelection(){return state.currentPart!=='all'&&new Set(state.allNotes.map(n=>n.partId||'all')).size>1;}
function notesForPart(part=state.currentPart){
  if(part!=='all')return state.partScores.get(part)||state.allNotes.filter(note=>note.partId===part);
  if(state.allPartScore)return state.allPartScore;
  const ids=[...new Set(state.allNotes.map(note=>note.partId||'all'))];
  return ids.flatMap(id=>state.partScores.get(id)||state.allNotes.filter(note=>(note.partId||'all')===id));
}
function chooseParts(notes){
  state.allNotes=notes;state.partScores.clear();state.allPartScore=null;
  const counts=new Map();
  for(const note of notes){const key=note.partId||'all';counts.set(key,(counts.get(key)||0)+1);}
  const parts=[...counts].map(([id,count])=>({id,count,label:notes.find(note=>(note.partId||'all')===id)?.part||'All notes'}));
  ui.partFilter.replaceChildren();
  if(parts.length>1){
    ui.partFilter.add(new Option(`All detected parts (${notes.length})`,'all'));
    for(const part of parts)ui.partFilter.add(new Option(`${part.label} (${part.count})`,part.id));
    const defaultPart=[...parts].sort((a,b)=>b.count-a.count||a.label.localeCompare(b.label))[0];
    state.currentPart='all';state.defaultPart=defaultPart.id;ui.partFilter.value='all';ui.partChoice.hidden=false;
  }else{state.currentPart=parts[0]?.id||'all';state.defaultPart=state.currentPart;ui.partChoice.hidden=true;}
  return notesForPart();
}
function resetWorkspace(){
  state.forceNoteGrab=false;projectUI.jointRow.hidden=true;
  exportsUI.stems.hidden=true;exportsUI.stems.removeAttribute("href");
  ui.frame.contentWindow?.postMessage({type:'notesep-editor-pause'},location.origin);
  ui.original.pause();ui.original.removeAttribute('src');ui.original.load();
  state.session=null;state.duration=0;state.sourceDuration=0;state.windowStart=0;state.score=[];state.scoreReviewed=false;ui.reviewed.checked=false;state.outputReview={active:false,audioReady:false,sumHeard:false,requiredNoteIds:[],heardNoteIds:new Set()};ui.outputReviewed.checked=false;state.audioUrl=null;state.allNotes=[];
  state.qualityWarning='';state.audioCheck=null;state.crossCheck=null;state.crossCheckStale=false;state.separatedCheck=null;state.reconstructionCheck=null;state.transcriptionProvider='';
  renderCrossCheck(null);
  ui.add.dataset.drawing='false';
  state.currentPart='all';state.defaultPart='all';state.partScores.clear();state.allPartScore=null;state.transcriptionDone=false;
  ui.recording.hidden=true;ui.scoreCard.hidden=true;ui.resultCard.hidden=true;
  ui.partChoice.hidden=true;ui.editScore.hidden=true;ui.transcribe.hidden=true;
  offerAlternateProvider(false);
  update();
}
async function transcribeDraft(){
  const provider=selectedProvider();
  const providerLabel=ui.provider.selectedOptions[0]?.textContent?.trim()||provider;
  offerAlternateProvider(false);
  const instrument=provider==='muscriptor_public'?ui.instrumentHint.value:'';
  showProgress(`Drafting score with ${providerLabel}`);
  report(provider==='muscriptor_public'?'Sending only the selected window to the public MuScriptor service…':'YourMT3+ is making a first-pass draft from the selected window.');
  let output;
  try{output=await call('transcribe',{session:state.session,instrument,provider});}
  catch(error){offerAlternateProvider(true);throw new Error(transcriptionFailureMessage(error));}
  const rawNotes=Array.isArray(output.notes)?output.notes:[];
  const rawMidiEvents=Number.isFinite(output.rawMidiNoteEvents)?output.rawMidiNoteEvents:null;
  const filteredEvents=rawMidiEvents===null?0:Math.max(0,rawMidiEvents-rawNotes.length);
  const notes=chooseParts(rawNotes);state.scoreReviewed=false;ui.reviewed.checked=false;state.transcriptionDone=true;state.transcriptionProvider=provider;
  state.crossCheck=null;state.crossCheckStale=false;state.separatedCheck=null;state.reconstructionCheck=null;renderCrossCheck(null);
  showProgress('Opening the editable score');
  const emptyTitle=rawMidiEvents>0?'No usable MIDI notes · try another draft':'No notes detected · add notes or try again';
  await scoreEditor(notes,notes.length?'Transcription draft · correct against the spectrogram':emptyTitle,output.audioCheck||null);
  guideScore();
  if(rawNotes.length){
    const partCount=new Set(rawNotes.map(note=>note.partId).filter(Boolean)).size;
    const hint=instrument?` Instrument hint: ${ui.instrumentHint.selectedOptions[0]?.textContent}.`:'';
    const filtered=filteredEvents?` ${filteredEvents} raw MIDI events could not be placed in this edit window; check for omitted notes.`:'';
    report(`${output.audioCheck?.summary||`${providerLabel} drafted ${rawNotes.length} notes. Check pitches and timing against the spectrogram before separation.`}${filtered}${partCount>1?` All detected parts are shown. Use the part selector to focus on one instrument.`:''}${hint}`);
  }else if(rawMidiEvents>0){offerAlternateProvider(true);report(`${providerLabel} returned ${rawMidiEvents} MIDI note events, but none could be placed in this edit window. Try the other transcriber, use aligned MIDI, or add notes manually.`,true);}
  else{offerAlternateProvider(true);report(`No notes came back from ${providerLabel}.${instrument?' Try Automatic / all parts, add notes manually, or import aligned MIDI.':' Add notes manually, import aligned MIDI, or try another excerpt.'}`,true);}
}
async function busy(work,action='other'){
  if(state.busy)return;state.busy=true;state.action=action;update();showProgress(action==='load'?'Preparing recording':'Working on your score');
  try{await work();}catch(error){const message=action==='separate'?separationFailureMessage(error):(error?.message||String(error));report(message,true);if(action==='load')uploadReport(message,true);console.error(error);}finally{hideProgress();state.busy=false;state.action=null;update();}
}
function syncAudioSelection(){applyAudioSelection(ui.audio.files?.[0]||null);}
function applyAudioSelection(file){
  if(file===state.selectedAudio)return;
  state.selectedAudio=file;
  ui.name.textContent=file?.name||'Choose audio';
  update();
  try{
    ui.sourcePreview.pause();ui.sourcePreview.removeAttribute('src');ui.sourcePreview.load();
    if(sourcePreviewUrl)URL.revokeObjectURL(sourcePreviewUrl);
    sourcePreviewUrl=null;state.sourceDuration=0;
    if(file){
      sourcePreviewUrl=URL.createObjectURL(file);ui.sourcePreviewRow.hidden=false;
      ui.windowStart.value='0';ui.windowSeconds.value='12';
      ui.sourcePreview.onloadedmetadata=()=>{setWindowControls(ui.sourcePreview.duration,true);update();};
      ui.sourcePreview.onerror=()=>uploadReport('The browser could not preview this file. The WAV/FLAC can still be tried if it is valid.');
      ui.sourcePreview.src=sourcePreviewUrl;
      uploadReport(`Ready to load ${file.name}. Listen and mark the rough edit window, then load it.`);
    }else{ui.sourcePreviewRow.hidden=true;uploadReport('Choose a WAV, FLAC, MP3 or AIFF file to begin.');}
  }catch(error){
    uploadReport(file?`Selected ${file.name}. Browser preview is unavailable; you can still load the recording.`:'Choose a WAV, FLAC, MP3 or AIFF file to begin.');
    console.warn('Audio preview setup failed; preserving the selected file for upload.',error);
  }finally{update();}
}
ui.audio.addEventListener('input',syncAudioSelection);
ui.audio.addEventListener('change',syncAudioSelection);
ui.windowSeconds.addEventListener('change',()=>setWindowControls(state.sourceDuration));
ui.windowStart.addEventListener('change',()=>setWindowControls(state.sourceDuration));
ui.usePlaybackPosition.addEventListener('click',()=>{
  if(!state.sourceDuration)return;
  const seconds=Math.min(12,Number(ui.windowSeconds.value)||12),maxStart=Math.max(0,state.sourceDuration-seconds);
  ui.windowStart.value=Math.min(maxStart,Math.max(0,ui.sourcePreview.currentTime||0)).toFixed(1);
});
const dropTarget=document.querySelector('.file-choice');
dropTarget.addEventListener('dragover',event=>{event.preventDefault();dropTarget.classList.add('drag-over');});
dropTarget.addEventListener('dragleave',()=>dropTarget.classList.remove('drag-over'));
dropTarget.addEventListener('drop',event=>{
  event.preventDefault();dropTarget.classList.remove('drag-over');
  if(event.dataTransfer?.files?.length){ui.audio.files=event.dataTransfer.files;ui.audio.dispatchEvent(new Event('change'));}
});
ui.midi.addEventListener('change',()=>{updateTranscriptionPrivacy();if(ui.midi.files?.length)uploadReport(`Aligned MIDI selected: ${ui.midi.files[0].name}. Loading it skips automatic transcription.`);update();});
ui.reviewed.addEventListener('change',()=>{state.scoreReviewed=ui.reviewed.checked;update();if(state.scoreReviewed)report('Score review confirmed. Run NoteSep, then compare the direct sum and audition notes that sound uncertain.');});
ui.outputReviewed.addEventListener('change',()=>{
  state.outputReview.reviewed=ui.outputReviewed.checked;
  update();
  report(state.outputReview.reviewed
    ?'Listening review recorded. If the separated audio does not match the score, edit the score or note audio and rerun as needed.'
    :'Listen through the complete NoteSep sum and every outlined note before treating this output as reviewed.');
});
ui.provider.addEventListener('change',()=>{offerAlternateProvider(false);updateTranscriptionPrivacy();});
ui.alternateTranscriber.addEventListener('click',()=>busy(async()=>{
  ui.provider.value=ui.alternateTranscriber.dataset.provider||'yourmt3';
  updateTranscriptionPrivacy();
  await transcribeDraft();
}));
ui.crossCheck.addEventListener('click',()=>busy(async()=>{
  const source=state.transcriptionProvider||selectedProvider();
  const provider=source==='muscriptor_public'?'yourmt3':'muscriptor_public';
  const label=provider==='yourmt3'?'YourMT3+':'MuScriptor';
  showProgress(`Cross-checking with ${label}`);
  report(`Running an independent ${label} draft. Your current score will not be replaced.`);
  const result=await call('cross_check',{session:state.session,notes_json:JSON.stringify(state.score),provider,
    instrument:provider==='muscriptor_public'?ui.instrumentHint.value:''});
  if((result.primaryOnly||[]).length||(result.independentOnly||[]).length){state.scoreReviewed=false;ui.reviewed.checked=false;}
  state.crossCheck=result;state.crossCheckStale=false;state.qualityWarning=result.summary;renderCrossCheck(result);
  const editor=await api();
  const quality=new Map();
  for(const check of [state.audioCheck,state.crossCheckStale?null:state.crossCheck,state.separatedCheck])for(const item of check?.notes||[]){
    const id=String(item.id),flags=quality.get(id)||[];quality.set(id,[...new Set([...flags,...(item.flags||[])])]);
  }
  editor.setQualityChecks([...quality].map(([id,flags])=>({id,flags})));
  update();report(result.summary);
}));
async function loadRecording(midiOverride=null){
  resetWorkspace();
  uploadReport('Loading your recording and preparing its spectrogram…');
  report('Loading and resampling the recording…');
  const audio=ui.audio.files?.[0]||state.selectedAudio;if(!audio)throw new Error('Choose a WAV, FLAC, MP3 or AIFF recording first.');
  if(!/\.(?:wav|flac|mp3|aif|aiff)$/i.test(audio.name)){const error=new Error('Choose a WAV, FLAC, MP3 or AIFF recording.');uploadReport(error.message,true);throw error;}
  const measuredDuration=await checkAudioDuration(audio);
  const requestedStart=Math.max(0,Number(ui.windowStart.value)||0);
  const requestedWindow=Math.min(12,Math.max(1.1,Number(ui.windowSeconds.value)||12));
  showProgress('Uploading and preparing the edit window');
  const prepared=await call('prepare',{audio_base64:await encoded(audio),filename:audio.name,midi_base64:midiOverride??(ui.midi.files?.[0]?await encoded(ui.midi.files[0],config.fileLimits?.midiBytes??3_000_000,'MIDI file'):''),window_start:requestedStart,window_seconds:requestedWindow});
  state.session=prepared.session;state.duration=prepared.duration;state.sourceDuration=prepared.sourceDuration??measuredDuration??prepared.duration;state.windowStart=prepared.windowStart??requestedStart;state.audioUrl=prepared.audioUrl;
  setWindowControls(state.sourceDuration);
  ui.original.src=prepared.audioUrl;ui.recording.hidden=false;ui.scoreCard.hidden=false;ui.resultCard.hidden=false;ui.editScore.hidden=true;ui.transcribe.hidden=false;
  const windowEnd=state.windowStart+state.duration;
  const windowDescription=`Edit window ${state.windowStart.toFixed(1)}–${windowEnd.toFixed(1)} s of ${state.sourceDuration.toFixed(1)} s`;
  ui.recordingLabel.textContent=windowDescription;
  ui.scoreSubtitle.textContent=`${windowDescription}. Check note pitches and timing against the spectrogram.`;
  const notes=chooseParts(prepared.notes||[]);
  showProgress('Opening the editable score');
  await scoreEditor(notes,notes.length?'Imported MIDI · correct the notes':'Transcribing a draft…',prepared.audioCheck||null);
  uploadReport(`Loaded ${windowDescription}. ${notes.length?'Opening the editable score…':'Making an editable first-pass score…'}`);
  let draftError=null;
  if(notes.length){
    state.transcriptionDone=true;
    state.transcriptionProvider=selectedProvider();
    ui.crossCheck.hidden=false;
    guideScore();
    report(`${prepared.audioCheck?.summary||`Loaded ${windowDescription} and ${notes.length} aligned MIDI notes. Review the score, then run NoteSep.`}`);
  }else{
    try{await transcribeDraft();}
    catch(error){
      draftError=error;
      state.transcriptionDone=true;
      await scoreEditor([], 'Transcription unavailable · add notes manually');
      report(`Audio and spectrogram are ready. ${error?.message||error}`,true);
    }
  }
  if(draftError)uploadReport('Audio is ready. Transcription could not complete; add notes manually or try a different edit window.',true);
  else uploadReport(`${windowDescription}. The score is ready below.`);
}
ui.load.addEventListener('click',()=>busy(loadRecording,'load'));
projectUI.file.addEventListener('change',update);
projectUI.joint.addEventListener('change',()=>{state.forceNoteGrab=!projectUI.joint.checked;state.scoreReviewed=false;update();});
async function reopenProject(file,settings=null){
  if(!file||!file.name.toLowerCase().endsWith('.zip'))throw new Error('Choose a NoteSep project ZIP.');
  if(file.size>180000000)throw new Error('Project ZIP exceeds 180 MB.');
  showProgress('Opening saved note stems');
  const example=await call('restore_project',{project_base64:await encoded(file,config.fileLimits?.projectBytes??180_000_000,'Project ZIP'),filename:file.name});
  resetWorkspace();ui.audio.value='';ui.midi.value='';state.selectedAudio=null;ui.sourcePreviewRow.hidden=true;
  state.projectSession=example.session;state.session=example.canReseparate?example.session:null;state.duration=example.duration;
  state.sourceDuration=example.sourceDuration;state.windowStart=example.windowStart;state.audioUrl=example.inputAudio;
  state.score=example.notes.map(({id,pitch,start,end})=>({id,pitch,start,end}));state.forceNoteGrab=example.forceNoteGrab;
  state.allNotes=state.score;state.allPartScore=state.score;state.transcriptionDone=true;
  ui.recording.hidden=!example.inputAudio;ui.scoreCard.hidden=false;ui.resultCard.hidden=false;
  if(example.inputAudio)ui.original.src=example.inputAudio;
  setWindowControls(state.sourceDuration);ui.windowStart.value=state.windowStart;ui.windowSeconds.value=state.duration;
  ui.recordingLabel.textContent=`Saved edit window ${state.windowStart.toFixed(1)}–${(state.windowStart+state.duration).toFixed(1)} s`;
  ui.scoreSubtitle.textContent='Saved estimates are ready to edit. Upload the original recording to choose another window.';
  ui.editScore.hidden=!example.canReseparate;ui.transcribe.hidden=!example.canReseparate;
  projectUI.jointRow.hidden=!example.canReseparate;projectUI.joint.checked=!state.forceNoteGrab;
  exportsUI.stems.href=example.projectArchive;exportsUI.stems.hidden=false;
  if(nativeBackend)nativeUI.start.value=Number(example.provenance?.daw_recording_start_seconds)||0;
  state.outputReview={active:true,audioReady:false,sumHeard:false,requiredNoteIds:[],heardNoteIds:new Set(),reviewed:false};
  const editor=await api();state.outputReview.audioReady=await editor.loadSession(example);
  editor.applyEditSettings(settings??example.edits??{});
  report(`Reopened ${example.notes.length} editable note estimates. Saved playback edits restored; audition the sum before continuing.`);
  update();return example;
}
projectUI.open.addEventListener('click',()=>busy(async()=>{
  const file=projectUI.file.files?.[0],settings=projectUI.settings.files?.[0];
  if(settings&&settings.size>1000000)throw new Error('Edit settings JSON is too large.');
  await reopenProject(file,settings?JSON.parse(await settings.text()):null);
},'load'));
exampleButton.addEventListener('click',()=>busy(async()=>{
  const response=await fetch('/static/examples/mixture.wav');
  if(!response.ok)throw new Error('The example could not load. Upload your own WAV, FLAC, MP3 or AIFF instead.');
  const file=new File([await response.blob()],'notesep-example.wav',{type:'audio/wav'});
  ui.audio.value='';ui.midi.value='';
  applyAudioSelection(file);
  await loadRecording();
},'load'));
alignedExampleButton.addEventListener('click',()=>busy(async()=>{
  const responses=await Promise.all([fetch('/static/examples/mixture.wav'),fetch('/static/examples/score.mid')]);
  if(responses.some(response=>!response.ok))throw new Error('The aligned example could not load.');
  const file=new File([await responses[0].blob()],'notesep-example.wav',{type:'audio/wav'});
  const midi=new File([await responses[1].blob()],'notesep-example.mid',{type:'audio/midi'});
  ui.audio.value='';ui.midi.value='';applyAudioSelection(file);
  await loadRecording(await encoded(midi,config.fileLimits?.midiBytes??3_000_000,'MIDI file'));
},'load'));
ui.transcribe.addEventListener('click',()=>busy(async()=>{
  if(state.score.length&&!window.confirm('Replace the current score with a new automatic draft?'))return;
  state.partScores.clear();
  await transcribeDraft();
}));
ui.partFilter.addEventListener('change',()=>busy(async()=>{
  const editor=await api();
  if(state.currentPart==='all'){
    const originalById=new Map(state.allNotes.map(note=>[String(note.id),note]));
    state.allPartScore=editor.scoreNotes().map(note=>{
      const original=originalById.get(String(note.id));
      return {...(original||{}),...note,partId:original?.partId||state.defaultPart,
        part:original?.part||state.allNotes.find(item=>item.partId===state.defaultPart)?.part||'Instrument'};
    });
    state.partScores.clear();
    for(const partId of new Set(state.allPartScore.map(note=>note.partId)))
      state.partScores.set(partId,state.allPartScore.filter(note=>note.partId===partId));
  }else{
    const originalPart=state.allNotes.filter(note=>note.partId===state.currentPart);
    const originalById=new Map(originalPart.map(note=>[String(note.id),note]));
    const edited=editor.scoreNotes().map(note=>({... (originalById.get(String(note.id))||{}),...note,
      partId:state.currentPart,part:originalPart[0]?.part||'Instrument'}));
    state.partScores.set(state.currentPart,edited);state.allPartScore=null;
  }
  state.currentPart=ui.partFilter.value;state.scoreReviewed=false;ui.reviewed.checked=false;
  if(state.crossCheck){state.crossCheckStale=true;renderCrossCheck(state.crossCheck,true);}
  const notes=notesForPart();
  await scoreEditor(notes,`Correct ${ui.partFilter.selectedOptions[0]?.textContent||'selected part'}`,state.audioCheck);
  guideScore();
  report('Instrument part changed. Edits already made to other parts are kept; review this part against the spectrogram.');
}));
ui.add.addEventListener('click',()=>busy(async()=>{
  const editor=await api();
  if(ui.editScore.hidden===false)await scoreEditor(state.score,'Correct this score');
  const drawing=editor.setDrawMode(ui.add.dataset.drawing!=='true');
  ui.add.dataset.drawing=String(drawing);ui.editScore.hidden=true;
  if(drawing)report('Draw Mode on: click an empty piano-roll row for a short note, or drag right to set its length. Click a note to erase it; press B to exit.');
  else report('Draw Mode off. Drag notes to move them, or drag their right edges to resize them.');
}));
ui.remove.addEventListener('click',()=>busy(async()=>{
  const editor=await api();if(!editor.removeSelected())throw new Error('Select a score note first, or switch to score editing.');
  state.score=editor.scoreNotes();report('Selected score note removed.');
}));
ui.editScore.addEventListener('click',()=>busy(async()=>{
  const editor=await api();state.score=editor.sourceScoreNotes();
  state.scoreReviewed=false;ui.reviewed.checked=false;
  state.outputReview={active:false,audioReady:false,sumHeard:false,requiredNoteIds:[],heardNoteIds:new Set()};ui.outputReviewed.checked=false;
  await scoreEditor(state.score,'Correct score · then reseparate');ui.editScore.hidden=true;
  guideScore();report('Edit pitches and boundaries, then run NoteSep again for new note estimates.');
}));
ui.separate.addEventListener('click',()=>busy(async()=>{
  const editor=await api();const score=editor.scoreNotes();
  if(score.length<1||score.length>96)throw new Error('Keep 1–96 notes in the selected score window.');
  const useASO=score.length>1&&!partialPartSelection()&&!state.forceNoteGrab;
  const method=useASO?'NoteSep':'NoteGrab';
  showProgress(useASO?`Separating ${score.length} notes`:`Extracting ${score.length} notes`);
  report(useASO?`Running NoteGrab and ASO on ${score.length} corrected notes…`:`Running NoteGrab on ${score.length} selected notes…`);
  const example=await call('separate',{session:state.session,notes_json:JSON.stringify(score),apply_aso:useASO});
  showProgress('Opening separated notes');
  exportsUI.stems.href=example.projectArchive;exportsUI.stems.hidden=!example.projectArchive;
  state.audioCheck=example.audioCheck||null;
  state.separatedCheck=example.separatedNoteCheck||null;
  state.reconstructionCheck=example.reconstructionCheck||null;
  const crossFlags=new Map((state.crossCheckStale?[]:state.crossCheck?.notes||[]).map(item=>[String(item.id),item.flags||[]]));
  example.notes=example.notes.map(note=>({...note,qualityFlags:[...new Set([...(note.qualityFlags||[]),...(crossFlags.get(String(note.id))||[])])]}));
  state.outputReview={active:true,audioReady:false,sumHeard:false,requiredNoteIds:[...new Set(example.notes.filter(note=>(note.qualityFlags||[]).length).map(note=>String(note.id)))],heardNoteIds:new Set(),reviewed:false};
  ui.outputReviewed.checked=false;
  state.outputReview.audioReady=await editor.loadSession(example);
  state.score=score;ui.editScore.hidden=false;ui.add.dataset.drawing='false';
  const reviews=[example.audioCheck,example.separatedNoteCheck].filter(item=>item?.reviewCount);
  const reconstruction=reconstructionSummary(example.reconstructionCheck);
  const crossSummary=state.crossCheckStale?'Cross-check is out of date after score edits; rerun it to check this version.':state.crossCheck?.summary||'';
  state.qualityWarning=[...reviews.map(item=>item.summary),crossSummary,reconstruction].filter(Boolean).join(' ')
    ||[example.audioCheck?.summary,example.separatedNoteCheck?.summary].filter(Boolean).join(' ');
  ui.scoreSubtitle.textContent=`${method} output is ready. ${state.qualityWarning} Select an outlined note to audition its estimate; drag it or use the sliders to edit. To correct the score, choose Edit score again and rerun ${method}.`;
  ui.scoreCard.scrollIntoView({behavior:'smooth',block:'start'});
  if(!state.outputReview.audioReady){
    state.qualityWarning=[state.qualityWarning,'Some separated note files did not load, so the output cannot be auditioned yet.'].filter(Boolean).join(' ');
    report(`The model returned ${score.length} estimates, but one or more audio files failed to load. Check the editor connection before reviewing this result.`,true);
  }else report(`${method} finished for ${score.length} note${score.length===1?'':'s'}. ${state.qualityWarning} Play the full separated sum and review any outlined notes before treating it as checked; edit the score and rerun if anything sounds wrong.`);
},'separate'));
let frameFitObserver=null,frameFitRequest=0;
async function fitFrame(){
  try{
    await api();
    const editor=ui.frame.contentDocument.querySelector('#editor');
    frameFitObserver?.disconnect();
    if(frameFitRequest)cancelAnimationFrame(frameFitRequest);
    frameFitRequest=0;
    const fit=()=>{
      frameFitRequest=0;
      const height=`${Math.max(660,Math.ceil(editor.getBoundingClientRect().height)+16)}px`;
      if(ui.frame.style.height!==height)ui.frame.style.height=height;
    };
    frameFitObserver=new ResizeObserver(()=>{
      if(!frameFitRequest)frameFitRequest=requestAnimationFrame(fit);
    });
    frameFitObserver.observe(editor);
    fit();
  }catch{}
}
ui.frame.addEventListener('load',fitFrame);
fitFrame();
window.addEventListener('message',event=>{
  if(event.origin!==location.origin)return;
  if(event.data?.type==='notesep-editor-height')ui.frame.style.height=`${Math.max(660,event.data.height+12)}px`;
  if(event.data?.type==='notesep-editor-play')ui.original.pause();
  if(event.data?.type==='notesep-editor-audio-edit'&&state.outputReview.active){
    const editedIds=[...new Set((event.data.noteIds||[]).map(String))];
    if(editedIds.length){
      state.outputReview.reviewed=false;
      state.outputReview.sumHeard=false;
      state.outputReview.requiredNoteIds=[...new Set([...state.outputReview.requiredNoteIds,...editedIds])];
      for(const id of editedIds)state.outputReview.heardNoteIds.delete(id);
      ui.outputReviewed.checked=false;
      update();
      report('A separated note was edited. Replay the full sum and audition each changed note before marking it reviewed.');
    }
  }
  if(event.data?.type==='notesep-editor-audition-complete'&&state.outputReview.active){
    if(event.data.mode==='sum')state.outputReview.sumHeard=true;
    if(event.data.mode==='note'&&event.data.noteId!=null)state.outputReview.heardNoteIds.add(String(event.data.noteId));
    update();
  }
  if(event.data?.type==='notesep-editor-draw-mode'){ui.add.dataset.drawing=String(Boolean(event.data.enabled));update();}
  if(event.data?.type==='notesep-editor-score-change'&&Array.isArray(event.data.notes)){
    state.score=event.data.notes.filter(n=>Number.isFinite(n.pitch)&&Number.isFinite(n.start)&&Number.isFinite(n.end));
    state.scoreReviewed=false;ui.reviewed.checked=false;
    if(state.crossCheck){state.crossCheckStale=true;renderCrossCheck(state.crossCheck,true);state.qualityWarning='Score changed; run the cross-check again after finishing edits.';}
    state.audioCheck=null;state.qualityWarning='Score changed. Audio checks refresh when you run NoteGrab or NoteSep.';
    state.separatedCheck=null;state.reconstructionCheck=null;
    ui.frame.contentWindow?.noteSepEditor?.setQualityChecks?.([]);
    if(state.session)update();
  }
});
ui.original.addEventListener('play',async()=>{try{(await api()).pause();}catch{}});
fetch('/notesep-config',{cache:'no-store'}).then(response=>response.ok?response.json():null).then(value=>{
  if(!value)return;config={...config,...value};
  if(Array.isArray(config.providers)){
    ui.provider.replaceChildren(...config.providers.map(item=>new Option(item.label,item.id)));
    ui.provider.value=config.defaultProvider||config.providers[0]?.id||'yourmt3';
  }
  updateTranscriptionPrivacy();
}).catch(()=>{});
report('Choose a short recording to begin.');update();

exportsUI.score.addEventListener("click",()=>busy(async()=>{(await api()).downloadScore();},"export"));
exportsUI.edit.addEventListener("click",()=>busy(async()=>{await (await api()).downloadEditedWav();},"export"));

exportsUI.settings.addEventListener("click",()=>busy(async()=>{(await api()).downloadEditSettings();},"export"));
async function projectArchiveUrl(){
  const origin=nativeBackend?Number(nativeUI.start.value):null;
  if(origin!==null&&(!Number.isFinite(origin)||origin<0))throw new Error('Use a nonnegative DAW recording position.');
  const filename=new URL(exportsUI.stems.href,location.origin).pathname.split('/').pop();
  const result=await call('save_project',{session:state.session||state.projectSession,filename,
    edits_json:JSON.stringify((await api()).editSettings()),recording_start:origin});
  exportsUI.stems.href=result.projectArchive;return exportsUI.stems.href;
}
exportsUI.stems.addEventListener('click',event=>{
  event.preventDefault();
  if(state.busy||!exportsUI.stems.hasAttribute('href'))return;
  busy(async()=>{
    // Fetch in the app's authenticated session, then save locally. A direct
    // attachment navigation can lose authentication on private hosted Spaces.
    const response=await fetch(await projectArchiveUrl(),{credentials:'same-origin'});
    if(!response.ok)throw new Error(`Project download failed (${response.status}). Run separation again if its temporary files expired.`);
    const blob=await response.blob(),header=new Uint8Array(await blob.slice(0,4).arrayBuffer());
    if(header[0]!==80||header[1]!==75||header[2]!==3||header[3]!==4)throw new Error('The server did not return a note project ZIP. Check your session and run separation again.');
    saveBlob(blob,'notesep-notes.zip');
    report('Downloaded the separated note stems and project.');
  },'export');
});

if(nativeBackend){
  nativeUI.row.hidden=false;
  nativeBackend.addEventListener('notesepRenderStatus',packet=>{
    report(packet.ok?'Edited passage received by the DAW. Rewind to audition it.':(packet.error||'The DAW rejected this render.'),!packet.ok);
  });
  nativeBackend.addEventListener('notesepRestore',packet=>busy(async()=>{
    const bytes=Uint8Array.from(atob(packet.projectBase64),c=>c.charCodeAt(0));
    await reopenProject(new File([bytes],'daw-session-notes.zip',{type:'application/zip'}),packet.editSettings||{});
    nativeUI.start.value=Number(packet.recordingStartSeconds)||0;
    report('Editable note bank restored from the DAW session.');
  },'load'));
  nativeBackend.addEventListener('notesepCapture',packet=>{
    if(state.busy){report('Finish the current task before loading a DAW capture.',true);return;}
    try{
      const bytes=Uint8Array.from(atob(packet.audioBase64),c=>c.charCodeAt(0));
      const file=new File([bytes],packet.filename||'daw-capture.wav',{type:'audio/wav'});
      const transfer=new DataTransfer();transfer.items.add(file);
      ui.midi.value='';ui.audio.files=transfer.files;
      nativeUI.start.value=Number(packet.dawStartSeconds)||0;
      ui.audio.dispatchEvent(new Event('change',{bubbles:true}));
      report('DAW capture loaded. Choose the edit window and create or import its score.');
    }catch(error){report(error.message,true);}
  });
  nativeUI.send.addEventListener('click',()=>busy(async()=>{
    const start=Number(nativeUI.start.value)+state.windowStart;
    if(!Number.isFinite(start)||start<0)throw new Error('Use a nonnegative DAW recording position.');
    const bytes=await (await api()).renderedWav();
    const response=await fetch(await projectArchiveUrl(),{credentials:'same-origin'});
    if(!response.ok)throw new Error('The saved note bank has expired. Rerun separation or reopen its downloaded project before sending to the DAW.');
    const project=await response.arrayBuffer();
    const base64=buffer=>{let binary='';const data=new Uint8Array(buffer);for(let i=0;i<data.length;i+=32768)binary+=String.fromCharCode(...data.subarray(i,i+32768));return btoa(binary);};
    nativeBackend.emitEvent('notesepRender',{audioBase64:base64(bytes),dawStartSeconds:start,
      projectBase64:base64(project),editSettings:(await api()).editSettings(),recordingStartSeconds:Number(nativeUI.start.value)});
    report('Rendered passage sent; waiting for the DAW to accept it.');
  },'export'));
  nativeBackend.emitEvent('notesepReady',{});
}
