"""Draft a bounded audio window, correct its notes, then separate through Python."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import soundfile as sf

from .assets import sha256
from .audio_windows import decode_audio_window, MAX_WINDOW_SECONDS, RATE
from .score_io import midi_notes, midi_note_event_count


def transcribe_midi(path, *, provider='yourmt3', instrument='', device='auto', model_file=None):
    """Return raw MIDI and provider identity for an already bounded window."""
    instrument = instrument.strip()
    if provider == 'yourmt3':
        if instrument:
            raise ValueError('Instrument hints are supported by MuScriptor, not YourMT3+')
        try:
            from .public_transcription import transcribe_with_yourmt3
            import gradio_client
        except ImportError as error:
            raise ValueError('Install notesep[transcription] for public transcription') from error
        return transcribe_with_yourmt3(Path(path)), {'provider': 'YourMT3+', 'reproducibility': 'external service; model revision is not controlled'}
    if provider == 'muscriptor_public':
        try:
            from .public_transcription import transcribe_with_public_muscriptor
        except ImportError as error:
            raise ValueError('Install notesep[transcription] for public transcription') from error
        return transcribe_with_public_muscriptor(Path(path), instrument), {'provider': 'MuScriptor public', 'reproducibility': 'external service; model revision is not controlled'}
    if provider == 'muscriptor_local':
        from .local_transcription import load_muscriptor, local_midi
        model, identity = load_muscriptor(device, model_file)
        return local_midi(model, path, instrument), identity
    raise ValueError('Choose yourmt3, muscriptor_public or muscriptor_local')


@dataclass
class TranscriptionDraft:
    """Editable window-relative notes with source/AMT provenance and original MIDI."""
    source_path: Path
    notes: list[dict]
    audio: np.ndarray
    midi: bytes
    provenance: dict = field(default_factory=dict)

    def export(self, directory):
        root = Path(directory)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Draft output directory must be empty')
        root.mkdir(parents=True, exist_ok=True)
        sf.write(root / 'window.wav', self.audio, RATE, subtype='FLOAT')
        (root / 'draft.mid').write_bytes(self.midi)
        (root / 'score.json').write_text(json.dumps(self.notes, indent=2) + '\n')
        metadata = {**self.provenance, 'score_timeline': 'window',
                    'note_count': len(self.notes), 'window_wav_sha256': sha256(root / 'window.wav'),
                    'score_sha256': sha256(root / 'score.json')}
        (root / 'transcription.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return root / 'score.json'

    def separate(self, model, notes=None, *, apply_aso=False):
        """Use corrected window-relative queries; enable ASO only for a complete bank."""
        if sha256(self.source_path) != self.provenance['source_sha256']:
            raise ValueError('The source recording changed after transcription')
        bank = model.separate(self.source_path, self.notes if notes is None else notes,
            window_start=self.provenance['window_start_seconds'],
            window_seconds=self.provenance['window_duration_seconds'],
            score_timeline='window', apply_aso=apply_aso)
        bank.provenance['transcription'] = dict(self.provenance)
        return bank


def transcribe_audio(audio_path, *, provider='yourmt3', instrument='', device='auto',
                     window_start=0.0, window_seconds=None, model_file=None):
    """Create a draft of the selected 1.1–12 s window; longer files use the first 12 s by default."""
    source = Path(audio_path).resolve()
    source_hash = sha256(source)
    if window_seconds is None:
        window_seconds = min(MAX_WINDOW_SECONDS, sf.info(source).duration)
    audio, actual_start, source_duration = decode_audio_window(source, window_start, window_seconds)
    with tempfile.TemporaryDirectory(prefix='notesep-transcribe-') as folder:
        path = Path(folder) / 'window.wav'
        sf.write(path, audio, RATE, subtype='FLOAT')
        data, identity = transcribe_midi(path, provider=provider, instrument=instrument,
                                         device=device, model_file=model_file)
        window_sha = sha256(path)
    notes = midi_notes(data, len(audio) / RATE)
    if sha256(source) != source_hash:
        raise ValueError('The source recording changed during transcription')
    return TranscriptionDraft(source, notes, audio, data, {
        **identity, 'source_name': source.name, 'source_sha256': source_hash,
        'window_start_seconds': actual_start, 'source_duration_seconds': source_duration,
        'window_duration_seconds': len(audio) / RATE, 'window_wav_sha256': window_sha,
        'score_timeline': 'window', 'instrument_hint': instrument,
        'midi_sha256': hashlib.sha256(data).hexdigest(), 'raw_midi_note_events': midi_note_event_count(data),
        'correction_policy': 'manual review; no automatic acceptance or deletion',
    })
