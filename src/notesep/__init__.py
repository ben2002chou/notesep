"""Separate score queries into aligned, individually editable note estimates."""
from .api import NoteBank, NoteSep
from .transcription import TranscriptionDraft, transcribe_audio

__all__ = ["NoteBank", "NoteSep", "TranscriptionDraft", "transcribe_audio"]
__version__ = "0.1.0"
