import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from notesep.public_transcription import (
    MUSCRIPTOR_PUBLIC_MIDI_URL,
    transcribe_with_public_muscriptor,
)


class PublicMuScriptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.temp.name) / "window.wav"
        self.audio_path.write_bytes(b"selected-window")

    def tearDown(self):
        self.temp.cleanup()

    @patch("notesep.public_transcription.requests.post")
    def test_sends_bounded_file_and_optional_instrument(self, post):
        response = Mock()
        response.headers = {"content-type": "audio/midi"}
        response.content = b"MThd\x00midi"
        post.return_value = response

        result = transcribe_with_public_muscriptor(self.audio_path, "acoustic_piano")

        self.assertEqual(result, response.content)
        args, kwargs = post.call_args
        self.assertEqual(args[0], MUSCRIPTOR_PUBLIC_MIDI_URL)
        self.assertEqual(kwargs["data"], {"detect_tempo": "false", "instruments": "acoustic_piano"})
        self.assertEqual(kwargs["files"]["file"][0], "window.wav")
        self.assertEqual(kwargs["timeout"], 120)
        response.raise_for_status.assert_called_once()

    @patch("notesep.public_transcription.requests.post")
    def test_service_failure_has_actionable_message(self, post):
        post.side_effect = requests.Timeout("timeout")
        with self.assertRaisesRegex(ValueError, r"Try YourMT3\+"):
            transcribe_with_public_muscriptor(self.audio_path)

    @patch("notesep.public_transcription.time.sleep")
    @patch("notesep.public_transcription.requests.post")
    def test_retries_only_the_public_service_busy_response(self, post, sleep):
        busy = requests.Response()
        busy.status_code = 503
        busy._content = b'{"detail":"server busy: another transcription is in progress"}'
        success = Mock()
        success.status_code = 200
        success.headers = {"content-type": "audio/midi"}
        success.content = b"MThd\x00midi"
        post.side_effect = [busy, success]

        self.assertEqual(transcribe_with_public_muscriptor(self.audio_path), success.content)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    @patch("notesep.public_transcription.requests.post")
    def test_rejects_non_midi_response(self, post):
        response = Mock()
        response.headers = {"content-type": "text/html"}
        response.content = b"upstream error page"
        post.return_value = response
        with self.assertRaisesRegex(ValueError, "did not return a valid MIDI"):
            transcribe_with_public_muscriptor(self.audio_path)


if __name__ == "__main__":
    unittest.main()
