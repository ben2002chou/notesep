"""Tests for selecting a bounded edit window from a full recording."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf

from notesep.audio_windows import decode_audio_window


class AudioWindowTests(unittest.TestCase):
    def test_reads_requested_region_downmixes_and_resamples(self):
        rate = 44_100
        time = np.arange(rate * 20, dtype=np.float32) / rate
        mono = np.sin(2 * np.pi * 880 * time).astype(np.float32) * 0.1
        stereo = np.column_stack([mono, mono])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            sf.write(path, stereo, rate)
            audio, start, source_duration = decode_audio_window(path, 7.25, 6.5)

        self.assertEqual(start, 7.25)
        self.assertAlmostEqual(source_duration, 20.0, places=3)
        self.assertEqual(len(audio), round(6.5 * 16_000))
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(float(np.max(np.abs(audio))), 0.09)

    def test_window_near_end_is_clamped_without_padding(self):
        rate = 16_000
        audio = np.full(rate * 8, 0.05, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.wav"
            sf.write(path, audio, rate)
            result, start, _ = decode_audio_window(path, 3.0, 12.0)

        self.assertEqual(start, 3.0)
        self.assertEqual(len(result), rate * 5)

    def test_mp3_window_matches_full_decoder_with_nonzero_start(self):
        from math import gcd
        from scipy.signal import resample_poly

        with tempfile.TemporaryDirectory() as directory:
            for rate in (8_000, 16_000, 22_050, 44_100, 48_000):
                time = np.arange(rate * 16, dtype=np.float32) / rate
                left = 0.1 * np.sin(2 * np.pi * (220 * time + 15 * time**2))
                right = 0.07 * np.sin(2 * np.pi * (440 * time + 25 * time**2))
                path = Path(directory) / "recording.mp3"
                sf.write(path, np.column_stack([left, right]), rate, format="MP3", subtype="MPEG_LAYER_III")
                decoded, decoded_rate = sf.read(path, dtype="float32", always_2d=True)
                self.assertEqual(decoded_rate, rate)
                for origin in (0.0, 0.4, 1.125, 3.25, 10.3):
                    with self.subTest(rate=rate, origin=origin):
                        result, start, duration = decode_audio_window(path, origin, 2.5)
                        first = round(origin * rate)
                        stop = first + round(2.5 * rate)
                        factor = gcd(rate, 16_000)
                        expected = resample_poly(decoded[first:stop].mean(axis=1),
                                                 16_000 // factor, rate // factor).astype(np.float32)
                        self.assertEqual(start, first / rate)
                        self.assertAlmostEqual(duration, len(decoded) / rate, places=6)
                        self.assertEqual(len(result), 2.5 * 16_000)
                        self.assertTrue(np.isfinite(result).all())
                        np.testing.assert_allclose(result, expected, rtol=0, atol=2e-6)

    def test_mp3_reads_are_bounded_independently_of_window_origin(self):
        rate = 16_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.mp3"
            sf.write(path, np.zeros(rate * 30, dtype=np.float32), rate,
                     format="MP3", subtype="MPEG_LAYER_III")
            read = sf.SoundFile.read
            requests = []

            def bounded_read(source, frames=-1, **kwargs):
                requests.append(frames)
                self.assertGreater(frames, 0)
                self.assertLessEqual(frames, 13 * rate)
                return read(source, frames=frames, **kwargs)

            with patch.object(sf.SoundFile, "read", bounded_read):
                result, start, duration = decode_audio_window(path, 17.5, 12.0)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(result), 12 * 16_000)
            self.assertEqual(start, 17.5)
            self.assertEqual(duration, 30.0)

    def test_aiff_window_preserves_stereo_downmix_and_source_origin(self):
        rate = 48_000
        time = np.arange(rate * 8, dtype=np.float32) / rate
        signal = np.sin(2 * np.pi * 330 * time).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.aiff"
            sf.write(path, np.column_stack([signal * 0.08, signal * 0.04]), rate,
                     format="AIFF", subtype="PCM_24")
            result, start, duration = decode_audio_window(path, 2.125, 3.5)

        self.assertEqual(start, 2.125)
        self.assertEqual(duration, 8.0)
        self.assertEqual(len(result), 3.5 * 16_000)
        self.assertTrue(np.isfinite(result).all())
        # Exclude the resampling boundary when checking the steady-state
        # amplitude: the finite window's zero extension can ring at its ends.
        interior = result[160:-160]
        self.assertGreater(float(np.max(np.abs(interior))), 0.059)
        self.assertLess(float(np.max(np.abs(interior))), 0.061)
        self.assertAlmostEqual(float(np.sqrt(np.mean(interior**2))), 0.06 / np.sqrt(2), places=4)

    def test_rejects_short_tail_and_invalid_ranges(self):
        rate = 16_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            sf.write(path, np.zeros(rate * 8, dtype=np.float32), rate)
            with self.assertRaisesRegex(ValueError, "at least 1.1 seconds"):
                decode_audio_window(path, 7.0, 2.0)
            with self.assertRaisesRegex(ValueError, "1.1 to 12 seconds"):
                decode_audio_window(path, 0.0, 13.0)
            with self.assertRaisesRegex(ValueError, "beyond the end"):
                decode_audio_window(path, 8.0, 2.0)


if __name__ == "__main__":
    unittest.main()
