"""Tests for SpeechSegmenter's hysteresis on top of raw per-frame VAD probability
-- in particular, that min_speech_ms actually rejects short non-speech blips
(breath, click, mic bump) instead of letting them open a segment that gets
sent to whisper and risks a hallucinated stock phrase (see text_filter.py).

Uses a scripted fake VAD instead of the real ONNX model so these run fast and
deterministically without a model file on disk.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from vad import SpeechSegmenter, VadConfig

FRAME_SIZE = 512
SAMPLE_RATE = 16000
FRAME_MS = 1000 * FRAME_SIZE / SAMPLE_RATE  # 32ms


class FakeVAD:
    """Returns a pre-scripted probability per call instead of running the ONNX model."""

    def __init__(self, probs: list[float]):
        self._probs = iter(probs)

    def speech_prob(self, chunk: np.ndarray) -> float:
        return next(self._probs)


def _frames(n: int) -> list[np.ndarray]:
    return [np.zeros(FRAME_SIZE, dtype=np.float32) for _ in range(n)]


def _frames_needed(ms: int) -> int:
    return round(ms / FRAME_MS)


class RecordingSegmenter:
    """Wraps SpeechSegmenter, recording every start/end callback invocation."""

    def __init__(self, probs: list[float], **cfg_kwargs):
        self.starts: list[tuple] = []
        self.ends: list[tuple] = []
        vad = FakeVAD(probs)
        cfg = VadConfig(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, **cfg_kwargs)
        self.segmenter = SpeechSegmenter(
            vad, cfg,
            on_speech_start=lambda start_s, audio: self.starts.append((start_s, len(audio))),
            on_speech_end=lambda start_s, dur_s, audio: self.ends.append((start_s, dur_s, len(audio))),
        )

    def feed(self, n: int) -> None:
        for frame in _frames(n):
            self.segmenter.feed(frame)


class TestMinSpeechFrames:
    def test_blip_shorter_than_min_speech_ms_never_starts_a_segment(self):
        min_speech_frames = _frames_needed(200)
        probs = [0.9] * (min_speech_frames - 1) + [0.1] * 20
        rec = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=200,
                                  min_silence_ms=600, speech_pad_ms=0)
        rec.feed(len(probs))
        assert rec.starts == []

    def test_speech_at_exactly_min_speech_ms_starts_a_segment(self):
        min_speech_frames = _frames_needed(200)
        probs = [0.9] * min_speech_frames + [0.1] * 20
        rec = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=200,
                                  min_silence_ms=600, speech_pad_ms=0)
        rec.feed(len(probs))
        assert len(rec.starts) == 1

    def test_raising_min_speech_ms_rejects_a_blip_that_used_to_pass(self):
        # A ~100ms noise burst (breath/click right after real speech ends) that
        # this app's old 100ms default would have opened a segment for...
        probs = [0.9] * _frames_needed(100) + [0.1] * 20
        old_default = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=100,
                                          min_silence_ms=600, speech_pad_ms=0)
        old_default.feed(len(probs))
        assert len(old_default.starts) == 1

        # ...but the new 200ms default rejects outright, so it's never sent to
        # whisper in the first place.
        new_default = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=200,
                                          min_silence_ms=600, speech_pad_ms=0)
        new_default.feed(len(probs))
        assert new_default.starts == []

    def test_silence_never_starts_a_segment(self):
        rec = RecordingSegmenter([0.1] * 50, threshold=0.5, min_speech_ms=200,
                                  min_silence_ms=600, speech_pad_ms=0)
        rec.feed(50)
        assert rec.starts == []


class TestSegmentEnd:
    def test_segment_ends_after_min_silence_ms(self):
        probs = [0.9] * _frames_needed(200) + [0.1] * _frames_needed(600)
        rec = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=200,
                                  min_silence_ms=600, speech_pad_ms=0)
        rec.feed(len(probs))
        assert len(rec.starts) == 1
        assert len(rec.ends) == 1

    def test_brief_dip_below_threshold_does_not_end_segment(self):
        # a dip shorter than min_silence_ms, then speech resumes -- must not
        # be treated as the end of the utterance
        probs = [0.9] * _frames_needed(200) + [0.1] * (_frames_needed(600) - 1) + [0.9] * 5
        rec = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=200,
                                  min_silence_ms=600, speech_pad_ms=0)
        rec.feed(len(probs))
        assert rec.ends == []

    def test_reported_duration_matches_fed_frames(self):
        speech_frames = _frames_needed(200)
        silence_frames = _frames_needed(600)
        probs = [0.9] * speech_frames + [0.1] * silence_frames
        rec = RecordingSegmenter(probs, threshold=0.5, min_speech_ms=200,
                                  min_silence_ms=600, speech_pad_ms=0)
        rec.feed(len(probs))
        assert len(rec.ends) == 1
        _, duration_s, _ = rec.ends[0]
        # kept audio is the speech frames plus the leading part of the silence
        # run up to the min_silence_ms trim -- always <= total fed audio.
        assert 0 < duration_s <= len(probs) * FRAME_MS / 1000
