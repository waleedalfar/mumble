"""Tests for AudioCapture."""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import List

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audio_capture import AudioCapture, find_input_device, FRAME_SIZE, SAMPLE_RATE
from main import _fallback_input_device


class TestAudioCapture:
    def test_stream_restarts_on_error(self):
        call_count = [0]

        def fake_stream(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise sd.PortAudioError("Simulated stream failure")
            mock_stream = MagicMock()
            return mock_stream

        capture = AudioCapture(device=0, max_restarts=5, restart_backoff_s=0.05)
        frames: List[np.ndarray] = []
        with patch.object(sd, "InputStream", side_effect=fake_stream):
            capture.start(on_frame=frames.append)
        assert call_count[0] == 2

    def test_raises_after_max_restarts(self):
        with patch.object(sd, "InputStream", side_effect=sd.PortAudioError("always fails")):
            capture = AudioCapture(device=0, max_restarts=2, restart_backoff_s=0.01)
            with pytest.raises(RuntimeError, match="Audio stream failed after 2 restarts"):
                capture.start(on_frame=lambda f: None)

    def test_default_frame_size(self):
        capture = AudioCapture()
        assert capture.frame_size == FRAME_SIZE

    def test_default_sample_rate(self):
        capture = AudioCapture()
        assert capture.sample_rate == SAMPLE_RATE

    def test_stop_closes_stream(self):
        mock_stream = MagicMock()
        capture = AudioCapture()
        capture._stream = mock_stream
        capture.stop()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert capture._stream is None

    def test_find_input_device_matches_name(self):
        fake_devices = [
            {"name": "Line 1", "max_input_channels": 0, "hostapi": 0},
            {"name": "Mic (AirPods Max)", "max_input_channels": 1, "hostapi": 0},
            {"name": "Output", "max_input_channels": 0, "hostapi": 0},
        ]
        def fake_hostapis(idx):
            return {"name": "Windows WASAPI"}
        with patch.object(sd, "query_devices", return_value=fake_devices):
            with patch.object(sd, "query_hostapis", side_effect=fake_hostapis):
                idx = find_input_device("AirPods Max")
                assert idx == 1

    def test_find_input_device_no_match(self):
        fake_devices = [
            {"name": "Line 1", "max_input_channels": 1, "hostapi": 0},
        ]
        def fake_hostapis(idx):
            return {"name": "Windows WASAPI"}
        with patch.object(sd, "query_devices", return_value=fake_devices):
            with patch.object(sd, "query_hostapis", side_effect=fake_hostapis):
                with pytest.raises(LookupError, match="No input device matching"):
                    find_input_device("nonexistent mic")

    def test_fallback_prefers_mic_named_device(self):
        fake_devices = [
            {"name": "Line 1", "max_input_channels": 0, "hostapi": 0},
            {"name": "Microphone Array", "max_input_channels": 1, "hostapi": 0},
            {"name": "Stereo Mix", "max_input_channels": 1, "hostapi": 0},
        ]
        with patch.object(sd, "query_devices", return_value=fake_devices):
            idx = _fallback_input_device()
            assert fake_devices[idx]["name"] == "Microphone Array"
