"""Tests for SimulatedCapture."""
from __future__ import annotations

import io
import math
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator import SimulatedCapture


def _make_wav(samples: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        clipped = np.clip(samples, -1.0, 1.0)
        wf.writeframes((clipped * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf


@pytest.fixture()
def sine_wav(tmp_path):
    samples = np.sin(2 * math.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    path = tmp_path / "sine.wav"
    path.write_bytes(_make_wav(samples, 16000).read())
    return path


class TestSimulatedCapture:
    def test_captures_frames(self, sine_wav):
        cap = SimulatedCapture([str(sine_wav)])
        frames = []
        cap.start(on_frame=frames.append)
        import time
        time.sleep(1.5)
        cap.stop()
        total_samples = sum(len(f) for f in frames)
        assert total_samples == 16384
        assert len(frames) == 32

    def test_frame_size(self, sine_wav):
        cap = SimulatedCapture([str(sine_wav)])
        frames = []
        cap.start(on_frame=frames.append)
        cap.stop()
        for f in frames:
            assert len(f) == cap._frame_size

    def test_missing_file(self, tmp_path, capsys):
        cap = SimulatedCapture([str(tmp_path / "nonexistent.wav")])
        frames = []
        cap.start(on_frame=frames.append)
        cap.stop()
        assert frames == []
        captured = capsys.readouterr()
        assert "missing file" in captured.out

    def test_stop_halts_playback(self, sine_wav):
        cap = SimulatedCapture([str(sine_wav)], loop=True)
        frames = []
        cap.start(on_frame=frames.append)
        import time
        time.sleep(0.1)
        cap.stop()
        assert len(frames) < 16000

    def test_multiple_files(self, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        a.write_bytes(_make_wav(np.ones(16000, dtype=np.float32)).read())
        b.write_bytes(_make_wav(np.zeros(16000, dtype=np.float32)).read())
        cap = SimulatedCapture([str(a), str(b)])
        frames = []
        cap.start(on_frame=frames.append)
        import time
        time.sleep(2.5)
        cap.stop()
        assert sum(len(f) for f in frames) == 32768

    def test_skips_non_mono(self, tmp_path, capsys):
        path = tmp_path / "stereo.wav"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
        path.write_bytes(buf.getvalue())
        cap = SimulatedCapture([str(path)])
        frames = []
        cap.start(on_frame=frames.append)
        cap.stop()
        assert frames == []
        assert "non-mono" in capsys.readouterr().out

    def test_skips_non_16bit(self, tmp_path, capsys):
        path = tmp_path / "8bit.wav"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1600, dtype=np.uint8).tobytes())
        path.write_bytes(buf.getvalue())
        cap = SimulatedCapture([str(path)])
        frames = []
        cap.start(on_frame=frames.append)
        cap.stop()
        assert frames == []
        assert "non-16-bit" in capsys.readouterr().out

    def test_skips_wrong_rate(self, tmp_path, capsys):
        path = tmp_path / "8k.wav"
        path.write_bytes(_make_wav(np.zeros(16000, dtype=np.float32), 8000).read())
        cap = SimulatedCapture([str(path)])
        frames = []
        cap.start(on_frame=frames.append)
        cap.stop()
        assert frames == []
        assert "Hz file" in capsys.readouterr().out
