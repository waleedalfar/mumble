"""Simulated microphone capture from WAV files for offline testing.

Reads 16-bit mono WAVs at 16 kHz, splits them into FRAME_SIZE frames, and
delivers them through the same on_frame callback used by AudioCapture, so
SpeechSegmenter sees identical input whether it comes from a mic or a file.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from audio_capture import FRAME_SIZE, SAMPLE_RATE


class SimulatedCapture:
    """Play back one or more WAV files as if they were live microphone frames."""

    def __init__(self, files: list[str | Path], *,
                 loop: bool = False,
                 sample_rate: int = SAMPLE_RATE,
                 frame_size: int = FRAME_SIZE) -> None:
        self._files = [Path(f) for f in files]
        self._loop = loop
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self._on_frame: Optional[Callable[[np.ndarray], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        self._on_frame = on_frame
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
            self._stop_event = None

    def _run(self) -> None:
        assert self._on_frame is not None
        frame_samples = self._frame_size
        frame_dur = frame_samples / self._sample_rate

        while not self._stop_event.is_set():
            for path in self._files:
                if not path.exists():
                    print(f"[simulator] missing file: {path}")
                    continue
                try:
                    with wave.open(str(path), "rb") as wf:
                        if wf.getnchannels() != 1:
                            print(f"[simulator] skipping non-mono file: {path}")
                            continue
                        if wf.getsampwidth() != 2:
                            print(f"[simulator] skipping non-16-bit file: {path}")
                            continue
                        if wf.getframerate() != self._sample_rate:
                            print(f"[simulator] skipping {self._sample_rate} Hz file: {path}")
                            continue
                        self._play(wf, frame_samples, frame_dur)
                except wave.Error as e:
                    print(f"[simulator] bad wav {path}: {e}")
                    continue
                if self._stop_event.is_set():
                    return
            if not self._loop:
                print("[simulator] done.")
                return

    def _play(self, wf: wave.Wave_read, frame_samples: int, frame_dur: float) -> None:
        while not self._stop_event.is_set():
            frames = wf.readframes(frame_samples)
            if not frames:
                break
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < frame_samples:
                padded = np.zeros(frame_samples, dtype=np.float32)
                padded[: len(audio)] = audio
                audio = padded
            self._on_frame(audio)
            self._stop_event.wait(timeout=frame_dur)
        if self._loop:
            wf.rewind()


import threading
