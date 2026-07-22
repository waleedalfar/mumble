"""Continuous microphone capture, delivered as fixed-size frames on a callback."""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32ms @ 16kHz — matches the silero-vad v5 model's expected chunk size


def find_input_device(name_substring: str) -> int:
    """Resolve a mic by name substring, preferring the WASAPI endpoint.

    Device indices shift whenever audio devices connect/disconnect, so devices
    must be selected by name at startup, never by a stored index.
    """
    matches = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name_substring.lower() in dev["name"].lower():
            api = sd.query_hostapis(dev["hostapi"])["name"]
            matches.append((idx, api))
    if not matches:
        raise LookupError(f"No input device matching {name_substring!r}. Available:\n"
                          + "\n".join(f"  {d['name']}" for d in sd.query_devices()
                                      if d["max_input_channels"] > 0))
    for idx, api in matches:
        if api == "Windows WASAPI":
            return idx
    return matches[0][0]


class AudioCapture:
    """Streams mono float32 audio from the mic in fixed FRAME_SIZE chunks.

    Automatically recovers from transient PortAudio errors (for example when
    a USB/Bluetooth device is disconnected and reconnected) by restarting the
    stream instead of crashing the app.
    """

    def __init__(self, device: Optional[int] = None, sample_rate: int = SAMPLE_RATE,
                 frame_size: int = FRAME_SIZE,
                 max_restarts: int = 10, restart_backoff_s: float = 0.5) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._max_restarts = max_restarts
        self._restart_backoff_s = restart_backoff_s
        self._stream: Optional[sd.InputStream] = None
        self._on_frame: Optional[Callable[[np.ndarray], None]] = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        self._on_frame = on_frame
        restarts = 0
        while True:
            try:
                self._start_stream()
                return
            except Exception as e:
                if restarts >= self._max_restarts:
                    raise RuntimeError(
                        f"Audio stream failed after {restarts} restarts: {e}") from e
                print(f"[audio] stream error ({e}); restarting in "
                      f"{self._restart_backoff_s}s ({restarts + 1}/{self._max_restarts})...")
                self._stop_stream()
                time.sleep(self._restart_backoff_s)
                restarts += 1

    def stop(self) -> None:
        self._stop_stream()

    def _start_stream(self) -> None:
        assert self._on_frame is not None

        def _callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] status: {status}")
            try:
                self._on_frame(indata[:, 0].copy())
            except Exception as e:
                print(f"[audio] callback error: {e}")

        extra_settings = None
        try:
            hostapi = sd.query_devices(self.device, kind="input")["hostapi"]
            if sd.query_hostapis(hostapi)["name"] == "Windows WASAPI":
                extra_settings = sd.WasapiSettings(auto_convert=True)
        except Exception:
            pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            device=self.device,
            callback=_callback,
            extra_settings=extra_settings,
        )
        self._stream.start()

    def _stop_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
