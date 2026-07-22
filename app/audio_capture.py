"""Continuous microphone capture, delivered as fixed-size frames on a callback."""
from __future__ import annotations

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
    """Streams mono float32 audio from the mic in fixed FRAME_SIZE chunks."""

    def __init__(self, device: Optional[int] = None, sample_rate: int = SAMPLE_RATE,
                 frame_size: int = FRAME_SIZE):
        self.device = device
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._stream: Optional[sd.InputStream] = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        def _callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] status: {status}")
            on_frame(indata[:, 0].copy())

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

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
