"""whisper.cpp wrapper: manages a persistent whisper-server process and sends it
speech segments over localhost HTTP.

The server loads the model into VRAM once at startup, so per-segment cost is just
inference plus a local HTTP round trip.
"""
from __future__ import annotations

import io
import os
import socket
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import requests

from audio_capture import SAMPLE_RATE

# whisper.cpp's encoder/decoder threading rarely benefits past ~8 threads
# (memory-bandwidth bound, not core-bound) and using every logical core can
# starve the rest of the app (audio callback, VAD, tray) -- so auto-detected
# thread count is capped, not just os.cpu_count() directly.
MAX_AUTO_THREADS = 8


def _auto_thread_count() -> int:
    return max(1, min(os.cpu_count() or 4, MAX_AUTO_THREADS))


class WhisperServer:
    def __init__(self, server_path: str, model_path: str,
                 host: str = "127.0.0.1", port: int = 8178, threads: int | None = None):
        self.server_path = Path(server_path)
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.threads = threads if threads is not None else _auto_thread_count()
        self.log_path = Path("whisper-server.log")
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self._url = f"http://{host}:{port}/inference"
        # Reused across calls: avoids a fresh TCP/HTTP handshake to the local
        # server on every segment.
        self._session = requests.Session()

    def start(self, timeout_s: float = 30.0) -> None:
        if not self.server_path.exists():
            raise FileNotFoundError(f"whisper-server not found: {self.server_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        # A leftover server from a killed run would answer on this port with a
        # possibly stale model; refuse to start until it is gone.
        try:
            with socket.create_connection((self.host, self.port), timeout=0.3):
                raise RuntimeError(
                    f"Port {self.port} is already in use — a whisper-server from a "
                    f"previous run is still alive. Kill it with: taskkill //F //IM whisper-server.exe")
        except OSError:
            pass

        self._log_file = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            [str(self.server_path),
             "-m", str(self.model_path),
             "--host", self.host,
             "--port", str(self.port),
             "-t", str(self.threads),
             "-l", "en",
             "-bo", "1",  # best-of 1: only matters on temperature-fallback decodes, cheap safety margin
             "-sns",  # suppress non-speech tokens ([BLANK_AUDIO], [Music], etc. at the model level
             "--no-timestamps"],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        # The server only starts listening after the model is loaded, so a
        # successful TCP connect means it is ready for inference.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"whisper-server exited with code {self._proc.returncode}; see whisper-server.log")
            try:
                with socket.create_connection((self.host, self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise TimeoutError(f"whisper-server did not start within {timeout_s}s; see whisper-server.log")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def transcribe(self, audio: np.ndarray, *, prompt: str = "", timeout_s: float = 60.0) -> str:
        """Transcribe a mono float32 16kHz segment; returns the text.

        `prompt` is forwarded as whisper-server's per-request initial-prompt
        field (confirmed supported by reading examples/server/server.cpp) --
        used by streaming mode to give the decoder the words already committed
        so far as textual context across repeated calls on a growing window.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        buf.seek(0)

        data = {"response_format": "json", "temperature": "0.0"}
        if prompt:
            data["prompt"] = prompt

        resp = self._session.post(
            self._url,
            files={"file": ("segment.wav", buf, "audio/wav")},
            data=data,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()
