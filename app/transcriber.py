"""whisper.cpp wrapper: manages a persistent whisper-server process and sends it
speech segments over localhost HTTP.

The server loads the model into VRAM once at startup, so per-segment cost is just
inference plus a local HTTP round trip.
"""
from __future__ import annotations

import ctypes
import io
import os
import socket
import subprocess
import time
import wave
from ctypes import wintypes
from pathlib import Path

import numpy as np
import requests

from audio_capture import SAMPLE_RATE

# whisper.cpp's encoder/decoder threading rarely benefits past ~8 threads
# (memory-bandwidth bound, not core-bound) and using every logical core can
# starve the rest of the app (audio callback, VAD, tray) -- so auto-detected
# thread count is capped, not just os.cpu_count() directly.
MAX_AUTO_THREADS = 8

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
# default ctypes restype is c_int, which truncates 64-bit handles/pointers on
# Win64 (see injector.py's set_clipboard for the same gotcha) -- these calls
# return/take real HANDLEs, so restype/argtypes must be declared explicitly.
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
_kernel32.SetInformationJobObject.restype = wintypes.BOOL
_kernel32.SetInformationJobObject.argtypes = (
    wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
_kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_void_p),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _make_kill_on_close_job() -> wintypes.HANDLE | None:
    """A Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: the OS
    itself force-kills every process assigned to it the instant this handle
    closes, which happens automatically when this app's process ends --
    however it ends. A `finally`/atexit-based cleanup only runs on a
    cooperative shutdown; it can't run if the app is killed outright (Task
    Manager "End task", a crash, a hard power-off). That's exactly how
    whisper-server.exe used to get orphaned, still holding the port for the
    next run. The job object moves the guarantee into the OS, so there's
    nothing left for our own code to forget to clean up.

    Returns None (caller falls back to best-effort terminate() on exit --
    see WhisperServer.start()'s warning when that happens) if job creation
    fails, which is possible but rare -- e.g. some sandboxed environments
    restrict CreateJobObjectW.
    """
    handle = _kernel32.CreateJobObjectW(None, None)
    if not handle:
        print(f"[whisper-server] CreateJobObjectW failed ({ctypes.WinError(ctypes.get_last_error())}); "
              f"a hard kill of this app won't auto-clean whisper-server.exe.")
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = _kernel32.SetInformationJobObject(
        handle, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        print(f"[whisper-server] SetInformationJobObject failed ({ctypes.WinError(ctypes.get_last_error())}); "
              f"a hard kill of this app won't auto-clean whisper-server.exe.")
        _kernel32.CloseHandle(handle)
        return None
    return handle


def _assign_to_job(job: wintypes.HANDLE, pid: int) -> bool:
    """Best-effort: some environments (older Windows, certain sandboxes) put
    their own process in a job that doesn't allow child processes to break
    away into a different one -- AssignProcessToJobObject then fails, and we
    just fall back to the existing terminate()-on-exit path instead."""
    proc_handle = _kernel32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
    if not proc_handle:
        print(f"[whisper-server] OpenProcess failed ({ctypes.WinError(ctypes.get_last_error())}); "
              f"a hard kill of this app won't auto-clean whisper-server.exe.")
        return False
    try:
        ok = bool(_kernel32.AssignProcessToJobObject(job, proc_handle))
        if not ok:
            print(f"[whisper-server] AssignProcessToJobObject failed "
                  f"({ctypes.WinError(ctypes.get_last_error())}); a hard kill of this app "
                  f"won't auto-clean whisper-server.exe.")
        return ok
    finally:
        _kernel32.CloseHandle(proc_handle)


# One job object for the whole app process: created once, held open for the
# process lifetime, closed automatically by the OS on exit -- see
# _make_kill_on_close_job's docstring for why that's what makes this airtight.
_JOB = _make_kill_on_close_job()


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

        # With the kill-on-close job object below, whisper-server.exe from a
        # previous run should never actually outlive that run, so this is no
        # longer the "an old run leaked" case it used to be -- the two things
        # left that can trip it are the app already running elsewhere, or an
        # environment where job assignment itself isn't permitted (see
        # _make_kill_on_close_job's docstring).
        try:
            with socket.create_connection((self.host, self.port), timeout=0.3):
                raise RuntimeError(
                    f"Port {self.port} is already in use. whisper-server.exe now dies "
                    f"automatically with this app (see the job-object setup at the top "
                    f"of this file), so this almost always means the app is already "
                    f"running in another window -- check for it there first. If it's "
                    f"genuinely an orphaned process, close it with: "
                    f"taskkill //F //IM whisper-server.exe")
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
        if _JOB is not None:
            # Accepted gap: if this app were hard-killed in the few
            # microseconds between Popen() returning and this line running,
            # the child would momentarily escape the job. Closing that fully
            # needs a suspended-at-birth process (raw CreateProcess with
            # CREATE_SUSPENDED, bypassing subprocess.Popen) resumed only
            # after assignment -- not worth the complexity for a window this
            # narrow.
            _assign_to_job(_JOB, self._proc.pid)

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
            # Safe even if the job object already killed this process (or it
            # died on its own): CPython's Popen.terminate() on Windows
            # already catches the resulting PermissionError from
            # TerminateProcess on a dead process and fills in returncode
            # instead of raising.
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
