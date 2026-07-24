"""Types text into the currently focused window via Windows SendInput (ctypes).

Uses KEYEVENTF_UNICODE so any character can be sent regardless of keyboard layout,
including punctuation and non-ASCII. Never sends Enter/Tab/control keys: in the
target apps (Claude chat, terminals) those would submit/execute, so newlines in
transcripts are converted to spaces before injection.
"""
from __future__ import annotations

import ctypes
import re
import time
from ctypes import wintypes

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_SendInput = ctypes.windll.user32.SendInput
_SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_SendInput.restype = wintypes.UINT


VK_RETURN = 0x0D


def press_enter() -> None:
    """Send a real Enter keypress. Only called for explicit spoken commands,
    never as part of transcript text."""
    events = (INPUT * 2)()
    events[0].type = INPUT_KEYBOARD
    events[0].ki = KEYBDINPUT(VK_RETURN, 0, 0, 0, 0)
    events[1].type = INPUT_KEYBOARD
    events[1].ki = KEYBDINPUT(VK_RETURN, 0, KEYEVENTF_KEYUP, 0, 0)
    sent = _SendInput(2, events, ctypes.sizeof(INPUT))
    if sent != 2:
        raise OSError(f"SendInput injected {sent}/2 events")


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def set_clipboard(text: str) -> None:
    """Clipboard-only output mode: place text on the clipboard instead of typing."""
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    # default ctypes restype is c_int, which truncates 64-bit handles/pointers
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE
    data = text.encode("utf-16-le") + b"\x00\x00"
    if not user32.OpenClipboard(None):
        raise OSError("OpenClipboard failed")
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        ptr = kernel32.GlobalLock(handle)
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise OSError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()


def _sanitize(text: str) -> str:
    # No Enter (submits in Claude chat, executes in terminals), no tabs (focus jumps
    # / autocomplete in browsers); collapse the result to single spaces.
    #
    # Only lstrip, never rstrip: callers deliberately pass a trailing space
    # (e.g. main.py's `text + " "`) to separate consecutive deliveries --
    # rstrip() was eating that exact space on every single call, which barely
    # showed with one delivery per sentence but became "every word runs
    # together" once streaming started delivering one word at a time.
    return re.sub(r"\s+", " ", text).lstrip()


def type_text(text: str, batch_chars: int = 32, inter_batch_delay_s: float = 0.005) -> None:
    """Type `text` into the focused window.

    Sent in small batches with a short delay between them: some apps (browsers
    with heavy pages, terminals under load) drop events from very large single
    SendInput bursts. Batch size and delay are tuning knobs.
    """
    text = _sanitize(text)
    if not text:
        return

    units = text.encode("utf-16-le")  # handles surrogate pairs for non-BMP chars
    code_units = [int.from_bytes(units[i:i + 2], "little") for i in range(0, len(units), 2)]

    for start in range(0, len(code_units), batch_chars):
        batch = code_units[start:start + batch_chars]
        events = (INPUT * (len(batch) * 2))()
        for i, cu in enumerate(batch):
            down = events[2 * i]
            down.type = INPUT_KEYBOARD
            down.ki = KEYBDINPUT(0, cu, KEYEVENTF_UNICODE, 0, 0)
            up = events[2 * i + 1]
            up.type = INPUT_KEYBOARD
            up.ki = KEYBDINPUT(0, cu, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)
        sent = _SendInput(len(events), events, ctypes.sizeof(INPUT))
        if sent != len(events):
            raise OSError(f"SendInput injected {sent}/{len(events)} events "
                          f"(blocked by a higher-integrity window?)")
        if start + batch_chars < len(code_units):
            time.sleep(inter_batch_delay_s)
