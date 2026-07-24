"""Global hotkey via Win32 RegisterHotKey (ctypes), not a keyboard-hook library.

RegisterHotKey asks the OS to deliver a WM_HOTKEY message only when the exact
combination is pressed -- it costs nothing on every other keystroke, unlike a
low-level keyboard hook (e.g. the `keyboard` pip package), which runs a
callback for every key event system-wide to check for a match. The tradeoff
is a smaller feature set: this only supports modifier+single-key combos, no
key-up detection or multi-key chords, which is all a pause/resume toggle needs.
"""
from __future__ import annotations

import ctypes
import re
import threading
from ctypes import wintypes

user32 = ctypes.windll.user32

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312

_MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT,
              "shift": MOD_SHIFT, "win": MOD_WIN, "super": MOD_WIN}

_NAMED_KEYS = {"space": 0x20, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
               "esc": 0x1B, "escape": 0x1B}
_NAMED_KEYS.update({f"f{i}": 0x70 + i - 1 for i in range(1, 13)})


class HotkeyParseError(ValueError):
    pass


def _parse_hotkey(spec: str) -> tuple[int, int]:
    """Parse "ctrl+alt+d" into (modifiers, virtual_key_code)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise HotkeyParseError(f"Empty hotkey spec: {spec!r}")
    *mod_names, key_name = parts
    modifiers = 0
    for name in mod_names:
        if name not in _MODIFIERS:
            raise HotkeyParseError(f"Unknown modifier {name!r} in hotkey {spec!r}")
        modifiers |= _MODIFIERS[name]

    if key_name in _NAMED_KEYS:
        vk = _NAMED_KEYS[key_name]
    elif len(key_name) == 1 and (key_name.isalnum()):
        vk = ord(key_name.upper())  # VK codes for '0'-'9'/'A'-'Z' match ASCII
    else:
        raise HotkeyParseError(f"Unknown key {key_name!r} in hotkey {spec!r}")
    return modifiers, vk


class GlobalHotkey:
    """Registers one system-wide hotkey and calls back on a dedicated thread."""

    _HOTKEY_ID = 1

    def __init__(self, spec: str, callback):
        self.spec = spec
        self.modifiers, self.vk = _parse_hotkey(spec)
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self._registration_error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._registration_error is not None:
            raise self._registration_error

    def stop(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        ok = user32.RegisterHotKey(None, self._HOTKEY_ID, self.modifiers | MOD_NOREPEAT, self.vk)
        if not ok:
            self._registration_error = OSError(
                f"RegisterHotKey failed for {self.spec!r} (already in use by another app?)")
            self._ready.set()
            return
        self._ready.set()

        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam == self._HOTKEY_ID:
                    try:
                        self._callback()
                    except Exception as e:
                        print(f"[hotkey] callback error: {e}")
        finally:
            user32.UnregisterHotKey(None, self._HOTKEY_ID)
