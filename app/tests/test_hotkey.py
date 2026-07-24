"""Tests for hotkey spec parsing (no real OS registration -- see GlobalHotkey
for the RegisterHotKey/message-loop side, which needs real Win32 state)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hotkey import HotkeyParseError, MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, _parse_hotkey


class TestParseHotkey:
    def test_single_modifier_and_letter(self):
        assert _parse_hotkey("ctrl+d") == (MOD_CONTROL, ord("D"))

    def test_multiple_modifiers(self):
        mods, vk = _parse_hotkey("ctrl+alt+d")
        assert mods == MOD_CONTROL | MOD_ALT
        assert vk == ord("D")

    def test_all_modifiers(self):
        mods, _ = _parse_hotkey("ctrl+alt+shift+win+d")
        assert mods == MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_WIN

    def test_case_insensitive(self):
        assert _parse_hotkey("CTRL+ALT+D") == _parse_hotkey("ctrl+alt+d")

    def test_digit_key(self):
        _, vk = _parse_hotkey("ctrl+5")
        assert vk == ord("5")

    def test_named_function_key(self):
        _, vk = _parse_hotkey("ctrl+f9")
        assert vk == 0x70 + 8  # F1=0x70, F9 is the 9th

    def test_named_special_key(self):
        _, vk = _parse_hotkey("ctrl+space")
        assert vk == 0x20

    def test_unknown_modifier_raises(self):
        with pytest.raises(HotkeyParseError, match="Unknown modifier"):
            _parse_hotkey("ctrl+bogus+d")

    def test_unknown_key_raises(self):
        with pytest.raises(HotkeyParseError, match="Unknown key"):
            _parse_hotkey("ctrl+bogus")

    def test_empty_spec_raises(self):
        with pytest.raises(HotkeyParseError):
            _parse_hotkey("")

    def test_no_modifier_just_key(self):
        mods, vk = _parse_hotkey("f9")
        assert mods == 0
        assert vk == 0x70 + 8
