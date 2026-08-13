"""System tray icon with three states (idle / listening / transcribing) plus paused.

Right-click menu: Pause-Resume, Open config, Reload config, Quit. No other UI.
"""
from __future__ import annotations

from typing import Callable

import pystray
from PIL import Image, ImageDraw

_COLORS = {
    "idle": (150, 150, 150),
    "listening": (60, 200, 90),
    "transcribing": (255, 160, 40),
    "paused": (70, 70, 70),
}


def _dot(color) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color + (255,))
    return img


class TrayUI:
    def __init__(self, on_toggle_pause: Callable[[], None],
                 on_open_config: Callable[[], None],
                 on_reload_config: Callable[[], None],
                 on_quit: Callable[[], None]):
        self._icons = {state: _dot(c) for state, c in _COLORS.items()}
        self._paused = False

        menu = pystray.Menu(
            pystray.MenuItem(lambda item: "Resume" if self._paused else "Pause",
                             lambda: on_toggle_pause()),
            pystray.MenuItem("Open config", lambda: on_open_config()),
            pystray.MenuItem("Reload config", lambda: on_reload_config()),
            pystray.MenuItem("Quit", lambda: on_quit()),
        )
        self._icon = pystray.Icon("mumble", self._icons["idle"], "Mumble: starting", menu)

    def set_state(self, state: str) -> None:
        if self._paused and state != "paused":
            return
        self._icon.icon = self._icons[state]
        self._icon.title = f"Mumble: {state}"

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._icon.icon = self._icons["paused" if paused else "idle"]
        self._icon.title = "Mumble: paused" if paused else "Mumble: idle"
        self._icon.update_menu()

    def run(self) -> None:
        """Blocks until stop() is called (run on the main thread)."""
        self._icon.run()

    def stop(self) -> None:
        self._icon.stop()
