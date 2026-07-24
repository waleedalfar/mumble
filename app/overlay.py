"""Small always-on-top status bar (Zoom-style "Listening..." indicator).

Built with tkinter (stdlib, no new dependency) so it's just a tiny label in a
borderless, always-on-top window. Two things matter more than looks here:
  - it must never be able to take keyboard focus (WS_EX_NOACTIVATE), or a
    dictated segment could get typed into it instead of your target window
  - it must be cheap: no animation loop, redraws only when the state changes
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from ctypes import windll

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000

_STYLES = {
    "idle": ("Idle", "#9a9a9a"),
    "listening": ("Listening...", "#3ec85a"),
    "transcribing": ("Transcribing...", "#ffa028"),
    "paused": ("Paused", "#c83c3c"),
}

_MARGIN = 24
_WIDTH, _HEIGHT = 170, 34


class StatusOverlay:
    def __init__(self, position: str = "bottom-right"):
        self.position = position
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        # Routed through the same queue `poll()` already drains on the Tk
        # thread, rather than calling tkinter methods directly from this
        # (foreign) thread, which Tcl's async-event handling isn't safe for.
        self._queue.put("__stop__")
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def set_state(self, state: str) -> None:
        if state not in _STYLES:
            return
        self._queue.put(state)

    # ---- Tk thread ----------------------------------------------------

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="#202020")

        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y = self._corner_xy(screen_w, screen_h)
        root.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")

        canvas = tk.Canvas(root, width=_WIDTH, height=_HEIGHT, bg="#202020",
                          highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        dot = canvas.create_oval(12, _HEIGHT // 2 - 6, 24, _HEIGHT // 2 + 6, fill="#9a9a9a", width=0)
        label = canvas.create_text(34, _HEIGHT // 2, text="Idle", fill="white",
                                   anchor="w", font=("Segoe UI", 10))

        root.update_idletasks()
        self._make_noactivate(root)

        def poll():
            try:
                while True:
                    state = self._queue.get_nowait()
                    if state == "__stop__":
                        root.quit()
                        return
                    text, color = _STYLES[state]
                    canvas.itemconfig(dot, fill=color)
                    canvas.itemconfig(label, text=text)
            except queue.Empty:
                pass
            root.after(50, poll)

        root.after(50, poll)
        self._ready.set()
        try:
            root.mainloop()
        except Exception:
            pass

    def _corner_xy(self, screen_w: int, screen_h: int) -> tuple[int, int]:
        top = self.position.startswith("top")
        left = self.position.endswith("left")
        x = _MARGIN if left else screen_w - _WIDTH - _MARGIN
        y = _MARGIN if top else screen_h - _HEIGHT - _MARGIN - 48  # clear the taskbar
        return x, y

    @staticmethod
    def _make_noactivate(root: tk.Tk) -> None:
        hwnd = windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
