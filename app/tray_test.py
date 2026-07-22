"""Minimal pystray check: shows a bright red tray icon for 15 seconds, then exits.

Usage: python tray_test.py
While it runs, look at the taskbar clock area AND the ^ overflow popup.
"""
import sys
import threading
import time

import pystray
from PIL import Image, ImageDraw

sys.stdout.reconfigure(line_buffering=True)

img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
ImageDraw.Draw(img).ellipse((8, 8, 56, 56), fill=(230, 40, 40, 255))

icon = pystray.Icon("tray_test", img, "TRAY TEST — red dot",
                    pystray.Menu(pystray.MenuItem("Quit", lambda: icon.stop())))

threading.Thread(target=icon.run, daemon=True).start()
print("Red test icon should now be visible near the clock or in the ^ overflow.")
for i in range(15, 0, -1):
    print(f"  ...{i}s remaining")
    time.sleep(1)
icon.stop()
print("Done. Did you see the red dot?")
