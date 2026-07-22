"""Live mic diagnostic for the configured headset: prints a volume bar + VAD speech
probability twice a second.

Usage:  python mic_test.py
"""
import sys
import time

import numpy as np

from audio_capture import AudioCapture, find_input_device
from config import load_config
from vad import SileroVAD

sys.stdout.reconfigure(line_buffering=True)


def main():
    device = find_input_device(load_config().mic_device)

    import sounddevice as sd
    print(f"Testing device [{device}] {sd.query_devices(device)['name']!r}")
    print("Speak now — the bar should jump and 'prob' should go above 0.5 while talking.")
    print("Ctrl+C to stop.\n")

    vad = SileroVAD("../models/silero_vad.onnx")
    frames = []

    def on_frame(frame):
        frames.append(frame)

    cap = AudioCapture(device=device)
    cap.start(on_frame=on_frame)

    total = 0
    try:
        while True:
            time.sleep(0.5)
            batch, frames[:] = frames[:], []
            total += len(batch)
            if not batch:
                print("!! no audio arriving from this device")
                continue
            peak = max(np.abs(f).max() for f in batch)
            prob = max(vad.speech_prob(f) for f in batch)
            bar = "#" * min(50, int(peak * 200))
            flag = "  <-- SPEECH" if prob >= 0.5 else ""
            print(f"peak={peak:.3f} prob={prob:.2f} |{bar:<50}|{flag}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        print(f"\nDone. {total} frames total ({total * 512 / 16000:.1f}s of audio).")


if __name__ == "__main__":
    main()
