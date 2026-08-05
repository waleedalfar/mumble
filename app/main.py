"""Background voice dictation: mic -> silero VAD -> whisper.cpp -> SendInput typing.

Run as `python main.py`. All tuning lives in config.yaml at the project root;
the tray icon (right-click) has Pause/Resume, Open config, Reload config, Quit.
"""
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime

# Git Bash (mintty) gives Python a pipe, not a console, so prints are block-buffered
# and appear to hang. Force line buffering so events show up as they happen.
sys.stdout.reconfigure(line_buffering=True)

import sounddevice as sd

from audio_capture import AudioCapture, SAMPLE_RATE, find_input_device
from config import CONFIG_PATH, load_config
from gpu_check import detect_cuda_capable
from hotkey import GlobalHotkey
from injector import press_enter, set_clipboard, type_text
from overlay import StatusOverlay
from simulator import SimulatedCapture
from streaming_transcriber import StreamingSession
from text_filter import clean_transcript
from transcriber import WhisperServer
from tray import TrayUI
from vad import SileroVAD, SpeechSegmenter, VadConfig

VAD_MODEL_PATH = "../models/silero_vad.onnx"
LOG_PATH = "dictation.log"


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def _list_input_devices() -> None:
    print("Available input devices (set 'mic_device' in config.yaml to a name substring):")
    for dev in sd.query_devices():
        if dev["max_input_channels"] > 0:
            print(f"  {dev['name']}")


def _fallback_input_device() -> int:
    best = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        if "Microphone" in dev["name"] or "Mic" in dev["name"]:
            return idx
        if best is None:
            best = idx
    return best if best is not None else 0


class DictationApp:
    def __init__(self):
        self.cfg = load_config()
        self.paused = False
        self.segments: queue.Queue = queue.Queue()
        self.log = open(LOG_PATH, "a", encoding="utf-8")
        self.server: WhisperServer | None = None
        self.capture: AudioCapture | SimulatedCapture | None = None
        self.segmenter: SpeechSegmenter | None = None
        self.hotkey: GlobalHotkey | None = None
        self.overlay: StatusOverlay | None = None
        self.streaming_active = False
        self._streaming_session: StreamingSession | None = None
        self.tray = TrayUI(self.toggle_pause, self.open_config, self.reload_config, self.quit)
        self._lock = threading.Lock()  # guards server/capture swaps during reload
        # Serializes SendInput/clipboard calls: with streaming, _deliver can be
        # called from the transcribe worker (final text) AND a StreamingSession's
        # own tick thread (partial commits) at the same time -- two concurrent
        # SendInput callers could in principle interleave at the OS input-queue
        # level. (The actual words-running-together bug turned out to be
        # injector._sanitize()'s rstrip() eating the deliberate trailing
        # separator space on every delivery, not this race -- see injector.py.
        # This lock is still worth keeping as real, separate protection against
        # genuinely concurrent SendInput calls.)
        self._deliver_lock = threading.Lock()

    # ---- pipeline lifecycle ----------------------------------------------

    def start_pipeline(self):
        cfg = self.cfg

        if cfg.overlay.enabled:
            self.overlay = StatusOverlay(position=cfg.overlay.position)
            self.overlay.start()
        else:
            self.overlay = None

        if cfg.hotkey.enabled:
            self.hotkey = GlobalHotkey(cfg.hotkey.toggle_pause, self.toggle_pause)
            try:
                self.hotkey.start()
                print(f"Global hotkey active: {cfg.hotkey.toggle_pause!r} toggles pause/resume.")
            except OSError as e:
                print(f"  !! hotkey registration failed: {e}")
                self.hotkey = None
        else:
            self.hotkey = None

        print("Starting whisper-server (loading model into VRAM)...")
        self.server = WhisperServer(str(cfg.whisper_server), str(cfg.whisper_model),
                                    port=cfg.server_port)
        self.server.start()
        print(f"whisper-server ready (model: {cfg.whisper_model.name}).")

        gpu_capable, reason = detect_cuda_capable(self.server.log_path)
        setting = cfg.streaming.enabled
        if setting == "false":
            self.streaming_active = False
        elif setting == "auto":
            self.streaming_active = gpu_capable
            print(f"Streaming: {'auto-enabled' if gpu_capable else 'auto-disabled'} ({reason}).")
        else:  # "true": still a hard ceiling -- never allow it on hardware that can't handle it
            self.streaming_active = gpu_capable
            if gpu_capable:
                print(f"Streaming: enabled ({reason}).")
            else:
                print(f"Streaming: forced OFF -- streaming.enabled=true in config, but {reason}.")

        vad = SileroVAD(VAD_MODEL_PATH)
        vad_cfg = VadConfig(threshold=cfg.vad.threshold,
                            min_speech_ms=cfg.vad.min_speech_ms,
                            min_silence_ms=cfg.vad.min_silence_ms,
                            speech_pad_ms=cfg.vad.speech_pad_ms)
        self.segmenter = SpeechSegmenter(vad, vad_cfg, self._on_speech_start, self._on_speech_end)

        sim_files = cfg.simulate.files
        if sim_files:
            self.capture = SimulatedCapture(sim_files, loop=cfg.simulate.loop)
            self.capture.start(on_frame=self._on_frame)
            loop_flag = " (loop)" if cfg.simulate.loop else ""
            print(f"Simulation mode: {len(sim_files)} file(s){loop_flag}.")
        else:
            if not cfg.mic_device:
                _list_input_devices()
                raise SystemExit(f"\nSet 'mic_device' in {CONFIG_PATH} or use simulate mode.")
            try:
                device = find_input_device(cfg.mic_device)
                print(f"Using mic device [{device}].")
            except LookupError as e:
                if not cfg.mic_fallback:
                    # No fallback to fall back to -- the full device list in
                    # `e` is exactly what the user needs to pick one by hand.
                    raise SystemExit(str(e)) from e
                device = _fallback_input_device()
                print(f"[mic] configured device not found (not connected yet?); "
                      f"using device [{device}] instead.")
            self.capture = AudioCapture(device=device)
            self.capture.start(on_frame=self._on_frame)

    def stop_pipeline(self):
        if self._streaming_session is not None:
            self._streaming_session.stop()
            self._streaming_session = None
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None
        if self.overlay is not None:
            self.overlay.stop()
            self.overlay = None

    # ---- audio path -------------------------------------------------------

    def _set_state(self, state: str):
        self.tray.set_state(state)
        if self.overlay is not None:
            self.overlay.set_state(state)

    def _on_frame(self, frame):
        if self.paused or self.segmenter is None:
            return
        # capture whether a session already existed *before* this feed() call:
        # if feed() synchronously creates a new one (via on_speech_start), its
        # initial_audio already includes this frame through the VAD's own
        # pre-roll, so adding it again here would duplicate it.
        session_before = self._streaming_session
        self.segmenter.feed(frame)
        session = self._streaming_session
        if session is not None and session is session_before:
            session.add_frame(frame)

    def _on_speech_start(self, start_s: float, initial_audio):
        self._set_state("listening")
        if self.streaming_active:
            session = StreamingSession(
                self._streaming_transcribe,
                self.cfg.streaming.step_ms,
                self.cfg.streaming.max_window_s,
                self.cfg.streaming.stability_confirmations,
                self._on_stream_commit,
                initial_audio,
                SAMPLE_RATE,
            )
            session.start()
            self._streaming_session = session

    def _on_speech_end(self, start_s: float, duration_s: float, audio):
        self._set_state("transcribing")
        session = self._streaming_session
        self._streaming_session = None
        if session is not None:
            session.stop()
        self.segments.put((time.monotonic(), duration_s, audio, session))

    def _streaming_transcribe(self, audio, prompt: str = "", timeout_s: float = 8.0) -> str:
        with self._lock:
            server = self.server
        if server is None:
            raise RuntimeError("no whisper-server (pipeline stopped)")
        return server.transcribe(audio, prompt=prompt, timeout_s=timeout_s)

    def _on_stream_commit(self, words: list[str]):
        # Reuses the same delivery path as a finished utterance (typing/
        # clipboard, enter-phrase check). Known accepted risk: a short
        # configured enter_phrase could in principle match a partial commit
        # before the sentence actually ends -- see README.
        text = clean_transcript(" ".join(words))
        if text:
            print(f'   ~ "{text}"')
            self._deliver(text)

    # ---- transcription worker --------------------------------------------

    def transcribe_worker(self):
        while True:
            item = self.segments.get()
            if item is None:
                return
            queued_at, duration_s, audio, session = item
            try:
                text = self._transcribe_with_recovery(audio)
            except Exception as e:
                print(f"  !! transcription failed: {e}")
                self._set_state("idle")
                continue
            latency_ms = (time.monotonic() - queued_at) * 1000
            if session is not None:
                tail_words = session.finalize(text)
                clean = clean_transcript(" ".join(tail_words)) if tail_words else ""
            else:
                clean = clean_transcript(text)
            if clean:
                print(f'>> "{text}"  ({duration_s:.1f}s audio, latency {latency_ms:.0f}ms)')
                self._deliver(clean)
            else:
                print(f'>> "{text}"  ({duration_s:.1f}s audio, latency {latency_ms:.0f}ms) '
                      f'[suppressed: non-speech]')
            self.log.write(f"{datetime.now():%H:%M:%S} dur={duration_s:.2f}s "
                           f"latency={latency_ms:.0f}ms text={text!r}\n")
            self.log.flush()
            if self.segments.empty():
                self._set_state("idle")

    def _transcribe_with_recovery(self, audio) -> str:
        """Transcribe a segment; if the server has died (killed, crashed), relaunch
        it once and retry so a dead server doesn't take down the whole session."""
        import requests as _requests
        with self._lock:
            server = self.server
        if server is None:
            raise RuntimeError("no whisper-server (pipeline stopped)")
        try:
            return server.transcribe(audio)
        except _requests.exceptions.ConnectionError:
            print("  !! whisper-server is gone — relaunching it...")
            with self._lock:
                server.stop()
                server.start()
            print("  whisper-server back up.")
            return server.transcribe(audio)

    def _deliver(self, text: str):
        if not text:
            return
        out = self.cfg.output
        with self._deliver_lock:
            try:
                if out.mode == "type":
                    if _normalize_phrase(text) in {_normalize_phrase(p) for p in self.cfg.enter_phrases}:
                        press_enter()
                        print("   (enter pressed)")
                    else:
                        type_text(text + " ", batch_chars=out.batch_chars,
                                  inter_batch_delay_s=out.inter_batch_delay_ms / 1000)
                else:
                    set_clipboard(text)
                    print("   (copied to clipboard)")
            except OSError as e:
                print(f"  !! output failed: {e}")

    # ---- tray callbacks ---------------------------------------------------

    def toggle_pause(self):
        self.paused = not self.paused
        self.tray.set_paused(self.paused)
        if self.overlay is not None:
            self.overlay.set_state("paused" if self.paused else "idle")
        print("Paused." if self.paused else "Resumed.")

    def open_config(self):
        os.startfile(CONFIG_PATH)

    def reload_config(self):
        # Runs on its own thread: a reload restarts whisper-server (seconds), and
        # tray menu callbacks must never block the tray's message loop.
        threading.Thread(target=self._do_reload, daemon=True).start()

    def _do_reload(self):
        print("Reloading config...")
        try:
            new_cfg = load_config()
        except SystemExit as e:
            print(f"  !! config invalid, keeping old settings: {e}")
            return
        with self._lock:
            self.stop_pipeline()
            self.cfg = new_cfg
            try:
                self.start_pipeline()
            except (SystemExit, Exception) as e:
                print(f"  !! restart failed: {e}")
                return
        if not self.paused:
            self._set_state("idle")
        print("Config reloaded.")

    def quit(self):
        self._quit_event.set()

    # ---- run --------------------------------------------------------------

    def run(self):
        self._quit_event = threading.Event()
        self.start_pipeline()
        worker = threading.Thread(target=self.transcribe_worker, daemon=True)
        worker.start()

        # Tray runs on a background thread so the main thread stays responsive
        # to Ctrl+C in the console.
        tray_thread = threading.Thread(target=self.tray.run, daemon=True)
        tray_thread.start()
        self._set_state("idle")
        print("Listening, tray icon active near your clock (right-click to pause/quit; Ctrl+C here also quits).")
        try:
            while not self._quit_event.wait(timeout=0.2):
                pass
        except KeyboardInterrupt:
            print("\nCtrl+C received, shutting down...")
        finally:
            self.tray.stop()
            self.stop_pipeline()
            self.segments.put(None)
            worker.join(timeout=5)
            self.log.close()
            print("Stopped.")


if __name__ == "__main__":
    DictationApp().run()
