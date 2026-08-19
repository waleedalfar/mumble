"""Config loader. The config file lives at the project root (config.yaml), is created
with commented defaults on first run, and is fully hot-editable: every tuning knob
(model, VAD, injection pacing, output mode) is here, never in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG = """\
# Mumble config. Edit freely; use the tray menu's "Reload config" to apply
# without restarting the app. Relative paths are resolved from the project root.

# Any ggml .bin dropped into models/ can be used here.
whisper_model: models/ggml-small.en.bin
whisper_server: whisper.cpp/build/bin/Release/whisper-server.exe
server_port: 8178

# Input device, matched by name substring. Leave empty ("") to print the available
# devices at startup and exit.
mic_device: ""
mic_fallback: true

vad:
  threshold: 0.5          # speech probability needed to count a frame as speech (0-1)
  min_speech_ms: 200      # this much consecutive speech starts a segment (rejects blips, e.g. a
                           # breath/click right after you stop talking that whisper can hallucinate
                           # a stock phrase like "Thank you." from -- see text_filter.py)
  min_silence_ms: 600     # this much consecutive silence ends a segment (lower = snappier, may split sentences)
  speech_pad_ms: 300      # audio kept before/after the detected speech (protects word edges)

# whisper often ends a sentence with a period at an ordinary clause pause,
# not just a real sentence break. A period found mid-utterance is downgraded
# to a comma (and the next word's capital lowered) unless the measured
# silence before it was at least this long. Keep this comfortably below
# vad.min_silence_ms above -- no pause inside one utterance can ever reach
# that value, or VAD would have ended the utterance there instead, so a
# threshold at or above it would downgrade every mid-utterance period
# unconditionally.
punctuation:
  pause_threshold_ms: 350

output:
  mode: type              # "type" = inject into focused window, "clipboard" = copy only
  batch_chars: 32         # characters per SendInput burst
  inter_batch_delay_ms: 5 # pause between bursts (raise if characters get dropped)

# If a whole utterance transcribes to exactly one of these phrases (case and
# punctuation ignored), the Enter key is pressed instead of typing the words.
# Only active in "type" mode.
enter_phrases:
  - press enter

# Global keyboard shortcut to pause/resume listening, from anywhere (works
# even when a different app has focus). Modifiers: ctrl, alt, shift, win --
# combine with + and a single letter/digit/F-key, e.g. "ctrl+alt+d", "f9".
# ctrl+f9 was picked as the default because F-keys are rarely bound by
# browsers/editors, unlike letter combos (ctrl+d = bookmark, alt+d = address
# bar, etc.) -- see README for other low-collision options if it clashes
# with something on your machine.
hotkey:
  enabled: true
  toggle_pause: "ctrl+f9"

# Small always-on-top status bar (like Zoom's recording indicator) showing
# idle/listening/transcribing/paused. position: bottom-right, bottom-left,
# top-right, or top-left.
overlay:
  enabled: true
  position: bottom-right

# Continuous "words appear as you talk" mode, instead of waiting for the
# whole sentence to finish. This re-transcribes the in-progress utterance
# every step_ms and types newly-confirmed words as they stabilize; a word
# that's already been typed can NEVER be corrected later, even if more
# context would have transcribed it differently -- see README for the full
# trade-off before turning this on.
#
# enabled: auto  -- turns on only if whisper-server reports a capable CUDA
#                   GPU at startup (see gpu_check.py); off otherwise. This is
#                   a hard ceiling: even "true" below is downgraded to off on
#                   an incapable machine, never silently left slow.
# enabled: true  -- same hard GPU check as "auto"; expresses intent to use it
#                   whenever hardware allows.
# enabled: false -- always off, regardless of hardware. Recommended for
#                   CPU-only machines: the repeated re-transcription adds
#                   real, constant load a GPU absorbs for free.
streaming:
  enabled: auto
  step_ms: 600
  max_window_s: 10.0
  stability_confirmations: 2
"""


@dataclass
class VadSettings:
    threshold: float = 0.5
    min_speech_ms: int = 200
    min_silence_ms: int = 600
    speech_pad_ms: int = 300


@dataclass
class PunctuationSettings:
    pause_threshold_ms: int = 350


@dataclass
class OutputSettings:
    mode: str = "type"
    batch_chars: int = 32
    inter_batch_delay_ms: int = 5


@dataclass
class SimulateSettings:
    files: list = field(default_factory=list)
    loop: bool = False


@dataclass
class HotkeySettings:
    enabled: bool = True
    toggle_pause: str = "ctrl+f9"


@dataclass
class OverlaySettings:
    enabled: bool = True
    position: str = "bottom-right"


@dataclass
class StreamingSettings:
    enabled: str = "auto"  # "auto" | "true" | "false" -- resolved against GPU capability at runtime
    step_ms: int = 600
    max_window_s: float = 10.0
    stability_confirmations: int = 2


@dataclass
class AppConfig:
    whisper_model: Path = PROJECT_ROOT / "models/ggml-small.en.bin"
    whisper_server: Path = PROJECT_ROOT / "whisper.cpp/build/bin/Release/whisper-server.exe"
    server_port: int = 8178
    mic_device: str = ""
    mic_fallback: bool = True
    vad: VadSettings = field(default_factory=VadSettings)
    punctuation: PunctuationSettings = field(default_factory=PunctuationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    enter_phrases: list = field(default_factory=lambda: ["press enter"])
    simulate: SimulateSettings = field(default_factory=SimulateSettings)
    hotkey: HotkeySettings = field(default_factory=HotkeySettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    streaming: StreamingSettings = field(default_factory=StreamingSettings)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _normalize_tristate(value) -> str:
    # YAML parses bare true/false as Python bool; also accept the literal
    # strings "auto"/"true"/"false" (case-insensitive).
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")
        print(f"Created default config: {CONFIG_PATH}")

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    vad_raw = raw.get("vad") or {}
    punct_raw = raw.get("punctuation") or {}
    out_raw = raw.get("output") or {}
    sim_raw = raw.get("simulate") or {}
    hotkey_raw = raw.get("hotkey") or {}
    overlay_raw = raw.get("overlay") or {}
    streaming_raw = raw.get("streaming") or {}
    cfg = AppConfig(
        whisper_model=_resolve(raw.get("whisper_model", "models/ggml-small.en.bin")),
        whisper_server=_resolve(raw.get("whisper_server",
                                        "whisper.cpp/build/bin/Release/whisper-server.exe")),
        server_port=int(raw.get("server_port", 8178)),
        mic_device=str(raw.get("mic_device") or ""),
        mic_fallback=bool(raw.get("mic_fallback", True)),
        vad=VadSettings(
            threshold=float(vad_raw.get("threshold", 0.5)),
            min_speech_ms=int(vad_raw.get("min_speech_ms", 200)),
            min_silence_ms=int(vad_raw.get("min_silence_ms", 600)),
            speech_pad_ms=int(vad_raw.get("speech_pad_ms", 300)),
        ),
        punctuation=PunctuationSettings(
            pause_threshold_ms=int(punct_raw.get("pause_threshold_ms", 350)),
        ),
        output=OutputSettings(
            mode=str(out_raw.get("mode", "type")).lower(),
            batch_chars=int(out_raw.get("batch_chars", 32)),
            inter_batch_delay_ms=int(out_raw.get("inter_batch_delay_ms", 5)),
        ),
        enter_phrases=[str(p) for p in (raw.get("enter_phrases") or ["press enter"])],
        simulate=SimulateSettings(
            files=[str(p) for p in (sim_raw.get("files") or [])],
            loop=bool(sim_raw.get("loop", False)),
        ),
        hotkey=HotkeySettings(
            enabled=bool(hotkey_raw.get("enabled", True)),
            toggle_pause=str(hotkey_raw.get("toggle_pause", "ctrl+f9")),
        ),
        overlay=OverlaySettings(
            enabled=bool(overlay_raw.get("enabled", True)),
            position=str(overlay_raw.get("position", "bottom-right")).lower(),
        ),
        streaming=StreamingSettings(
            enabled=_normalize_tristate(streaming_raw.get("enabled", "auto")),
            step_ms=int(streaming_raw.get("step_ms", 600)),
            max_window_s=float(streaming_raw.get("max_window_s", 10.0)),
            stability_confirmations=int(streaming_raw.get("stability_confirmations", 2)),
        ),
    )

    if not cfg.whisper_model.exists():
        models_dir = PROJECT_ROOT / "models"
        available = sorted(p.name for p in models_dir.glob("*.bin"))
        listing = "\n".join(f"  models/{n}" for n in available) or "  (none found)"
        raise SystemExit(
            f"Config error: whisper_model not found: {cfg.whisper_model}\n"
            f"Models available in {models_dir}:\n{listing}\n"
            f"Fix 'whisper_model' in {CONFIG_PATH}")
    if not cfg.whisper_server.exists():
        raise SystemExit(
            f"Config error: whisper_server not found: {cfg.whisper_server}\n"
            f"Fix 'whisper_server' in {CONFIG_PATH}")
    if cfg.output.mode not in ("type", "clipboard"):
        raise SystemExit(
            f"Config error: output.mode must be 'type' or 'clipboard', got {cfg.output.mode!r}")
    if cfg.punctuation.pause_threshold_ms <= 0:
        raise SystemExit(
            f"Config error: punctuation.pause_threshold_ms must be > 0, "
            f"got {cfg.punctuation.pause_threshold_ms}")
    if cfg.hotkey.enabled:
        from hotkey import HotkeyParseError, _parse_hotkey
        try:
            _parse_hotkey(cfg.hotkey.toggle_pause)
        except HotkeyParseError as e:
            raise SystemExit(f"Config error: hotkey.toggle_pause: {e}\nFix it in {CONFIG_PATH}")
    if cfg.overlay.position not in ("bottom-right", "bottom-left", "top-right", "top-left"):
        raise SystemExit(
            f"Config error: overlay.position must be one of bottom-right/bottom-left/"
            f"top-right/top-left, got {cfg.overlay.position!r}")
    if cfg.streaming.enabled not in ("auto", "true", "false"):
        raise SystemExit(
            f"Config error: streaming.enabled must be auto/true/false, got {cfg.streaming.enabled!r}")
    if cfg.streaming.step_ms <= 0:
        raise SystemExit(f"Config error: streaming.step_ms must be > 0, got {cfg.streaming.step_ms}")
    if cfg.streaming.max_window_s <= 0:
        raise SystemExit(
            f"Config error: streaming.max_window_s must be > 0, got {cfg.streaming.max_window_s}")
    if cfg.streaming.stability_confirmations < 1:
        raise SystemExit(
            "Config error: streaming.stability_confirmations must be >= 1, "
            f"got {cfg.streaming.stability_confirmations}")
    return cfg
