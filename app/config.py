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
# Voice dictation config. Edit freely; use the tray menu's "Reload config" to apply
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
  min_speech_ms: 100      # this much consecutive speech starts a segment (rejects blips)
  min_silence_ms: 600     # this much consecutive silence ends a segment (lower = snappier, may split sentences)
  speech_pad_ms: 300      # audio kept before/after the detected speech (protects word edges)

output:
  mode: type              # "type" = inject into focused window, "clipboard" = copy only
  batch_chars: 32         # characters per SendInput burst
  inter_batch_delay_ms: 5 # pause between bursts (raise if characters get dropped)

# If a whole utterance transcribes to exactly one of these phrases (case and
# punctuation ignored), the Enter key is pressed instead of typing the words.
# Only active in "type" mode.
enter_phrases:
  - press enter
"""


@dataclass
class VadSettings:
    threshold: float = 0.5
    min_speech_ms: int = 100
    min_silence_ms: int = 600
    speech_pad_ms: int = 300


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
class AppConfig:
    whisper_model: Path = PROJECT_ROOT / "models/ggml-small.en.bin"
    whisper_server: Path = PROJECT_ROOT / "whisper.cpp/build/bin/Release/whisper-server.exe"
    server_port: int = 8178
    mic_device: str = ""
    mic_fallback: bool = True
    vad: VadSettings = field(default_factory=VadSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    enter_phrases: list = field(default_factory=lambda: ["press enter"])
    simulate: SimulateSettings = field(default_factory=SimulateSettings)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")
        print(f"Created default config: {CONFIG_PATH}")

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    vad_raw = raw.get("vad") or {}
    out_raw = raw.get("output") or {}
    sim_raw = raw.get("simulate") or {}
    cfg = AppConfig(
        whisper_model=_resolve(raw.get("whisper_model", "models/ggml-small.en.bin")),
        whisper_server=_resolve(raw.get("whisper_server",
                                        "whisper.cpp/build/bin/Release/whisper-server.exe")),
        server_port=int(raw.get("server_port", 8178)),
        mic_device=str(raw.get("mic_device") or ""),
        mic_fallback=bool(raw.get("mic_fallback", True)),
        vad=VadSettings(
            threshold=float(vad_raw.get("threshold", 0.5)),
            min_speech_ms=int(vad_raw.get("min_speech_ms", 100)),
            min_silence_ms=int(vad_raw.get("min_silence_ms", 600)),
            speech_pad_ms=int(vad_raw.get("speech_pad_ms", 300)),
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
    return cfg
