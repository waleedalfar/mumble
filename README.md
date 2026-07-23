# Voice Dictation

Background voice dictation for Windows: continuous mic capture → voice activity
detection (silero-vad) → speech-to-text (whisper.cpp) → typed into whatever
window currently has focus (via `SendInput`). Built for dictating into a
browser chat box or a terminal — not for games or exotic apps.

Runs entirely locally. No audio or text leaves your machine.

## Prerequisites

- **Windows 10/11**
- **Python 3.10+**
- **Git**
- **A C++ build toolchain**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
  with the "Desktop development with C++" workload (or Visual Studio Community,
  which includes it and a bundled CMake). MinGW also works if you already have
  a C++ toolchain you prefer.
- **Optional — NVIDIA GPU**: if you have an NVIDIA GPU and want faster
  transcription, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  (12.8+) matching your driver. Not required — this app runs fine CPU-only.

## Setup

### 1. Clone this repo and whisper.cpp

```bash
git clone <this-repo-url>
cd <this-repo>
git clone https://github.com/ggml-org/whisper.cpp
```

`whisper.cpp` is cloned as a sibling folder inside the project root, not a git
submodule — it has its own upstream and its own build.

### 2. Build whisper.cpp

If your Visual Studio installation doesn't already have `cmake` on your PATH,
use the one bundled with it, e.g.
`C:\Program Files\Microsoft Visual Studio\<version>\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

**CPU-only (recommended for a standard laptop/desktop with no NVIDIA GPU):**

```bash
cd whisper.cpp
cmake -B build
cmake --build build --config Release -j 8
cd ..
```

This automatically uses AVX2/AVX512 CPU acceleration where your processor
supports it — no flags needed.

**With an NVIDIA GPU** (replace `120` with your GPU's compute capability,
e.g. `89` for a 40-series card, `120` for a 50-series card):

```bash
cd whisper.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j 8
cd ..
```

Either way, this produces `whisper.cpp/build/bin/Release/whisper-server.exe`
and `whisper-quantize.exe`, which the app uses.

### 3. Download the models

```bash
mkdir models
```

**Voice activity detection model** (required, small, always CPU):

Download `silero_vad.onnx` from
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
into `models/`.

**Speech-to-text model** (required — pick one based on your hardware):

| Your hardware | Recommended model | Notes |
|---|---|---|
| CPU only | `ggml-base.en.bin` | Fastest on CPU; noticeably less accurate than small.en |
| CPU only, willing to trade some speed for accuracy | `ggml-small.en.bin`, then quantize it (see below) | |
| NVIDIA GPU | `ggml-small.en.bin` | Fast enough unquantized (~100-250ms per sentence) |

Download from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<filename>`
into `models/`, e.g.:

```bash
curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

**Optional: quantize a model for a better CPU speed/accuracy tradeoff.** A
quantized model is smaller and faster with a small accuracy cost:

```bash
whisper.cpp/build/bin/Release/whisper-quantize.exe models/ggml-small.en.bin models/ggml-small.en-q5_1.bin q5_1
```

Any `.bin` file dropped into `models/` can be used — just point `whisper_model`
in `config.yaml` at it (see below). No restart needed, no code changes.

### 4. Set up the Python environment

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 5. Run it

```bash
cd app
../venv/Scripts/python.exe main.py
```

**First run**: since no `config.yaml` exists yet, one is created automatically
with defaults. Because `mic_device` starts empty, the app will print your
available microphones and ask you to fill it in — open `config.yaml` at the
project root, set `mic_device` to a name (or part of a name) from the printed
list, and run again.

A tray icon should now appear (check the `^` overflow arrow next to your
clock if you don't see it). Click into a browser chat box or a terminal and
talk — text should appear where your cursor is.

## Tray icon

Right-click it for:
- **Pause / Resume** — stop/start listening without closing the app
- **Open config** — opens `config.yaml` in your default editor
- **Reload config** — re-reads `config.yaml` and restarts the transcription
  server with new settings, without restarting the whole app
- **Quit**

Icon colors: gray = idle, green = hearing you, orange = transcribing, dark
gray = paused.

## config.yaml reference

| Key | What it does |
|---|---|
| `whisper_model` | Path to the `.bin` model whisper-server should load |
| `whisper_server` | Path to `whisper-server.exe` |
| `server_port` | Local port used between the app and whisper-server |
| `mic_device` | Microphone selected by name substring |
| `mic_fallback` | If `mic_device` isn't found, fall back to any available mic instead of refusing to start |
| `vad.threshold` | Speech probability (0-1) needed to count a frame as speech |
| `vad.min_speech_ms` | How much consecutive speech is needed to start a segment (rejects short blips/clicks) |
| `vad.min_silence_ms` | How much consecutive silence is needed to end a segment — **the main latency knob**, see below |
| `vad.speech_pad_ms` | Audio kept before/after detected speech, so word edges aren't clipped |
| `output.mode` | `type` = inject into the focused window; `clipboard` = copy only, don't type |
| `output.batch_chars` / `output.inter_batch_delay_ms` | How text is chunked/paced while typing — raise the delay if characters get dropped in a particular app |
| `enter_phrases` | If a whole utterance transcribes to exactly one of these phrases, the Enter key is pressed instead of typing the words |
| `simulate.files` / `simulate.loop` | Feed WAV files through the pipeline instead of the mic, for testing without touching hardware |

## Tuning latency

Actual transcription is fast (well under a second on both CPU and GPU for a
typical sentence). Most of the perceived delay is `vad.min_silence_ms` — the
app waits this long after you stop talking before it's confident you're
actually done, to avoid cutting sentences off mid-pause. The default (600ms)
is conservative. Try lowering it in steps (e.g. to 450ms, then 350ms) and
watch for two failure modes as you go lower: sentences getting split into
fragments on a normal breath/comma pause, and words at the very end getting
clipped. Reload config after each change to test it live.

## Troubleshooting

- **"Port already in use"**: an old `whisper-server.exe` from a previous run
  is still alive. Close the previous instance of this app first; if it's
  truly orphaned, `taskkill /F /IM whisper-server.exe` (only when the app
  isn't currently running).
- **No microphone detected / nothing happens when you talk**: check Windows
  Settings → System → Sound → Input, select your mic, and confirm the level
  meter moves when you talk. Also check Settings → Privacy & security →
  Microphone → "Let desktop apps access your microphone" is on.
- **Transcription stopped working mid-session**: the app detects a dead
  whisper-server and relaunches it automatically on the next segment — you
  should see `whisper-server is gone — relaunching it...` in the console.

## Tests

```bash
cd app
../venv/Scripts/python.exe -m pip install pytest
../venv/Scripts/python.exe -m pytest tests -q
```
