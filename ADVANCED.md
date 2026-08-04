# Advanced guide

Everything beyond the quick setup in [README.md](README.md): manual setup
(if `setup.ps1` can't finish something), hardware profiles, every feature in
detail, the full `config.yaml` reference, tuning, and troubleshooting.

## Choosing your profile

Two starter configs are provided in `profiles/`, matched to hardware class —
`setup.ps1` picks one automatically, but if you're setting up by hand:

- **`profiles/config.gpu.yaml.example`** — for a machine with a dedicated
  NVIDIA GPU. Uses the unquantized `small.en` model and turns on continuous
  streaming mode (see below).
- **`profiles/config.portable.yaml.example`** — for a regular office
  laptop/desktop with no dedicated GPU. Uses the smaller/faster `base.en`
  model and leaves streaming off (it would just add CPU load for no benefit
  without a GPU to absorb it).

```powershell
copy profiles\config.gpu.yaml.example config.yaml        # or config.portable.yaml.example
```

Same underlying app either way — nothing about the code differs between
"versions," just these starting defaults. Feel free to copy one and adjust
individual settings afterward.

## Manual setup (what `setup.ps1` automates)

Only needed if the script can't finish something on your machine (e.g. no
build toolchain found) — otherwise `.\setup.ps1` from the project root does
all of this for you.

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
| CPU only, weak/older CPU (no AVX2, or just a couple cores) | `ggml-tiny.en.bin` | Smallest and fastest; a real accuracy drop from base.en, but the right first thing to try if base.en still feels slow — see "Notes on CPU performance" below |
| CPU only | `ggml-base.en.bin` | Fastest on CPU; noticeably less accurate than small.en |
| CPU only, willing to trade some speed for accuracy | `ggml-small.en.bin`, then quantize it (see below) | |
| NVIDIA GPU | `ggml-small.en.bin` | Fast enough unquantized (~100-250ms per sentence) |

**Notes on CPU performance**: everything except whisper's own transcription
step (VAD, tray, overlay, hotkey, typing) is already negligible cost on any
hardware. Transcription speed is what actually varies with CPU quality — a
modern CPU with AVX2 running `base.en` should feel workable (real but
tolerable delay per sentence); an older/weaker CPU without AVX2, or a
low-core-count machine, will be noticeably slower and may need `tiny.en` to
feel responsive. The app auto-detects your CPU core count to pick a sensible
thread count for whisper-server (capped at 8 — more rarely helps and can
starve the rest of the app); this isn't something you need to configure.

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

## Tray icon

Right-click it for:
- **Pause / Resume** — stop/start listening without closing the app
- **Open config** — opens `config.yaml` in your default editor
- **Reload config** — re-reads `config.yaml` and restarts the transcription
  server with new settings, without restarting the whole app
- **Quit**

Icon colors: gray = idle, green = hearing you, orange = transcribing, dark
gray = paused.

## Global hotkey

`hotkey.toggle_pause` in `config.yaml` (default `ctrl+f9`) pauses/resumes
listening from anywhere, without switching to the tray. It's a real Win32
hotkey registration (`RegisterHotKey`), not a keyboard hook — it costs
nothing on every other keystroke, since the OS only wakes the app when your
exact combo is pressed. Change it to any `modifier+key` or
`modifier+modifier+key` combo (`ctrl`/`alt`/`shift`/`win`, plus a single
letter, digit, or F-key) and Reload config. If registration fails (usually
because another app already owns that combo), the console tells you and the
app keeps running without the hotkey — the tray toggle still works.

**Picking a combo**: a single modifier + key (e.g. `ctrl+f9`) is faster to
hit than a three-key chord, but the more "normal" the combo, the likelier it
collides with something else that's already bound to it — `ctrl+d` (browser
bookmark), `alt+d` (browser address bar), `ctrl+space` (Windows IME switch),
and `alt+space`/`alt+f4` (window menu/close) are all claimed. `Win+`letter is
mostly claimed by Windows itself (`Win+H` is literally Windows' own dictation
toggle). Function keys are the safest lane since almost nothing binds them
globally — good two-key alternatives if `ctrl+f9` conflicts with something on
your machine: `ctrl+f10`, `ctrl+f11`, `alt+f9`. Fall back to a three-key combo
like `ctrl+alt+d` for close to zero collision risk if you'd rather not think
about it.

## Status overlay

A small always-on-top bar (like Zoom's recording indicator) shows Idle /
Listening / Transcribing / Paused, so you don't have to check the tray. It's
a plain tkinter window — no new dependency — set to never take keyboard
focus, so it can't accidentally steal a dictated segment from your actual
target window. Configurable via `overlay.enabled` and `overlay.position`
(`bottom-right`/`bottom-left`/`top-right`/`top-left`).

Measured cost of adding it (Python process, idle, before vs. after creating
the overlay window): about **+10-15MB memory** (the Tk/Tcl runtime) and
**effectively 0% CPU** at rest — it only redraws when the state actually
changes, a few times per utterance, not on a timer/animation loop.

## Continuous streaming mode

Normally, text only appears after you finish a sentence and pause (that
pause is what `vad.min_silence_ms` waits for). Streaming mode instead
re-transcribes the in-progress utterance every `streaming.step_ms` and types
newly-confirmed words as you go, so you see words appear while still
talking.

**The trade-off, stated plainly**: once a word is typed, it can never be
corrected — not even by the final, most-accurate pass at the end of the
sentence. whisper.cpp has no way to incrementally revise text it's already
produced (confirmed by reading its own source, including its own real-time
reference tool, which behaves the same way: commit forward, never revise).
A word typed early, with less context than the model will have a second
later, occasionally stays wrong. This is a genuine accuracy trade-off for
continuous feedback, not a strict improvement — try it and decide whether
you prefer the immediacy or the current wait-for-the-sentence accuracy.

**`streaming.enabled` accepts three values:**
- `auto` (default) — turns on only if whisper-server reports a capable CUDA
  GPU at startup (checked by reading its own startup log for a CUDA backend
  with enough VRAM — see `gpu_check.py`). Off otherwise.
- `true` — same hard capability check as `auto`. This is **not** a way to
  force it on regardless of hardware: on a machine that isn't capable, it's
  still downgraded to off, with a console message explaining why. Streaming
  re-decodes a growing window many times per utterance instead of once,
  which is cheap on a GPU but real, constant load on a CPU — the app won't
  silently let that happen.
- `false` — always off.

Other knobs: `streaming.step_ms` (how often to re-check, default 600ms),
`streaming.max_window_s` (cap on how much trailing audio one tick re-decodes,
default 10s), `streaming.stability_confirmations` (how many ticks in a row a
word must hold before it's typed — higher is slower but steadier, default 2).

**Known limitation**: a short configured `enter_phrases` entry could in
principle get partially typed before the sentence actually ends, since
streaming commits happen before the whole-utterance enter-phrase check runs.
If you rely on voice commands like "press enter" heavily, leave streaming off
for now.

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
| `hotkey.enabled` / `hotkey.toggle_pause` | Global keyboard shortcut to pause/resume from any app |
| `overlay.enabled` / `overlay.position` | On-screen Idle/Listening/Transcribing/Paused status bar |
| `streaming.enabled` | `auto`/`true`/`false` — continuous mid-sentence transcription, hard-gated to capable GPUs regardless of setting (see "Continuous streaming mode") |
| `streaming.step_ms` / `streaming.max_window_s` / `streaming.stability_confirmations` | Streaming re-check interval, max re-decoded window, and word-stability threshold |

## Tuning latency

Actual transcription is fast (well under a second on both CPU and GPU for a
typical sentence). Most of the perceived delay is `vad.min_silence_ms` — the
app waits this long after you stop talking before it's confident you're
actually done, to avoid cutting sentences off mid-pause. The default (600ms)
is conservative. Try lowering it in steps (e.g. to 450ms, then 350ms) and
watch for two failure modes as you go lower: sentences getting split into
fragments on a normal breath/comma pause, and words at the very end getting
clipped. Reload config after each change to test it live.

If waiting for the pause at all feels too slow, that's what streaming mode
addresses directly (see "Continuous streaming mode" above) — a different
trade-off (accuracy for immediacy), not a tuning value.

## Non-speech filtering

whisper occasionally transcribes a non-speech sound (a breath, a click, room
noise the VAD let through) as a bracketed tag like `[BLANK_AUDIO]` or
`(silence)`, or appends a trailing "..." to a low-confidence ending. None of
that is ever typed or copied — a segment that's nothing but tag(s) is
dropped entirely, and a trailing ellipsis is trimmed from otherwise-real
text. The raw, unfiltered transcript is still written to `dictation.log`
(and shown in the console, marked `[suppressed: non-speech]` when nothing
was delivered) so you can see what actually happened. This is on by default
and isn't currently a config option, since suppressing it is expected to
always be wanted.

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
- **Streaming won't turn on**: check the console line printed at startup
  (`Streaming: auto-enabled (...)` / `auto-disabled (...)` / `forced OFF (...)`)
  — it states exactly why, based on what whisper-server's own log reported
  about CUDA/VRAM at startup.
- **Words run together with no spaces between them**: if you're on an older
  checkout, update — this was a real bug (`injector.py`'s text sanitizer was
  stripping the trailing space between deliveries) that showed rarely with
  normal sentence-at-a-time delivery but affected every single word once
  streaming mode started delivering word-by-word. Fixed; if you still see it,
  it's a regression worth reporting.

## Tests

```bash
cd app
../venv/Scripts/python.exe -m pip install pytest
../venv/Scripts/python.exe -m pytest tests -q
```
