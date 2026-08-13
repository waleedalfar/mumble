# Mumble

Mumble is background voice dictation for Windows. Talk, and your words get
typed wherever your cursor is — a browser chat box, a terminal, any text field.

Runs entirely on your own computer. No audio or text ever leaves your machine.

## How it works

Your mic feeds a small local voice-activity model ([Silero VAD](https://github.com/snakers4/silero-vad))
that detects when you're actually speaking, then the speech is sent to
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) — a fast, local,
open-source implementation of OpenAI's Whisper speech-to-text model — for
transcription. The resulting text is typed into whatever window has focus
using Windows' native input APIs. Nothing about this talks to the internet
or any cloud service. More detail on the full pipeline is in
[ADVANCED.md](ADVANCED.md).

**There are two ways to run whisper.cpp, and this app supports both from the
same code**: a **CPU-only build** (works on any Windows PC, no special
hardware needed) and a **CUDA-accelerated build** (noticeably faster, needs
an NVIDIA GPU). You don't have to choose by hand — the setup script below
detects your hardware and builds whichever one fits, automatically.

## What you need

- Windows 10 or 11
- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) —
  during install, check the **"Desktop development with C++"** workload
- An NVIDIA GPU is optional. With one, setup builds the CUDA-accelerated
  version of whisper.cpp for faster transcription; without one, it builds
  the CPU-only version instead, which runs fine on any regular Windows PC.

## Set it up

Clone the repo, then from inside the project folder in PowerShell:

```powershell
.\setup.ps1
```

This one command detects whether you have a working NVIDIA/CUDA setup and
builds whisper.cpp accordingly — CUDA-accelerated if you do, CPU-only if you
don't — downloads the required models, and gets the app ready to run. It
takes a few minutes, and it's safe to run again if anything interrupts it
partway.

## Run it

Double-click `start.bat` in the project folder (or the "Mumble" shortcut on
your desktop, if you've set one up) — a console window opens showing status,
and the app starts. Closing that window stops the app.

Prefer a terminal? Same thing, spelled out:

```bash
cd app
../venv/Scripts/python.exe main.py
```

**The very first time**, it won't know which microphone to use — it'll print
a list of your available microphones and ask you to pick one. Open the
`config.yaml` file that just appeared in the project folder, set
`mic_device` to your microphone's name (or part of it), and run it again.

After that, a small icon appears near your clock (you may need to click the
`^` arrow to see it) — right-click it any time to pause, resume, or quit.
Click into any text box, start talking, and your words will appear.

## Everyday use

- **Pause/resume from anywhere**: press `Ctrl+F9` — no need to switch to the
  tray first. This works even while a different app has focus. Want a
  different combo? Edit `hotkey.toggle_pause` in `config.yaml` (see
  [ADVANCED.md](ADVANCED.md) for good alternatives) and use the tray's
  "Reload config."
- **Say "press enter"** and it sends the Enter key instead of typing those
  words out — handy for sending a chat message right after dictating it.
- **Status indicator**: a small floating bar shows what's happening —
  gray = idle, green = listening, orange = transcribing, red = paused. Move
  or turn it off via `overlay.position` / `overlay.enabled` in `config.yaml`.
- **Tray icon** (right-click): Pause/Resume, Open config, Reload config, Quit.
  Config edits apply after "Reload config" — no need to restart the app.
- **Mic unplugged or renamed?** If your configured microphone isn't found at
  startup, the app automatically falls back to another available input
  device instead of refusing to start (as long as `mic_fallback: true`, the
  default).
- Every transcript (including bits that got filtered out as non-speech) is
  logged to `dictation.log` in the project folder, timestamped — useful if
  you want to check what was actually heard.

## Want more control?

Everything else — configuration options, tuning, an optional "words appear
as you talk" mode, and troubleshooting — is in [ADVANCED.md](ADVANCED.md).
