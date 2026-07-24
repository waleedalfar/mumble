# Voice Dictation

Background voice dictation for Windows. Talk, and your words get typed
wherever your cursor is — a browser chat box, a terminal, any text field.

Runs entirely on your own computer. No audio or text ever leaves your machine.

## What you need

- Windows 10 or 11
- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) —
  during install, check the **"Desktop development with C++"** workload
- An NVIDIA GPU makes it faster, but it works fine without one too.

## Set it up

Clone the repo, then from inside the project folder in PowerShell:

```powershell
.\setup.ps1
```

This one command downloads and builds everything it needs (automatically
detecting whether you have a GPU) and gets the app ready to run. It takes a
few minutes, and it's safe to run again if anything interrupts it partway.

## Run it

```bash
cd app
../venv/Scripts/python.exe main.py
```

**The very first time**, it won't know which microphone to use — it'll print
a list of your available microphones and ask you to pick one. Open the
`config.yaml` file that just appeared in the project folder, set
`mic_device` to your microphone's name (or part of it), and run the command
again.

After that, a small icon appears near your clock (you may need to click the
`^` arrow to see it) — right-click it any time to pause, resume, or quit.
Click into any text box, start talking, and your words will appear.

## Want more control?

Everything else — configuration options, tuning, an optional "words appear
as you talk" mode, and troubleshooting — is in [ADVANCED.md](ADVANCED.md).
