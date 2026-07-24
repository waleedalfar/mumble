"""Detects whether the running whisper-server actually activated a CUDA
backend with enough VRAM to comfortably support streaming mode's repeated
re-transcription of a growing window.

Reads the server's own startup log rather than probing hardware separately
(no nvidia-smi call, no new dependency) -- the log is the authoritative
record of what THIS build + THIS GPU combination actually did at runtime,
which is what matters: a CPU-only build won't show a CUDA line even if an
NVIDIA GPU is physically present, and that's exactly the case we need to
catch.
"""
from __future__ import annotations

import re
from pathlib import Path

# Streaming re-decodes a growing window many times per utterance instead of
# once -- a GPU with only a couple GB tends to be an old/low-end card that
# would struggle with that repeated load even though CUDA technically
# initializes. This is a soft, documented cutoff, not a hard hardware limit.
MIN_VRAM_MIB = 4000

_CUDA_LINE = re.compile(r"ggml_cuda_init: found (\d+) CUDA devices? \(Total VRAM: (\d+) MiB\)")


def detect_cuda_capable(log_path: Path) -> tuple[bool, str]:
    """Returns (capable, reason) by inspecting a whisper-server startup log."""
    try:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False, "could not read whisper-server log"

    match = _CUDA_LINE.search(log_text)
    if not match:
        return False, "no CUDA backend detected (CPU-only build or no compatible GPU)"

    n_devices, vram_mib = int(match.group(1)), int(match.group(2))
    if vram_mib < MIN_VRAM_MIB:
        return False, f"CUDA GPU found but only {vram_mib}MiB VRAM (need >= {MIN_VRAM_MIB}MiB for streaming)"
    return True, f"CUDA GPU detected ({n_devices} device(s), {vram_mib}MiB VRAM)"
