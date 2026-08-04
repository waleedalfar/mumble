"""Continuous mid-utterance transcription: periodically re-transcribes the
in-progress utterance and commits newly-stabilized words as they're
confirmed, instead of waiting for the whole sentence to finish.

whisper has no incremental-decode API (confirmed by reading whisper.cpp's own
source, including its "stream" reference tool): every tick fully re-decodes
whatever audio window it's given. TextReconciler's job is to decide, from
each tick's full re-transcription, which words are safe to type now --
appended forward only, NEVER edited once committed. This matches how
whisper.cpp's own stream example already behaves (it commits printed text
and moves on), so it isn't an extra constraint bolted on top, it's the
natural fit for what the model actually gives us.

The accepted trade-off: once a word is committed, a later pass with fuller
context can never correct it, even if it would have transcribed it
differently. Continuous feedback in exchange for occasional uncorrectable
errors -- see README's "Continuous streaming mode" section.
"""
from __future__ import annotations

import difflib
import re
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np

_PUNCT_RE = re.compile(r"[^\w']")


def _normalize(word: str) -> str:
    return _PUNCT_RE.sub("", word.lower())


class TextReconciler:
    """Pure logic, no I/O/threads -- the crux of streaming mode."""

    def __init__(self, stability_confirmations: int = 2):
        self._stability = max(1, stability_confirmations)
        self.committed_words: list[str] = []
        self._committed_norm: list[str] = []
        self._pending_norm: list[str] = []
        self._pending_counts: list[int] = []

    def reconcile(self, raw_transcript: str, *, is_final: bool) -> list[str]:
        raw_words = raw_transcript.split()
        norm_words = [_normalize(w) for w in raw_words]
        # drop tokens that normalize to nothing (pure punctuation/symbols)
        pairs = [(w, n) for w, n in zip(raw_words, norm_words) if n]
        words = [w for w, _ in pairs]
        norm = [n for _, n in pairs]

        k = self._find_overlap(norm)
        cand_words, cand_norm = words[k:], norm[k:]

        if not is_final and cand_words:
            # the window's right edge is an arbitrary timer cut, likely
            # mid-word/mid-context -- never trust the very last candidate yet
            cand_words, cand_norm = cand_words[:-1], cand_norm[:-1]

        counts = self._vote(cand_norm)

        promoted: list[str] = []
        i = 0
        while i < len(cand_words) and (is_final or counts[i] >= self._stability):
            promoted.append(cand_words[i])
            i += 1

        self.committed_words.extend(promoted)
        self._committed_norm.extend(_normalize(w) for w in promoted)
        self._pending_norm = cand_norm[i:]
        self._pending_counts = counts[i:]

        return promoted

    def _find_overlap(self, window_norm: list[str]) -> int:
        committed = self._committed_norm
        if not committed or not window_norm:
            return 0
        # An exact-match suffix/prefix comparison breaks entirely on a single
        # interior word coming out differently between passes (e.g. a later,
        # fuller-context pass transcribing "wifi" as "wi-fi") -- one mismatch
        # anywhere would make the whole committed text look like it has zero
        # overlap with the new window, and get retyped in full.
        # SequenceMatcher finds the real alignment around such substitutions.
        matcher = difflib.SequenceMatcher(None, committed, window_norm, autojunk=False)
        a_end = b_end = 0
        for a, b, size in matcher.get_matching_blocks():
            if size and a + size > a_end:
                a_end, b_end = a + size, b + size
        if b_end == 0:
            # no shared words at all -- genuinely unrelated content (e.g. a
            # VAD hiccup), not worth guessing a positional alignment for.
            return 0
        # Committed words past the last matched block, if any, are presumed
        # re-transcribed differently rather than genuinely new -- carry them
        # forward positionally so the alignment isn't lost over a handful of
        # substitutions.
        tail = len(committed) - a_end
        return min(b_end + tail, len(window_norm))

    def _vote(self, cand_norm: list[str]) -> list[int]:
        """A word matching its previous tick's guess at the same position
        increments a confirmation counter; any mismatch -- at that position or
        anything after it -- resets to 1, since a changed word invalidates
        confidence in everything that follows."""
        counts = []
        mismatch_seen = False
        for i, cn in enumerate(cand_norm):
            if not mismatch_seen and i < len(self._pending_norm) and cn == self._pending_norm[i]:
                counts.append(self._pending_counts[i] + 1)
            else:
                mismatch_seen = True
                counts.append(1)
        return counts


class StreamingSession:
    """Owns per-utterance audio buffering and the periodic tick thread."""

    def __init__(self, transcribe_fn: Callable[..., str], step_ms: int, max_window_s: float,
                 stability_confirmations: int, on_commit: Callable[[list[str]], None],
                 initial_audio: np.ndarray, sample_rate: int, tick_timeout_s: float = 8.0):
        self._transcribe = transcribe_fn
        self._step_s = step_ms / 1000
        self._max_samples = int(max_window_s * sample_rate)
        self._on_commit = on_commit
        self._tick_timeout_s = tick_timeout_s
        self.reconciler = TextReconciler(stability_confirmations)

        self._buffer: deque = deque()
        self._buffer_samples = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._add_audio(initial_audio)

    def add_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._add_audio(frame)

    def _add_audio(self, audio: np.ndarray) -> None:
        if len(audio) == 0:
            return
        self._buffer.append(audio)
        self._buffer_samples += len(audio)
        while self._buffer_samples > self._max_samples and len(self._buffer) > 1:
            dropped = self._buffer.popleft()
            self._buffer_samples -= len(dropped)

    def _snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self._buffer))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._step_s):
            self._tick()

    def _tick(self) -> None:
        audio = self._snapshot()
        if len(audio) == 0:
            return
        prompt = " ".join(self.reconciler.committed_words[-32:])
        try:
            text = self._transcribe(audio, prompt=prompt, timeout_s=self._tick_timeout_s)
        except Exception as e:
            print(f"[streaming] tick failed: {e}")
            return
        new_words = self.reconciler.reconcile(text, is_final=False)
        if new_words:
            self._on_commit(new_words)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def finalize(self, final_text: str) -> list[str]:
        """Called once, by the existing end-of-utterance transcribe worker
        (no extra HTTP call introduced) -- reconciles the authoritative full
        transcript against whatever streaming already committed, promoting
        everything that's left."""
        return self.reconciler.reconcile(final_text, is_final=True)
