"""Strips whisper.cpp non-speech artifacts before delivery (typing/clipboard).

whisper occasionally transcribes a segment of non-speech audio (breath, mouse
click, background noise the VAD let through) as a bracketed/parenthesized tag
like "[BLANK_AUDIO]", "(silence)", "[whoosh]", or appends a trailing ellipsis
to a low-confidence ending. None of that should ever be typed or copied — but
the raw text is still worth keeping in the log for debugging VAD/model
behavior, so this only trims what gets delivered, never what gets logged.

Separately, whisper (trained heavily on YouTube/podcast captions) also
sometimes hallucinates a stock closing phrase -- "Thank you.", "Thanks for
watching." -- outright, on ambiguous non-speech audio the VAD let through as
a "speech" segment. These come out as confident, well-formed sentences, not
the garbled/low-probability output whisper.cpp's own no_speech/logprob/
entropy thresholds are built to catch, so nothing upstream filters them.
_HALLUCINATION_PHRASES below is a plain denylist for that -- the same
mitigation the wider whisper.cpp/faster-whisper community has landed on.

Two more pure text transforms live here: apply_symbol_words() turns spoken
symbol names ("slash", "underscore", ...) into the literal character, and
join_segments() (used by transcriber.py) softens whisper's own trigger-happy
mid-utterance periods using real per-segment pause timing.
"""
from __future__ import annotations

import re

_TAG_GROUP = re.compile(r"[\[(][^\])]*[\])]")
# 2+ literal dots, or the single-glyph unicode ellipsis character, either one
# optionally trailed by more dots/ellipses/whitespace. A single "." is left
# alone -- that's normal sentence punctuation, not a trailing-off artifact.
_TRAILING_ELLIPSIS = re.compile(r"(?:\.{2,}|…)[.…\s]*$")
_NON_ALPHA = re.compile(r"[^a-z ]")

_HALLUCINATION_PHRASES = frozenset({
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "thanks for listening",
    "bye",
    "bye bye",
    "goodbye",
    "see you next time",
    "you",
})

# Every real occurrence logged so far landed well under this (0.6-1.3s
# segments, immediately after real speech ended) -- kept generous above that
# observed max so near-boundary cases are still caught. A deliberately
# dictated one of these short phrases is rare enough in a dictation context
# that trading it away is worth it to silence the far more common false
# positive; this only ever suppresses a segment whose *entire* transcript is
# one of these phrases, never one that merely contains it.
_HALLUCINATION_MAX_DURATION_S = 1.5


def _normalize(text: str) -> str:
    return _NON_ALPHA.sub("", text.lower()).strip()


def clean_transcript(text: str, duration_s: float | None = None) -> str:
    """Return `text` with a trailing ellipsis trimmed and, if the whole segment
    turns out to be nothing but non-speech tag(s) or a short hallucinated
    stock phrase, an empty string.

    Real dictated sentences that merely contain a parenthetical are left
    completely untouched — only a segment that is *entirely* tag(s) (nothing
    left after removing every bracket/paren group) is suppressed.

    `duration_s`, when given, is the audio duration of the segment that
    produced `text`; a segment whose whole transcript exactly matches a known
    whisper hallucination (see _HALLUCINATION_PHRASES) is suppressed only if
    it's also short (see _HALLUCINATION_MAX_DURATION_S) -- callers that can't
    supply a duration (e.g. per-tick streaming commits) simply skip this
    check by leaving it at the default None.
    """
    text = _TRAILING_ELLIPSIS.sub("", text).strip()
    if not text:
        return ""
    if _TAG_GROUP.search(text):
        remainder = _TAG_GROUP.sub("", text).strip(" .")
        if not remainder:
            return ""
    if duration_s is not None and duration_s <= _HALLUCINATION_MAX_DURATION_S \
            and _normalize(text) in _HALLUCINATION_PHRASES:
        return ""
    return text


# Longest phrase first so "at sign" (two words) is tried before any
# single-word alternative could partially shadow it.
_SYMBOL_WORDS = {
    "at sign": "@",
    "underscore": "_",
    "slash": "/",
    "hyphen": "-",
    "dash": "-",
    "colon": ":",
}
_SYMBOL_RE = re.compile(
    r"\s*\b(" + "|".join(re.escape(p) for p in sorted(_SYMBOL_WORDS, key=len, reverse=True)) + r")\b\s*",
    re.IGNORECASE,
)


def apply_symbol_words(text: str) -> str:
    """Replace spoken symbol names with the literal character, consuming the
    whitespace on both sides so the result reads like a path/identifier
    rather than a sentence -- "foo slash bar" -> "foo/bar", not "foo / bar".
    """
    return _SYMBOL_RE.sub(lambda m: _SYMBOL_WORDS[m.group(1).lower()], text)


def join_segments(segments: list[dict], pause_threshold_s: float) -> str:
    """Join whisper's per-segment texts into one string, downgrading a
    trailing "." on an interior segment to "," when the measured silence gap
    before the next segment is shorter than `pause_threshold_s`.

    whisper places these interior periods from grammar alone, not from a
    genuinely measured pause -- a segment handed a few seconds of one
    continuous, unbroken thought will still often get split into separate
    "sentences" at ordinary clause boundaries. Downgrading a short-gap period
    to a comma (and lowercasing the following segment's first letter, so
    "Claude." / "There's" becomes "Claude," / "there's") corrects that
    without needing to touch whisper's own decoding.

    The last segment's own trailing punctuation is left untouched -- that
    boundary belongs to vad.min_silence_ms, not this function. Segments
    missing "start"/"end" (no timing info available) are left untouched too,
    rather than guessed at.
    """
    if not segments:
        return ""
    parts = [str(seg.get("text", "")).strip() for seg in segments]
    for i in range(len(parts) - 1):
        cur = parts[i]
        if not cur.endswith(".") or cur.endswith(".."):
            continue
        end_cur = segments[i].get("end")
        start_next = segments[i + 1].get("start")
        if end_cur is None or start_next is None:
            continue
        if start_next - end_cur < pause_threshold_s:
            parts[i] = cur[:-1] + ","
            nxt = parts[i + 1]
            if nxt and nxt[0].isalpha():
                parts[i + 1] = nxt[0].lower() + nxt[1:]
    return " ".join(p for p in parts if p)
