"""Strips whisper.cpp non-speech artifacts before delivery (typing/clipboard).

whisper occasionally transcribes a segment of non-speech audio (breath, mouse
click, background noise the VAD let through) as a bracketed/parenthesized tag
like "[BLANK_AUDIO]", "(silence)", "[whoosh]", or appends a trailing ellipsis
to a low-confidence ending. None of that should ever be typed or copied — but
the raw text is still worth keeping in the log for debugging VAD/model
behavior, so this only trims what gets delivered, never what gets logged.
"""
from __future__ import annotations

import re

_TAG_GROUP = re.compile(r"[\[(][^\])]*[\])]")
# 2+ literal dots, or the single-glyph unicode ellipsis character, either one
# optionally trailed by more dots/ellipses/whitespace. A single "." is left
# alone -- that's normal sentence punctuation, not a trailing-off artifact.
_TRAILING_ELLIPSIS = re.compile(r"(?:\.{2,}|…)[.…\s]*$")


def clean_transcript(text: str) -> str:
    """Return `text` with a trailing ellipsis trimmed and, if the whole segment
    turns out to be nothing but non-speech tag(s), an empty string.

    Real dictated sentences that merely contain a parenthetical are left
    completely untouched — only a segment that is *entirely* tag(s) (nothing
    left after removing every bracket/paren group) is suppressed.
    """
    text = _TRAILING_ELLIPSIS.sub("", text).strip()
    if not text:
        return ""
    if not _TAG_GROUP.search(text):
        return text
    remainder = _TAG_GROUP.sub("", text).strip(" .")
    return "" if not remainder else text
