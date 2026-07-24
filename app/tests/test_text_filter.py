"""Tests for the non-speech-artifact transcript filter."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from text_filter import clean_transcript


class TestCleanTranscript:
    def test_pure_bracket_tag_is_suppressed(self):
        assert clean_transcript("[BLANK_AUDIO]") == ""

    def test_pure_paren_tag_is_suppressed(self):
        assert clean_transcript("(silence)") == ""

    def test_multiple_tags_are_suppressed(self):
        assert clean_transcript("[BLANK_AUDIO] [whoosh]") == ""

    def test_whitespace_padded_tag_is_suppressed(self):
        assert clean_transcript("  [BLANK_AUDIO]  ") == ""

    def test_empty_and_whitespace_only_are_suppressed(self):
        assert clean_transcript("") == ""
        assert clean_transcript("   ") == ""

    def test_bare_ellipsis_is_suppressed(self):
        assert clean_transcript("...") == ""

    def test_trailing_ascii_ellipsis_is_trimmed(self):
        assert clean_transcript("And so my fellow Americans...") == "And so my fellow Americans"

    def test_trailing_unicode_ellipsis_is_trimmed(self):
        assert clean_transcript("And so my fellow Americans" + chr(0x2026)) == "And so my fellow Americans"

    def test_single_trailing_period_is_kept(self):
        assert clean_transcript("Ask not what your country can do for you.") == \
            "Ask not what your country can do for you."

    def test_real_sentence_with_parenthetical_is_untouched(self):
        text = "Please remember to check (the second one) before you go."
        assert clean_transcript(text) == text

    def test_plain_sentence_untouched(self):
        assert clean_transcript("Hi there.") == "Hi there."
