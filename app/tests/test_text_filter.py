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


class TestHallucinationFilter:
    """"Thank you." and similar stock phrases whisper hallucinates outright on
    short, ambiguous non-speech audio -- see text_filter.py's module docstring."""

    def test_short_thank_you_is_suppressed(self):
        assert clean_transcript("Thank you.", duration_s=0.8) == ""

    def test_short_single_word_you_is_suppressed(self):
        assert clean_transcript("you", duration_s=0.67) == ""

    def test_matching_is_case_and_punctuation_insensitive(self):
        assert clean_transcript("THANK YOU!!", duration_s=1.0) == ""
        assert clean_transcript("thank, you?", duration_s=1.0) == ""

    def test_every_known_phrase_is_suppressed_when_short(self):
        phrases = [
            "Thank you.", "Thanks for watching.", "Thank you for watching.",
            "Please subscribe.", "Subscribe to my channel.", "Like and subscribe.",
            "Thanks for listening.", "Bye.", "Bye bye.", "Goodbye.",
            "See you next time.", "You",
        ]
        for phrase in phrases:
            assert clean_transcript(phrase, duration_s=1.0) == "", f"{phrase!r} should be suppressed"

    def test_no_duration_supplied_is_kept(self):
        # Streaming per-tick commits don't have a per-commit duration to pass
        # -- omitting it must never suppress anything, only opt out of this check.
        assert clean_transcript("Thank you.") == "Thank you."

    def test_long_duration_is_kept(self):
        # A genuinely long segment that happens to transcribe to just "Thank
        # you." wasn't a blip -- e.g. the user paused a long time mid-VAD-segment.
        assert clean_transcript("Thank you.", duration_s=5.0) == "Thank you."

    def test_at_threshold_boundary_is_suppressed(self):
        from text_filter import _HALLUCINATION_MAX_DURATION_S
        assert clean_transcript("Thank you.", duration_s=_HALLUCINATION_MAX_DURATION_S) == ""

    def test_just_above_threshold_is_kept(self):
        from text_filter import _HALLUCINATION_MAX_DURATION_S
        assert clean_transcript("Thank you.", duration_s=_HALLUCINATION_MAX_DURATION_S + 0.01) \
            == "Thank you."

    def test_short_non_hallucination_text_is_kept(self):
        # Duration alone is never enough to suppress -- only a short segment
        # whose *whole* transcript is a known stock phrase.
        assert clean_transcript("Reloaded the config.", duration_s=0.5) == "Reloaded the config."

    def test_sentence_merely_containing_a_phrase_is_kept(self):
        # Must match the *entire* cleaned transcript, not just contain it --
        # same whole-segment-only rule the tag filter above already follows.
        text = "Thank you very much for your help today."
        assert clean_transcript(text, duration_s=1.0) == text

    def test_hallucination_check_runs_after_ellipsis_trim(self):
        assert clean_transcript("Thank you...", duration_s=1.0) == ""
