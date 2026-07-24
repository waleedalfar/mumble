"""Regression test for the words-running-together bug: _sanitize() must
never strip the trailing separator space callers deliberately append (e.g.
main.py's `text + " "`) before every delivery. rstrip()-ing it was invisible
with one delivery per full sentence, but became "every word runs together"
once streaming started delivering one word (or a few) at a time."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from injector import _sanitize


class TestSanitize:
    def test_trailing_separator_space_is_preserved(self):
        assert _sanitize("Hello ") == "Hello "

    def test_leading_whitespace_is_stripped(self):
        assert _sanitize("  Hello") == "Hello"

    def test_internal_whitespace_runs_collapse_to_one_space(self):
        assert _sanitize("Hello\nworld\t\tagain") == "Hello world again"

    def test_consecutive_deliveries_join_with_exactly_one_space(self):
        # simulates two separate type_text() calls concatenating into one
        # target text field, the way streaming or back-to-back utterances do
        delivered = _sanitize("Hello ") + _sanitize("world ")
        assert delivered == "Hello world "

    def test_no_double_space_when_input_already_ends_in_one(self):
        assert _sanitize("Hello  ") == "Hello "
