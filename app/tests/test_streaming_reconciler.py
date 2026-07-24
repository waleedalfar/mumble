"""Tests for TextReconciler -- the core append-only stabilization algorithm
behind continuous streaming mode. Pure logic: no I/O, no threads, no model."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from streaming_transcriber import TextReconciler


class TestTextReconciler:
    def test_stable_append_across_ticks(self):
        r = TextReconciler(stability_confirmations=2)
        assert r.reconcile("hello how", is_final=False) == []
        assert r.reconcile("hello how are", is_final=False) == ["hello"]
        assert r.reconcile("hello how are you", is_final=False) == ["how"]
        assert r.committed_words == ["hello", "how"]

    def test_word_revised_before_stabilizing_updates_silently(self):
        r = TextReconciler(stability_confirmations=2)
        # tick 1: window suggests "alpha" as the pending (non-boundary) word
        assert r.reconcile("alpha beta", is_final=False) == []
        # tick 2: fuller context revises it to "gamma" instead -- must NOT
        # have committed "alpha", so the revision is silent/free
        assert r.reconcile("gamma beta delta", is_final=False) == []
        assert r.committed_words == []
        # tick 3: "gamma" and "beta" have now both held their position
        # across two consecutive ticks -> both commit together
        promoted = r.reconcile("gamma beta delta epsilon", is_final=False)
        assert promoted == ["gamma", "beta"]
        assert "alpha" not in r.committed_words

    def test_word_revised_after_commit_is_not_corrected(self):
        r = TextReconciler(stability_confirmations=1)  # commit on first sight
        assert r.reconcile("the cat", is_final=False) == ["the"]
        assert r.committed_words == ["the"]
        # a later, better-contextualized pass would say "hat", not "cat" --
        # but "the" is already committed and must never be rewritten
        r.reconcile("the hat sat", is_final=False)
        assert r.committed_words[0] == "the"

    def test_boundary_word_excluded_on_non_final_tick(self):
        r = TextReconciler(stability_confirmations=1)
        # single-word window: the only candidate is also the boundary word,
        # so nothing should be promoted yet
        assert r.reconcile("hello", is_final=False) == []
        assert r.committed_words == []

    def test_no_overlap_fallback_treats_whole_window_as_new(self):
        r = TextReconciler(stability_confirmations=1)
        r.reconcile("one two three", is_final=False)
        first_committed = list(r.committed_words)
        # a completely unrelated transcript (e.g. VAD hiccup) shares no
        # overlap with committed text -- k=0, whole window becomes candidates
        promoted = r.reconcile("completely different words here", is_final=False)
        assert isinstance(promoted, list)  # doesn't crash; may promote some
        assert r.committed_words[: len(first_committed)] == first_committed

    def test_is_final_promotes_everything_remaining(self):
        r = TextReconciler(stability_confirmations=5)  # high bar, nothing would stabilize normally
        r.reconcile("hello how are you", is_final=False)
        assert r.committed_words == []  # nothing hit the stability threshold
        promoted = r.reconcile("hello how are you today", is_final=True)
        assert promoted == ["hello", "how", "are", "you", "today"]

    def test_is_final_on_first_call_commits_everything(self):
        r = TextReconciler(stability_confirmations=2)
        promoted = r.reconcile("just one final segment", is_final=True)
        assert promoted == ["just", "one", "final", "segment"]

    def test_empty_transcript_promotes_nothing(self):
        r = TextReconciler()
        assert r.reconcile("", is_final=False) == []
        assert r.reconcile("   ", is_final=True) == []

    def test_punctuation_ignored_for_overlap_matching(self):
        r = TextReconciler(stability_confirmations=1)
        r.reconcile("Hello, world", is_final=False)
        # committed = ["Hello,"] (boundary "world" excluded); next tick's
        # transcript rephrases punctuation on the same word -- should still
        # be recognized as overlapping, not duplicated
        promoted = r.reconcile("Hello world again", is_final=False)
        assert "Hello," in r.committed_words or "Hello" in r.committed_words
        assert r.committed_words.count("Hello,") + r.committed_words.count("Hello") == 1
