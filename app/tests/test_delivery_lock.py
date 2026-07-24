"""Regression test for the words-running-together bug: concurrent _deliver()
calls (e.g. a StreamingSession's tick thread committing partial words while
the transcribe worker delivers a different utterance's final text) must never
overlap, since concurrent SendInput callers can interleave at the OS input
queue and drop/displace characters -- including the space between words."""
from __future__ import annotations

import os
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import main as main_module


class TestDeliverLock:
    def test_concurrent_deliver_calls_never_overlap(self):
        app = main_module.DictationApp()
        try:
            intervals: list[tuple[float, float]] = []
            lock = threading.Lock()

            def slow_type_text(text, **kwargs):
                start = time.monotonic()
                time.sleep(0.05)  # long enough that a race would be caught
                end = time.monotonic()
                with lock:
                    intervals.append((start, end))

            with patch.object(main_module, "type_text", side_effect=slow_type_text):
                threads = [threading.Thread(target=app._deliver, args=(f"word{i}",))
                          for i in range(5)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5)

            assert len(intervals) == 5
            intervals.sort()
            for (s1, e1), (s2, _) in zip(intervals, intervals[1:]):
                assert e1 <= s2, "two _deliver() calls overlapped -- the lock isn't serializing them"
        finally:
            app.log.close()
