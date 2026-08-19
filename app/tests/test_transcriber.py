"""Tests for the Windows Job Object that guarantees whisper-server.exe can
never outlive this app, even if the app is killed outright (crash, Task
Manager "End task", power loss) rather than exited cleanly."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transcriber


class TestKillOnCloseJob:
    def test_job_creation_succeeds(self):
        job = transcriber._make_kill_on_close_job()
        assert job is not None
        assert job > 0
        transcriber._kernel32.CloseHandle(job)

    def test_assign_to_bogus_pid_fails_gracefully(self):
        job = transcriber._make_kill_on_close_job()
        try:
            assert transcriber._assign_to_job(job, 0x7FFFFFFF) is False
        finally:
            transcriber._kernel32.CloseHandle(job)

    def test_closing_job_handle_kills_assigned_process(self):
        # Closing the job handle is the same terminal event that happens
        # automatically -- and unconditionally -- when this app's own process
        # ends, by any means. This is the actual guarantee being tested: no
        # cooperative cleanup code runs here at all, only CloseHandle.
        job = transcriber._make_kill_on_close_job()
        assert job is not None
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            assert transcriber._assign_to_job(job, proc.pid) is True
            transcriber._kernel32.CloseHandle(job)
            assert proc.wait(timeout=5) is not None
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_module_level_job_was_created_on_import(self):
        # The whole guarantee depends on this existing for the life of the
        # app's own process -- if job creation silently failed, _JOB would be
        # None and WhisperServer.start() falls back to best-effort cleanup
        # only, losing the hard-kill guarantee.
        assert transcriber._JOB is not None

    def test_create_job_failure_returns_none(self, capsys):
        with patch.object(transcriber._kernel32, "CreateJobObjectW", return_value=0):
            assert transcriber._make_kill_on_close_job() is None
        assert "CreateJobObjectW failed" in capsys.readouterr().out

    def test_set_information_failure_closes_the_leaked_handle(self, capsys):
        with patch.object(transcriber._kernel32, "SetInformationJobObject", return_value=0), \
             patch.object(transcriber._kernel32, "CloseHandle") as mock_close:
            assert transcriber._make_kill_on_close_job() is None
        mock_close.assert_called_once()
        assert "SetInformationJobObject failed" in capsys.readouterr().out

    def test_assign_open_process_failure_returns_false(self, capsys):
        job = transcriber._make_kill_on_close_job()
        try:
            with patch.object(transcriber._kernel32, "OpenProcess", return_value=0):
                assert transcriber._assign_to_job(job, os.getpid()) is False
        finally:
            transcriber._kernel32.CloseHandle(job)
        assert "OpenProcess failed" in capsys.readouterr().out

    def test_assign_job_object_failure_returns_false(self, capsys):
        job = transcriber._make_kill_on_close_job()
        try:
            with patch.object(transcriber._kernel32, "AssignProcessToJobObject", return_value=0):
                assert transcriber._assign_to_job(job, os.getpid()) is False
        finally:
            transcriber._kernel32.CloseHandle(job)
        assert "AssignProcessToJobObject failed" in capsys.readouterr().out


class TestPortAlreadyInUse:
    def test_start_raises_when_port_is_occupied(self, tmp_path):
        fake_server = tmp_path / "whisper-server.exe"
        fake_server.write_bytes(b"")
        fake_model = tmp_path / "model.bin"
        fake_model.write_bytes(b"")

        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        try:
            server = transcriber.WhisperServer(str(fake_server), str(fake_model), port=port)
            try:
                server.start()
                assert False, "expected RuntimeError for an occupied port"
            except RuntimeError as e:
                assert f"Port {port} is already in use" in str(e)
        finally:
            blocker.close()


class TestTranscribeRequest:
    def _server(self, **kwargs) -> transcriber.WhisperServer:
        return transcriber.WhisperServer("fake-server.exe", "fake-model.bin", **kwargs)

    def _mock_response(self, segments):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"segments": segments}
        return resp

    def test_request_asks_for_segment_timestamps_without_language_probs(self):
        server = self._server()
        server._session.post = MagicMock(
            return_value=self._mock_response([{"text": "hi", "start": 0.0, "end": 0.5}]))

        server.transcribe(np.zeros(1600, dtype=np.float32))

        _, kwargs = server._session.post.call_args
        assert kwargs["data"]["response_format"] == "verbose_json"
        assert kwargs["data"]["no_timestamps"] == "false"
        assert kwargs["data"]["no_language_probabilities"] == "true"

    def test_segments_are_routed_through_join_segments(self):
        # Two segments with a short gap: real behavior of join_segments()
        # (downgrade + lowercase-continue) must show up in transcribe()'s
        # return value, not just in the raw top-level "text" field.
        server = self._server(pause_threshold_s=0.35)
        segments = [
            {"text": " Claude.", "start": 0.0, "end": 1.0},
            {"text": " There's a lot of leeway.", "start": 1.1, "end": 3.0},
        ]
        server._session.post = MagicMock(return_value=self._mock_response(segments))

        result = server.transcribe(np.zeros(1600, dtype=np.float32))

        assert result == "Claude, there's a lot of leeway."

    def test_no_segments_returns_empty_string(self):
        server = self._server()
        server._session.post = MagicMock(return_value=self._mock_response([]))

        assert server.transcribe(np.zeros(1600, dtype=np.float32)) == ""
