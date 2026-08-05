"""Tests for the Windows Job Object that guarantees whisper-server.exe can
never outlive this app, even if the app is killed outright (crash, Task
Manager "End task", power loss) rather than exited cleanly."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from unittest.mock import patch

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
