# -*- coding: utf-8 -*-
"""Regression test for start_training.py's stale-lock detection (2026-11-19).

Context: the user's very first launch attempt crashed (a separate bug, see
tests/test_auto_train_bootstrap.py), leaving runs/training.lock and
runs/training.pid behind. The *next* launch attempt then failed with
"Training lock already exists" even though the old process was confirmed
dead (Get-Process on Windows reported it gone) -- start_training.py's
stale-lock cleanup never ran.

Root cause: it checked liveness with ``os.kill(pid, 0)``. That is a safe,
non-destructive existence probe on POSIX, but ``os.kill`` has no such
special case on Windows -- passing signal 0 there calls
``TerminateProcess(handle, 0)``, which actually terminates whatever process
currently holds that PID. Windows recycles PIDs aggressively, so the old
dead training run's PID was very likely reassigned to some other, unrelated
process by the time the user retried; the call would have (a) silently
killed that unrelated process and (b) returned without raising, so the
stale-lock cleanup branch (written for the POSIX "process is gone" case)
never ran -- reproducing exactly the "lock already exists" failure.

Fixed with a platform-aware ``_process_alive()``: query-only (``tasklist``)
on Windows, real ``os.kill(pid, 0)`` on POSIX. These tests run on this
(POSIX) CI/sandbox machine, so they exercise the POSIX branch directly and
the Windows branch via a monkeypatched ``sys.platform`` + a fake
``subprocess.run`` -- asserting in particular that ``os.kill`` is never
called once ``sys.platform`` is ``"win32"``, which is the actual bug this
fixes.
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from start_training import _process_alive  # noqa: E402


class ProcessAliveTests(unittest.TestCase):
    def test_posix_current_process_is_alive(self) -> None:
        self.assertTrue(_process_alive(os.getpid()))

    def test_posix_dead_process_is_not_alive(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()  # guaranteed to have exited, so its pid is free to query
        self.assertFalse(_process_alive(child.pid))

    def test_windows_branch_queries_tasklist_and_never_signals(self) -> None:
        """The actual regression: must not call os.kill() on Windows."""
        fake_result = mock.Mock(stdout="python.exe   1234 Console   1   12,345 K\n")
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("start_training.subprocess.run", return_value=fake_result) as run, \
             mock.patch("start_training.os.kill", side_effect=AssertionError("must not call os.kill on win32")) as kill:
            self.assertTrue(_process_alive(1234))
            run.assert_called_once()
            self.assertTrue(any("1234" in str(part) for part in run.call_args.args[0]))
            kill.assert_not_called()

    def test_windows_branch_reports_dead_when_pid_absent_from_tasklist(self) -> None:
        fake_result = mock.Mock(stdout="INFO: No tasks are running which match the specified criteria.\n")
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("start_training.subprocess.run", return_value=fake_result), \
             mock.patch("start_training.os.kill", side_effect=AssertionError("must not call os.kill on win32")):
            self.assertFalse(_process_alive(9999))


if __name__ == "__main__":
    unittest.main()
