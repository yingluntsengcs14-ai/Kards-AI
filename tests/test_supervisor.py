# -*- coding: utf-8 -*-
"""tools/supervisor.py: only commits marked [auto-restart] trigger a pull and
restart; [fresh-runs] additionally sets runs/ aside (renamed, never deleted)."""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import supervisor  # noqa: E402


def fake_git(log_output):
    def run(*args, check=False):
        out = log_output if args[0] == "log" else ""
        return subprocess.CompletedProcess(["git", *args], 0, out, "")
    return run


class SupervisorTests(unittest.TestCase):
    def test_pending_commits_splits_full_messages(self) -> None:
        output = "aaa Fix thing\n\nbody [auto-restart]\n\x00bbb Docs only\n\x00"
        with mock.patch.object(supervisor, "git", fake_git(output)):
            commits = supervisor.pending_commits("main")
        self.assertEqual(len(commits), 2)
        self.assertTrue(supervisor.RESTART_MARKER in commits[0])
        self.assertFalse(supervisor.RESTART_MARKER in commits[1])

    def test_one_loop_restarts_and_renames_runs_on_fresh_marker(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "runs").mkdir()
            calls = []
            commits = [["abc Fix [auto-restart] [fresh-runs]"], []]
            with mock.patch.object(supervisor, "ROOT", root), \
                 mock.patch.object(supervisor, "LOG", root / "supervisor.log"), \
                 mock.patch.object(supervisor, "pending_commits", lambda branch: commits.pop(0) if commits else []), \
                 mock.patch.object(supervisor, "training_alive", lambda runs: True), \
                 mock.patch.object(supervisor, "stop_training", lambda runs: calls.append("stop")), \
                 mock.patch.object(supervisor, "start_training", lambda config: calls.append("start") or True), \
                 mock.patch.object(supervisor, "git", fake_git("")), \
                 mock.patch.object(supervisor.time, "sleep", side_effect=[None, KeyboardInterrupt]), \
                 mock.patch.object(sys, "argv", ["supervisor.py"]):
                with self.assertRaises(KeyboardInterrupt):
                    supervisor.main()
            self.assertEqual(calls, ["stop", "start"])
            self.assertFalse((root / "runs").exists())
            self.assertEqual(len(list(root.glob("runs_*"))), 1)


if __name__ == "__main__":
    unittest.main()
