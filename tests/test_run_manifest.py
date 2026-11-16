# -*- coding: utf-8 -*-
"""Regression test for ``ai.runtime.write_run_manifest`` (2026-11-16).

Context: ``claude/訓練工作流.md``'s run-folder design calls for a
``manifest.json`` per run recording git provenance (commit + dirty flag),
which rule generation produced the self-play data (``data/rules/card_rules.json``
hash + implemented/partial/unresolved counts), the environment
(Python/PyTorch/CUDA/GPU), the config used, and a seed -- none of which
``auto_train.py`` previously wrote anywhere (only a per-candidate-checkpoint
``git_commit`` field existed, no reproducibility snapshot for the whole run).

``write_run_manifest`` is called once at ``auto_train.py`` startup and again
at the top of every cycle, so it must create the file on first call and
*refresh* it (new git/rule state) on every later call while keeping
``run_id``/``start_time_utc`` stable -- that's what these tests check.
"""
from pathlib import Path
import json
import tempfile
import time
import unittest

from ai.runtime import write_run_manifest


class RunManifestTests(unittest.TestCase):
    def test_first_write_populates_all_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            path = write_run_manifest(run_dir, "configs/formal_training.yaml", seed=42)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["seed"], 42)
            self.assertEqual(data["run_id"], "run")
            self.assertIn("commit", data["git"])
            self.assertEqual(data["rules"]["total"], 1646)
            self.assertIn("implemented", data["rules"]["by_status"])
            self.assertIn("python", data["environment"])

    def test_refresh_keeps_run_id_and_start_time_but_updates_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            first = json.loads(write_run_manifest(run_dir, "c.yaml", seed=7).read_text(encoding="utf-8"))
            time.sleep(0.01)
            second = json.loads(write_run_manifest(run_dir, "c.yaml", extra={"last_cycle": 2}).read_text(encoding="utf-8"))
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["start_time_utc"], second["start_time_utc"])
            self.assertNotEqual(first["last_updated_utc"], second["last_updated_utc"])
            self.assertEqual(second["seed"], 7, "seed should persist when a later call omits it")
            self.assertEqual(second["last_cycle"], 2)


if __name__ == "__main__":
    unittest.main()
