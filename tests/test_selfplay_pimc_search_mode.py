# -*- coding: utf-8 -*-
"""Regression test for ``selfplay.py --search pimc`` (2026-11-16).

Context: self-play training data had only ever been generated with
``MCTSAgent`` searching directly on the real ``GameState`` -- a "perfect
information" search that can see the opponent's hand, which does not match
what the trained model will face in an actual (hidden-information) match.
``ai/pimc.py``'s ``PIMCAgent`` already existed and was already wired into
``ai/selfplay.py``'s agent-agnostic ``SelfPlayRunner.run()`` (used by
``tools/eval_pimc_vs_mcts.py``), but ``selfplay.py`` -- the actual CLI that
``auto_train.py`` calls every cycle -- had no way to select it; it always
built ``MCTSAgent`` instances.

Added ``--search {perfect,pimc}`` and ``--pimc-worlds`` to ``selfplay.py``.
``pimc`` only supports ``--workers 1``: ``run_parallel``'s shared/async
inference worker pool is wired specifically for ``MCTSAgent``, and
``workers=1`` is also the measured-fastest setting on the actual training
desktop (single GTX 1070 Ti; see ``configs/formal_training.yaml``), so this
is not a real-world restriction.

These are subprocess-level tests (same style as exercising any other CLI
entry point) with tiny simulation counts/episode counts to stay fast.
"""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]


class SelfplayPimcSearchModeTests(unittest.TestCase):
    def _bootstrap_checkpoint(self, path: Path) -> None:
        subprocess.run(
            [sys.executable, "-c",
             f"from ai.network import KARDSNet; KARDSNet().save_checkpoint({str(path)!r})"],
            cwd=ROOT, check=True,
        )

    def test_pimc_search_produces_replay_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.pt"
            self._bootstrap_checkpoint(model)
            replay = tmp_path / "replay.pkl"
            metrics = tmp_path / "metrics.jsonl"
            result = subprocess.run(
                [sys.executable, "selfplay.py",
                 "--episodes", "1", "--mcts-simulations", "2", "--workers", "1",
                 "--max-actions", "150", "--device", "cpu",
                 "--replay", str(replay), "--metrics", str(metrics),
                 "--model", str(model), "--meta-decks",
                 "--search", "pimc", "--pimc-worlds", "2"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SelfPlayReport", result.stdout)
            self.assertTrue(replay.exists(), "pimc self-play should still write a replay buffer")

    def test_pimc_with_workers_greater_than_one_is_rejected_with_a_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.pt"
            self._bootstrap_checkpoint(model)
            result = subprocess.run(
                [sys.executable, "selfplay.py",
                 "--episodes", "1", "--mcts-simulations", "2", "--workers", "2",
                 "--max-actions", "100", "--device", "cpu",
                 "--replay", str(tmp_path / "replay.pkl"), "--metrics", str(tmp_path / "metrics.jsonl"),
                 "--model", str(model), "--meta-decks", "--search", "pimc"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("workers 1", result.stderr)

    def test_default_search_is_still_perfect_information_mcts(self) -> None:
        """No --search flag given -> unchanged legacy behaviour (MCTSAgent on
        the real GameState), so existing runs/configs that never pass
        --search are not silently switched to pimc."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.pt"
            self._bootstrap_checkpoint(model)
            replay = tmp_path / "replay.pkl"
            result = subprocess.run(
                [sys.executable, "selfplay.py",
                 "--episodes", "1", "--mcts-simulations", "2", "--workers", "1",
                 "--max-actions", "150", "--device", "cpu",
                 "--replay", str(replay), "--metrics", str(tmp_path / "metrics.jsonl"),
                 "--model", str(model), "--meta-decks"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(replay.exists())


if __name__ == "__main__":
    unittest.main()
