# -*- coding: utf-8 -*-
"""Regression test for ``tools/render_replay.py`` (2026-11-19).

Context: the user asked whether the final project presentation could
include a "對戰覆盤" (battle replay) -- picked a graphical board over a
plain text log. ``tools/watch_game.py`` already writes a text-log HTML for
rule debugging, but it only plays with the cheap active/random/rule
baselines, not the trained checkpoint, and its board is a <pre> text
snapshot. ``render_replay.py`` is a separate, presentation-focused tool:
it can drive the real NeuralAgent/MCTSAgent/PIMCAgent on a checkpoint
(exactly like selfplay.py/evaluate.py build them) and writes a single
self-contained HTML page with an embedded step list and inline JS that
draws a card-table board (HQ bars, frontline/support-line boxes) with
prev/next/autoplay controls -- verified against the real desktop with
Playwright separately; these are subprocess-level CLI smoke tests.
"""
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]


class RenderReplayTests(unittest.TestCase):
    def test_baseline_agents_produce_a_playable_html_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "replay.html"
            result = subprocess.run(
                [sys.executable, "tools/render_replay.py",
                 "--p1", "Germany:Italy", "--p2", "Soviet:Britain", "--seed", "7",
                 "--agent1", "rule", "--agent2", "random", "--max-actions", "80",
                 "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            # The embedded step data must be valid JSON with a non-trivial
            # board (at least one side's frontline/support populated at some
            # point), since an empty replay would silently "work" but render
            # a blank board for the whole game.
            match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', text, re.S)
            self.assertIsNotNone(match)
            data = json.loads(match.group(1))
            self.assertGreater(len(data["steps"]), 1)
            any_units = any(
                step["board"]["players"][pid]["frontline"] or step["board"]["players"][pid]["support"]
                for step in data["steps"] for pid in ("p1", "p2")
            )
            self.assertTrue(any_units, "no unit ever appeared on either board across the whole game")

    def test_network_driven_agents_do_not_crash(self) -> None:
        """mcts vs pimc on a freshly-bootstrapped (untrained) checkpoint --
        the same construction selfplay.py/evaluate.py use, just small/fast
        enough for a test."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.pt"
            subprocess.run(
                [sys.executable, "-c",
                 f"from ai.network import KARDSNet; KARDSNet().save_checkpoint({str(model)!r})"],
                cwd=ROOT, check=True,
            )
            out = tmp_path / "replay.html"
            result = subprocess.run(
                [sys.executable, "tools/render_replay.py", "--meta-decks", "--seed", "3",
                 "--agent1", "mcts", "--agent2", "pimc", "--model", str(model), "--device", "cpu",
                 "--mcts-simulations", "4", "--pimc-worlds", "2", "--max-actions", "30",
                 "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.exists())

    def test_neural_agent_without_search_also_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.pt"
            subprocess.run(
                [sys.executable, "-c",
                 f"from ai.network import KARDSNet; KARDSNet().save_checkpoint({str(model)!r})"],
                cwd=ROOT, check=True,
            )
            out = tmp_path / "replay.html"
            result = subprocess.run(
                [sys.executable, "tools/render_replay.py", "--agent1", "neural", "--agent2", "rule",
                 "--model", str(model), "--device", "cpu", "--max-actions", "40", "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
