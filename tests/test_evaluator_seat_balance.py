# -*- coding: utf-8 -*-
"""Regression test (2026-11-23): Evaluator always seated the candidate as p1,
so first-move advantage (~58% in self-play) inflated every gate result."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai import evaluator as evaluator_module  # noqa: E402
from ai.selfplay import SelfPlayReport  # noqa: E402


class SeatBalanceTests(unittest.TestCase):
    def test_candidate_plays_half_the_games_from_each_seat(self) -> None:
        calls = []

        def fake_run(self_, episodes, p1, p2, **kwargs):
            calls.append((episodes, p1, p2))
            # p1 always wins: a seat-blind evaluator would report 100%.
            return SelfPlayReport(episodes, 0, episodes, 0, 0, 10.0, 5.0, 3.0)

        candidate, opponent = object(), object()
        with mock.patch.object(evaluator_module.SelfPlayRunner, "run", fake_run):
            report = evaluator_module.Evaluator(None, None, seed=1).evaluate(candidate, opponent, games=10)
        self.assertEqual(calls, [(5, candidate, opponent), (5, opponent, candidate)])
        self.assertEqual((report.wins, report.losses, report.win_rate), (5, 5, 0.5))


if __name__ == "__main__":
    unittest.main()
