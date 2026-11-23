# -*- coding: utf-8 -*-
"""Regression test (2026-11-23): one simulator rule bug in one self-play game
(a stale unit id in USS MISSOURI's retreat_adjacent lookup) crashed the whole
formal run at 04:55 and it sat idle until morning. SelfPlayRunner now drops
that game and carries on, but still raises if errors are systemic."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.agents import RandomAgent  # noqa: E402
from ai.observation import ObservationEncoder  # noqa: E402
from ai.replay_buffer import ReplayBuffer  # noqa: E402
from ai.selfplay import SelfPlayRunner  # noqa: E402
from simulator.actions.validator import ActionValidationError  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402

CARDS = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
ENCODER = ObservationEncoder(CARDS)
REAL_STEP = Simulator.step_fast


class EngineErrorSkipTests(unittest.TestCase):
    def _run(self, episodes: int, fail_games: set[int]):
        state = {"game": -1}
        real_reset = Simulator.reset

        def reset(self_, *args, **kwargs):
            state["game"] += 1
            return real_reset(self_, *args, **kwargs)

        def step(self_, action):
            if state["game"] in fail_games:
                raise ActionValidationError("Unknown unit instance: test-unit")
            return REAL_STEP(self_, action)

        runner = SelfPlayRunner(CARDS, ENCODER, ReplayBuffer(), max_actions=40, seed=3)
        with mock.patch.object(Simulator, "reset", reset), mock.patch.object(Simulator, "step_fast", step):
            report = runner.run(episodes, RandomAgent(1), RandomAgent(2), deck_pool=[("Germany", "Italy")])
        return runner, report

    def test_one_broken_game_is_skipped_and_the_rest_complete(self) -> None:
        runner, report = self._run(4, {1})
        self.assertEqual(len(runner.engine_errors), 1)
        self.assertEqual(report.p1_wins + report.p2_wins + report.draws, 3)
        self.assertEqual(len(runner.game_records), 3)

    def test_systemic_errors_still_raise(self) -> None:
        with self.assertRaises(ActionValidationError):
            self._run(10, set(range(10)))


if __name__ == "__main__":
    unittest.main()
