# -*- coding: utf-8 -*-
"""ai/heuristic.py: board score is antisymmetric and bounded, and the greedy
agent always returns one of the legal actions."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.agents import RandomAgent  # noqa: E402
from ai.heuristic import GreedyHeuristicAgent, evaluate  # noqa: E402
from ai.observation import ObservationEncoder  # noqa: E402
from ai.replay_buffer import ReplayBuffer  # noqa: E402
from ai.selfplay import SelfPlayRunner  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.testing import build_random_deck  # noqa: E402

CARDS = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")


class HeuristicTests(unittest.TestCase):
    def test_score_is_antisymmetric_and_bounded(self) -> None:
        deck = build_random_deck(CARDS, seed=5, main_nation="Germany", ally_nation="Italy")
        env = Simulator(CARDS); env.reset(deck, deck, nations=("Germany", "Germany"), ally_nations=("Italy", "Italy"), seed=1)
        state = env.get_state()
        state.players["p1"].hq.current_health = 7
        a, b = evaluate(state, "p1"), evaluate(state, "p2")
        self.assertAlmostEqual(a, -b)
        self.assertTrue(-1.0 < a < 0.0)

    def test_greedy_agent_plays_full_games(self) -> None:
        runner = SelfPlayRunner(CARDS, ObservationEncoder(CARDS), ReplayBuffer(), max_actions=200, seed=2)
        report = runner.run(2, GreedyHeuristicAgent(), RandomAgent(3), deck_pool=[("Germany", "Italy")])
        self.assertEqual(report.p1_wins + report.p2_wins + report.draws, 2)
        self.assertEqual(runner.engine_errors, [])


if __name__ == "__main__":
    unittest.main()
