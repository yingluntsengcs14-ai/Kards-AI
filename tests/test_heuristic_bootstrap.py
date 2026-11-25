# -*- coding: utf-8 -*-
"""2026-11-25: heuristic bootstrap and MCTS value blending.

* A fresh run with ``training.bootstrap`` first plays greedy-heuristic games
  and trains the brand-new Champion on them before self-play.
* ``self_play.heuristic_value_weight`` blends ai/heuristic.py's board score
  into MCTS leaf values.
* GreedyHeuristicAgent exposes a proper soft policy target.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_auto_train_bootstrap import TINY_CONFIG  # noqa: E402
from ai.heuristic import GreedyHeuristicAgent, evaluate  # noqa: E402
from ai.mcts import MCTS, MCTSNode  # noqa: E402
from ai.observation import ObservationEncoder  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.testing import build_random_deck  # noqa: E402

CARDS = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")


def fresh_state():
    deck = build_random_deck(CARDS, seed=5, main_nation="Germany", ally_nation="Italy")
    env = Simulator(CARDS); env.reset(deck, deck, nations=("Germany", "Germany"), ally_nations=("Italy", "Italy"), seed=1)
    return env.get_state(), env


class HeuristicBootstrapTests(unittest.TestCase):
    def test_soft_policy_is_a_distribution_peaked_on_the_choice(self) -> None:
        state, env = fresh_state()
        agent = GreedyHeuristicAgent(); agent.set_state(state, state.current_player, CARDS)
        legal = env.get_available_actions()
        choice = agent.select_action({}, legal)
        self.assertAlmostEqual(sum(agent.last_policy), 1.0, places=6)
        self.assertEqual(agent.last_policy.index(max(agent.last_policy)), legal.index(choice))

    def test_mcts_blend_mixes_network_and_board_score(self) -> None:
        state, _ = fresh_state()
        state.players["p2"].hq.current_health = 4
        search = MCTS(None, ObservationEncoder(CARDS), simulations=1)
        node = MCTSNode(state, state.current_player)
        self.assertEqual(search._blend(node, "p1", 0.2), 0.2)  # weight 0: unchanged
        search.heuristic_value_weight = 0.5
        self.assertAlmostEqual(search._blend(node, "p1", 0.2), 0.5 * 0.2 + 0.5 * evaluate(state, "p1"))

    def test_fresh_run_bootstraps_from_heuristic_games(self) -> None:
        import torch
        config = TINY_CONFIG.replace("  updates_per_cycle: 2\n", "  updates_per_cycle: 2\n  bootstrap:\n    games: 2\n    updates: 3\n") \
                            .replace("  replay_capacity: 500\n", "  replay_capacity: 500\n  heuristic_value_weight: 0.5\n  heuristic_value_decay: 0.9\n") \
                            .replace("opponents: [random]", "opponents: [heuristic]")
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"; path = Path(tmp) / "cfg.yaml"; path.write_text(config, encoding="utf-8")
            env = os.environ.copy(); env["KARDS_RUNS_DIR"] = str(runs)
            result = subprocess.run([sys.executable, "auto_train.py", "--config", str(path), "--cycles", "1", "--device", "cpu"],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn("bootstrap: 2 heuristic games", result.stdout)
            trainee = torch.load(runs / "trainee.pt", map_location="cpu", weights_only=False)
            self.assertEqual(int(trainee["training_step"]), 3 + 2)  # bootstrap updates, then one cycle


if __name__ == "__main__":
    unittest.main()
