# -*- coding: utf-8 -*-
"""Regression test for an MCTS sign error across turn boundaries (2026-11-22).

``_backpropagate`` stores each node's value from the perspective of the player
to move *at that node*. ``_select`` then maximised ``child.value`` directly.
That is only right when the child has the same player to move as the parent.
After an end-turn action the child belongs to the opponent, so the parent was
maximising the *opponent's* value -- end turn looked best exactly when it
helped the opponent. With a value net that rates "opponent to move with fresh
resources" highly, end turn was systematically over-visited, and training on
those visit counts made the formal run's policy pass more and play fewer cards
(end-turn share 31% -> 45% of actions, win rate vs random 87% -> ~63%).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.mcts import MCTS, MCTSNode  # noqa: E402


class TurnBoundarySignTests(unittest.TestCase):
    def test_parent_scores_opponent_child_from_its_own_perspective(self) -> None:
        parent = MCTSNode(state=None, player_to_move="p1")  # type: ignore[arg-type]
        parent.actions = ["play_card", "end_turn"]  # type: ignore[list-item]
        parent.priors = [0.5, 0.5]; parent.prior_order = [0, 1]
        same_turn = MCTSNode(state=None, player_to_move="p1", parent=parent, prior_probability=0.5)  # type: ignore[arg-type]
        same_turn.visit_count, same_turn.value_sum = 10, 0.0      # p1's view: even
        opponent_turn = MCTSNode(state=None, player_to_move="p2", parent=parent, prior_probability=0.5)  # type: ignore[arg-type]
        opponent_turn.visit_count, opponent_turn.value_sum = 10, 9.0  # p2's view: +0.9, i.e. bad for p1
        parent.children = {0: same_turn, 1: opponent_turn}; parent.visit_count = 20

        search = MCTS(None, None, simulations=1, c_puct=0.0)  # pure exploitation
        self.assertIs(search._select(parent, None), same_turn)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
