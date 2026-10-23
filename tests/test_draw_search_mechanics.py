"""Focused regression tests for Milestone 2 draw/search/generate/copy/discard/mill.

These assert final GameState and event log, not merely "no exception".
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction


ROOT = Path(__file__).parents[1]


class DrawSearchMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")

    def _state(self, p1_hand=(), p2_hand=(), p1_units=(), p1_deck=("garrison",)):
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "USA", deck=list(p1_deck), hand=list(p1_hand),
                                   resources=ResourceState(10, 10),
                                   units=[UnitState(uid, cid, 2, 3, "p1", "support_line") for uid, cid in p1_units]),
                "p2": PlayerState("p2", "Germany", deck=["garrison"], hand=list(p2_hand),
                                   resources=ResourceState(10, 10)),
            },
        )

    def test_mill_removes_top_cards_of_deck(self) -> None:
        # alpenfestung: "Remove the top 10 cards of your deck."
        deck = ["garrison", "home_guard", "radar", "garrison", "home_guard", "radar",
                "garrison", "home_guard", "radar", "garrison", "home_guard", "radar"]
        state = self._state(p1_hand=("alpenfestung",), p1_deck=deck)
        before = len(state.players["p1"].deck)
        PlayCardAction("p1", "alpenfestung").execute(state, self.cards)
        self.assertEqual(len(state.players["p1"].deck), before - 10)
        self.assertEqual(len(state.removed_cards), 10)
        self.assertTrue(any(e.get("event") == "cards_milled" and e.get("count") == 10
                            for e in state.event_log))

    def test_copy_unit_deploys_token(self) -> None:
        # hell_on_wheels: "Duplicate a friendly tank ..." (the rest is unresolved).
        state = self._state(p1_hand=("hell_on_wheels",), p1_units=(("t1", "greif"),),
                            p1_deck=("garrison",))
        before = len(state.players["p1"].units)
        PlayCardAction("p1", "hell_on_wheels").execute(state, self.cards)
        after = len(state.players["p1"].units)
        self.assertEqual(after, before + 1)
        copies = [u for u in state.players["p1"].units if u.card_id == "greif" and "copy" in u.instance_id]
        self.assertEqual(len(copies), 1)
        self.assertTrue(any(e.get("event") == "unit_copied" for e in state.event_log))

    def test_draw_top_matching_draws_matching_unit(self) -> None:
        # free_french_navy: "Draw the top unit of your deck with an operation cost of 3."
        state = self._state(p1_hand=("free_french_navy",), p1_deck=("royal_fusiliers", "garrison"))
        PlayCardAction("p1", "free_french_navy").execute(state, self.cards)
        self.assertIn("royal_fusiliers", state.players["p1"].hand)
        self.assertEqual(state.players["p1"].hq.current_health, 23)  # HQ defense is its health

    def test_draw_top_matching_skips_non_matching(self) -> None:
        # Top of deck is a unit with op cost != 3 -> must NOT be drawn.
        state = self._state(p1_hand=("free_french_navy",), p1_deck=("garrison", "royal_fusiliers"))
        PlayCardAction("p1", "free_french_navy").execute(state, self.cards)
        self.assertNotIn("garrison", state.players["p1"].hand)

    def test_discard_hand_empties_hand(self) -> None:
        state = self._state(p1_hand=("garrison", "home_guard"), p1_deck=("garrison",))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(RuleAction("discard_hand", "owner"), state, EffectContext("p1"))
        self.assertEqual(len(state.players["p1"].hand), 0)
        self.assertEqual(len(state.graveyard["p1"]), 2)
        self.assertTrue(any(e.get("event") == "hand_discarded" and e.get("count") == 2
                            for e in state.event_log))


if __name__ == "__main__":
    unittest.main()
