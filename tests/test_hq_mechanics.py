"""Focused regression tests for HQ mechanics (M19-M20).

Each handler gets at least one test asserting final State and event log
(not merely "no exception").  Per task book: "每条通用 Action 至少一个测试."
"""

import unittest
from pathlib import Path

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction
from simulator.effects.resolver import EffectContext

ROOT = Path(__file__).parents[1]


class HQMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_hand=(), p2_hand=(), p1_units=(), p2_units=(),
               p1_deck=None, p2_deck=None) -> GameState:
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain",
                                   deck=p1_deck or ["garrison"]*15,
                                   hand=list(p1_hand),
                                   resources=ResourceState(10, 10),
                                   units=[UnitState(uid, cid, 2, 3, "p1", "support_line") for uid, cid in p1_units]),
                "p2": PlayerState("p2", "Germany",
                                   deck=p2_deck or ["garrison"]*15,
                                   hand=list(p2_hand),
                                   resources=ResourceState(10, 10),
                                   units=[UnitState(uid, cid, 2, 3, "p2", "support_line") for uid, cid in p2_units]),
            },
        )

    # --- hq_take_damage ---

    def test_hq_take_damage_via_five_year_plan(self) -> None:
        """'Your HQ takes 5 damage' reduces current_health through defense_modifier."""
        state = self._state(p1_hand=("five_year_plan",), p2_hand=(),
                            p1_deck=["garrison"]*10, p2_deck=["garrison"]*10)
        before = state.players["p1"].hq.current_health
        defense = state.players["p1"].hq.defense_modifier
        PlayCardAction("p1", "five_year_plan").execute(state, self.cards)
        # Damage is reduced by defense_modifier, capped at 0.
        expected_damage = max(0, 5 - defense)
        self.assertEqual(state.players["p1"].hq.current_health, before - expected_damage,
                         "HQ health should decrease by damage-minus-defense")
        self.assertTrue(
            any(e.get("event") == "hq_damaged" and e.get("player_id") == "p1"
                for e in state.event_log),
            "event log must contain hq_damaged for p1")

    def test_hq_take_damage_rule_action_direct(self) -> None:
        """Direct handler call: hq_take_damage reduces health via HQResolver."""
        state = self._state()
        before = state.players["p1"].hq.current_health
        action = RuleAction("hq_take_damage", "owner", amount=3)
        self.engine._execute_action(action, state, EffectContext("p1", "test_card"))
        self.assertEqual(state.players["p1"].hq.current_health, before - 3,
                         "hq_take_damage(3) should reduce health by 3 (no defense)")

    # --- set_hq_defense ---

    def test_set_hq_defense_via_alpenfestung(self) -> None:
        """'Set your HQ's defense to 25': the HQ's health becomes 25 (defense is health)."""
        state = self._state(p1_hand=("alpenfestung",), p2_hand=(),
                            p1_deck=["garrison"]*15, p2_deck=["garrison"]*15)
        PlayCardAction("p1", "alpenfestung").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.current_health, 25)
        self.assertEqual(state.players["p1"].hq.defense_modifier, 0, "defense is health, not damage reduction")
        self.assertTrue(
            any(e.get("event") == "hq_defense_set" and e.get("player_id") == "p1"
                for e in state.event_log),
            "event log must contain hq_defense_set for p1")

    def test_set_hq_defense_rule_action_direct(self) -> None:
        """Direct handler call: set_hq_defense sets the HQ's health."""
        state = self._state()
        action = RuleAction("set_hq_defense", "owner", amount=10)
        self.engine._execute_action(action, state, EffectContext("p1", "test_card"))
        self.assertEqual(state.players["p1"].hq.current_health, 10)

    # --- hq_immune ---

    def test_hq_immune_sets_flag(self) -> None:
        """hq_immune sets immune_until_end_of_turn flag."""
        state = self._state()
        self.assertFalse(state.players["p1"].hq.immune_until_end_of_turn)
        action = RuleAction("hq_immune", "owner")
        self.engine._execute_action(action, state, EffectContext("p1", "test_card"))
        self.assertTrue(state.players["p1"].hq.immune_until_end_of_turn,
                        "immune_until_end_of_turn should be True after hq_immune")
        self.assertTrue(
            any(e.get("event") == "hq_immune_set" and e.get("player_id") == "p1"
                for e in state.event_log),
            "event log must contain hq_immune_set for p1")

    def test_hq_immune_cleared_at_end_of_turn(self) -> None:
        """hq_immune flag is reset by the end-of-turn cleanup phase."""
        state = self._state()
        state.players["p1"].hq.immune_until_end_of_turn = True
        # Simulate end-turn cleanup (the engine's _process_phase with phase="end")
        # The end_turn handler runs via TurnManager, but for a direct flag reset
        # we simulate what _process_phase does.
        from simulator.core.turn import TurnManager
        # Advance turn → cleanup runs
        clone = state
        # Trigger end_turn action via engine to hit the cleanup code path.
        action = RuleAction("end_turn", "owner")
        self.engine._execute_action(action, clone, EffectContext("p1", "end_turn"))
        self.assertFalse(clone.players["p1"].hq.immune_until_end_of_turn,
                         "immune_until_end_of_turn should be reset at end of turn")


if __name__ == "__main__":
    unittest.main()
