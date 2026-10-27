"""M8: condition-engine evaluation + event triggers + computed values.

Locks:
- Conditional actions are skipped when their condition is not satisfied.
- Event-gated self-buffs resolve to the SOURCE unit (not selected_target).
- "equal to [amount]" computed values resolve (845th_rifles -> HQ gains the
  exact damage dealt).
"""

import sys
import unittest

sys.path.insert(0, ".")

from simulator.cards.loader import CardDatabase
from simulator.effects.resolver import EffectContext
from simulator.rules.condition import evaluate
from simulator.rules.native import NativeRuleEngine
from simulator.core.state import GameState, PlayerState, UnitState


class ConditionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file("data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    # --- ConditionEvaluator mapping (direct unit checks) ---
    def test_evaluator_event_gating(self) -> None:
        ctx = EffectContext("p1", event="on_enemy_hq_damaged")
        self.assertTrue(evaluate("when it damages the enemy hq.", "on_enemy_hq_damaged", ctx, self.cards))
        self.assertFalse(evaluate("when it damages the enemy hq.", "on_damage", ctx, self.cards))
        self.assertFalse(evaluate("when it survives combat.", "on_enemy_hq_damaged", ctx, self.cards))

    def test_evaluator_played_card_filter(self) -> None:
        intel_ctx = EffectContext("p1", event="on_friendly_card_played", metadata={"played_card_id": "resourcefulness"})
        self.assertTrue(evaluate("when you play a card with Intel.", "on_friendly_card_played", intel_ctx, self.cards))
        radar_ctx = EffectContext("p1", event="on_friendly_card_played", metadata={"played_card_id": "radar"})
        self.assertFalse(evaluate("when you play a card with Intel.", "on_friendly_card_played", radar_ctx, self.cards))
        # "give an order" only fires for order-type plays.
        self.assertTrue(evaluate("when you give an order.", "on_friendly_card_played", radar_ctx, self.cards))
        # A played unit (not an order) must not satisfy "give an order".
        unit_ctx = EffectContext("p1", event="on_friendly_card_played", metadata={"played_card_id": "hurricane_mk_i"})
        self.assertFalse(evaluate("when you give an order.", "on_friendly_card_played", unit_ctx, self.cards))
        self.assertFalse(evaluate("when you play a card with Intel.", "on_friendly_card_played", unit_ctx, self.cards))

    def test_evaluator_unmodeled_condition_is_not_satisfied(self) -> None:
        """看不懂的條件不成立，並記一筆事件（以前一律當成立）。"""
        state = GameState(current_player="p1", players={"p1": PlayerState("p1", "Britain"), "p2": PlayerState("p2", "Germany")})
        self.assertFalse(evaluate("if the moon is full.", "on_play", EffectContext("p1"), self.cards, state))
        self.assertTrue(any(e.get("event") == "native_rule_condition_unmodeled" for e in state.event_log))
        # Without a state there is no board to read, so a board condition is unmodeled too.
        self.assertFalse(evaluate("if enemy has 3 or more units.", "on_play", EffectContext("p1"), self.cards))

    def test_controls_predicate_reads_nation_type_and_threshold(self) -> None:
        state = GameState(current_player="p1", players={
            "p1": PlayerState("p1", "Germany", units=[UnitState("t", "panzer_ivh", 5, 5, "p1", "support_line")]),
            "p2": PlayerState("p2", "Britain", units=[]),
        }, battlefield={"frontline": [], "support_line": ["t"]})
        ctx = EffectContext("p1")
        self.assertTrue(evaluate("if you control a german tank", "on_play", ctx, self.cards, state))
        self.assertTrue(evaluate("if you control a tank with 5 or more attack.", "on_play", ctx, self.cards, state))
        self.assertFalse(evaluate("if you control a tank with 6 or more attack.", "on_play", ctx, self.cards, state))
        self.assertFalse(evaluate("if you control a bomber", "on_play", ctx, self.cards, state))
        self.assertFalse(evaluate("if the enemy controls an air unit", "on_play", ctx, self.cards, state))
        # An unknown descriptor ("commando") is not silently ignored.
        self.assertFalse(evaluate("if you control a commando", "on_play", ctx, self.cards, state))
        self.assertTrue(any(e.get("event") == "native_rule_condition_unmodeled" for e in state.event_log))

    def test_evaluator_models_common_board_state_conditions(self) -> None:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain", units=[UnitState("self", "garrison", 4, 3, "p1", "frontline")]),
                "p2": PlayerState("p2", "Germany", units=[UnitState("e1", "m3a3_honey", 2, 3, "p2"), UnitState("e2", "m3a3_honey", 2, 3, "p2")]),
            },
            battlefield={"frontline": ["self"], "support_line": ["e1", "e2"]},
        )
        ctx = EffectContext("p1", source_unit_id="self")
        self.assertTrue(evaluate("if enemy has 2 or more units.", "on_play", ctx, self.cards, state))
        self.assertTrue(evaluate("if you control a unit with 4 or more attack.", "on_play", ctx, self.cards, state))
        self.assertFalse(evaluate("if you control another unit with 4 or more attack.", "on_play", ctx, self.cards, state))
        self.assertTrue(evaluate("if you control the frontline.", "on_play", ctx, self.cards, state))
        self.assertTrue(evaluate("if you have no cards in hand.", "on_play", ctx, self.cards, state))

    # --- End-to-end: condition NOT met -> action skipped ---
    def test_nakajima_does_not_buff_on_non_intel_play(self) -> None:
        from simulator.actions.action import PlayCardAction
        from simulator.core.state import GameState, PlayerState, ResourceState, UnitState

        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Japan", hand=["radar"], units=[UnitState("u", "nakajima_b5n2", 2, 2, "p1", "support_line")]),
                "p2": PlayerState("p2", "Germany", units=[]),
            },
            battlefield={"frontline": [], "support_line": ["u"]},
        )
        unit = state.players["p1"].units[0]
        before = (unit.attack, unit.defense)
        # radar is an order but NOT intel -> nakajima must NOT buff.
        PlayCardAction("p1", "radar").execute(state, self.cards)
        self.assertEqual((unit.attack, unit.defense), before)

    def test_navy_listener_runs_once_per_turn(self) -> None:
        """Rule-level once-per-turn applies after the matching Navy event."""
        from simulator.actions.action import PlayCardAction
        from simulator.core.state import ResourceState

        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState(
                    "p1", "Germany", hand=["free_french_navy", "free_french_navy"],
                    units=[UnitState("fw", "fw_189", 2, 2, "p1", "support_line")],
                    resources=ResourceState(20, 20),
                ),
                "p2": PlayerState("p2", "Britain"),
            },
            battlefield={"frontline": [], "support_line": ["fw"]},
        )
        PlayCardAction("p1", "free_french_navy").execute(state, self.cards)
        PlayCardAction("p1", "free_french_navy").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand.count("production"), 1)

    # --- End-to-end: computed value (equal to damage) ---
    def test_equal_to_damage_adds_exact_hq_defense(self) -> None:
        from simulator.effects.resolver import EffectResolver
        from simulator.core.state import GameState, PlayerState, ResourceState, UnitState

        state = GameState(
            current_player="p2",
            players={
                "p1": PlayerState("p1", "USA", units=[UnitState("r", "845th_rifles", 2, 5, "p1", "frontline")]),
                "p2": PlayerState("p2", "Germany", units=[]),
            },
            battlefield={"frontline": ["r"], "support_line": []},
        )
        unit = state.players["p1"].units[0]
        start = len(state.event_log)
        EffectResolver(self.cards).resolve(
            {"type": "damage", "target": "selected_target", "value": {"amount": 3}},
            state, EffectContext("p2", target_unit_id=unit.instance_id),
        )
        self.engine.emit_damage_since(state, start)
        self.assertEqual(state.players["p1"].hq.current_health, 23)  # HQ defense is its health


if __name__ == "__main__":
    unittest.main()
