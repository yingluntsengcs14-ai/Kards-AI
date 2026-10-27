"""Regression tests for documented, system-level KARDS rule requirements."""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, _operation_cost
from simulator.actions.validator import ActionValidationError
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.core.game import Simulator
from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction

ROOT = Path(__file__).parents[1]


class CoreRuleGapsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")

    def _state(self) -> GameState:
        attacker = UnitState("attacker", "greif", 3, 4, "p1", "frontline")
        guard = UnitState("guard", "garrison", 1, 1, "p2", "support_line")
        return GameState(
            current_player="p1", battlefield={"frontline": ["attacker"], "support_line": ["guard"]},
            players={
                "p1": PlayerState("p1", resources=ResourceState(10, 10), units=[attacker], deck=["garrison"]),
                "p2": PlayerState("p2", resources=ResourceState(10, 10), units=[guard]),
            },
        )

    def test_guard_adjacent_to_hq_blocks_normal_hq_attack(self) -> None:
        with self.assertRaises(ActionValidationError):
            AttackAction("p1", "attacker").validate(self._state(), self.cards)

    def test_terminal_damage_stops_remaining_effect_actions(self) -> None:
        state = self._state()
        state.players["p2"].units.clear()
        state.battlefield["support_line"].clear()
        EffectResolver(self.cards).resolve(
            {"actions": [
                {"type": "damage", "target": "enemy_hq", "value": {"amount": 20}},
                {"type": "draw", "value": {"amount": 1}},
            ]}, state, EffectContext("p1"),
        )
        self.assertEqual(state.players["p2"].hq.current_health, 0)
        self.assertEqual(state.players["p1"].hand, [])

    def test_opponent_observation_hides_countermeasure_payment_resources(self) -> None:
        simulator = Simulator(self.cards)
        simulator.state = self._state()
        simulator.state.players["p2"].resources.kredits = 3
        observation = simulator.get_observation("p1")
        self.assertIsNone(observation["opponent"]["kredits"])

    def test_pin_immunity_applies_to_generic_effects(self) -> None:
        state = self._state()
        target = state.players["p2"].units[0]
        target.status["cannot_be_pinned"] = True
        EffectResolver(self.cards).resolve(
            {"type": "pin", "target": "enemy_units"}, state, EffectContext("p1"),
        )
        self.assertNotIn("pinned", target.status)

    def test_covert_operation_cost_is_one_even_when_printed_cost_differs(self) -> None:
        unit = UnitState("covert", "kawanishi_e7k", 1, 1, "p1")
        unit.status["covert"] = True
        self.assertEqual(_operation_cost(unit, self.cards.get(unit.card_id)), 1)

    def test_shock_prevents_one_return_attack_then_is_consumed(self) -> None:
        attacker = UnitState("shock", "humber_mk_iv", 3, 4, "p1", "frontline")
        defender = UnitState("defender", "greif", 1, 10, "p2", "frontline")
        state = GameState(
            current_player="p1", battlefield={"frontline": ["shock", "defender"], "support_line": []},
            players={
                "p1": PlayerState("p1", resources=ResourceState(10, 10), units=[attacker]),
                "p2": PlayerState("p2", units=[defender]),
            },
        )
        AttackAction("p1", "shock", "defender").execute(state, self.cards)
        self.assertEqual(attacker.defense, 4)
        self.assertIn("shock", attacker.status["removed_abilities"])

    def test_convert_replaces_the_selected_battlefield_unit(self) -> None:
        state = self._state()
        target = state.players["p2"].units[0]
        NativeRuleEngine(self.cards)._execute_action(
            RuleAction("convert_to", "selected_target", card_name="GREIF"),
            state, EffectContext("p1", target_unit_id=target.instance_id),
        )
        self.assertEqual(target.card_id, "greif")
        self.assertEqual((target.attack, target.defense), (self.cards.get("greif").attack, self.cards.get("greif").defense))

    def test_develop_honors_the_selected_candidate(self) -> None:
        state = self._state()
        NativeRuleEngine(self.cards)._execute_action(
            RuleAction("develop_options", "owner", card_name="GREIF|GARRISON"),
            state, EffectContext("p1", metadata={"selected_option": "GARRISON"}),
        )
        # Develop adds the chosen card to hand, units included (confirmed in game).
        self.assertIn("garrison", state.players["p1"].hand)
        self.assertNotIn("greif", state.players["p1"].hand)

    def test_convert_replaces_selected_hand_card_in_place(self) -> None:
        state = self._state()
        state.players["p1"].hand = ["garrison"]
        NativeRuleEngine(self.cards)._execute_action(
            RuleAction("convert_to", "owner", card_name="GREIF"), state,
            EffectContext("p1", metadata={"selected_card_id": "garrison"}),
        )
        self.assertEqual(state.players["p1"].hand, ["greif"])

    def test_convert_replaces_selected_deck_card_in_place(self) -> None:
        state = self._state()
        state.players["p1"].deck = ["garrison"]
        NativeRuleEngine(self.cards)._execute_action(
            RuleAction("convert_to", "owner", card_name="GREIF"), state,
            EffectContext("p1", metadata={"selected_card_id": "garrison", "selected_card_zone": "deck"}),
        )
        self.assertEqual(state.players["p1"].deck, ["greif"])
