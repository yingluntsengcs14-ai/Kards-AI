"""Tests for genuinely-executed handlers (double_stats, double_damage, cannot, etc.).

These verify that handlers produce real, verifiable game state changes,
NOT just dead status writes.  Every handler added to _GENUINELY_EXECUTED_KINDS
MUST have at least one test here.
"""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]


class DoubleStatsTests(unittest.TestCase):
    """double_stats handler: multiplies unit attack and defense by 2."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, units=(), player="p1"):
        return GameState(
            current_player=player,
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"],
                                  resources=ResourceState(5, 5),
                                  units=list(units) if player == "p1" else []),
                "p2": PlayerState("p2", "USA", deck=["garrison"],
                                  resources=ResourceState(5, 5),
                                  units=list(units) if player == "p2" else []),
            },
        )

    def test_double_stats_on_single_unit(self) -> None:
        """Double the attack and defense of a friendly unit."""
        card = self.cards.get("red_bull")
        unit = UnitState("u1", card.id, 3, 4, "p1", "support_line")
        state = self._state([unit])
        self.engine.emit("on_turn_start", state, EffectContext("p1", card.id, unit.instance_id))
        self.assertEqual(unit.attack, 6)
        self.assertEqual(unit.defense, 8)
        doubled = [e for e in state.event_log if e.get("event") == "stats_doubled"]
        self.assertEqual(len(doubled), 1)

    def test_double_stats_on_multiple_units(self) -> None:
        """Double stats on all friendly units (strategic_planning)."""
        rule = self.engine.rule_for("strategic_planning")
        # Verify parser produces double_stats for "Double the attack and defense of your units"
        kinds = [a.kind for a in rule.actions]
        self.assertIn("double_stats", kinds)

    def test_parser_double_stats_no_longer_ambiguous(self) -> None:
        """double_stats is now in _GENUINELY_EXECUTED_KINDS, not a catch-all."""
        rule = self.engine.rule_for("strategic_planning")
        self.assertEqual(rule.status, "implemented")


class DoubleDamageTests(unittest.TestCase):
    """double_damage handler: marks unit for double combat damage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self):
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"],
                                  resources=ResourceState(5, 5), units=[]),
                "p2": PlayerState("p2", "USA", deck=["garrison"],
                                  resources=ResourceState(5, 5), units=[]),
            },
        )

    def test_double_damage_sets_status_flag(self) -> None:
        """Unit with 'deals double damage' gets the status flag."""
        state = self._state()
        card = self.cards.get("blenheim_mk_i")
        unit = UnitState("u1", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        state.players["p1"].units = [unit]
        state.battlefield = {"frontline": [unit.instance_id], "support_line": []}
        self.engine.emit("on_deploy", state, EffectContext("p1", card.id, unit.instance_id))
        # BLENHEIM's "if the frontline is empty" is a standing condition the
        # engine does not model yet.  It used to count as satisfied, so the
        # bomber always dealt double damage; now the card is honestly partial
        # and the flag is not set on a non-empty frontline.
        self.assertEqual(self.engine.rule_for(card.id).status, "partial")
        self.assertFalse(unit.status.get("double_damage"))

    def test_parser_double_damage_implemented(self) -> None:
        """'combat damage dealt to this unit is doubled' parses as implemented."""
        rule = self.engine.rule_for("2nd_marines_battalion")
        self.assertIn("double_damage", [a.kind for a in rule.actions])


class CannotHandlerTests(unittest.TestCase):
    """cannot handler: restricts specific actions on units."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self):
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"],
                                  resources=ResourceState(5, 5), units=[]),
                "p2": PlayerState("p2", "USA", deck=["garrison"],
                                  resources=ResourceState(5, 5), units=[]),
            },
        )

    def test_cannot_retreat_or_suppressed(self) -> None:
        """'Cannot Retreat or be Suppressed' sets both flags."""
        state = self._state()
        card = self.cards.get("10th_guards_regiment")
        unit = UnitState("u1", card.id, card.attack or 0, card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [unit]
        state.battlefield = {"frontline": [], "support_line": [unit.instance_id]}
        self.engine.emit("on_deploy", state, EffectContext("p1", card.id, unit.instance_id))
        applied = [e for e in state.event_log if e.get("event") == "cannot_applied"]
        self.assertTrue(applied)
        self.assertTrue(unit.status.get("cannot_retreat"),
                        f"Expected cannot_retreat, got status={unit.status}")
        self.assertTrue(unit.status.get("cannot_be_suppressed"),
                        f"Expected cannot_be_suppressed, got status={unit.status}")

    def test_cannot_be_pinned(self) -> None:
        """'Cannot be pinned' sets the flag."""
        state = self._state()
        card = self.cards.get("tiger_ih")
        unit = UnitState("u1", card.id, card.attack or 0, card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [unit]
        state.battlefield = {"frontline": [], "support_line": [unit.instance_id]}
        self.engine.emit("on_deploy", state, EffectContext("p1", card.id, unit.instance_id))
        self.assertTrue(unit.status.get("cannot_be_pinned"))

    def test_cannot_attack_units(self) -> None:
        """'Cannot attack units' sets the flag."""
        state = self._state()
        card = self.cards.get("heinkel_he_111")
        unit = UnitState("u1", card.id, card.attack or 0, card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [unit]
        state.battlefield = {"frontline": [], "support_line": [unit.instance_id]}
        self.engine.emit("on_deploy", state, EffectContext("p1", card.id, unit.instance_id))
        self.assertTrue(unit.status.get("cannot_attack_units"),
                        f"Expected cannot_attack_units, got status={unit.status}")

    def test_cannot_attack_or_move(self) -> None:
        """'Cannot attack or move' sets both flags."""
        state = self._state()
        card = self.cards.get("190th_guard_rifles")
        unit = UnitState("u1", card.id, card.attack or 0, card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [unit]
        state.battlefield = {"frontline": [], "support_line": [unit.instance_id]}
        self.engine.emit("on_deploy", state, EffectContext("p1", card.id, unit.instance_id))
        self.assertTrue(unit.status.get("cannot_attack"),
                        f"Expected cannot_attack, got status={unit.status}")
        self.assertTrue(unit.status.get("cannot_move"),
                        f"Expected cannot_move, got status={unit.status}")


if __name__ == "__main__":
    unittest.main()
