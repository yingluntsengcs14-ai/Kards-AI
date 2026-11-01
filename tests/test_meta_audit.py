"""Cards the 2026-11-01 meta-deck audit found doing something other than their text.

The board comes from tools/card_probe.py: p1 (Germany) holds an infantry, a tank
and a fighter in its support line; p2 holds the frontline with an infantry and a
tank, and an artillery and a fighter in support.  Kredits 9 of 9.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from card_probe import CATALOG, board  # noqa: E402
from simulator.actions.action import PassAction, PlayCardAction  # noqa: E402
from simulator.actions.validator import ActionValidationError  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.core.state import UnitState  # noqa: E402
from simulator.core.turn import TurnManager  # noqa: E402
from simulator.rules.native import engine_for  # noqa: E402


class MetaAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = CardDatabase.from_file(CATALOG)

    def play(self, card_id, **kw):
        state = board(card_id, self.cards)
        state.players["p1"].resources.kredits = state.players["p1"].resources.max_kredits = 12
        PlayCardAction("p1", card_id, **kw).execute(state, self.cards)
        return state

    def ids(self, state, pid):
        return sorted(u.instance_id for u in state.players[pid].units)

    def options(self, card_id):
        sim = Simulator(self.cards)
        sim.state = board(card_id, self.cards)
        return {a.selected_option for a in sim.get_available_actions()
                if isinstance(a, PlayCardAction) and a.card_id == card_id}

    def add_unit(self, state, uid, card_id, owner="p1", pos="support_line"):
        card = self.cards.get(card_id)
        state.players[owner].units.append(UnitState(uid, card_id, card.attack, card.defense, owner, pos))
        state.battlefield[pos].append(uid)

    # ── "all" effects ────────────────────────────────────────────────────────
    def test_all_enemies_includes_the_hq(self):
        state = self.play("desperate_measures")
        self.assertEqual(state.players["p2"].hq.current_health, 16)
        self.assertEqual(self.ids(state, "p2"), ["E_ftr", "E_tank"])      # 2-defense units died

    def test_to_the_last_man_destroys_every_unit(self):
        state = self.play("to_the_last_man")
        self.assertEqual((self.ids(state, "p1"), self.ids(state, "p2")), ([], []))

    def test_banzai_spares_units_over_4_attack(self):
        state = board("banzai", self.cards)
        state.players["p2"].units[0].attack = 5
        PlayCardAction("p1", "banzai").execute(state, self.cards)
        self.assertEqual(self.ids(state, "p2"), ["E_inf"])

    def test_repel_the_attack_retreats_every_enemy_unit(self):
        state = self.play("repel_the_attack")
        self.assertTrue(all(u.position == "support_line" for u in state.players["p2"].units))

    def test_strategic_focus_is_a_choice_of_ground_or_air(self):
        self.assertEqual(self.options("strategic_focus"), {"ground units", "air units"})
        state = self.play("strategic_focus", selected_option="air units")
        self.assertEqual(self.ids(state, "p2"), ["E_art", "E_inf", "E_tank"])

    def test_overcast_helps_both_players(self):
        state = self.play("overcast", selected_option="rain")
        self.assertEqual(state.players["p2"].hq.current_health, 20)
        self.assertEqual(len(state.players["p2"].hand), 4)

    # ── Choose One ───────────────────────────────────────────────────────────
    def test_research_orders_offer_both_cards(self):
        self.assertEqual(self.options("royal_research"), {"RADAR", "EXPANDED ROYAL RESEARCH"})
        state = self.play("royal_research", selected_option="EXPANDED ROYAL RESEARCH")
        self.assertIn("expanded_royal_research", state.players["p1"].hand)

    # ── counted amounts ──────────────────────────────────────────────────────
    def test_by_the_sword_draws_one_per_enemy_unit(self):
        state = self.play("by_the_sword")
        self.assertEqual(len(state.players["p1"].hand), 2 + 4)

    def test_conquest_draws_one_per_unit_type(self):
        state = self.play("conquest")
        self.assertEqual(len(state.players["p1"].hand), 2 + 3)

    def test_counter_offensive_scales_with_damaged_units(self):
        state = board("counter_offensive", self.cards)
        state.players["p2"].units[1].defense = 1                         # a second damaged unit
        PlayCardAction("p1", "counter_offensive", target_unit_id="F_inf").execute(state, self.cards)
        unit = next(u for u in state.players["p1"].units if u.instance_id == "F_inf")
        self.assertEqual((unit.attack, unit.defense), (1 + 4, 3 + 2))

    # ── missing parts ────────────────────────────────────────────────────────
    def test_overpower_destroys_every_unit_with_the_name(self):
        state = self.play("overpower", target_unit_id="S_tank")
        self.assertNotIn("S_tank", self.ids(state, "p1"))
        self.assertNotIn("E_tank", self.ids(state, "p2"))

    def test_anders_army_hurts_your_own_hq(self):
        state = self.play("anders_army")
        self.assertEqual(state.players["p1"].hq.current_health, 16)

    def test_liberation_adds_three_resistance(self):
        state = self.play("liberation")
        self.assertEqual(state.players["p2"].deck.count("resistance_b"), 3)

    def test_resistance_plays_itself_when_drawn(self):
        state = board("garrison", self.cards)
        state.players["p2"].deck.insert(0, "resistance_b")
        TurnManager.draw_card(state, "p2")
        self.assertNotIn("resistance_b", state.players["p2"].hand)
        self.assertEqual(state.players["p2"].hq.current_health, 17)
        self.assertTrue(any(u.status.get("pinned") for u in state.players["p2"].units))

    def test_lotta_svard_lowers_operation_cost(self):
        state = self.play("lotta_svrd")
        self.assertTrue(all(any(m.get("amount") == -1 for m in u.modifiers) for u in state.players["p1"].units))

    def test_gordon_highlanders_puts_a_free_order_on_top(self):
        state = self.play("gordon_highlanders", selected_card_id="blast")
        self.assertEqual(state.players["p1"].deck[0], "blast")
        from simulator.actions.action import card_play_cost
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("blast"), self.cards), 0)

    # ── listeners and auras ──────────────────────────────────────────────────
    def test_shibata_hits_the_hq_when_the_enemy_gives_an_order(self):
        state = board("garrison", self.cards)
        self.add_unit(state, "SH", "shibata_regiment")
        state.current_player = "p2"
        PlayCardAction("p2", "blast", target_unit_id="F_inf").execute(state, self.cards)
        self.assertEqual(state.players["p2"].hq.current_health, 17)

    def test_kv1_hits_the_hq_when_the_enemy_draws(self):
        state = board("garrison", self.cards)
        self.add_unit(state, "KV", "kv1_1941")
        start = len(state.event_log)
        TurnManager.draw_card(state, "p2")
        engine_for(self.cards).emit_draws_since(state, start)
        self.assertEqual(state.players["p2"].hq.current_health, 16)
        start = len(state.event_log)
        TurnManager.draw_card(state, "p1")                               # its owner drawing: nothing
        engine_for(self.cards).emit_draws_since(state, start)
        self.assertEqual(state.players["p2"].hq.current_health, 16)

    def test_type_93_aura_follows_the_board(self):
        state = self.play("type_93")
        f_inf = next(u for u in state.players["p1"].units if u.instance_id == "F_inf")
        self.assertEqual(f_inf.attack, 2)
        type93 = next(u for u in state.players["p1"].units if u.card_id == "type_93")
        self.assertEqual(type93.attack, 1)                                 # "other" units only
        from simulator.effects import basic_effects
        from simulator.rules import meta_mechanics
        basic_effects.destroy(state, type93)
        meta_mechanics.refresh_auras(state, self.cards)
        self.assertEqual(f_inf.attack, 1)

    def test_909th_gives_blitz_to_infantry_only(self):
        from simulator.rules.keywords import KeywordEngine
        state = self.play("909th_roslavl")
        unit = {u.instance_id: u for u in state.players["p1"].units}
        self.assertTrue(KeywordEngine.has(self.cards.get("garrison"), "blitz", unit["F_inf"]))
        self.assertNotIn("aura_abilities", unit["S_ftr"].status)

    def test_60th_cavalry_allows_one_order_a_turn(self):
        state = board("blast", self.cards)
        state.players["p1"].hand = ["blast", "blast"]
        self.add_unit(state, "C60", "60th_cavalry_regiment", owner="p2")
        PlayCardAction("p1", "blast", target_unit_id="E_inf").execute(state, self.cards)
        with self.assertRaises(ActionValidationError):
            PlayCardAction("p1", "blast", target_unit_id="E_tank").execute(state, self.cards)

    def test_85_pioneer_discounts_only_the_first_order(self):
        from simulator.actions.action import card_play_cost
        state = board("radar", self.cards)
        self.add_unit(state, "P85", "85_pioneer_company")
        radar = self.cards.get("radar")
        self.assertEqual(card_play_cost(state, "p1", radar, self.cards), max(0, radar.kredits - 1))
        state.players["p1"].hand = ["blast", "radar"]
        PlayCardAction("p1", "blast", target_unit_id="E_inf").execute(state, self.cards)
        self.assertEqual(card_play_cost(state, "p1", radar, self.cards), radar.kredits)

    def test_war_bonds_draws_an_extra_card_next_turn(self):
        state = self.play("war_bonds")
        PassAction("p1").execute(state, self.cards)
        before = len(state.players["p1"].hand)
        PassAction("p2").execute(state, self.cards)
        self.assertEqual(len(state.players["p1"].hand) - before, 2)


if __name__ == "__main__":
    unittest.main()
