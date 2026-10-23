"""Rules found broken by simulated games (tools/watch_game.py, 2026-10-23).

Each card here was marked "implemented" while doing something else in play.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from simulator.actions.action import PassAction, PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects import basic_effects
from simulator.rules.native import engine_for
from simulator.effects.resolver import EffectContext

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class SimulationFindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, hand=(), p1_units=(), p2_units=(), p1_hand_extra=()) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Soviet", deck=["garrison"] * 10, hand=[*hand, *p1_hand_extra],
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line") for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Germany", deck=["garrison"] * 10, resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", "support_line") for uid, cid, atk, dfn in p2_units]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=5,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield["support_line"].append(unit.instance_id)
        return state

    def _end_turn(self, state) -> None:
        PassAction(state.current_player).execute(state, self.cards)

    def _ids(self, state, player, card_id) -> list[str]:
        """Units of a card; for a name shared by a spawn-only copy (T-70), either id."""
        name = self.cards.get(card_id).name
        return [u.instance_id for u in state.players[player].units if self.cards.get(u.card_id).name == name]

    # ── "Destroy it/them at the end of ..." ────────────────────────────────
    def test_mobile_defense_deploys_two_t70_and_leaves_the_enemy_alone(self) -> None:
        """MOBILE DEFENSE 以前不出 T-70，反而立刻消滅被選的敵方單位。"""
        state = self._state(hand=("mobile_defense",), p2_units=(("fw", "fw_189", 2, 2),))
        PlayCardAction("p1", "mobile_defense").execute(state, self.cards)
        self.assertEqual(len(self._ids(state, "p1", "t70")), 2)
        self.assertEqual(self._ids(state, "p2", "fw_189"), ["fw"])

    def test_mobile_defense_t70s_die_at_the_end_of_your_next_turn(self) -> None:
        state = self._state(hand=("mobile_defense",))
        PlayCardAction("p1", "mobile_defense").execute(state, self.cards)
        self._end_turn(state)                       # p1 第一個回合結束：還在
        self.assertEqual(len(self._ids(state, "p1", "t70")), 2)
        self._end_turn(state)                       # p2 回合結束：還在
        self.assertEqual(len(self._ids(state, "p1", "t70")), 2)
        self._end_turn(state)                       # p1 下一個回合結束：消滅
        self.assertEqual(self._ids(state, "p1", "t70"), [])

    def test_honor_buffs_now_and_destroys_at_end_of_turn(self) -> None:
        """HONOR 以前只在回合結束觸發，打出時完全沒有效果。"""
        state = self._state(hand=("honor",), p1_units=(("u", "garrison", 1, 3),))
        PlayCardAction("p1", "honor", target_unit_id="u").execute(state, self.cards)
        unit = state.players["p1"].units[0]
        self.assertEqual((unit.attack, unit.defense), (5, 7))
        self._end_turn(state)
        self.assertEqual(state.players["p1"].units, [])
        self.assertIn("garrison", state.graveyard["p1"])

    def test_final_push_destroys_at_end_of_turn_not_immediately(self) -> None:
        state = self._state(hand=("final_push",), p1_units=(("u", "garrison", 1, 3),))
        PlayCardAction("p1", "final_push", target_unit_id="u").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].attack, 4)
        self._end_turn(state)
        self.assertEqual(state.players["p1"].units, [])

    # ── Destruction effects firing on deploy ─────────────────────────────────
    def test_destruction_effect_waits_for_destruction(self) -> None:
        """T-60「Destruction: Add a T-34 1942」：部署時不加牌，被消滅時才加。"""
        state = self._state(hand=("t60",))
        PlayCardAction("p1", "t60").execute(state, self.cards)
        self.assertNotIn("t34_1942", state.players["p1"].hand)
        unit = state.players["p1"].units[0]
        start = len(state.event_log)
        basic_effects.destroy(state, unit)
        engine_for(self.cards).emit_deaths_since(state, start)
        self.assertEqual(len(state.players["p1"].hand), 1)
        self.assertEqual(self.cards.get(state.players["p1"].hand[0]).name, "T-34 1942")

    # ── Generated cards ────────────────────────────────────────────────────
    def test_generated_cards_respect_the_hand_limit(self) -> None:
        state = self._state(p1_hand_extra=["garrison"] * 8)
        engine = engine_for(self.cards)
        for _ in range(3):
            engine._execute_action(
                engine.rule_for("protected_convoy").actions[0], state,
                EffectContext("p1", "protected_convoy"))
        self.assertEqual(len(state.players["p1"].hand), 9)

    def test_a_card_named_with_and_adds_both(self) -> None:
        """PROTECTED CONVOY：「a PRODUCTION and a PLAN」要加兩張不同的牌。"""
        rule = engine_for(self.cards).rule_for("protected_convoy")
        state = self._state()
        add = next(a for a in rule.actions if a.kind == "add_card")
        engine_for(self.cards)._execute_action(add, state, EffectContext("p1", "protected_convoy"))
        self.assertEqual(sorted(state.players["p1"].hand), ["plan", "production"])

    def test_rule_deployed_units_get_unique_ids(self) -> None:
        engine = engine_for(self.cards)
        rule = engine.rule_for("tractor_factories")
        deploy = next(a for a in rule.actions if a.kind == "deploy_named")
        state = self._state()
        engine._execute_action(deploy, state, EffectContext("p1", "tractor_factories"))
        basic_effects.destroy(state, state.players["p1"].units[0])
        engine._execute_action(deploy, state, EffectContext("p1", "tractor_factories"))
        ids = [u.instance_id for u in state.players["p1"].units]
        self.assertEqual(len(ids), len(set(ids)))

    # ── HQ defense is health (confirmed in game) ─────────────────────────────
    def test_great_patriotic_war_sets_both_hqs_to_12(self) -> None:
        """以前做成「每次受傷先減 12」，HQ 幾乎打不動。"""
        state = self._state(hand=("great_patriotic_war",))
        state.players["p2"].hq.current_health = 7
        PlayCardAction("p1", "great_patriotic_war").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.current_health, 12)
        self.assertEqual(state.players["p2"].hq.current_health, 12)
        from simulator.core.hq import HQResolver
        HQResolver.damage(state, "p2", 3)
        self.assertEqual(state.players["p2"].hq.current_health, 9)

    def test_hq_defense_gain_can_exceed_the_starting_20(self) -> None:
        """遊戲內確認：「+N defense」可以超過起始的 20；修復才有上限。"""
        from simulator.core.hq import HQResolver
        state = self._state()
        HQResolver.modify_defense(state, "p1", 4)
        self.assertEqual(state.players["p1"].hq.current_health, 24)

    # ── BLAST (spawn-only card missing from the kards.info catalog) ─────────
    def test_blast_exists_and_deals_one_damage(self) -> None:
        self.assertIn("blast", self.cards)
        card = self.cards.get("blast")
        self.assertEqual((card.nation, card.type, card.kredits), ("Neutral", "order", 0))
        state = self._state(hand=("blast",), p2_units=(("e", "garrison", 1, 3),))
        PlayCardAction("p1", "blast", target_unit_id="e").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].defense, 2)

    def test_866th_rifles_adds_two_blasts_when_destroyed(self) -> None:
        state = self._state(hand=("866th_rifles",))
        PlayCardAction("p1", "866th_rifles").execute(state, self.cards)
        self.assertNotIn("blast", state.players["p1"].hand)
        start = len(state.event_log)
        basic_effects.destroy(state, state.players["p1"].units[0])
        engine_for(self.cards).emit_deaths_since(state, start)
        self.assertEqual(state.players["p1"].hand.count("blast"), 2)

    # ── Kredits never below 0 (confirmed in game) ───────────────────────────
    def test_lose_kredits_next_turn_waits_and_stops_at_zero(self) -> None:
        """99th KHOLM：「下回合開始時失去 1 kredit」，以前部署當下就扣、還會扣到負數。"""
        state = self._state(hand=("99th_kholm",))
        state.players["p1"].resources.kredits = 1
        PlayCardAction("p1", "99th_kholm").execute(state, self.cards)
        self.assertEqual(state.players["p1"].resources.kredits, 0)
        self._end_turn(state)
        self._end_turn(state)
        resources = state.players["p1"].resources
        self.assertEqual(resources.kredits, resources.max_kredits - 1)

    def test_countermeasure_charges_the_price_it_was_validated_at(self) -> None:
        state = self._state(hand=("dowding_system",))
        state.players["p1"].cost_modifiers.append({"kind": "hand_cost", "amount": -2, "scope": "all"})
        state.players["p1"].resources.kredits = 1
        PlayCardAction("p1", "dowding_system").execute(state, self.cards)
        self.assertEqual(state.players["p1"].resources.kredits, 0)


if __name__ == "__main__":
    unittest.main()
