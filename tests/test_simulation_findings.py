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

    # ── States that should never occur ───────────────────────────────────────
    def test_attack_never_goes_below_zero(self) -> None:
        """ROOT OUT THE ENEMY 打在 0 攻單位上，以前會變成 -1。"""
        state = self._state(hand=("root_out_the_enemy",), p2_units=(("e", "329th_engineer_battalion", 0, 4),))
        PlayCardAction("p1", "root_out_the_enemy").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].attack, 0)

    def test_howitzer_weakens_the_unit_it_damages_not_itself(self) -> None:
        """B-4 203mm：被它打到的單位 -3 攻到回合結束；以前是部署時自己 -3。"""
        from simulator.actions.action import AttackAction
        card = self.cards.get("b4_203mm_howitzer")
        state = self._state(hand=("b4_203mm_howitzer",), p2_units=(("e", "garrison", 2, 9),))
        PlayCardAction("p1", "b4_203mm_howitzer").execute(state, self.cards)
        gun = state.players["p1"].units[0]
        self.assertEqual(gun.attack, card.attack)
        gun.status.pop("deployed_this_turn", None)
        AttackAction("p1", gun.instance_id, "e").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].attack, 0)        # 2 - 3，夾在 0
        self._end_turn(state)
        self.assertEqual(state.players["p2"].units[0].attack, 2)        # 回合結束還原成 2，不是 3

    def test_confusion_takes_a_cheap_unit_to_your_support_line_until_end_of_turn(self) -> None:
        state = self._state(hand=("confusion",), p2_units=(("e", "garrison", 1, 3),))
        enemy = state.players["p2"].units[0]
        state.battlefield["support_line"].remove("e"); state.battlefield["frontline"].append("e")
        enemy.position = "frontline"
        PlayCardAction("p1", "confusion", target_unit_id="e").execute(state, self.cards)
        self.assertEqual([u.instance_id for u in state.players["p1"].units], ["e"])
        self.assertEqual(enemy.position, "support_line")
        self.assertNotIn("e", state.battlefield["frontline"])
        self._end_turn(state)
        self.assertEqual([u.instance_id for u in state.players["p2"].units], ["e"])

    def test_confusion_respects_its_cost_limit(self) -> None:
        from simulator.actions.validator import ActionValidationError
        expensive = next(c.id for c in self.cards if c.is_unit and c.kredits >= 5 and c.attack and c.defense)
        state = self._state(hand=("confusion",), p2_units=(("e", expensive, 3, 3),))
        with self.assertRaises(ActionValidationError):
            PlayCardAction("p1", "confusion", target_unit_id="e").execute(state, self.cards)

    def test_deployment_buff_with_a_this_turn_condition_is_permanent_and_on_itself(self) -> None:
        """10th PARA：「本回合下過指令就 +1+1」，加成給自己而且是永久的。"""
        state = self._state(hand=("10th_para_battalion",))
        state.players["p1"].status["orders_played_this_turn"] = 1
        PlayCardAction("p1", "10th_para_battalion").execute(state, self.cards)
        rule = engine_for(self.cards).rule_for("10th_para_battalion")
        self.assertEqual([(a.target, a.duration) for a in rule.actions if a.kind == "buff"], [("source", None)])

    def test_unit_sent_to_a_full_hand_is_lost_not_a_tenth_card(self) -> None:
        """單位被送回手牌；對方手牌已 9 張時，以前會變 10 張。"""
        from simulator.rules.parser import RuleAction
        state = self._state(p2_units=(("e", "garrison", 1, 3),))
        state.players["p2"].hand = ["garrison"] * 9
        engine_for(self.cards)._execute_action(RuleAction("return_to_hand", "selected_target"), state,
                                               EffectContext("p1", "test", target_unit_id="e"))
        self.assertEqual(len(state.players["p2"].hand), 9)
        self.assertEqual(state.players["p2"].units, [])

    # ── "...send it back at the end of turn" (2026-10-27) ───────────────────
    def test_eagle_day_puts_a_german_air_unit_from_hand_into_play_then_back(self) -> None:
        """EAGLE DAY 以前把選中的單位立刻送回手牌，連敵方單位都會被送回去。"""
        from simulator.core.game import Simulator
        state = self._state(hand=("eagle_day",), p1_hand_extra=["arado_ar_196", "1_infantry_regiment"],
                            p2_units=(("e", "garrison", 1, 3),))
        sim = Simulator(self.cards); sim.state = state
        plays = [a for a in sim.get_available_actions() if isinstance(a, PlayCardAction) and a.card_id == "eagle_day"]
        self.assertEqual({a.selected_card_id for a in plays}, {"arado_ar_196"})    # 只能選德國空軍
        PlayCardAction("p1", "eagle_day", selected_card_id="arado_ar_196").execute(state, self.cards)
        plane = state.players["p1"].units[0]
        self.assertEqual(plane.card_id, "arado_ar_196")
        self.assertIn("blitz", plane.status["added_abilities"])
        self.assertNotIn("arado_ar_196", state.players["p1"].hand)
        self.assertEqual(len(state.players["p2"].units), 1)                          # 敵方單位不受影響
        self._end_turn(state)
        self.assertEqual(state.players["p1"].units, [])
        self.assertIn("arado_ar_196", state.players["p1"].hand)

    def test_night_attack_returns_the_unit_fresh_then_sends_it_to_hand(self) -> None:
        state = self._state(hand=("night_attack",), p1_units=(("u", "1_infantry_regiment", 1, 1),))
        PlayCardAction("p1", "night_attack", target_unit_id="u").execute(state, self.cards)
        unit = state.players["p1"].units[0]
        card = self.cards.get("1_infantry_regiment")
        self.assertEqual((unit.card_id, unit.attack, unit.defense), ("1_infantry_regiment", card.attack, card.defense))
        self.assertEqual(sorted(unit.status["added_abilities"]), ["blitz", "shock"])
        self.assertEqual(state.players["p1"].hand, [])                                # 不是立刻回手牌
        self._end_turn(state)
        self.assertEqual(state.players["p1"].units, [])
        self.assertEqual(state.players["p1"].hand, ["1_infantry_regiment"])

    def test_sloped_armor_deploys_is2_which_fights_then_returns_to_hand(self) -> None:
        """SLOPED ARMOR 以前只在回合結束觸發，打出時沒有效果。"""
        state = self._state(hand=("sloped_armor",), p2_units=(("e", "garrison", 1, 12),))
        PlayCardAction("p1", "sloped_armor", target_unit_id="e").execute(state, self.cards)
        tank = state.players["p1"].units[0]
        self.assertEqual(tank.card_id, "iosef_stalin_ii")
        self.assertEqual(state.players["p2"].units[0].defense, 12 - 8)                 # IS-II 8 攻
        self.assertEqual(tank.defense, 8)                                              # 反擊 1，被重裝甲 2 擋掉
        self._end_turn(state)
        self.assertIn("iosef_stalin_ii", state.players["p1"].hand)
        # IS-II：「被送回手牌時，在支援線加一個複製」——這張卡的組合技。
        board = state.players["p1"].units
        self.assertEqual([u.card_id for u in board], ["iosef_stalin_ii"])
        self.assertEqual(board[0].defense, 8)                                          # 新的一台，不是受傷的那台

    def test_second_front_returns_the_unit_as_two_after_the_next_enemy_turn(self) -> None:
        state = self._state(hand=("second_front",), p1_units=(("g", "1_infantry_regiment", 1, 1),))
        PlayCardAction("p1", "second_front", target_unit_id="g").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units, [])
        self._end_turn(state)                        # 自己回合結束：還沒回來
        self.assertEqual(state.players["p1"].units, [])
        self._end_turn(state)                        # 敵方回合結束：回場並複製
        self.assertEqual([u.card_id for u in state.players["p1"].units], ["1_infantry_regiment"] * 2)
        self.assertNotIn("1_infantry_regiment", state.players["p1"].hand)

    def test_second_front_only_takes_a_german_ground_unit(self) -> None:
        from simulator.actions.validator import ActionValidationError
        state = self._state(hand=("second_front",), p1_units=(("a", "arado_ar_196", 1, 2),))
        with self.assertRaises(ActionValidationError):
            PlayCardAction("p1", "second_front", target_unit_id="a").execute(state, self.cards)

    # ── Conditions (2026-10-27) ─────────────────────────────────────────────
    def test_m2a4_has_blitz_only_while_it_has_3_or_more_attack(self) -> None:
        """M2A4「攻擊力 3 以上才有閃擊」：以前條件沒檢查，永遠有閃擊。"""
        from simulator.rules.keywords import KeywordEngine
        state = self._state(hand=("m2a4",))
        PlayCardAction("p1", "m2a4").execute(state, self.cards)
        unit = state.players["p1"].units[0]
        card = self.cards.get("m2a4")
        self.assertLess(unit.attack, 3)
        self.assertFalse(KeywordEngine.has(card, "blitz", unit))
        unit.attack = 3                      # 例如場上的加攻效果
        self.assertTrue(KeywordEngine.has(card, "blitz", unit))

    def test_no_implemented_rule_carries_an_unmodeled_condition(self) -> None:
        """看不懂條件的卡不能標成 implemented（擋住以後新增的卡）。"""
        from simulator.rules.condition import handled_by_kind, is_modeled
        engine = engine_for(self.cards)
        offenders = []
        for card in self.cards:
            rule = engine.rule_for(card.id)
            if rule.status != "implemented":
                continue
            for action in rule.actions:
                if action.condition and not handled_by_kind(action.kind, action.condition) \
                        and not is_modeled(action.condition, action.on or rule.triggers, self.cards):
                    offenders.append((card.id, action.condition))
        self.assertEqual(offenders, [])

    def test_confusion_removes_the_unit_when_its_owner_has_no_room(self) -> None:
        """遊戲內確認：歸還時原主人支援線已滿 → 移除（不是消滅，不觸發消滅效果）。"""
        state = self._state(hand=("confusion",), p2_units=(("e", "13th_rifle_regiment", 2, 2),))
        PlayCardAction("p1", "confusion", target_unit_id="e").execute(state, self.cards)
        for index in range(4):
            uid = "f{0}".format(index)
            state.players["p2"].units.append(UnitState(uid, "garrison", 1, 3, "p2", "support_line"))
            state.battlefield["support_line"].append(uid)
        hand_before = list(state.players["p2"].hand)
        self._end_turn(state)
        self.assertNotIn("e", [u.instance_id for p in state.players.values() for u in p.units])
        self.assertIn("13th_rifle_regiment", state.removed_cards)
        self.assertNotIn("13th_rifle_regiment", state.graveyard.get("p2", []))
        # 13th RIFLE 的「消滅時加一張蘇聯步兵」不觸發（p2 回合開始只抽了正常那一張）
        self.assertEqual(len(state.players["p2"].hand), len(hand_before) + 1)

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
