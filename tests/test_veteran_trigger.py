# -*- coding: utf-8 -*-
"""回歸測試：「成為老兵」事件（on_veteran）與「if possible.」條件。

背景（F. 條件補完，2026-11-04）：對 150 局 self-play 的掃描顯示「條件看不懂
（當成不成立）」佔可疑事件的大宗，其中最常見的兩個模板是：

* "Target unit moves into the frontline, if possible." —— "if possible" 不是
  真正的條件閘門，move_to_frontline/move_to_support_line 本身早就會在目的地
  滿了時安靜跳過（見 native._move_units），所以這句話恆成立。
* "When this/a/an/the (friendly/enemy) unit becomes Veteran" —— 老兵換卡
  （native._promote_to_veteran）之前沒有廣播任何事件，所以監聽它的卡（換卡
  本身的被動、或別的單位的「有友方單位變老兵時」）永遠不會觸發。

本檔案釘住這兩個修法，並特別守住一個差一點就犯的假陽性：「Becomes Veteran
when it damages the enemy HQ.」（PANZER I-B／22nd GUARDS BRIGADE 的換卡**條
件**本身）絕對不能被誤判成「有人變老兵時」的監聽句——那是完全相反的方向。
"""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules import condition
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"
PLAIN_GROUND = "1_infantry_regiment"


class IfPossibleConditionTests(unittest.TestCase):
    """"if possible." 恆成立，不管事件或盤面是什麼。"""

    def test_if_possible_is_always_satisfied(self) -> None:
        self.assertTrue(condition.evaluate("if possible.", "on_deploy", EffectContext("p1"), None))
        self.assertTrue(condition.evaluate("If Possible", "on_attack", EffectContext("p1"), None))

    def test_unrelated_condition_is_not_shortcut_by_if_possible(self) -> None:
        # 確認沒有不小心把比對寫得太寬，隨便一句含 "possible" 的話不該被攔截。
        self.assertFalse(condition.evaluate("if it is possible to deploy 3 units this turn.",
                                            "on_deploy", EffectContext("p1"), None))


class VeteranTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_units=(), p2_units=()) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=[PLAIN_GROUND] * 10,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", pos)
                                         for uid, cid, atk, dfn, pos in p1_units]),
                "p2": PlayerState("p2", "USA", deck=[PLAIN_GROUND] * 10,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", pos)
                                         for uid, cid, atk, dfn, pos in p2_units]),
            },
            graveyard={"p1": [], "p2": []},
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
        return state

    # ------------------------------------------------------------------
    # "When THIS unit becomes Veteran" -- 換卡後的新卡自己的被動。
    # ------------------------------------------------------------------
    def test_this_unit_becomes_veteran_fires_own_ability(self) -> None:
        """22nd GUARDS BRIGADE 的老兵版：「變成老兵時，對每個敵方單位造成 3 傷害。」"""
        card = self.cards.get("22nd_guards_brigade")
        enemy = self.cards.get(PLAIN_GROUND)
        state = self._state(
            p1_units=(("u1", card.id, card.attack or 0, card.defense or 0, "frontline"),),
            p2_units=(("e1", enemy.id, enemy.attack or 0, enemy.defense or 0, "frontline"),),
        )
        unit = state.players["p1"].units[0]
        enemy_unit = state.players["p2"].units[0]
        before = enemy_unit.defense
        self.engine._promote_to_veteran(state, unit)
        self.assertEqual(unit.card_id, "22nd_guards_brigade_vet")
        self.assertEqual(enemy_unit.defense, before - 3,
                         "22nd GUARDS BRIGADE 換成老兵版後沒有對敵方單位造成 3 點傷害")

    def test_this_unit_becomes_veteran_does_not_fire_for_someone_elses_promotion(self) -> None:
        """同一回合另一個單位變老兵，不該讓 22nd GUARDS BRIGADE 的「THIS unit」被動誤觸發。"""
        card = self.cards.get("22nd_guards_brigade")
        other = self.cards.get("50_infantry_regiment")
        enemy = self.cards.get(PLAIN_GROUND)
        state = self._state(
            p1_units=(
                ("u1", card.id, card.attack or 0, card.defense or 0, "frontline"),
                ("u2", other.id, other.attack or 0, other.defense or 0, "support_line"),
            ),
            p2_units=(("e1", enemy.id, enemy.attack or 0, enemy.defense or 0, "frontline"),),
        )
        guards_unit = state.players["p1"].units[0]
        other_unit = state.players["p1"].units[1]
        enemy_unit = state.players["p2"].units[0]
        before = enemy_unit.defense
        self.engine._promote_to_veteran(state, other_unit)
        self.assertEqual(guards_unit.card_id, card.id, "22nd GUARDS BRIGADE 自己沒換老兵，不該被動換卡")
        self.assertEqual(enemy_unit.defense, before,
                         "別的單位變老兵，不該觸發 22nd GUARDS BRIGADE『THIS unit』的被動")

    # ------------------------------------------------------------------
    # "When A FRIENDLY unit becomes Veteran" -- 其他單位監聽隊友換卡。
    # ------------------------------------------------------------------
    def test_friendly_unit_becomes_veteran_buffs_listener(self) -> None:
        """MAORI BATTALION：「有友方單位變老兵時，獲得 +2+2。」"""
        maori = self.cards.get("maori_battalion")
        promotee = self.cards.get("50_infantry_regiment")
        state = self._state(p1_units=(
            ("m1", maori.id, maori.attack or 0, maori.defense or 0, "frontline"),
            ("u1", promotee.id, promotee.attack or 0, promotee.defense or 0, "support_line"),
        ))
        maori_unit = state.players["p1"].units[0]
        promotee_unit = state.players["p1"].units[1]
        before_atk, before_def = maori_unit.attack, maori_unit.defense
        self.engine._promote_to_veteran(state, promotee_unit)
        self.assertEqual(maori_unit.attack, before_atk + 2)
        self.assertEqual(maori_unit.defense, before_def + 2)

    def test_friendly_unit_becomes_veteran_ignores_enemy_promotion(self) -> None:
        """敵方單位變老兵，不該觸發我方 MAORI BATTALION 的「friendly unit」被動。"""
        maori = self.cards.get("maori_battalion")
        promotee = self.cards.get("50_infantry_regiment")
        state = self._state(
            p1_units=(("m1", maori.id, maori.attack or 0, maori.defense or 0, "frontline"),),
            p2_units=(("e1", promotee.id, promotee.attack or 0, promotee.defense or 0, "support_line"),),
        )
        maori_unit = state.players["p1"].units[0]
        enemy_promotee = state.players["p2"].units[0]
        before_atk, before_def = maori_unit.attack, maori_unit.defense
        self.engine._promote_to_veteran(state, enemy_promotee)
        self.assertEqual(maori_unit.attack, before_atk)
        self.assertEqual(maori_unit.defense, before_def)

    # ------------------------------------------------------------------
    # 假陽性守門：換卡「條件」本身絕對不是「有人換卡時」的監聽句。
    # ------------------------------------------------------------------
    def test_promotion_condition_itself_is_not_a_veteran_listener(self) -> None:
        """PANZER I-B：「打到敵方 HQ 時變成老兵」—— 這是換卡的條件，不是監聽別人換卡。"""
        rule = self.engine.rule_for("panzer_ib")
        self.assertNotIn("on_veteran", rule.triggers,
                         "『Becomes Veteran when it damages the enemy HQ』被誤判成 on_veteran 監聽句")
        rule_vet = self.engine.rule_for("panzer_ib_vet")
        self.assertIn("on_veteran", rule_vet.triggers,
                      "PANZER I-B 老兵版自己的『When this unit becomes Veteran』沒有接上 on_veteran")


if __name__ == "__main__":
    unittest.main()
