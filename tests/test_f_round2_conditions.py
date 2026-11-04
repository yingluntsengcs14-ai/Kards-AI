# -*- coding: utf-8 -*-
"""回歸測試：F. 條件補完第二輪（2026-11-04），接在 if possible./becomes veteran/
E9 crash 修法之後。

本輪涵蓋兩類修法：

1. 新增的簡單狀態判斷（``if it is pinned``、``if this is your only unit``、
   ``if this destroys it``）——都在 ``condition._board_predicate`` 裡新增分支。

2. 新增的廣播事件──``on_hq_gained_defense``／``on_kredit_slot_gained``／
   ``on_kredit_slot_lost``／``on_destroys_unit`` 觸發接線。後三者沿用
   ``_promote_to_veteran`` 的廣播寫法：native 的動作處理完事後，對同一個玩家
   （或攻擊者本身）盤面上「有在聽」的單位重播一次。

額外釘住一個在調查這批卡時發現的獨立 parser bug：好幾張卡的句型是
"<真正效果> when your HQ gains defense."／"<真正效果> when you lose a kredit
slot."（觸發子句在句尾，不是句首），但 ``_parse_sentence`` 會先整句丟給
``_parse_sentence_core``，而裡面幾個用 ``original_lower``／``lower`` 字串比對
的「泛用」模板（``gain_hq_defense``、``gain_kredit_slot``、``lose_kredit_slot``）
不管那段文字是不是在觸發子句裡，照樣命中——於是把觸發子句本身誤判成「真正要執行
的動作」，吃掉了原本的效果（73e RÉGIMENT D'INFANTERIE 的 +1+1、COMMANDOS
KIEFFER 的傷害、SB2C HELLDIVER 的入手、MURASE BATTALION 的獲得 kredit 等）。
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


class BoardPredicateTests(unittest.TestCase):
    """純函式層：不需要整套引擎,但 _board_predicate 需要 state.players 裡有這個
    player_id 才會進入判斷（否則直接回 None → UNMODELED），所以給一個最小 state。
    """

    def _minimal_state(self) -> GameState:
        return GameState(
            current_player="p1",
            players={"p1": PlayerState("p1", "Germany", resources=ResourceState(6, 6)),
                     "p2": PlayerState("p2", "USA", resources=ResourceState(6, 6))},
            graveyard={"p1": [], "p2": []},
        )

    def test_if_it_is_pinned_true_and_false(self) -> None:
        state = self._minimal_state()
        ctx_pinned = EffectContext("p1", metadata={"target_pinned": True})
        ctx_unpinned = EffectContext("p1", metadata={"target_pinned": False})
        self.assertTrue(condition.evaluate("If it is pinned", "on_play", ctx_pinned, None, state))
        self.assertFalse(condition.evaluate("If it is pinned", "on_play", ctx_unpinned, None, state))

    def test_already_pinned_wording_still_works(self) -> None:
        # 原本就存在的措辭("if it is already pinned")不該被新規則擠掉。
        state = self._minimal_state()
        ctx = EffectContext("p1", metadata={"target_pinned": True})
        self.assertTrue(condition.evaluate("if it is already pinned", "on_play", ctx, None, state))


class EngineBackedConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_units=(), p2_units=()) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=[PLAIN_GROUND] * 10,
                                  resources=ResourceState(6, 6),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", pos)
                                         for uid, cid, atk, dfn, pos in p1_units]),
                "p2": PlayerState("p2", "USA", deck=[PLAIN_GROUND] * 10,
                                  resources=ResourceState(6, 6),
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
    # "If this is your only unit" -- 不靠 metadata,直接數盤面。
    # ------------------------------------------------------------------
    def test_only_unit_buff_fires_when_alone(self) -> None:
        """989th INFANTRY REGIMENT：「部署時，如果這是你唯一的單位，獲得 +2+2。」"""
        card = self.cards.get("989_infantry_regiment")
        state = self._state(p1_units=(("u1", card.id, card.attack or 0, card.defense or 0, "frontline"),))
        unit = state.players["p1"].units[0]
        before_atk, before_def = unit.attack, unit.defense
        self.engine.execute(card.id, "on_deploy", state, EffectContext("p1", card.id, "u1", event="on_deploy"))
        self.assertEqual(unit.attack, before_atk + 2)
        self.assertEqual(unit.defense, before_def + 2)

    def test_only_unit_buff_does_not_fire_with_a_second_unit(self) -> None:
        card = self.cards.get("989_infantry_regiment")
        other = self.cards.get("50_infantry_regiment")
        state = self._state(p1_units=(
            ("u1", card.id, card.attack or 0, card.defense or 0, "frontline"),
            ("u2", other.id, other.attack or 0, other.defense or 0, "support_line"),
        ))
        unit = state.players["p1"].units[0]
        before_atk, before_def = unit.attack, unit.defense
        self.engine.execute(card.id, "on_deploy", state, EffectContext("p1", card.id, "u1", event="on_deploy"))
        self.assertEqual(unit.attack, before_atk, "盤面上還有別的友軍單位，不該觸發「唯一單位」加成")
        self.assertEqual(unit.defense, before_def)

    # ------------------------------------------------------------------
    # "If this destroys it" -- 同一張卡前一個動作的目標是否已被摧毀。
    # ------------------------------------------------------------------
    def test_this_destroys_it_fires_when_target_dies(self) -> None:
        """AWOKEN GIANT：對前線敵方單位造成 2 傷害，摧毀的話抽一張牌。"""
        card = self.cards.get("awoken_giant")
        weak_target = self.cards.get(PLAIN_GROUND)  # 1 atk / 1 def -- 2 傷害必死
        state = self._state(p2_units=(("t1", weak_target.id, weak_target.attack or 0, 1, "frontline"),))
        state.players["p1"].hand = [card.id]
        before_hand = len(state.players["p1"].hand) - 1  # 這張卡本身會在 PlayCardAction 裡先離手
        self.engine.execute(card.id, "on_play", state,
                            EffectContext("p1", card.id, None, "t1", event="on_play"))
        self.assertNotIn("t1", [u.instance_id for u in state.players["p2"].units])
        self.assertGreater(len(state.players["p1"].hand), before_hand,
                           "AWOKEN GIANT 摧毀目標後沒有抽牌")

    def test_this_destroys_it_does_not_fire_when_target_survives(self) -> None:
        card = self.cards.get("awoken_giant")
        tough_target = self.cards.get(PLAIN_GROUND)
        state = self._state(p2_units=(("t1", tough_target.id, tough_target.attack or 0, 9, "frontline"),))
        state.players["p1"].hand = []
        before_hand = len(state.players["p1"].hand)
        self.engine.execute(card.id, "on_play", state,
                            EffectContext("p1", card.id, None, "t1", event="on_play"))
        self.assertIn("t1", [u.instance_id for u in state.players["p2"].units], "目標不該被 2 傷害摧毀")
        self.assertEqual(len(state.players["p1"].hand), before_hand,
                         "目標沒被摧毀，不該抽牌")

    # ------------------------------------------------------------------
    # "When your HQ gains defense" -- native.gain_hq_defense 分支廣播。
    # ------------------------------------------------------------------
    def test_hq_gained_defense_broadcasts_to_listener(self) -> None:
        """111th INDIAN BRIGADE：「你的 HQ 獲得防禦力時，額外獲得 1 防禦力。」
        用 BALUCH REGIMENT（部署：HQ +3 防禦）當觸發源，驗證「任何來源」都算。
        """
        listener = self.cards.get("111th_indian_brigade")
        source = self.cards.get("baluch_regiment")
        state = self._state(p1_units=(
            ("l1", listener.id, listener.attack or 0, listener.defense or 0, "frontline"),
            ("s1", source.id, source.attack or 0, source.defense or 0, "support_line"),
        ))
        before = state.players["p1"].hq.current_health
        self.engine.execute(source.id, "on_deploy", state,
                            EffectContext("p1", source.id, "s1", event="on_deploy"))
        # +3（BALUCH REGIMENT 自己）+ 1（111th INDIAN BRIGADE 監聽到之後額外加的）
        self.assertEqual(state.players["p1"].hq.current_health, before + 4)

    def test_hq_gained_defense_buff_card_gets_correct_action(self) -> None:
        """回歸：73e RÉGIMENT D'INFANTERIE 的「+1+1 when your HQ gains defense」
        以前被 parser 誤判成「gain_hq_defense」動作本身，吃掉了真正的 +1+1。"""
        rule = self.engine.rule_for("73e_rgiment_dinfanterie")
        self.assertIn("on_hq_gained_defense", rule.triggers)
        kinds = [a.kind for a in rule.actions]
        self.assertIn("buff", kinds, "73e RÉGIMENT D'INFANTERIE 的 +1+1 效果不見了")
        self.assertNotIn("gain_hq_defense", kinds,
                         "觸發子句被誤判成了『獲得 HQ 防禦力』這個動作本身")

    def test_hq_gained_defense_damage_card_gets_correct_action(self) -> None:
        """回歸：COMMANDOS KIEFFER 一樣的 parser bug,真正效果是對隨機敵人造成傷害。"""
        rule = self.engine.rule_for("commandos_kieffer")
        self.assertIn("on_hq_gained_defense", rule.triggers)
        kinds = [a.kind for a in rule.actions]
        self.assertIn("damage", kinds)
        self.assertNotIn("gain_hq_defense", kinds)

    # ------------------------------------------------------------------
    # "When you gain an extra kredit slot" / "...lose a kredit slot" --
    # native.gain_kredit_slot / lose_kredit_slot 分支廣播。
    # ------------------------------------------------------------------
    def test_kredit_slot_gained_broadcasts_to_listener(self) -> None:
        """106th QUARTERMASTERS：「你獲得額外 kredit 槽時，抽一張牌。」
        用 WAR BONDS（Deployment: 獲得 2 個額外 kredit 槽）當觸發源。
        """
        listener = self.cards.get("106th_quartermasters")
        source = self.cards.get("war_bonds")
        state = self._state(p1_units=(("l1", listener.id, listener.attack or 0, listener.defense or 0, "frontline"),))
        state.players["p1"].hand = []
        before_hand = len(state.players["p1"].hand)
        self.engine.execute(source.id, "on_play", state,
                            EffectContext("p1", source.id, event="on_play"))
        self.assertGreater(len(state.players["p1"].hand), before_hand,
                           "106th QUARTERMASTERS 沒有在隊友獲得額外 kredit 槽時抽牌")

    def test_kredit_slot_gained_card_keeps_its_own_effect(self) -> None:
        """回歸：SB2C HELLDIVER 的「獲得額外 kredit 槽時，複製一張到手牌」
        以前被誤判成「gain_kredit_slot」動作本身,吃掉了入手效果。"""
        rule = self.engine.rule_for("sb2c_helldiver")
        self.assertIn("on_kredit_slot_gained", rule.triggers)
        kinds = [a.kind for a in rule.actions]
        self.assertIn("add_card", kinds)
        self.assertNotIn("gain_kredit_slot", kinds)

    def test_kredit_slot_lost_card_keeps_its_own_effect(self) -> None:
        """回歸：MURASE BATTALION 的「失去 kredit 槽時，額外獲得 3 kredit」
        以前被誤判成「lose_kredit_slot」動作本身,吃掉了獲得 kredit 效果。"""
        rule = self.engine.rule_for("murase_battalion")
        self.assertIn("on_kredit_slot_lost", rule.triggers)
        kinds = [a.kind for a in rule.actions]
        self.assertIn("gain_kredits", kinds)
        self.assertNotIn("lose_kredit_slot", kinds)

    # ------------------------------------------------------------------
    # "When this unit destroys an enemy unit" -- AttackAction 早就有 emit，
    # 缺的只是 parser 的觸發接線。
    # ------------------------------------------------------------------
    def test_destroys_unit_trigger_wired_up(self) -> None:
        rule_hq = self.engine.rule_for("142nd_infantry_regiment")
        rule_draw = self.engine.rule_for("p47d_thunderbolt")
        self.assertIn("on_destroys_unit", rule_hq.triggers)
        self.assertIn("on_destroys_unit", rule_draw.triggers)

    def test_destroys_unit_fires_hq_defense_gain(self) -> None:
        """142nd INFANTRY REGIMENT：「這個單位摧毀敵方單位時，HQ +2 防禦。」"""
        from simulator.actions.action import AttackAction

        attacker_card = self.cards.get("142nd_infantry_regiment")
        weak_target = self.cards.get(PLAIN_GROUND)
        state = self._state(
            p1_units=(("atk", attacker_card.id, attacker_card.attack or 10, attacker_card.defense or 10, "frontline"),),
            p2_units=(("tgt", weak_target.id, weak_target.attack or 0, 1, "frontline"),),
        )
        before = state.players["p1"].hq.current_health
        AttackAction("p1", "atk", "tgt").execute(state, self.cards)
        self.assertNotIn("tgt", [u.instance_id for u in state.players["p2"].units])
        self.assertEqual(state.players["p1"].hq.current_health, before + 2)


if __name__ == "__main__":
    unittest.main()
