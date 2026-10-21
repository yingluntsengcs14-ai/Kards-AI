# -*- coding: utf-8 -*-
"""v9 規則對照測試（parity tests）。

來源：作者自己寫的 kards_ai_v9_hidden_info.ipynb 內 `rules_test()`,
以及 HANDOFF_交接.md 裡已在遊戲內確認過的規則。

這個檔案的用途是把「已驗證的規則真相」釘在 Kards-AI 引擎上。
測試失敗代表 Kards-AI 的實作和真實 KARDS 行為不符，不是測試寫錯。
每個測試的 docstring 都寫明規則來源與期望值。

放置位置：tests/test_v9_rule_parity.py
執行：python -m unittest tests.test_v9_rule_parity -v
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction
from simulator.testing import build_random_deck

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"

# 一張沒有任何效果文字的德國步兵，當作「乾淨的地面單位載體」使用；
# 攻擊力/防禦力由 UnitState 直接指定，不受卡面效果干擾。
PLAIN_GROUND = "1_infantry_regiment"


class V9RuleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _state(self, p1_units=(), p2_units=(), p1_hand=()) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=[PLAIN_GROUND] * 10, hand=list(p1_hand),
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

    def _deploy_110(self, state: GameState, unit_id: str = "pz") -> None:
        """讓 110 裝甲擲彈兵的光環生效（跑它的 on_deploy 規則）。"""
        self.engine.execute("110_panzergrenadier", "on_deploy", state,
                            EffectContext("p1", "110_panzergrenadier", unit_id))

    # ------------------------------------------------------------------
    # 1. 110 裝甲擲彈兵
    #    規則（已在遊戲內確認）：攻擊力 >=4 的友方陸軍「造成的傷害」+3,
    #    攻擊力本身不變。攻擊、反擊、部署效果的傷害都會加；
    #    指令卡造成的傷害不加。
    # ------------------------------------------------------------------
    def test_110_attack_hq_damage_plus_3(self) -> None:
        """110 在場時，5 攻陸軍打敵方總部應造成 5+3=8。"""
        state = self._state(
            p1_units=(("pz", "110_panzergrenadier", 5, 5, "support_line"),
                      ("tk", PLAIN_GROUND, 5, 5, "frontline")),
        )
        self._deploy_110(state)
        before = state.players["p2"].hq.current_health
        AttackAction("p1", "tk").execute(state, self.cards)
        self.assertEqual(before - state.players["p2"].hq.current_health, 8,
                         "110 光環沒有加到攻擊總部的戰鬥傷害")

    def test_110_attack_unit_damage_plus_3(self) -> None:
        """110 在場時，5 攻陸軍攻擊敵方單位應造成 5+3=8 傷害。"""
        state = self._state(
            p1_units=(("pz", "110_panzergrenadier", 5, 5, "support_line"),
                      ("tk", PLAIN_GROUND, 5, 5, "frontline")),
            p2_units=(("enemy", PLAIN_GROUND, 0, 30, "frontline"),),
        )
        self._deploy_110(state)
        AttackAction("p1", "tk", "enemy").execute(state, self.cards)
        enemy = state.players["p2"].units[0]
        self.assertEqual(30 - enemy.defense, 8,
                         "110 光環沒有加到攻擊單位的戰鬥傷害")

    def test_110_counter_attack_damage_plus_3(self) -> None:
        """110 在場時，被攻擊的 5 攻陸軍反擊應造成 5+3=8 傷害。"""
        state = self._state(
            p1_units=(("pz", "110_panzergrenadier", 5, 5, "support_line"),
                      ("tk", PLAIN_GROUND, 5, 30, "frontline")),
            p2_units=(("enemy", PLAIN_GROUND, 1, 30, "frontline"),),
        )
        self._deploy_110(state)
        state.current_player = "p2"
        AttackAction("p2", "enemy", "tk").execute(state, self.cards)
        enemy = state.players["p2"].units[0]
        self.assertEqual(30 - enemy.defense, 8,
                         "110 光環沒有加到反擊傷害")

    def test_110_deploy_effect_damage_plus_3(self) -> None:
        """黑桃 A 師（8 攻）部署效果 3 傷害，在 110 光環下應變成 6。"""
        state = self._state(
            p1_units=(("pz", "110_panzergrenadier", 5, 5, "support_line"),),
            p2_units=(("enemy", PLAIN_GROUND, 0, 30, "frontline"),),
            p1_hand=("ace_of_spades",),
        )
        self._deploy_110(state)
        PlayCardAction("p1", "ace_of_spades", target_unit_id="enemy").execute(state, self.cards)
        enemy = next(u for u in state.players["p2"].units if u.instance_id == "enemy")
        self.assertEqual(30 - enemy.defense, 6,
                         "110 光環沒有加到部署效果傷害")

    def test_110_does_not_boost_order_damage(self) -> None:
        """指令卡（提爾皮茨號 4 點總部傷害）不是單位造成的傷害，不該被 110 加成。"""
        state = self._state(
            p1_units=(("pz", "110_panzergrenadier", 5, 5, "support_line"),),
            p1_hand=("km_tirpitz",),
        )
        self._deploy_110(state)
        before = state.players["p2"].hq.current_health
        PlayCardAction("p1", "km_tirpitz").execute(state, self.cards)
        self.assertEqual(before - state.players["p2"].hq.current_health, 4,
                         "110 光環錯誤地加到指令卡傷害")

    # ------------------------------------------------------------------
    # 2. 老兵
    #    規則：變成老兵是「換成一張全新的老兵卡」,
    #    原本的傷害、加成、操作成本調整全部不保留。
    # ------------------------------------------------------------------
    def test_veteran_transformation_resets_unit(self) -> None:
        """成為老兵後，單位應換成老兵卡並重置數值（傷害與 +攻 加成都消失）。"""
        state = self._state(p1_units=(("u1", "50_infantry_regiment", 3, 2, "frontline"),))
        unit = state.players["p1"].units[0]
        unit.defense = 1                      # 受過傷
        unit.attack = 5                       # 吃過 +2 攻擊
        self.engine._execute_action(
            RuleAction("grant_ability", "source", card_name="veteran"),
            state, EffectContext("p1", "50_infantry_regiment", "u1"),
        )
        self.assertNotEqual(unit.card_id, "50_infantry_regiment",
                            "成為老兵後 card_id 沒有換成老兵版卡牌")
        self.assertEqual(unit.defense, self.cards.get(unit.card_id).defense,
                         "老兵沒有重置防禦力（舊的傷害被保留了）")
        self.assertEqual(unit.attack, self.cards.get(unit.card_id).attack,
                         "老兵沒有重置攻擊力（舊的加成被保留了）")

    def test_conditional_veteran_does_not_promote_on_deploy(self) -> None:
        """條件式老兵（190 近衛步兵：擊毀 4 個敵方單位）不該在部署時就換卡。

        condition.evaluate 對沒有建模的條件一律回傳 True，所以換卡必須另外
        用「無條件」來把關，否則單位一部署就變成老兵。
        """
        card = self.cards.get("190th_guard_rifles")
        state = self._state(p1_units=(("u1", card.id, card.attack or 0, card.defense or 0, "support_line"),))
        unit = state.players["p1"].units[0]
        self.engine.emit("on_deploy", state, EffectContext("p1", card.id, "u1"))
        self.engine.execute(card.id, "on_deploy", state, EffectContext("p1", card.id, "u1"))
        self.assertEqual(unit.card_id, "190th_guard_rifles",
                         "條件還沒滿足就把卡換成老兵版了")
        self.assertTrue(unit.status.get("cannot_attack"),
                        "卡面的『無法攻擊或移動』被換卡流程清掉了")

    def test_catalog_contains_veteran_counterpart(self) -> None:
        """卡池應包含老兵版卡牌（目前缺 44 張 Special 老兵版，見 HANDOFF）。"""
        veterans = [card.id for card in self.cards if card.id.endswith("_vet")]
        self.assertIn("50_infantry_regiment_vet", veterans,
                      "卡池裡找不到 50 步兵團的老兵版")

    # ------------------------------------------------------------------
    # 3. 撤退與爆牌
    #    規則：前線 -> 支援線最右邊；支援線滿 4 個 -> 回手牌；
    #    手牌也滿 9 張 -> 移出遊戲（不觸發亡語）。
    # ------------------------------------------------------------------
    def _retreat(self, state: GameState, unit_id: str) -> None:
        self.engine._execute_action(
            RuleAction("move_to_support_line", "selected_target"),
            state, EffectContext("p1", target_unit_id=unit_id),
        )

    def test_retreat_to_rightmost_support_slot(self) -> None:
        """撤退後單位應在支援線最右邊。"""
        state = self._state(p1_units=(("a", PLAIN_GROUND, 1, 2, "support_line"),
                                      ("f", PLAIN_GROUND, 1, 2, "frontline")))
        self._retreat(state, "f")
        unit = next(u for u in state.players["p1"].units if u.instance_id == "f")
        self.assertEqual(unit.position, "support_line")
        self.assertEqual(state.battlefield["support_line"][-1], "f",
                         "撤退的單位沒有放到支援線最右邊")

    def test_retreat_with_full_support_returns_to_hand(self) -> None:
        """支援線已滿 4 個時，撤退的單位應回到手牌。"""
        state = self._state(
            p1_units=tuple((f"s{i}", PLAIN_GROUND, 1, 2, "support_line") for i in range(4))
            + (("f", PLAIN_GROUND, 1, 2, "frontline"),),
        )
        self._retreat(state, "f")
        ids = {u.instance_id for u in state.players["p1"].units}
        self.assertNotIn("f", ids, "支援線滿時撤退的單位沒有離場")
        self.assertIn(PLAIN_GROUND, state.players["p1"].hand,
                      "支援線滿時撤退的單位沒有回到手牌")

    def test_retreat_with_full_support_and_full_hand_removes_unit(self) -> None:
        """支援線滿、手牌也滿 9 張時，撤退的單位應直接移出遊戲。"""
        state = self._state(
            p1_units=tuple((f"s{i}", PLAIN_GROUND, 1, 2, "support_line") for i in range(4))
            + (("f", PLAIN_GROUND, 1, 2, "frontline"),),
            p1_hand=tuple([PLAIN_GROUND] * 9),
        )
        self._retreat(state, "f")
        ids = {u.instance_id for u in state.players["p1"].units}
        self.assertNotIn("f", ids, "支援線滿、手牌滿時撤退的單位沒有離場")
        self.assertEqual(len(state.players["p1"].hand), 9,
                         "手牌已滿 9 張還是被塞進第 10 張")

    # ------------------------------------------------------------------
    # 4. 起手與抽牌
    #    規則：先手 4 張且第一回合不抽；後手 5 張，第一回合抽 1 共 6 張。
    # ------------------------------------------------------------------
    def test_opening_hand_sizes(self) -> None:
        """先手起手 4 張、後手 5 張。"""
        sim = Simulator(cards=self.cards, record_replay=False)
        deck = build_random_deck(self.cards, seed=1, main_nation="Germany")
        sim.reset(list(deck), list(deck), nations=("Germany", "Germany"), seed=1)
        self.assertEqual(len(sim.state.players["p1"].hand), 4)
        self.assertEqual(len(sim.state.players["p2"].hand), 5)

    def test_mulligan_conserves_card_count(self) -> None:
        """換牌不會改變手牌加牌庫的總張數。"""
        sim = Simulator(cards=self.cards, record_replay=False)
        deck = build_random_deck(self.cards, seed=1, main_nation="Germany")
        sim.reset(list(deck), list(deck), nations=("Germany", "Germany"), seed=1)
        player = sim.state.players["p1"]
        self.assertEqual(len(player.hand) + len(player.deck), 40,
                         "換牌後手牌加牌庫總張數不是 40")


if __name__ == "__main__":
    unittest.main()
