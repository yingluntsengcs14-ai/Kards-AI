# -*- coding: utf-8 -*-
"""天氣卡本身的效果。

熱浪、暴雨、旋風逐張驗。另外釘住一個既有 bug 的修復：`all_units` 這個
selector 原本在 native 與 resolver 都沒有實作，導致「所有單位 +N」的卡
（熱浪、叢林熱、root_out_the_enemy、enemy_at_the_gates）效果靜默失效，
而其中三張還被標成 implemented。
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"

INFANTRY = ("g1", "garrison", 0, 3)        # 步兵 0/3
AIR = ("a1", "duck", 1, 1)                 # 戰鬥機 1/1


class WeatherCardEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, hand=(), p1_units=(INFANTRY, AIR), p2_units=(("e1", "garrison", 0, 3),)) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "USA", deck=["garrison"] * 8, hand=list(hand),
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Japan", deck=["garrison"] * 8,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", "support_line")
                                         for uid, cid, atk, dfn in p2_units]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=7,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield["support_line"].append(unit.instance_id)
        return state

    # --- 熱浪 ---

    def test_heatwave_gives_ground_units_plus_two_attack(self) -> None:
        """熱浪：陸軍 +2 攻擊。"""
        state = self._state(hand=("heatwave",))
        PlayCardAction("p1", "heatwave").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].attack, 2)

    def test_heatwave_gives_air_units_plus_three_instead(self) -> None:
        """熱浪：空軍是 +3 取代 +2（1/1 的鴨式 → 攻擊 4）。"""
        state = self._state(hand=("heatwave",))
        PlayCardAction("p1", "heatwave").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[1].attack, 4)

    def test_heatwave_affects_both_players(self) -> None:
        """卡面是「所有單位」，敵方單位也會吃到（2026-10-20 遊戲內確認）。"""
        state = self._state(hand=("heatwave",))
        PlayCardAction("p1", "heatwave").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].attack, 2)

    # --- 暴雨 ---

    def test_deluge_distributes_exactly_six_defense(self) -> None:
        """暴雨：總共正好分配 6 點防禦給友方目標（含總部）。"""
        state = self._state(hand=("deluge",))
        before_units = [unit.defense for unit in state.players["p1"].units]
        before_hq = state.players["p1"].hq.defense_modifier
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        gained = sum(unit.defense - old for unit, old in zip(state.players["p1"].units, before_units))
        gained += state.players["p1"].hq.defense_modifier - before_hq
        self.assertEqual(gained, 6)

    def test_deluge_does_not_buff_the_enemy(self) -> None:
        state = self._state(hand=("deluge",))
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].defense, 3)

    def test_deluge_is_deterministic_for_a_given_seed(self) -> None:
        """同樣的 seed 要分配出同樣的結果（self-play 需要可重現）。"""
        results = []
        for _ in range(2):
            state = self._state(hand=("deluge",))
            PlayCardAction("p1", "deluge").execute(state, self.cards)
            results.append([unit.defense for unit in state.players["p1"].units]
                           + [state.players["p1"].hq.defense_modifier])
        self.assertEqual(results[0], results[1])

    # --- 旋風 ---

    def test_cyclone_draws_three_cards(self) -> None:
        state = self._state(hand=("cyclone",))
        PlayCardAction("p1", "cyclone").execute(state, self.cards)
        self.assertEqual(len(state.players["p1"].hand), 3)

    def test_cyclone_damages_own_hq_by_hand_size(self) -> None:
        """旋風：抽 3 張後，對自己總部造成等同手牌數的傷害。"""
        state = self._state(hand=("cyclone", "garrison", "garrison"))
        before = state.players["p1"].hq.current_health
        PlayCardAction("p1", "cyclone").execute(state, self.cards)
        # 打掉旋風後手上剩 2 張，抽 3 張 → 5 張 → 扣 5。
        self.assertEqual(len(state.players["p1"].hand), 5)
        self.assertEqual(before - state.players["p1"].hq.current_health, 5)

    def test_cyclone_does_not_touch_the_enemy_hq(self) -> None:
        state = self._state(hand=("cyclone",))
        before = state.players["p2"].hq.current_health
        PlayCardAction("p1", "cyclone").execute(state, self.cards)
        self.assertEqual(state.players["p2"].hq.current_health, before)

    # --- all_units selector 的回歸測試 ---

    def test_all_units_selector_is_implemented(self) -> None:
        """「所有單位 +N」不能靜默失效（叢林熱：全體 +3 攻擊）。"""
        state = self._state(hand=("jungle_fever",), p1_units=(INFANTRY,), p2_units=(("e1", "garrison", 0, 3),))
        PlayCardAction("p1", "jungle_fever").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].attack, 3, "友方沒吃到 +3")
        self.assertEqual(state.players["p2"].units[0].attack, 3, "敵方沒吃到 +3")

    # --- 預報卡 ---

    def test_forecast_tokens_carry_no_action(self) -> None:
        """預報卡只是宣告即將到來的天氣，機制在 weather.py，本身沒有效果。"""
        for card_id in ("blue_sky", "mist", "gale"):
            rule = self.engine.rule_for(card_id)
            self.assertEqual(rule.actions, (), card_id)
            self.assertEqual(rule.status, "implemented", card_id)


if __name__ == "__main__":
    unittest.main()
