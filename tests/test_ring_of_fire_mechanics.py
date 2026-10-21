# -*- coding: utf-8 -*-
"""火山環帶機制：手牌位置、指定銷毀、最左邊條件。

這些卡在併入火山環帶後是 partial 或行為錯誤：
- 手牌位置（最左／最右）引擎完全沒有這個概念。
- FEROCIOUS ASSAULT 的「銷毀目標步兵」沒被解析（"target" 不在 destroy 模板的
  冠詞清單裡），所以只會失去指揮點槽，核心效果沒發生。
- TASK FORCE 44 的 2 點與 3 點傷害**都會**造成（"instead" 與條件都沒生效）。
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules import hand_position
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class HandPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_hand=(), p1_units=(), p2_units=(), kredits=12) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Anzac", deck=["garrison"] * 8, hand=list(p1_hand),
                                  resources=ResourceState(kredits, kredits),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Japan", deck=["garrison"] * 8,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", "support_line")
                                         for uid, cid, atk, dfn in p2_units]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=3,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield["support_line"].append(unit.instance_id)
        return state

    # --- 約定 ---

    def test_leftmost_is_index_zero(self) -> None:
        """最左邊是 hand[0]，最右邊是 hand[-1]。"""
        self.assertTrue(hand_position.is_leftmost(0, 3))
        self.assertFalse(hand_position.is_leftmost(1, 3))
        self.assertTrue(hand_position.is_rightmost(2, 3))
        self.assertFalse(hand_position.is_rightmost(0, 3))

    def test_edge_helpers_are_false_without_a_position(self) -> None:
        self.assertFalse(hand_position.is_leftmost(None, None))
        self.assertFalse(hand_position.is_rightmost(None, 0))

    # --- 丟棄最左邊 ---

    def test_7th_brigade_discards_the_leftmost_card(self) -> None:
        """7 旅（部署：丟棄最左邊的手牌）。"""
        state = self._state(p1_hand=("7th_brigade", "radar", "garrison"))
        PlayCardAction("p1", "7th_brigade").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, ["garrison"])
        self.assertIn("radar", state.graveyard["p1"])

    def test_discarding_an_empty_hand_is_safe(self) -> None:
        state = self._state(p1_hand=("7th_brigade",))
        PlayCardAction("p1", "7th_brigade").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, [])

    def test_forward_base_discards_and_damages(self) -> None:
        """前進基地：丟棄最左邊的手牌，並對目標造成等同其費用的傷害。"""
        state = self._state(p1_hand=("forward_base", "radar", "garrison"),
                            p2_units=(("e1", "ace_of_spades", 0, 30),))
        PlayCardAction("p1", "forward_base", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, ["garrison"])
        cost = self.cards.get("ace_of_spades").kredits
        self.assertEqual(30 - state.players["p2"].units[0].defense, cost)

    # --- 移到最左邊 ---

    def test_hmas_warramunga_moves_the_chosen_card_leftmost(self) -> None:
        """HMAS WARRAMUNGA：把選到的手牌移到最左邊。"""
        state = self._state(p1_hand=("hmas_warramunga", "radar", "garrison"))
        PlayCardAction("p1", "hmas_warramunga", selected_card_id="garrison").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand[0], "garrison")

    def test_moving_a_card_not_in_hand_is_safe(self) -> None:
        state = self._state(p1_hand=("hmas_warramunga", "radar"))
        PlayCardAction("p1", "hmas_warramunga", selected_card_id="not_a_card").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, ["radar"])

    # --- 最左邊條件 ---

    def test_task_force_44_deals_three_when_leftmost(self) -> None:
        """第 44 特遣艦隊在最左邊時造成 3 點（不是 2+3=5）。"""
        state = self._state(p1_hand=("task_force_44", "garrison"),
                            p2_units=(("e1", "1_infantry_regiment", 0, 30),))
        PlayCardAction("p1", "task_force_44", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(30 - state.players["p2"].units[0].defense, 3)

    def test_task_force_44_deals_two_when_not_leftmost(self) -> None:
        state = self._state(p1_hand=("garrison", "task_force_44"),
                            p2_units=(("e1", "1_infantry_regiment", 0, 30),))
        PlayCardAction("p1", "task_force_44", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(30 - state.players["p2"].units[0].defense, 2)


class FerociousAssaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, enemy_attack: int, slots: int) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Japan", deck=["garrison"] * 8, hand=["ferocious_assault"],
                                  resources=ResourceState(slots, slots)),
                "p2": PlayerState("p2", "USA", deck=["garrison"] * 8,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState("e1", "1_infantry_regiment", enemy_attack, 20,
                                                   "p2", "support_line")]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=3,
        )
        state.battlefield["support_line"].append("e1")
        return state

    def test_destroys_the_target_infantry(self) -> None:
        """核心效果：銷毀目標步兵（原本完全沒發生）。"""
        state = self._state(enemy_attack=5, slots=12)
        PlayCardAction("p1", "ferocious_assault", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units, [])

    def test_keeps_the_kredit_slot_when_attack_is_not_greater(self) -> None:
        """攻擊力沒有大於指揮點槽數時，不該失去槽。"""
        state = self._state(enemy_attack=5, slots=12)
        PlayCardAction("p1", "ferocious_assault", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(state.players["p1"].resources.max_kredits, 12)

    def test_loses_a_kredit_slot_when_attack_is_greater(self) -> None:
        """攻擊力大於指揮點槽數時失去一槽。"""
        state = self._state(enemy_attack=5, slots=3)
        PlayCardAction("p1", "ferocious_assault", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(state.players["p1"].resources.max_kredits, 2)
        self.assertEqual(state.players["p2"].units, [])


if __name__ == "__main__":
    unittest.main()
