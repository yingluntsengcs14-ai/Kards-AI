# -*- coding: utf-8 -*-
"""回歸測試：「在手牌裡時，每次 X 發生，這張卡的費用就減 N（會累加）」（F 第四
輪，2026-11-10）：WELLINGTON（你下指令）、STURMBRIGADE RHODOS（你的反制卡
觸發）、MIG-3（敵方反制卡觸發）、34th GUARDS（你的 HQ 受到友傷）。

這 4 張卡本來被解析成 `modify_hand_cost`，一次性套用固定 -2，`condition="If
in your hand"` 只是順手塞在同一顆動作裡、從來沒被當成觸發閘門檢查——跟卡面
「每次 X 發生就再減 2，會累加」的語意完全不符。

做法仿照既有的 `hand_cost_on_enemy_draw`（4th GUARDS RIFLES，`TurnManager.
_enemy_drew` 接線）同一套 per-card、累加式折扣機制：
- `simulator/rules/parser.py` 的 `_META_DECK_CARD_RULES` 幫這 4 張卡各配一個
  專屬的 `RuleAction` kind（`hand_cost_on_order_given`／
  `hand_cost_on_own_countermeasure`／`hand_cost_on_enemy_countermeasure`／
  `hand_cost_on_hq_friendly_damage`），`amount=2`（WELLINGTON 等的折扣是 2，
  不是 4th GUARDS RIFLES 的 1）。
- `NativeRuleEngine.bump_hand_cost_reaction`（新函式）：給定玩家＋kind，掃過
  該玩家手牌裡「有這個 kind 動作」的卡，把 `player.status[kind][card_id]`
  累加 `action.amount`。
- 在真正的事件發生處各接一行呼叫：
  - `action.py` 的 `PlayCardAction.execute`（`card.type == "order"` 分支）→
    `hand_cost_on_order_given`。
  - `countermeasures.py` 的 `CountermeasureResolver.intercept()` 記
    `countermeasure_triggered` 事件後 → 同時對觸發方（`hand_cost_on_own_
    countermeasure`）跟對手（`hand_cost_on_enemy_countermeasure`）各呼叫一次。
  - `core/hq.py` 的 `HQResolver.damage()` 記 `hq_damaged` 事件後，若
    `source_player_id == target_player_id`（自傷）→
    `hand_cost_on_hq_friendly_damage`。
- `action.py` 的 `card_play_cost()` 讀這 4 個 status dict 扣費；`action.py`
  的 `PlayCardAction.execute()` 跟 `core/turn.py` 的 `draw_card()` 都補上跟
  既有 `hand_cost_on_enemy_draw` 一樣的「卡離開手牌就清掉舊紀錄」防呆，避免
  同張卡的新副本繼承前一張副本的折扣歷史。
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, PlayCardAction, card_play_cost
from simulator.cards.loader import CardDatabase
from simulator.core.hq import HQResolver
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules.native import engine_for

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class HandReactiveCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        engine_for(cls.cards)   # make sure NativeRuleEngine._last_engine is set

    def _state(self, **players) -> GameState:
        state = GameState(current_player="p1", players=players, graveyard={pid: [] for pid in players})
        for player in state.players.values():
            for unit in player.units:
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
        return state

    def test_wellington_discounts_two_per_order_given(self) -> None:
        card = self.cards.get("wellington")
        order_card = next(c for c in self.cards if c.type == "order" and c.nation == card.nation)
        state = self._state(
            p1=PlayerState("p1", card.nation, deck=[], hand=["wellington", order_card.id], resources=ResourceState(50, 50)),
            p2=PlayerState("p2", "USA", deck=[], resources=ResourceState(50, 50)),
        )
        base = card_play_cost(state, "p1", card, self.cards)
        PlayCardAction("p1", order_card.id).execute(state, self.cards)
        self.assertEqual(card_play_cost(state, "p1", card, self.cards), base - 2)

    def test_sturmbrigade_and_mig3_react_to_own_vs_enemy_countermeasure(self) -> None:
        srh = self.cards.get("sturmbrigade_rhodos")
        mig = self.cards.get("mig_3")
        attacker_card = self.cards.get("50_infantry_regiment")
        target_card = self.cards.get("1_infantry_regiment")
        state = self._state(
            p1=PlayerState("p1", "Germany", deck=[], hand=["mig_3"], resources=ResourceState(50, 50),
                            units=[UnitState("atk", attacker_card.id, attacker_card.attack or 0,
                                              attacker_card.defense or 0, "p1", "frontline")]),
            p2=PlayerState("p2", "USA", deck=[], hand=["sturmbrigade_rhodos"], resources=ResourceState(50, 50),
                            active_countermeasures=[{"card_id": "tactical_retreat", "activated_turn": 0}],
                            units=[UnitState("tgt", target_card.id, target_card.attack or 0,
                                              target_card.defense or 0, "p2", "frontline")]),
        )
        mig_base = card_play_cost(state, "p1", mig, self.cards)
        srh_base = card_play_cost(state, "p2", srh, self.cards)
        AttackAction("p1", "atk", "tgt").execute(state, self.cards)
        self.assertIn("countermeasure_triggered", [e.get("event") for e in state.event_log])
        # p2's own countermeasure triggered: p2's STURMBRIGADE RHODOS reacts.
        self.assertEqual(card_play_cost(state, "p2", srh, self.cards), srh_base - 2)
        # ... which, from p1's side, was "an enemy countermeasure" triggering:
        # p1's MIG-3 reacts too.
        self.assertEqual(card_play_cost(state, "p1", mig, self.cards), mig_base - 2)

    def test_34th_guards_reacts_only_to_friendly_hq_damage(self) -> None:
        card = self.cards.get("34th_guards")
        state = self._state(
            p1=PlayerState("p1", card.nation, deck=[], hand=["34th_guards"], resources=ResourceState(50, 50)),
            p2=PlayerState("p2", "USA", deck=[], resources=ResourceState(50, 50)),
        )
        base = card_play_cost(state, "p1", card, self.cards)
        HQResolver.damage(state, "p1", 3, "p2")   # enemy-inflicted: should NOT react
        self.assertEqual(card_play_cost(state, "p1", card, self.cards), base)
        HQResolver.damage(state, "p1", 3, "p1")   # self-inflicted: should react
        self.assertEqual(card_play_cost(state, "p1", card, self.cards), base - 2)

    def test_discount_does_not_carry_to_a_fresh_copy_drawn_later(self) -> None:
        """一張副本打出去後，折扣紀錄留在 status dict 裡；下一張同名卡從牌庫
        抽出來時，不該繼承這段歷史（跟既有 hand_cost_on_enemy_draw 防呆一致）。"""
        card = self.cards.get("wellington")
        order_card = next(c for c in self.cards if c.type == "order" and c.nation == card.nation)
        state = self._state(
            p1=PlayerState("p1", card.nation, deck=["wellington"], hand=["wellington", order_card.id],
                            resources=ResourceState(50, 50)),
            p2=PlayerState("p2", "USA", deck=[], resources=ResourceState(50, 50)),
        )
        PlayCardAction("p1", order_card.id).execute(state, self.cards)
        self.assertEqual(
            state.players["p1"].status.get("hand_cost_on_order_given", {}).get("wellington"), 2)
        PlayCardAction("p1", "wellington").execute(state, self.cards)
        self.assertNotIn("wellington", state.players["p1"].hand)
        self.assertNotIn("wellington", state.players["p1"].status.get("hand_cost_on_order_given", {}))
        from simulator.core.turn import TurnManager
        TurnManager.draw_card(state, "p1")
        self.assertIn("wellington", state.players["p1"].hand)
        self.assertEqual(card_play_cost(state, "p1", card, self.cards), card.kredits)


if __name__ == "__main__":
    unittest.main()
