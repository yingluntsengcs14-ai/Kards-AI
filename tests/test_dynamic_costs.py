# -*- coding: utf-8 -*-
"""自己重新定價的卡（simulator/rules/dynamic_costs.py）。

這兩張卡的減費原本被解析成「打出後才執行的效果」，所以：
- SURPRISE ENGAGEMENT 讓**其他**手牌便宜 2，自己的費用從沒變過。
- DETAILED RECON 的減費完全沒發生。

現在費用在打出的那一刻從 state 讀出來。
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, PlayCardAction, card_play_cost
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules import counters

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class DynamicCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, hand=(), p1_units=(), nation="Germany") -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", nation, deck=["garrison"] * 8, hand=list(hand),
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "frontline")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Japan", deck=["garrison"] * 8,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState("e1", "1_infantry_regiment", 0, 30,
                                                   "p2", "support_line")]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=3,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
        return state

    def test_detailed_recon_costs_less_after_operating(self) -> None:
        """DETAILED RECON：費用依本回合花在行動上的點數遞減。"""
        state = self._state(hand=("detailed_recon",), p1_units=(("u1", "garrison", 2, 5),))
        before = card_play_cost(state, "p1", self.cards.get("detailed_recon"), self.cards)
        AttackAction("p1", "u1").execute(state, self.cards)
        spent = counters.operating_spend_this_turn(state, "p1")
        self.assertGreater(spent, 0)
        after = card_play_cost(state, "p1", self.cards.get("detailed_recon"), self.cards)
        self.assertEqual(before - after, spent)

    def test_surprise_engagement_costs_two_less_after_rain(self) -> None:
        """SURPRISE ENGAGEMENT：本回合打過雨卡則便宜 2。"""
        state = self._state(hand=("surprise_engagement", "deluge"), nation="USA")
        before = card_play_cost(state, "p1", self.cards.get("surprise_engagement"), self.cards)
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        after = card_play_cost(state, "p1", self.cards.get("surprise_engagement"), self.cards)
        self.assertEqual(before - after, 2)

    def test_surprise_engagement_does_not_discount_other_cards(self) -> None:
        """原本的 bug：減費跑到其他手牌上。"""
        state = self._state(hand=("surprise_engagement", "deluge", "garrison"), nation="USA")
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("garrison"), self.cards),
                         self.cards.get("garrison").kredits)

    def test_no_discount_without_rain(self) -> None:
        state = self._state(hand=("surprise_engagement",), nation="USA")
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("surprise_engagement"), self.cards),
                         self.cards.get("surprise_engagement").kredits)

    def test_cost_never_goes_below_zero(self) -> None:
        state = self._state(hand=("detailed_recon",))
        counters.note_operating_spend(state, "p1", 99)
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("detailed_recon"), self.cards), 0)

    def test_ordinary_cards_are_untouched(self) -> None:
        state = self._state(hand=("radar",))
        counters.note_operating_spend(state, "p1", 5)
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("radar"), self.cards),
                         self.cards.get("radar").kredits)


if __name__ == "__main__":
    unittest.main()
