# -*- coding: utf-8 -*-
"""回歸測試（E9，2026-11-04）：反制卡在攻擊時摧毀目標，不該讓攻擊當掉。

真實當掉（150 局 sweep，seed 967974037，USA:Poland vs Germany:Britain）：
``ActionValidationError('Unknown unit instance: 111th_indian_brigade-1')``

追到的實際順序（``Simulator.step`` 丟例外前的 event_log 尾端）：
    countermeasure_triggered  counter_id=entrapment trigger=on_attack
    unit_died                 unit_id=111th_indian_brigade-1
    （crash）AttackAction._target_tax 再去 find_unit(target_unit_id) 時找不到

根因：``AttackAction.execute()`` 先呼叫 ``CountermeasureResolver.intercept()``
處理 "on_attack" 反制卡，*之後*才算 ``operating_cost``（含 ``_target_tax``，
它會用 ``self.target_unit_id`` 再查一次那個單位）。ENTRAPMENT（「單位攻擊時，
摧毀它」）沒有 ``cancel`` 動作──攻擊本身沒有被取消，但目標已經被摧毀了，
``_target_tax`` 卻還拿著舊的 ``target_unit_id`` 去 ``find_unit``，找不到就直
接讓整局當掉。

修法：``_target_tax`` 改成找不到目標時回 0（跟它後面本來就有的「目標在戰鬥
判定前消失，安靜結束攻擊」那段處理一致），而不是讓例外往外傳。

（附帶發現、不在本修法範圍內：TACTICAL RETREAT「被攻擊的友方單位送回手牌」
那張反制卡，條件是 `condition.py` 的「friendly unit is attacked」分支，依賴
`metadata["event_target_owner_id"]`，但 `countermeasures.py` 組
`counter_context.metadata` 時沒有填這個欄位，所以這張反制卡目前永遠不會真
的觸發——這是另一個獨立的 bug，已記進 `未完成清單.md`，不在這次修法處理。）
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class AttackTargetRemovedByCountermeasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, p2_active, attacker_id="50_infantry_regiment", target_id="1_infantry_regiment") -> GameState:
        attacker_card = self.cards.get(attacker_id)
        target_card = self.cards.get(target_id)
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState(
                    "p1", "Germany", deck=[attacker_card.id] * 10, resources=ResourceState(12, 12),
                    units=[UnitState("atk", attacker_card.id, attacker_card.attack or 0,
                                     attacker_card.defense or 0, "p1", "frontline")],
                ),
                "p2": PlayerState(
                    "p2", "USA", deck=[target_card.id] * 10, hand=[], resources=ResourceState(12, 12),
                    active_countermeasures=list(p2_active),
                    units=[UnitState("tgt", target_card.id, target_card.attack or 0,
                                     target_card.defense or 0, "p2", "frontline")],
                ),
            },
            graveyard={"p1": [], "p2": []},
        )

    def _place_on_battlefield(self, state: GameState) -> None:
        for player in state.players.values():
            for unit in player.units:
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)

    def test_entrapment_destroys_target_without_crashing(self) -> None:
        """ENTRAPMENT：攻擊時摧毀目標，_target_tax 不該因為目標已死而當掉。"""
        state = self._state(p2_active=[{"card_id": "entrapment", "activated_turn": 0}])
        self._place_on_battlefield(state)
        action = AttackAction("p1", "atk", "tgt")

        action.execute(state, self.cards)  # must not raise

        self.assertNotIn("tgt", [u.instance_id for u in state.players["p2"].units],
                         "ENTRAPMENT 應該摧毀被攻擊的目標單位")
        self.assertIn("countermeasure_triggered", [e["event"] for e in state.event_log])

    def test_original_crash_reproduction(self) -> None:
        """原始當掉案例的縮小重現：B17 FLYING FORTRESS 攻擊帶 ENTRAPMENT 的一方。"""
        state = self._state(
            p2_active=[{"card_id": "entrapment", "activated_turn": 0}],
            attacker_id="b17_flying_fortress", target_id="111th_indian_brigade",
        )
        self._place_on_battlefield(state)
        action = AttackAction("p1", "atk", "tgt")

        action.execute(state, self.cards)  # used to raise ActionValidationError

        self.assertNotIn("tgt", [u.instance_id for u in state.players["p2"].units])


if __name__ == "__main__":
    unittest.main()
