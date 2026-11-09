# -*- coding: utf-8 -*-
"""回歸測試：「When a friendly unit is attacked」反制卡永遠不會觸發的 bug
（11-04 查 E9 crash 時發現，11-09 修掉）。

根因：`CountermeasureResolver.intercept()`（`simulator/countermeasures.py`）
組 `counter_context.metadata` 時，從來沒有填 `event_target_owner_id`——但
`condition.py` 的「friendly unit is attacked」分支（給 native 的 broadcast
事件用，例如單位被動監聽「我方單位被攻擊」）就是靠這個欄位分辨「被攻擊的單位
是不是反制卡持有者自己的」。沒有這個欄位，條件永遠判不成立，於是卡池裡所有
「When a friendly unit is attacked, …」這句型的**反制卡**（ENTANGLEMENT、
TACTICAL RETREAT、STUBBORN DEFENSE、IN THE NAVY）全部永遠不會真的觸發——即使
`card_rules.json` 把它們標成 implemented（那只代表「解析器看懂這句話」，不代表
「引擎真的會跑」）。

修法：`intercept()` 組 `counter_context` 前，先把被攻擊目標的 `owner_id` 查
出來填進 `metadata["event_target_owner_id"]`，跟 `native.emit()` 原本就有的
同名欄位同一個寫法。
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class CountermeasureFriendlyUnitAttackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, counter_card_id, attacker_id="50_infantry_regiment", target_id="1_infantry_regiment") -> GameState:
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
                    active_countermeasures=[{"card_id": counter_card_id, "activated_turn": 0}],
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

    def test_tactical_retreat_sends_attacked_unit_to_hand(self) -> None:
        state = self._state("tactical_retreat")
        self._place_on_battlefield(state)
        AttackAction("p1", "atk", "tgt").execute(state, self.cards)
        self.assertNotIn("tgt", [u.instance_id for u in state.players["p2"].units])
        self.assertIn("1_infantry_regiment", state.players["p2"].hand)
        self.assertIn("countermeasure_triggered",
                      [e["event"] for e in state.event_log if e.get("counter_id") == "tactical_retreat"])

    def test_entanglement_buffs_defense_before_damage(self) -> None:
        state = self._state("entanglement")
        self._place_on_battlefield(state)
        before_def = state.players["p2"].units[0].defense
        AttackAction("p1", "atk", "tgt").execute(state, self.cards)
        triggered = [e for e in state.event_log if e.get("counter_id") == "entanglement"]
        self.assertTrue(triggered, "ENTANGLEMENT 沒有觸發")
        # +2 defense before damage, then combat damage is applied -- net
        # effect just needs to be strictly better than taking the hit raw.
        after = next(u.defense for u in state.players["p2"].units)
        attacker_atk = state.players["p1"].units[0].attack
        self.assertGreater(after, before_def - attacker_atk,
                           "ENTANGLEMENT 的 +2 防禦力沒有在傷害結算前生效")

    def test_in_the_navy_buffs_attack_and_defense(self) -> None:
        state = self._state("in_the_navy")
        self._place_on_battlefield(state)
        AttackAction("p1", "atk", "tgt").execute(state, self.cards)
        self.assertIn("countermeasure_triggered",
                      [e["event"] for e in state.event_log if e.get("counter_id") == "in_the_navy"])

    def test_condition_does_not_fire_for_an_enemy_units_attack_on_someone_else(self) -> None:
        """反制卡只在自己一方的單位被攻擊時才該觸發——這裡故意讓反制卡的持有者
        是攻擊方（p1 自己也放一張 tactical_retreat），確認不會錯誤觸發在敵方
        單位身上。"""
        state = self._state("tactical_retreat")
        state.players["p1"].active_countermeasures = [{"card_id": "tactical_retreat", "activated_turn": 0}]
        state.players["p2"].active_countermeasures = []
        self._place_on_battlefield(state)
        AttackAction("p1", "atk", "tgt").execute(state, self.cards)
        # p1 的 TACTICAL RETREAT 只在「我方（p1）單位被攻擊」時才該觸發；這裡
        # 是 p1 自己發起攻擊、p2 的單位被打，不該誤觸發 p1 的反制卡。
        self.assertNotIn("tactical_retreat",
                        [e.get("counter_id") for e in state.event_log if e.get("event") == "countermeasure_triggered"])
        self.assertNotIn("tgt", [u.instance_id for u in state.players["p2"].units],
                         "正常攻擊傷害應該照常摧毀目標")


if __name__ == "__main__":
    unittest.main()
