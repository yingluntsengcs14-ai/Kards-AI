# -*- coding: utf-8 -*-
"""回歸測試：「加一張同國家、費用比剛摧毀單位高/低 N」選取器（F 第三輪，11-10）。

根因：RED BANNER／UPGRADE／URAL FACTORIES 這 3 張卡的句型都是「摧毀一個友方
單位，加一張同國家、費用比它高（或低）N 的卡」。解析器本身是對的（action
已經正確拆成 `destroy selected_friendly` + `add_card` with
`condition="costs N more/less"`），但有兩個地方讓它完全不會真的執行：

1. `native._execute_action` 在跑任何 action 前，會先用
   `condition.handled_by_kind`／`condition.evaluate` 檢查 `action.condition`；
   "costs N more" 不在 `condition.py` 認得的任何模式裡，於是被當成「看不懂 ->
   不成立」，整個 `add_card` 動作直接被跳過（`native_rule_condition_false`），
   `_execute_action` 根本進不到 `add_card` 的 kind 處理邏輯。
   -> 修法：把 `("add_card", r"costs\\s+\\d+\\s+(more|less)\\b")` 加進
   `condition.py` 的 `_HANDLER_OWNED`，讓這個條件改由 `add_card` 自己的
   handler 判斷（它不是布林閘門，是「要選哪張卡」的選取邏輯）。
2. 就算條件檢查放行了，`native.py` 原本也沒有任何選取器知道怎麼找「同國家、
   費用 = 剛摧毀單位的費用 + N」的卡——`action.card_name` 是 "unit" 或
   "tank from the same nation" 這種泛用描述，不是真正的卡名，
   `_card_ids_for`／`_last_destroyed` 兩個既有 helper 都找不到東西。
   -> 新增 `NativeRuleEngine._nation_cost_offset_candidate`：讀
   `counters.destroyed_card_ids(state, player_id)` 拿到這張卡剛摧毀的那個
   單位的 card_id（跟 `_last_destroyed` 讀「你上一個被摧毀的單位」同一套
   infrastructure），算出目標費用，再按國家／費用／（可選的）單位類型過濾
   卡池，隨機選一張。

這批跟 E9/E10 的性質不同：那兩輪是「條件已經接對、只是某個欄位沒填」，這批是
「選取器功能原本整個不存在」，所以除了 condition.py 的接線，還真的新寫了一個
選取函式。
"""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import engine_for

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class CostOffsetSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = engine_for(cls.cards)

    def _state(self, order_card_id: str, friendly_unit_card_id: str) -> GameState:
        order_card = self.cards.get(order_card_id)
        friendly_card = self.cards.get(friendly_unit_card_id)
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState(
                    "p1", order_card.nation, deck=[], hand=[order_card_id],
                    resources=ResourceState(12, 12),
                    units=[UnitState("u1", friendly_unit_card_id, friendly_card.attack or 0,
                                     friendly_card.defense or 0, "p1", "frontline")],
                ),
                "p2": PlayerState("p2", "USA", deck=[], resources=ResourceState(12, 12)),
            },
            graveyard={"p1": [], "p2": []},
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
        return state

    def _cheapest(self, nation: str, unit_type: str | None = None):
        candidates = [c for c in self.cards if c.is_unit and c.nation == nation
                      and (unit_type is None or c.type == unit_type)]
        return min(candidates, key=lambda c: c.kredits)

    def test_red_banner_adds_card_of_same_nation_costing_two_more(self) -> None:
        card = self.cards.get("red_banner")
        cheap = self._cheapest(card.nation)
        state = self._state("red_banner", cheap.id)
        self.engine.execute("red_banner", "on_play", state, EffectContext("p1", "red_banner", target_unit_id="u1"))
        added = [cid for cid in state.players["p1"].hand if cid != "red_banner"]
        self.assertEqual(len(added), 1, "應該正好加一張卡到手牌")
        added_card = self.cards.get(added[0])
        self.assertEqual(added_card.nation, cheap.nation)
        self.assertEqual(added_card.kredits, cheap.kredits + 2)
        self.assertNotIn("native_rule_condition_false",
                         [e.get("event") for e in state.event_log])

    def test_upgrade_only_adds_a_tank(self) -> None:
        card = self.cards.get("upgrade")
        cheap_tank = self._cheapest(card.nation, "tank")
        state = self._state("upgrade", cheap_tank.id)
        self.engine.execute("upgrade", "on_play", state, EffectContext("p1", "upgrade", target_unit_id="u1"))
        added = [cid for cid in state.players["p1"].hand if cid != "upgrade"]
        self.assertEqual(len(added), 1)
        added_card = self.cards.get(added[0])
        self.assertEqual(added_card.type, "tank", "UPGRADE 的替代卡必須也是戰車，不能是任意兵種")
        self.assertEqual(added_card.nation, cheap_tank.nation)
        self.assertEqual(added_card.kredits, cheap_tank.kredits + 3)

    def test_ural_factories_adds_card_costing_one_more(self) -> None:
        card = self.cards.get("ural_factories")
        cheap = self._cheapest(card.nation)
        state = self._state("ural_factories", cheap.id)
        self.engine.execute("ural_factories", "on_play", state, EffectContext("p1", "ural_factories", target_unit_id="u1"))
        added = [cid for cid in state.players["p1"].hand if cid != "ural_factories"]
        self.assertEqual(len(added), 1)
        added_card = self.cards.get(added[0])
        self.assertEqual(added_card.nation, cheap.nation)
        self.assertEqual(added_card.kredits, cheap.kredits + 1)

    def test_no_matching_card_logs_unresolved_instead_of_crashing(self) -> None:
        """目標費用超出該國牌池時（例如摧毀全場最貴的單位再 +1），找不到候選
        卡——應該安靜記一個事件，不是加錯卡或整局崩掉。"""
        card = self.cards.get("ural_factories")
        priciest = max((c for c in self.cards if c.is_unit and c.nation == card.nation),
                       key=lambda c: c.kredits)
        state = self._state("ural_factories", priciest.id)
        self.engine.execute("ural_factories", "on_play", state, EffectContext("p1", "ural_factories", target_unit_id="u1"))
        added = [cid for cid in state.players["p1"].hand if cid != "ural_factories"]
        self.assertEqual(added, [], "找不到符合費用的卡時不該亂加")
        self.assertIn("native_rule_unresolved_card_name",
                      [e.get("event") for e in state.event_log])


if __name__ == "__main__":
    unittest.main()
