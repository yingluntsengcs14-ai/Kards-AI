# -*- coding: utf-8 -*-
"""Selector 覆蓋率回歸測試。

`tools/build_card_rules.py` 的 honest-status 白名單只檢查 action kind 有沒有
handler，不檢查 target selector 解不解析得出來，所以一張卡可以被標成
implemented 但效果從來沒發生過。`tools/scan_selector_coverage.py` 是實測掃描，
這個檔案把掃到並修掉的破洞釘住。
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.actions.validator import ActionValidationError
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class SelectorCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_units=(), p2_units=(), p1_hand=()) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Japan", deck=["garrison"] * 8, hand=list(p1_hand),
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", pos)
                                         for uid, cid, atk, dfn, pos in p1_units]),
                "p2": PlayerState("p2", "Soviet", deck=["garrison"] * 8,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", pos)
                                         for uid, cid, atk, dfn, pos in p2_units]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=3,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
        return state

    # --- all_units ---

    def test_all_units_resolves_to_both_sides(self) -> None:
        """`all_units` 必須解析得出雙方單位（原本解析成空的，效果靜默失效）。"""
        state = self._state(p1_units=(("u1", "garrison", 0, 3, "support_line"),),
                            p2_units=(("e1", "garrison", 0, 3, "support_line"),))
        self.engine._execute_action(
            RuleAction("modify_attack", "all_units", amount=2),
            state, EffectContext("p1", "garrison", "u1"))
        self.assertEqual(state.players["p1"].units[0].attack, 2)
        self.assertEqual(state.players["p2"].units[0].attack, 2)

    # --- 移動：source / random_enemy ---

    def test_move_to_frontline_accepts_source(self) -> None:
        """「把這個單位移到前線」原本因為 source 沒被處理而靜默失效。"""
        state = self._state(p1_units=(("u1", "garrison", 0, 3, "support_line"),))
        self.engine._execute_action(
            RuleAction("move_to_frontline", "source"),
            state, EffectContext("p1", "garrison", "u1"))
        self.assertEqual(state.players["p1"].units[0].position, "frontline")

    def test_move_to_support_line_accepts_source(self) -> None:
        state = self._state(p1_units=(("u1", "garrison", 0, 3, "frontline"),))
        self.engine._execute_action(
            RuleAction("move_to_support_line", "source"),
            state, EffectContext("p1", "garrison", "u1"))
        self.assertEqual(state.players["p1"].units[0].position, "support_line")

    def test_move_accepts_random_enemy(self) -> None:
        """327 傘兵（回合開始時隨機一個敵方單位撤退）：random_enemy 的改寫分支
        在移動分支之後，所以移動類動作拿不到隨機目標。"""
        state = self._state(p1_units=(("u1", "327th_pathfinders_vet", 3, 3, "support_line"),),
                            p2_units=(("e1", "garrison", 0, 3, "frontline"),))
        self.engine._execute_action(
            RuleAction("move_to_support_line", "random_enemy"),
            state, EffectContext("p1", "327th_pathfinders_vet", "u1"))
        self.assertEqual(state.players["p2"].units[0].position, "support_line")

    # --- set_defense / set_attack 群體目標 ---

    def test_set_defense_supports_group_targets(self) -> None:
        """季風腐蝕（每個敵方單位防禦力設為 1）原本把 1 塞進 set_cost、amount 是 0。"""
        state = self._state(p1_units=(("u1", "garrison", 0, 3, "support_line"),),
                            p2_units=(("e1", "garrison", 0, 3, "support_line"),
                                      ("e2", "duck", 1, 5, "support_line")),
                            p1_hand=("monsoon_rot",))
        PlayCardAction("p1", "monsoon_rot").execute(state, self.cards)
        self.assertEqual([unit.defense for unit in state.players["p2"].units], [1, 1])
        self.assertEqual(state.players["p1"].units[0].defense, 3, "友方不該被影響")

    def test_monsoon_rot_rule_uses_set_defense(self) -> None:
        rule = self.engine.rule_for("monsoon_rot")
        self.assertTrue(any(action.kind == "set_defense" and action.amount == 1
                            for action in rule.actions))

    # --- resolver 的 selected_enemy / selected_friendly ---

    def test_resolver_handles_selected_enemy(self) -> None:
        """`selected_enemy` 原本在 EffectResolver 裡解析不出目標。

        native 的 _resolve_units 認得它，但 EffectResolver._targets 不認得，
        所以走通用傷害路徑的卡（39 個 damage action）傷害全部落空。
        """
        from simulator.effects.resolver import EffectResolver
        state = self._state(p1_units=(("u1", "garrison", 0, 3, "support_line"),),
                            p2_units=(("e1", "garrison", 0, 9, "support_line"),))
        EffectResolver(self.cards).resolve(
            {"type": "damage", "target": "selected_enemy", "value": {"amount": 4}},
            state, EffectContext("p1", "garrison", "u1", "e1"))
        self.assertEqual(state.players["p2"].units[0].defense, 5)

    def test_resolver_handles_selected_friendly(self) -> None:
        from simulator.effects.resolver import EffectResolver
        state = self._state(p1_units=(("u1", "garrison", 0, 3, "support_line"),
                                      ("u2", "garrison", 0, 9, "support_line")))
        EffectResolver(self.cards).resolve(
            {"type": "modify_defense", "target": "selected_friendly", "value": {"amount": 2}},
            state, EffectContext("p1", "garrison", "u1", "u2"))
        self.assertEqual(state.players["p1"].units[1].defense, 11)

    # --- 誠實狀態 ---

    def test_deck_manipulation_is_not_parsed_as_unit_movement(self) -> None:
        """「把牌庫的卡移到頂端」原本被誤判成 move_to_frontline，永遠不會發生。
        現在誠實標成未解析，而不是帶一個永遠無效的動作。"""
        rule = self.engine.rule_for("atlantic_command")
        self.assertEqual(rule.actions, ())
        self.assertNotEqual(rule.status, "implemented")
        self.assertTrue(rule.unresolved_fragments)


if __name__ == "__main__":
    unittest.main()
