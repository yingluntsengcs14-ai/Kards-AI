# -*- coding: utf-8 -*-
"""Selector 契約：宣告出來的 selector 必須真的解析得出東西。

`simulator/rules/selectors.py` 宣告每條解析路徑支援哪些 target，
`tools/build_card_rules.py` 用它判斷一張卡能不能算 implemented。

宣告本身不會自己成立 —— 這個檔案做反向驗證：對每條路徑、每個宣告的
selector，在一個正常盤面上實際跑一次，確認它真的解析得到目標。
`all_units`、`move_to_frontline + source`、`move_to_support_line + random_enemy`、
`damage + selected_enemy` 這四個破洞都是「宣告以外沒人檢查」造成的。
"""

from pathlib import Path
import json
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.rules import selectors
from simulator.rules.native import NativeRuleEngine, _resolve_units
from simulator.rules.parser import RuleAction

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"
RULES = ROOT / "data/rules/card_rules.json"


class SelectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self) -> GameState:
        """一個雙方都有單位、來源在支援線、敵方不佔前線的正常盤面。"""
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"] * 8,
                                  hand=["garrison", "radar"],
                                  resources=ResourceState(12, 12),
                                  units=[UnitState("src", "garrison", 2, 5, "p1", "support_line"),
                                         UnitState("f1", "garrison", 2, 5, "p1", "support_line")]),
                "p2": PlayerState("p2", "Soviet", deck=["garrison"] * 8,
                                  hand=["garrison"],
                                  resources=ResourceState(12, 12),
                                  units=[UnitState("e1", "garrison", 2, 5, "p2", "support_line"),
                                         UnitState("e2", "duck", 2, 5, "p2", "support_line")]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=5,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield["support_line"].append(unit.instance_id)
        return state

    def _context(self) -> EffectContext:
        return EffectContext("p1", "garrison", "src", "e1")

    # --- 每條路徑的宣告都要成立 ---

    def test_unit_path_resolves_every_declared_selector(self) -> None:
        """native._resolve_units 宣告支援的 selector 都要解析得到單位。"""
        for target in sorted(selectors.UNIT_SELECTORS):
            resolved = _resolve_units(target, self._state(), self._context())
            self.assertTrue(resolved, "UNIT_SELECTORS 宣告了 {0} 但解析不到單位".format(target))

    def test_effect_path_resolves_every_declared_selector(self) -> None:
        """EffectResolver._targets 宣告支援的 selector 都要解析得到目標。

        玩家類的 selector 解析成 player id，單位類的解析成 UnitState，
        兩者都算有解析到。
        """
        resolver = EffectResolver(self.cards)
        for target in sorted(selectors.EFFECT_SELECTORS):
            state = self._state()
            resolved = resolver._targets(target, state, self._context())
            self.assertTrue(resolved, "EFFECT_SELECTORS 宣告了 {0} 但解析不到目標".format(target))

    def test_movement_path_moves_for_every_declared_selector(self) -> None:
        """_move_units 宣告支援的 selector 都要真的讓單位換線。"""
        for target in sorted(selectors.MOVEMENT_SELECTORS):
            state = self._state()
            before = {unit.instance_id: unit.position
                      for player in state.players.values() for unit in player.units}
            self.engine._execute_action(
                RuleAction("move_to_frontline", target), state, self._context())
            after = {unit.instance_id: unit.position
                     for player in state.players.values() for unit in player.units}
            self.assertNotEqual(before, after,
                                "MOVEMENT_SELECTORS 宣告了 {0} 但沒有任何單位移動".format(target))

    def test_random_targets_are_rewritten_before_the_effect_layer(self) -> None:
        """`random_enemy` 不是 resolver 解析的 —— native 會先把它改寫成被選中的目標。

        所以它不在 EFFECT_SELECTORS 裡，但走完整路徑時必須有效。
        """
        for target in sorted(selectors.RANDOM):
            state = self._state()
            before = [unit.defense for player in state.players.values() for unit in player.units]
            self.engine._execute_action(
                RuleAction("damage", target, amount=2), state, self._context())
            after = [unit.defense for player in state.players.values() for unit in player.units]
            self.assertNotEqual(before, after,
                                "{0} 走完整路徑後沒有造成任何傷害".format(target))

    def test_enemy_hand_is_read_by_the_player_scoped_kinds(self) -> None:
        """`enemy_hand` 代表「對手」，由 kredit 槽那類 handler 自己判斷，
        不是 EffectResolver._targets 解析的。"""
        self.assertIn("enemy_hand", selectors.PLAYER_SELECTORS)
        self.assertNotIn("enemy_hand", selectors.EFFECT_SELECTORS)
        state = self._state()
        state.players["p2"].resources.max_kredits = 5
        self.engine._execute_action(
            RuleAction("gain_kredit_slot", "enemy_hand", amount=1), state, self._context())
        self.assertEqual(state.players["p2"].resources.max_kredits, 6)

    # --- 規則存檔不能用到沒人能解析的 target ---

    def test_every_target_in_the_rule_store_is_resolvable(self) -> None:
        """規則存檔裡出現的每個 target 都必須在 ALL_SELECTORS 裡。"""
        store = json.loads(RULES.read_text(encoding="utf-8"))
        unknown = sorted({
            action["target"] for rule in store["rules"] for action in rule["actions"]
            if action.get("target") and action["target"] not in selectors.ALL_SELECTORS
        })
        self.assertEqual(unknown, [], "沒有任何路徑能解析這些 target：{0}".format(unknown))

    def test_declared_families_flag_nothing_in_the_current_store(self) -> None:
        """宣告的 selector 家族目前不該攔下任何卡。

        這是守門測試：若有人改了解析器、讓某個 kind 指向它處理不了的 target，
        這裡會直接指名是哪張卡，而不是等到 self-play 跑了一百萬局才發現那張卡
        其實沒有效果。
        """
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from build_card_rules import unsupported_selector
        store = json.loads(RULES.read_text(encoding="utf-8"))
        offenders = []
        for rule in store["rules"]:
            for action in rule["actions"]:
                problem = unsupported_selector(action["kind"], str(action.get("target", "")))
                if problem and rule["status"] == "implemented":
                    offenders.append((rule["card_id"], problem))
        self.assertEqual(offenders, [], "被標成 implemented 卻解析不到目標：{0}".format(offenders))

    def test_unit_effects_do_not_claim_player_selectors(self) -> None:
        """單位效果對玩家類 selector 無效，這是正確行為，要被宣告出來。"""
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from build_card_rules import unsupported_selector
        self.assertIsNotNone(unsupported_selector("modify_attack", "owner"))
        self.assertIsNotNone(unsupported_selector("buff", "enemy_hq"))
        self.assertIsNone(unsupported_selector("buff", "friendly_units"))
        self.assertIsNone(unsupported_selector("buff", "random_enemy"))

    # --- 建置時的檢查本身 ---

    def test_build_check_rejects_an_unresolvable_target(self) -> None:
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from build_card_rules import unsupported_selector
        self.assertIsNotNone(unsupported_selector("damage", "nowhere"))
        self.assertIsNone(unsupported_selector("damage", "selected_target"))

    def test_build_check_rejects_a_movement_target_the_path_cannot_resolve(self) -> None:
        """移動類是 target 真的決定結果的那一族，三次破洞都在這裡。"""
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from build_card_rules import unsupported_selector
        self.assertIsNotNone(unsupported_selector("move_to_frontline", "enemy_hq"))
        self.assertIsNone(unsupported_selector("move_to_frontline", "source"))
        self.assertIsNone(unsupported_selector("move_to_support_line", "random_enemy"))

    def test_chosen_target_is_only_allowed_for_kinds_that_read_it(self) -> None:
        """`chosen` 是 choose_one 設定的連結，只有少數 kind 會讀它。"""
        self.assertTrue(selectors.chosen_link_ok("add_card", "chosen"))
        self.assertFalse(selectors.chosen_link_ok("damage", "chosen"))
        self.assertTrue(selectors.chosen_link_ok("damage", "selected_target"))

    # --- 宣告集合之間的關係 ---

    def test_paths_agree_on_the_shared_core(self) -> None:
        """三條路徑都必須支援來源與被選中的單位 —— 跨路徑不一致就是 bug 的來源。

        `damage + selected_enemy` 當初就是 native 認得、resolver 不認得。
        """
        core = selectors.SOURCE | selectors.CHOSEN
        for name in ("unit", "movement", "effect"):
            missing = sorted(core - selectors.FAMILIES[name])
            self.assertEqual(missing, [], "{0} 路徑少了共用的核心 selector：{1}".format(name, missing))


if __name__ == "__main__":
    unittest.main()
