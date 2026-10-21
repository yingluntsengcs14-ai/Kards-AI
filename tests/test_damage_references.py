# -*- coding: utf-8 -*-
"""「造成等同 X 的傷害」。

20 張卡的傷害是一個動態數量而不是固定數字。解析器原本把它們變成
`damage amount=0` 加一句註記 "equal to referenced value"，所以這 20 張
**全部造成 0 傷害**，而且全部標成 implemented。

現在由 `simulator/rules/references.py` 把具名數量算出來；需要引擎還沒有的
長期計數器（本局失去的槽數、本局被摧毀的某國單位數、本回合總部受到的傷害、
本回合花在行動上的 kredits）的那幾張，誠實標成 partial／unresolved。
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.actions.action import AttackAction
from simulator.core.hq import HQResolver
from simulator.core.turn import TurnManager
from simulator.effects import basic_effects
from simulator.rules import counters
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"

TOUGH_ENEMY = ("e1", "1_infantry_regiment", 0, 30)


class DamageReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, card_id, p1_units=(), p2_units=(TOUGH_ENEMY,), kredits=12, enemy_hand=0) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "USA", deck=["garrison"] * 8, hand=[card_id],
                                  resources=ResourceState(kredits, kredits),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Japan", deck=["garrison"] * 8,
                                  hand=["garrison"] * enemy_hand,
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

    def _damage(self, state) -> int:
        return 30 - state.players["p2"].units[0].defense

    def test_damage_equal_to_units_you_control(self) -> None:
        """MASS ATTACK：等同你控制的單位數。"""
        units = tuple((f"u{i}", "garrison", 0, 3) for i in range(3))
        state = self._state("mass_attack", p1_units=units)
        PlayCardAction("p1", "mass_attack", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(self._damage(state), 3)

    def test_damage_equal_to_enemy_hand(self) -> None:
        """DEFEND IN DEPTH：等同敵方手牌數。"""
        state = self._state("defend_in_depth", enemy_hand=4)
        PlayCardAction("p1", "defend_in_depth", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(self._damage(state), 4)

    def test_damage_equal_to_total_air_attack(self) -> None:
        """POUND：等同你空軍的總攻擊力。"""
        state = self._state("pound", p1_units=(("a1", "duck", 3, 3), ("a2", "duck", 4, 3)))
        PlayCardAction("p1", "pound", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(self._damage(state), 7)

    def test_damage_equal_to_air_unit_count(self) -> None:
        """REGIA AERONAUTICA：等同友方空軍數（對每個敵人）。"""
        state = self._state("regia_aeronautica", p1_units=(("a1", "duck", 3, 3), ("a2", "duck", 4, 3)))
        PlayCardAction("p1", "regia_aeronautica").execute(state, self.cards)
        self.assertEqual(self._damage(state), 2)

    def test_damage_equal_to_unit_types_plus_one(self) -> None:
        """JOINT OPERATION：等同 1 加上你控制的單位類型數。"""
        state = self._state("joint_operation", p1_units=(("u1", "garrison", 0, 3), ("a1", "duck", 3, 3)))
        PlayCardAction("p1", "joint_operation", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(self._damage(state), 3)

    def test_damage_equal_to_target_cost(self) -> None:
        """FORWARD BASE：等同目標的費用。"""
        state = self._state("forward_base", p2_units=(("e1", "ace_of_spades", 0, 30),))
        PlayCardAction("p1", "forward_base", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(self._damage(state), self.cards.get("ace_of_spades").kredits)

    def test_zero_is_no_longer_the_silent_default(self) -> None:
        """這些卡都不該再造成 0 傷害。"""
        for card_id, units in (("mass_attack", (("u1", "garrison", 0, 3),)),
                               ("pound", (("a1", "duck", 3, 3),))):
            state = self._state(card_id, p1_units=units)
            PlayCardAction("p1", card_id, target_unit_id="e1").execute(state, self.cards)
            self.assertGreater(self._damage(state), 0, card_id)

    def test_counter_backed_cards_carry_their_reference(self) -> None:
        """靠長期計數器的卡（simulator/rules/counters.py）現在都接上了引用。"""
        expected = {
            "144th_infantry_regiment": "kredit_slots_lost_this_battle",
            "shinyo_motorboats": "kredit_slots_lost_this_battle_plus_one",
            "uprising": "nation_units_destroyed_this_battle",
            "push": "hq_damage_this_turn",
            "sustained_pressure": "kredits_spent_operating_this_turn",
        }
        for card_id, reference in expected.items():
            rule = self.engine.rule_for(card_id)
            self.assertTrue(any(action.kind == "damage_equal_to" and action.card_name == reference
                                for action in rule.actions), card_id)

    def test_within_card_quantity_stays_honest(self) -> None:
        """THE TIDE TURNS 的「洗入的海軍卡數」是同一張卡前一句產生的暫時量，
        不是長期計數器，所以仍誠實標成未解析。"""
        rule = self.engine.rule_for("the_tide_turns")
        self.assertNotEqual(rule.status, "implemented")
        self.assertTrue(rule.unresolved_fragments)

    def test_implemented_reference_cards_carry_a_reference(self) -> None:
        """已實作的那批必須帶具名引用，而不是 amount=0。"""
        for card_id in ("pound", "defend_in_depth", "mass_attack", "forward_base",
                        "joint_operation", "regia_aeronautica", "a24_banshee",
                        "bypass", "1re_bfl", "act_on_the_intel", "audacity",
                        "kriegsmarine", "shark_mk_iii", "the_empire_strikes"):
            rule = self.engine.rule_for(card_id)
            self.assertTrue(any(action.kind == "damage_equal_to" and action.card_name
                                for action in rule.actions), card_id)


class CounterTests(unittest.TestCase):
    """長期計數器本身（simulator/rules/counters.py）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, hand=(), p1_units=(), p2_units=(TOUGH_ENEMY,), nation="USA") -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", nation, deck=["garrison"] * 8, hand=list(hand),
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "frontline")
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
                state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
        return state

    def test_hq_damage_is_counted_for_the_damaged_player(self) -> None:
        state = self._state()
        HQResolver.damage(state, "p1", 4)
        self.assertEqual(counters.hq_damage_this_turn(state, "p1"), 4)
        self.assertEqual(counters.hq_damage_this_turn(state, "p2"), 0)

    def test_operating_spend_is_counted_on_attack(self) -> None:
        state = self._state(p1_units=(("u1", "garrison", 2, 5),))
        AttackAction("p1", "u1").execute(state, self.cards)
        self.assertGreater(counters.operating_spend_this_turn(state, "p1"), 0)

    def test_turn_counters_reset_for_both_players_at_turn_start(self) -> None:
        """「本回合」的計數在新回合開始時對雙方都要清掉。"""
        state = self._state()
        HQResolver.damage(state, "p1", 3)
        HQResolver.damage(state, "p2", 2)
        TurnManager.start_turn(state, "p2", draw_card=False)
        self.assertEqual(counters.hq_damage_this_turn(state, "p1"), 0)
        self.assertEqual(counters.hq_damage_this_turn(state, "p2"), 0)

    def test_battle_counters_survive_a_turn_change(self) -> None:
        """本局累計的計數不該被回合邊界清掉。"""
        state = self._state()
        counters.note_kredit_slots_lost(state, "p1", 2)
        TurnManager.start_turn(state, "p2", draw_card=False)
        self.assertEqual(counters.kredit_slots_lost(state, "p1"), 2)

    def test_destroyed_units_are_counted_by_nation(self) -> None:
        """被摧毀的單位按國家計數，而且卡面的形容詞要對得上卡池的國名。"""
        polish = [card.id for card in self.cards if card.nation == "Poland" and card.is_unit][:2]
        state = self._state(p1_units=tuple((f"u{i}", card_id, 1, 1) for i, card_id in enumerate(polish)))
        for unit in state.players["p1"].units:
            unit.defense = 0
        basic_effects.remove_dead_units(state)
        self.assertEqual(counters.units_destroyed_of_nation(state, "polish", self.cards), 2)
        self.assertEqual(counters.units_destroyed_of_nation(state, "poland", self.cards), 2)
        self.assertEqual(counters.units_destroyed_of_nation(state, "german", self.cards), 0)

    def test_push_damages_equal_to_hq_damage_this_turn(self) -> None:
        """PUSH：自己總部受 3 點，然後對目標造成 3 點。"""
        state = self._state(hand=("push",))
        PlayCardAction("p1", "push", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(30 - state.players["p2"].units[0].defense, 3)

    def test_sustained_pressure_reads_operating_spend(self) -> None:
        """SUSTAINED PRESSURE：先攻擊花掉行動費，傷害等於那個花費。"""
        state = self._state(hand=("sustained_pressure",), p1_units=(("u1", "garrison", 2, 5),))
        AttackAction("p1", "u1").execute(state, self.cards)
        spent = counters.operating_spend_this_turn(state, "p1")
        PlayCardAction("p1", "sustained_pressure", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(30 - state.players["p2"].units[0].defense, spent)

    def test_shinyo_counts_the_slot_it_just_lost(self) -> None:
        """SHINYO MOTORBOATS：失去一槽後，傷害是「已失去槽數 + 1」。"""
        state = self._state(hand=("shinyo_motorboats",), nation="Japan")
        PlayCardAction("p1", "shinyo_motorboats", target_unit_id="e1").execute(state, self.cards)
        self.assertEqual(counters.kredit_slots_lost(state, "p1"), 1)
        self.assertEqual(30 - state.players["p2"].units[0].defense, 2)


if __name__ == "__main__":
    unittest.main()
