"""Rules for cards in the tournament decks (data/decks/meta_decks.json).

Each test quotes the card; most of these conditions used to be assumed true.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from simulator.actions.action import PassAction, PlayCardAction
from simulator.actions.validator import ActionValidationError
from simulator.cards.deck_code import load_meta_decks
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules.native import engine_for

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class MetaDeckRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def _state(self, hand=(), p1_units=(), p2_units=(), kredits=12) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain", deck=["garrison", "radar"] * 6, hand=list(hand),
                                  resources=ResourceState(kredits, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", pos) for uid, cid, atk, dfn, pos in p1_units]),
                "p2": PlayerState("p2", "Germany", deck=["garrison"] * 12, hand=["garrison"] * 4,
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", pos) for uid, cid, atk, dfn, pos in p2_units]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=7,
        )
        for player in state.players.values():
            for unit in player.units:
                state.battlefield[unit.position].append(unit.instance_id)
        return state

    # ── the deck file ────────────────────────────────────────────────────────
    def test_all_tournament_decks_decode_to_legal_39_card_decks(self) -> None:
        from simulator.rules.deck import DeckValidator
        decks = load_meta_decks(self.cards)
        self.assertEqual(len(decks), 17)
        for deck in decks:
            self.assertEqual(len(deck.cards), 39, deck.name)
            result = DeckValidator(self.cards).validate(list(deck.cards), deck.main_nation, deck.ally_nation)
            self.assertTrue(result.valid, (deck.name, result.errors))

    # ── board predicates ────────────────────────────────────────────────────
    def test_yokosuka_draws_only_at_zero_kredits(self) -> None:
        """YOKOSUKA E14Y：「沒有 kredits 時抽一張」。"""
        for kredits, draws in ((3, 0), (4, 0)):
            state = self._state(hand=("yokosuka_e14y",), kredits=kredits)
            before = len(state.players["p1"].deck)
            PlayCardAction("p1", "yokosuka_e14y", selected_option="sunny").execute(state, self.cards)
            spent_to_zero = state.players["p1"].resources.kredits == 0
            self.assertEqual(before - len(state.players["p1"].deck), 1 if spent_to_zero else 0)

    def test_suppression_draws_only_with_an_air_unit(self) -> None:
        for planes, expected in (((), 0), ((("f", "gladiator_mk_i", 1, 2, "support_line"),), 1)):
            state = self._state(hand=("suppression",), p1_units=planes, p2_units=(("e", "garrison", 1, 3, "frontline"),))
            before = len(state.players["p1"].deck)
            PlayCardAction("p1", "suppression", target_unit_id="e").execute(state, self.cards)
            self.assertEqual(before - len(state.players["p1"].deck), expected)

    def test_arado_draws_an_order_when_not_ahead_on_units(self) -> None:
        state = self._state(hand=("arado_ar_196",), p2_units=(("e", "garrison", 1, 3, "frontline"),))
        PlayCardAction("p1", "arado_ar_196").execute(state, self.cards)     # 1 對 1：抽
        self.assertIn("radar", state.players["p1"].hand)
        state = self._state(hand=("arado_ar_196",), p1_units=(("u", "garrison", 1, 3, "support_line"),))
        PlayCardAction("p1", "arado_ar_196").execute(state, self.cards)     # 2 對 0：不抽
        self.assertNotIn("radar", state.players["p1"].hand)

    def test_wave_after_wave_deals_4_only_as_the_second_order(self) -> None:
        """WAVE AFTER WAVE：本回合第二張指令改成 4 點；以前「4 點」打到敵方 HQ。"""
        state = self._state(hand=("wave_after_wave",), p2_units=(("e", "garrison", 1, 9, "frontline"),))
        hq = state.players["p2"].hq.current_health
        PlayCardAction("p1", "wave_after_wave").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].defense, 7)
        self.assertEqual(state.players["p2"].hq.current_health, hq)
        state = self._state(hand=("radar", "wave_after_wave"), p2_units=(("e", "garrison", 1, 9, "frontline"),))
        PlayCardAction("p1", "radar").execute(state, self.cards)
        PlayCardAction("p1", "wave_after_wave").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].defense, 5)

    def test_forward_observers_checks_british_infantry_deployed_last_turn(self) -> None:
        state = self._state(hand=("5th_parachute_brigade", "forward_observers"))
        PlayCardAction("p1", "5th_parachute_brigade").execute(state, self.cards)
        PassAction("p1").execute(state, self.cards)
        PassAction("p2").execute(state, self.cards)
        before = len(state.players["p1"].deck)
        PlayCardAction("p1", "forward_observers").execute(state, self.cards)
        self.assertEqual(before - len(state.players["p1"].deck), 1)
        state = self._state(hand=("forward_observers",))
        before = len(state.players["p1"].deck)
        PlayCardAction("p1", "forward_observers").execute(state, self.cards)
        self.assertEqual(before - len(state.players["p1"].deck), 0)

    def test_13e_dragons_cannot_follow_another_deployment(self) -> None:
        state = self._state(hand=("5th_parachute_brigade", "13e_dragons"))
        PlayCardAction("p1", "13e_dragons").execute(state, self.cards)        # 第一個部署：可以
        state = self._state(hand=("5th_parachute_brigade", "13e_dragons"))
        PlayCardAction("p1", "5th_parachute_brigade").execute(state, self.cards)
        with self.assertRaises(ActionValidationError):
            PlayCardAction("p1", "13e_dragons").execute(state, self.cards)

    def test_10th_para_buffs_itself_after_an_order(self) -> None:
        state = self._state(hand=("radar", "10th_para_battalion"))
        PlayCardAction("p1", "radar").execute(state, self.cards)
        PlayCardAction("p1", "10th_para_battalion").execute(state, self.cards)
        card = self.cards.get("10th_para_battalion")
        unit = state.players["p1"].units[0]
        self.assertEqual((unit.attack, unit.defense), (card.attack + 1, card.defense + 1))

    # ── "it" ───────────────────────────────────────────────────────────────
    def test_15th_wolves_draws_two_only_for_a_friendly_target(self) -> None:
        for owner_units, enemy_units, target, expected in (
                ((("u", "garrison", 1, 3, "support_line"),), (), "u", 2),
                ((), (("e", "garrison", 1, 3, "frontline"),), "e", 0)):
            state = self._state(hand=("15th_wolves_regiment",), p1_units=owner_units, p2_units=enemy_units)
            before = len(state.players["p1"].deck)
            PlayCardAction("p1", "15th_wolves_regiment", target_unit_id=target).execute(state, self.cards)
            self.assertEqual(before - len(state.players["p1"].deck), expected)

    def test_buzz_bomb_can_hit_the_hq_and_then_draws(self) -> None:
        state = self._state(hand=("buzz_bomb",), p2_units=(("e", "garrison", 1, 3, "frontline"),))
        sim = Simulator(self.cards); sim.state = state
        targets = {a.target_unit_id for a in sim.get_available_actions()
                   if isinstance(a, PlayCardAction) and a.card_id == "buzz_bomb"}
        self.assertIn(None, targets)                     # HQ 是合法目標
        hq, before = state.players["p2"].hq.current_health, len(state.players["p1"].deck)
        PlayCardAction("p1", "buzz_bomb").execute(state, self.cards)
        self.assertEqual(state.players["p2"].hq.current_health, hq - 1)
        self.assertEqual(before - len(state.players["p1"].deck), 1)
        state = self._state(hand=("buzz_bomb",), p2_units=(("e", "garrison", 1, 3, "frontline"),))
        before = len(state.players["p1"].deck)
        PlayCardAction("p1", "buzz_bomb", target_unit_id="e").execute(state, self.cards)
        self.assertEqual(before - len(state.players["p1"].deck), 0)

    def test_sally_effect_depends_on_the_target_type(self) -> None:
        state = self._state(hand=("sally",), p2_units=(("t", "panzer_iia", 2, 3, "frontline"),))
        PlayCardAction("p1", "sally", target_unit_id="t").execute(state, self.cards)
        tank = state.players["p2"].units[0]
        self.assertTrue(tank.status.get("pinned"))
        self.assertFalse(tank.status.get("suppressed"))
        state = self._state(hand=("sally",), p2_units=(("a", "25_pounder", 1, 3, "support_line"),))
        PlayCardAction("p1", "sally", target_unit_id="a").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units, [])

    def test_cruiser_scouts_reveals_three_only_at_zero_kredits(self) -> None:
        state = self._state(hand=("cruiser_scouts",), p2_units=(("e", "garrison", 1, 3, "frontline"),), kredits=2)
        PlayCardAction("p1", "cruiser_scouts", target_unit_id="e", selected_option="rain").execute(state, self.cards)
        self.assertEqual(len(state.players["p1"].status.get("known_enemy_hand", [])), 3)


if __name__ == "__main__":
    unittest.main()
