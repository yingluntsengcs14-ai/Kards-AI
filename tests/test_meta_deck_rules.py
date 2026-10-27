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

    # ── events (batch 2) ────────────────────────────────────────────────────
    def _attack(self, state, attacker, target=None) -> None:
        from simulator.actions.action import AttackAction
        for unit in state.players["p1"].units:
            unit.status.pop("deployed_this_turn", None)
        AttackAction("p1", attacker, target).execute(state, self.cards)

    def test_katyusha_adds_one_blast_per_damage_dealt(self) -> None:
        """STZ-5 KATYUSHA：造成多少戰鬥傷害就加多少張 BLAST（以前部署就加）。"""
        state = self._state(hand=("stz5_katyusha",), p2_units=(("e", "garrison", 1, 9, "frontline"),))
        PlayCardAction("p1", "stz5_katyusha").execute(state, self.cards)
        self.assertNotIn("blast", state.players["p1"].hand)
        gun = state.players["p1"].units[0]
        gun.attack = 3
        self._attack(state, gun.instance_id, "e")
        self.assertEqual(state.players["p1"].hand.count("blast"), 3)

    def test_738_jager_draws_when_its_marked_unit_dies(self) -> None:
        from simulator.effects import basic_effects
        state = self._state(hand=("738_jger_regiment",), p2_units=(("e", "garrison", 1, 3, "frontline"),))
        PlayCardAction("p1", "738_jger_regiment", target_unit_id="e").execute(state, self.cards)
        before = len(state.players["p1"].deck)
        start = len(state.event_log)
        basic_effects.destroy(state, state.players["p2"].units[0])
        engine_for(self.cards).emit_deaths_since(state, start)
        self.assertEqual(before - len(state.players["p1"].deck), 1)

    def test_dornier_pays_once_per_turn_when_a_unit_leaves_undestroyed(self) -> None:
        from simulator.rules.parser import RuleAction
        from simulator.effects.resolver import EffectContext
        state = self._state(p1_units=(("d", "dornier_do_17z", 2, 4, "support_line"),),
                            p2_units=(("e1", "garrison", 1, 3, "frontline"), ("e2", "garrison", 1, 3, "frontline")), kredits=0)
        engine = engine_for(self.cards)
        before = len(state.players["p1"].deck)
        for target in ("e1", "e2"):
            engine._execute_action(RuleAction("return_to_hand", "selected_target"), state,
                                   EffectContext("p1", "test", target_unit_id=target))
        self.assertEqual(state.players["p1"].resources.kredits, 2)     # 一回合一次
        self.assertEqual(before - len(state.players["p1"].deck), 1)

    def test_77th_regiment_adds_20th_after_destroying_a_unit(self) -> None:
        state = self._state(p1_units=(("r", "77th_regiment", 3, 4, "frontline"),),
                            p2_units=(("e", "garrison", 1, 2, "frontline"),))
        state.battlefield["frontline"] = ["r"]; state.players["p2"].units[0].position = "support_line"
        state.battlefield["support_line"] = ["e"]
        self._attack(state, "r", "e")
        self.assertIn("20th INFANTRY REGIMENT", [self.cards.get(u.card_id).name for u in state.players["p1"].units])

    def test_3_panzergrenadier_grows_when_you_operate_a_german_unit(self) -> None:
        from simulator.actions.action import MoveUnitAction
        state = self._state(p1_units=(("p", "3_panzergrenadier", 3, 3, "support_line"),
                                      ("t", "panzer_iia", 1, 3, "support_line")))
        state.players["p1"].nation = "Germany"
        MoveUnitAction("p1", "t").execute(state, self.cards)
        pzg = state.players["p1"].units[0]
        self.assertEqual((pzg.attack, pzg.defense), (4, 4))

    def test_216th_guards_hurts_your_hq_and_becomes_veteran_on_the_second_turn(self) -> None:
        state = self._state(p1_units=(("g", "216th_guards", 3, 5, "support_line"),))
        hq = state.players["p1"].hq.current_health
        for _ in range(2):
            PassAction("p1").execute(state, self.cards)
            PassAction("p2").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.current_health, hq - 4)
        self.assertEqual(state.players["p1"].units[0].card_id, "216th_guards_vet")

    def test_4th_guards_rifles_gets_cheaper_as_the_enemy_draws(self) -> None:
        from simulator.actions.action import card_play_cost
        state = self._state(hand=("4th_guards_rifles",))
        card = self.cards.get("4th_guards_rifles")
        PassAction("p1").execute(state, self.cards)          # p2 抽一張
        self.assertEqual(card_play_cost(state, "p1", card, self.cards), card.kredits - 1)

    def test_114th_makes_friendly_destruction_effects_trigger_twice(self) -> None:
        from simulator.effects import basic_effects
        state = self._state(p1_units=(("j", "114th_infantry_regiment", 3, 4, "support_line"),
                                      ("t", "t60", 2, 4, "support_line")))
        start = len(state.event_log)
        basic_effects.destroy(state, state.players["p1"].units[1])
        engine_for(self.cards).emit_deaths_since(state, start)
        self.assertEqual(len(state.players["p1"].hand), 2)       # T-60：兩張 T-34 1942

    # ── misread cards (batch 3) ─────────────────────────────────────────────
    def test_24th_ulan_hurts_itself_unless_a_unit_died_this_turn(self) -> None:
        from simulator.effects import basic_effects
        state = self._state(p1_units=(("u", "24th_uan", 3, 4, "support_line"),),
                            p2_units=(("e", "garrison", 1, 3, "frontline"),))
        PassAction("p1").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].defense, 3)
        PassAction("p2").execute(state, self.cards)
        basic_effects.destroy(state, state.players["p2"].units[0])
        PassAction("p1").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].defense, 3)          # 有單位被消滅：不扣

    def test_3rd_morskoi_deploys_light_infantry_and_adds_a_blast(self) -> None:
        state = self._state(hand=("3rd_morskoi_pekhoty",))
        PlayCardAction("p1", "3rd_morskoi_pekhoty").execute(state, self.cards)
        self.assertIn("LIGHT INFANTRY", [self.cards.get(u.card_id).name for u in state.players["p1"].units])
        self.assertIn("blast", state.players["p1"].hand)

    def test_10th_recon_shuffles_the_card_the_player_picks(self) -> None:
        state = self._state(hand=("10th_recon", "radar", "fortification"))
        sim = Simulator(self.cards); sim.state = state
        picks = {a.selected_card_id for a in sim.get_available_actions()
                 if isinstance(a, PlayCardAction) and a.card_id == "10th_recon"}
        self.assertEqual(picks, {"radar", "fortification"})
        PlayCardAction("p1", "10th_recon", selected_card_id="fortification").execute(state, self.cards)
        self.assertNotIn("fortification", state.players["p1"].hand)
        self.assertIn("fortification", state.players["p1"].deck)

    def test_surprise_attack_pins_then_destroys_a_pinned_unit(self) -> None:
        state = self._state(hand=("surprise_attack", "surprise_attack"), p2_units=(("e", "garrison", 1, 3, "frontline"),))
        PlayCardAction("p1", "surprise_attack", target_unit_id="e").execute(state, self.cards)
        self.assertTrue(state.players["p2"].units[0].status.get("pinned"))
        PlayCardAction("p1", "surprise_attack", target_unit_id="e").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units, [])

    def test_mosquito_b_converts_itself_at_end_of_turn(self) -> None:
        state = self._state(p1_units=(("m", "mosquito_b_mk_iv", 4, 3, "support_line"),))
        PassAction("p1").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].card_id, "mosquito_f_mk_ii")

    def test_last_ditch_force_destroys_all_and_costs_both_players_slots(self) -> None:
        state = self._state(hand=("last_ditch_force",), p1_units=(("u", "garrison", 1, 3, "support_line"),),
                            p2_units=(("e", "garrison", 1, 3, "frontline"),))
        state.players["p1"].deck = ["5th_parachute_brigade"] + state.players["p1"].deck
        PlayCardAction("p1", "last_ditch_force").execute(state, self.cards)
        self.assertEqual((state.players["p1"].units, state.players["p2"].units), ([], []))
        self.assertEqual(state.players["p1"].resources.max_kredits, 6)
        self.assertEqual(state.players["p2"].resources.max_kredits, 6)
        # 牌庫裡的單位之後打出來 +3 攻
        state.players["p1"].hand.append(state.players["p1"].deck.pop(0))
        state.players["p1"].resources.kredits = 9
        PlayCardAction("p1", "5th_parachute_brigade").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].attack, self.cards.get("5th_parachute_brigade").attack + 3)

    def test_fog_of_war_puts_two_copies_on_top_of_the_owners_deck(self) -> None:
        state = self._state(hand=("fog_of_war",), p2_units=(("e", "panzer_iia", 1, 3, "frontline"),))
        PlayCardAction("p1", "fog_of_war", target_unit_id="e").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units, [])
        self.assertEqual(state.players["p2"].deck[:2], ["panzer_iia", "panzer_iia"])

    def test_salamander_leaves_after_attacking_and_returns_next_turn(self) -> None:
        state = self._state(p1_units=(("s", "salamander", 2, 3, "frontline"),))
        state.battlefield["frontline"] = ["s"]
        self._attack(state, "s", None)
        self.assertEqual(state.players["p1"].units, [])
        PassAction("p1").execute(state, self.cards)
        PassAction("p2").execute(state, self.cards)
        self.assertEqual([u.card_id for u in state.players["p1"].units], ["salamander"])

    def test_mi5_puts_an_enemy_hand_card_on_top_of_their_deck(self) -> None:
        state = self._state(hand=("mi5",))
        state.players["p2"].hand = ["radar"]
        PlayCardAction("p1", "mi5").execute(state, self.cards)
        self.assertEqual(state.players["p2"].hand, [])
        self.assertEqual(state.players["p2"].deck[0], "radar")

    def test_rm_roma_can_hit_the_hq_and_makes_the_top_card_navy(self) -> None:
        from simulator.rules.condition import _is_navy
        state = self._state(hand=("rm_roma",))
        hq = state.players["p2"].hq.current_health
        top = state.players["p1"].deck[0]
        PlayCardAction("p1", "rm_roma").execute(state, self.cards)
        self.assertEqual(state.players["p2"].hq.current_health, hq - 4)
        self.assertTrue(_is_navy(self.cards.get(top), state, "p1"))

    def test_336th_copies_a_cheap_hand_unit_to_the_support_line(self) -> None:
        state = self._state(hand=("336th_guards_naval", "5th_parachute_brigade"))
        PlayCardAction("p1", "336th_guards_naval", selected_card_id="5th_parachute_brigade").execute(state, self.cards)
        self.assertIn("5th_parachute_brigade", [u.card_id for u in state.players["p1"].units])
        self.assertIn("5th_parachute_brigade", state.players["p1"].hand)      # 手上那張還在

    # ── Develop and batch 3b ─────────────────────────────────────────────────
    def test_develop_adds_the_chosen_card_to_hand(self) -> None:
        """遊戲內確認：Develop 選到的卡加進手牌（以前單位會直接部署）。"""
        state = self._state(hand=("41st_infantry_regiment",))
        sim = Simulator(self.cards); sim.state = state
        options = {a.selected_option for a in sim.get_available_actions()
                   if isinstance(a, PlayCardAction) and a.card_id == "41st_infantry_regiment"}
        self.assertEqual(options, {"DEATH FROM ABOVE", "UNITY IS STRENGTH", "WE CAN DO IT"})
        PlayCardAction("p1", "41st_infantry_regiment", selected_option="UNITY IS STRENGTH").execute(state, self.cards)
        self.assertEqual([self.cards.get(card_id).name for card_id in state.players["p1"].hand], ["UNITY IS STRENGTH"])

    def test_pilot_escape_offers_three_main_nation_air_units_at_a_discount(self) -> None:
        state = self._state(hand=("pilot_escape",))
        state.players["p1"].nation = "Japan"
        sim = Simulator(self.cards); sim.state = state
        offered = {a.selected_card_id for a in sim.get_available_actions()
                   if isinstance(a, PlayCardAction) and a.card_id == "pilot_escape"}
        self.assertEqual(len(offered), 3)
        self.assertTrue(all(self.cards.get(c).nation == "Japan" and self.cards.get(c).type in {"fighter", "bomber"} for c in offered))
        pick = sorted(offered)[0]
        PlayCardAction("p1", "pilot_escape", selected_card_id=pick).execute(state, self.cards)
        from simulator.actions.action import card_play_cost
        self.assertEqual(state.players["p1"].hand, [pick])
        self.assertEqual(card_play_cost(state, "p1", self.cards.get(pick), self.cards), max(0, self.cards.get(pick).kredits - 1))

    def test_tropical_storm_reshuffles_both_hands(self) -> None:
        state = self._state(hand=("tropical_storm", "radar", "radar"))
        state.players["p2"].hand = ["garrison"] * 5
        PlayCardAction("p1", "tropical_storm").execute(state, self.cards)
        self.assertEqual(len(state.players["p1"].hand), 5)      # 最多洗回 5 張
        self.assertEqual(len(state.players["p2"].hand), 5)

    def test_peoples_army_fills_hand_and_support_line(self) -> None:
        state = self._state(hand=("the_peoples_army",))
        PlayCardAction("p1", "the_peoples_army").execute(state, self.cards)
        names = {self.cards.get(c).name for c in state.players["p1"].hand}
        self.assertEqual((len(state.players["p1"].hand), names), (9, {"LIGHT INFANTRY"}))
        self.assertEqual(len(state.players["p1"].units), 4)
        self.assertTrue(all("blitz" in u.status.get("added_abilities", []) for u in state.players["p1"].units))

    def test_shifting_attack_makes_the_returned_unit_free_this_turn(self) -> None:
        from simulator.actions.action import card_play_cost
        state = self._state(hand=("shifting_attack",), p1_units=(("t", "panzer_iia", 1, 3, "support_line"),))
        PlayCardAction("p1", "shifting_attack", target_unit_id="t").execute(state, self.cards)
        self.assertIn("panzer_iia", state.players["p1"].hand)
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("panzer_iia"), self.cards), 0)

    def test_uss_missouri_destroys_and_retreats_the_neighbours(self) -> None:
        state = self._state(hand=("uss_missouri",), p2_units=(("a", "garrison", 1, 3, "frontline"),
                                                               ("b", "garrison", 1, 3, "frontline"),
                                                               ("c", "garrison", 1, 3, "frontline")))
        PlayCardAction("p1", "uss_missouri", target_unit_id="b").execute(state, self.cards)
        positions = {u.instance_id: u.position for u in state.players["p2"].units}
        self.assertEqual(positions, {"a": "support_line", "c": "support_line"})

    def test_surveil_and_kv85_convert_the_chosen_hand_card(self) -> None:
        from simulator.effects import basic_effects
        state = self._state(hand=("surveil", "radar"), p2_units=(("e", "panzer_iia", 1, 3, "frontline"),))
        PlayCardAction("p1", "surveil", target_unit_id="e", selected_card_id="radar").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, ["panzer_iia"])
        state = self._state(hand=("kv85", "radar"), p1_units=(("g", "garrison", 1, 3, "support_line"),))
        basic_effects.destroy(state, state.players["p1"].units[0])
        PlayCardAction("p1", "kv85", selected_card_id="radar").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, ["garrison"])

    def test_punish_draws_only_for_a_known_card(self) -> None:
        state = self._state(hand=("punish",))
        state.players["p2"].hand = ["radar"]
        state.players["p1"].status["known_enemy_hand"] = ["radar"]
        before = len(state.players["p1"].deck)
        PlayCardAction("p1", "punish").execute(state, self.cards)
        self.assertEqual((state.players["p2"].hand, before - len(state.players["p1"].deck)), ([], 1))

    # ── standing mechanics (batch 3c) ───────────────────────────────────────
    def test_meteor_leaves_after_attacking_and_reforges_double_into_the_deck(self) -> None:
        state = self._state(p1_units=(("m", "meteor", 1, 1, "frontline"),))
        state.battlefield["frontline"] = ["m"]
        self._attack(state, "m", None)
        self.assertEqual(state.players["p1"].units, [])
        self.assertIn("meteor", state.players["p1"].deck)
        state.players["p1"].deck.remove("meteor"); state.players["p1"].hand.append("meteor")
        PlayCardAction("p1", "meteor").execute(state, self.cards)
        unit = state.players["p1"].units[0]
        self.assertEqual((unit.attack, unit.defense), (2, 2))

    def test_511th_never_drops_below_1_and_becomes_veteran_when_hit(self) -> None:
        from simulator.effects import basic_effects
        state = self._state(hand=("511th_regiment",))
        PlayCardAction("p1", "511th_regiment").execute(state, self.cards)
        unit = state.players["p1"].units[0]
        start = len(state.event_log)
        basic_effects.damage(state, unit, 99)
        engine_for(self.cards).emit_damage_since(state, start)
        self.assertEqual(len(state.players["p1"].units), 1)               # 99 點傷害也不死
        self.assertEqual(state.players["p1"].units[0].card_id, "511th_regiment_vet")

    def test_162nd_caps_combat_damage_to_friendly_infantry(self) -> None:
        state = self._state(p1_units=(("a", "panzer_iia", 6, 3, "frontline"),),
                            p2_units=(("r", "162nd_regiment", 4, 5, "support_line"), ("i", "garrison", 1, 9, "frontline")))
        state.battlefield["frontline"] = ["a", "i"]
        self._attack(state, "a", "i")
        self.assertEqual(state.players["p2"].units[1].defense, 7)

    def test_sd_kfz_222_gives_adjacent_ground_units_one_attack_in_combat(self) -> None:
        state = self._state(p1_units=(("s", "sd_kfz_222", 1, 1, "frontline"), ("t", "panzer_iia", 1, 3, "frontline")),
                            p2_units=(("e", "garrison", 0, 9, "support_line"),))
        self._attack(state, "t", "e")
        self.assertEqual(state.players["p2"].units[0].defense, 7)        # 1 + 1

    def test_sd_kfz_10_gets_two_more_when_another_card_buffs_it(self) -> None:
        """FINAL PUSH 給它 +3 攻 → 它再自己 +2（到回合結束）。"""
        state = self._state(hand=("final_push",), p1_units=(("p", "sd_kfz_10_pak_38", 1, 2, "support_line"),))
        PlayCardAction("p1", "final_push", target_unit_id="p").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[0].attack, 1 + 3 + 2)

    def test_27_fusiliers_discounts_german_units_entering_your_hand(self) -> None:
        from simulator.actions.action import card_play_cost
        from simulator.core.turn import TurnManager
        state = self._state(p1_units=(("f", "27_fsiliers", 2, 4, "support_line"),))
        state.players["p1"].deck = ["panzer_iia", "radar"]
        TurnManager.draw_card(state, "p1")
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("panzer_iia"), self.cards), 1)   # 最低 1
        state.players["p1"].deck = ["3_panzergrenadier"]
        TurnManager.draw_card(state, "p1")
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("3_panzergrenadier"), self.cards), 2)

    def test_2nd_west_africa_draws_the_cheapest_order(self) -> None:
        state = self._state(hand=("2nd_west_africa",))
        state.players["p1"].deck = ["fortification", "radar", "garrison"]
        state.players["p1"].status["starting_deck"] = ["fortification", "garrison", "radar"]
        PlayCardAction("p1", "2nd_west_africa").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hand, ["radar"])            # 0 費、原本就在牌庫：不重複

    def test_corps_command_arms_the_edge_cards_in_hand(self) -> None:
        state = self._state(hand=("radar", "garrison", "corps_command", "fortification"),
                            p2_units=(("e", "garrison", 1, 3, "frontline"),))
        PlayCardAction("p1", "corps_command").execute(state, self.cards)
        PlayCardAction("p1", "radar").execute(state, self.cards)        # 最左側那張
        self.assertEqual(state.players["p2"].units, [])

    def test_front_and_homeland_is_drawn_on_turn_4_and_discounts_odd_german_cards(self) -> None:
        from simulator.actions.action import card_play_cost
        from simulator.core.turn import TurnManager
        state = self._state()
        state.players["p1"].deck = ["front_and_homeland"] + ["garrison"] * 6
        state.turn_number = 7
        TurnManager.start_turn(state, "p1")
        self.assertIn("front_and_homeland", state.players["p1"].hand)
        state.players["p1"].deck = ["3_panzergrenadier", "panzer_iia"]       # 3 費（奇數）、1 費（奇數）
        PlayCardAction("p1", "front_and_homeland").execute(state, self.cards)
        TurnManager.draw_card(state, "p1")
        self.assertEqual(card_play_cost(state, "p1", self.cards.get("3_panzergrenadier"), self.cards), 2)

    def test_frontal_assault_adds_two_light_infantry_when_you_have_none(self) -> None:
        state = self._state(hand=("frontal_assault",))
        PlayCardAction("p1", "frontal_assault").execute(state, self.cards)
        self.assertEqual([self.cards.get(c).name for c in state.players["p1"].hand], ["LIGHT INFANTRY"] * 2)

    def test_lightning_strike_punishes_an_extra_draw(self) -> None:
        from simulator.core.turn import TurnManager
        state = self._state()
        state.players["p2"].active_countermeasures.append({"card_id": "lightning_strike", "activated_turn": 0})
        hq = state.players["p1"].hq.current_health
        TurnManager.draw_card(state, "p1")
        self.assertEqual(state.players["p1"].hq.current_health, hq - 4)

    def test_national_fire_service_caps_hq_damage_for_the_turn(self) -> None:
        from simulator.core.hq import HQResolver
        state = self._state()
        state.players["p2"].active_countermeasures.append({"card_id": "national_fire_service", "activated_turn": 0})
        hq = state.players["p2"].hq.current_health
        HQResolver.damage(state, "p2", 5, "p1")
        HQResolver.damage(state, "p2", 5, "p1")
        self.assertEqual(state.players["p2"].hq.current_health, hq - 2)

    def test_developed_cards_from_tournament_units(self) -> None:
        """740th／41st 開發出來的卡：以前 ARCTIC CONVOY 只抽 1、BURNING SKY 打任何單位、WE CAN DO IT 沒加防。"""
        state = self._state(hand=("arctic_convoy",))
        state.players["p1"].deck = ["garrison", "radar", "panzer_iia"]
        PlayCardAction("p1", "arctic_convoy").execute(state, self.cards)
        self.assertEqual(sorted(state.players["p1"].hand), ["garrison", "panzer_iia"])
        state = self._state(hand=("burning_sky",), p2_units=(("e", "garrison", 1, 3, "frontline"),))
        with self.assertRaises(ActionValidationError):
            PlayCardAction("p1", "burning_sky", target_unit_id="e").execute(state, self.cards)
        state = self._state(hand=("we_can_do_it",), p1_units=(("u", "garrison", 1, 3, "support_line"),))
        hq = state.players["p1"].hq.current_health
        PlayCardAction("p1", "we_can_do_it").execute(state, self.cards)
        self.assertEqual((state.players["p1"].units[0].defense, state.players["p1"].hq.current_health), (6, hq + 3))
        state = self._state(hand=("unity_is_strength",), p1_units=(("a", "garrison", 1, 3, "support_line"),
                                                                     ("b", "5th_parachute_brigade", 1, 1, "support_line")))
        PlayCardAction("p1", "unity_is_strength", target_unit_id="b").execute(state, self.cards)
        self.assertEqual(state.players["p1"].units[1].attack, 1 + 2)


if __name__ == "__main__":
    unittest.main()
