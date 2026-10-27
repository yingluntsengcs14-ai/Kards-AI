"""Regression tests for the kards.info-only card database migration."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, PassAction, PlayCardAction
from simulator.cards.card import CARD_TYPES, UNIT_TYPES
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.rules.deck import DeckValidator
from simulator.testing import build_random_deck


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class KardsInfoMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def test_catalog_loads_all_cards_and_ids_are_unique(self) -> None:
        # The catalog grows when a new card set is merged in (see
        # tools/merge_weather_cards.py), so the invariant is internal
        # consistency with metadata, not a frozen total.  1488 is the size of
        # the original 2026-05 dump and acts as a floor against truncation.
        total = len(self.cards)
        self.assertGreaterEqual(total, 1488)
        self.assertEqual(self.cards.report.total_cards, total)
        self.assertEqual(self.cards.report.metadata_total_cards, total)
        self.assertEqual(sum(self.cards.report.type_counts.values()), total)
        self.assertEqual(set(self.cards.report.type_counts), CARD_TYPES)

    def test_catalog_types_and_stats_follow_kards_info_contract(self) -> None:
        for card in self.cards:
            self.assertIn(card.type, CARD_TYPES)
            if card.type in UNIT_TYPES:
                self.assertIsInstance(card.attack, int)
                self.assertIsInstance(card.defense, int)
                self.assertIsInstance(card.operationCost, int)
            else:
                self.assertIsNone(card.attack)
                self.assertIsNone(card.defense)
                self.assertIsNone(card.operationCost)

    def test_only_spawnable_cards_cannot_enter_decks(self) -> None:
        token = next(card for card in self.cards if card.set == "OnlySpawnable")
        self.assertTrue(token.is_token)
        result = DeckValidator(self.cards).validate([token.id] * 39, token.nation)
        self.assertFalse(result.valid)
        self.assertTrue(any("OnlySpawnable" in error for error in result.errors))

    def test_retired_cards_remain_catalogued_but_are_not_deck_legal(self) -> None:
        self.assertIn("night_bombing", self.cards)
        result = DeckValidator(self.cards).validate(["night_bombing"] * 39, "Britain")
        self.assertFalse(result.valid)
        self.assertTrue(any("Retired card" in error for error in result.errors))

    def test_deck_builder_creates_39_legal_collectible_cards(self) -> None:
        deck = build_random_deck(self.cards, seed=17, main_nation="France")
        self.assertEqual(len(deck), 39)
        self.assertTrue(DeckValidator(self.cards).validate(deck, "France").valid)
        self.assertFalse(any(self.cards.get(card_id).is_token for card_id in deck))

    def test_image_url_uses_kards_info_host(self) -> None:
        card = self.cards.get("active_sonar")
        self.assertEqual(card.image_url(), "https://kards.info" + card.imageUrl)
        self.assertEqual(card.image_url(local=True, thumbnail=True), "https://kards.info" + card.localThumbUrl)

    def test_simulator_can_open_draw_play_attack_and_pass(self) -> None:
        unit = min(
            (card for card in self.cards if card.nation == "France" and card.type in {"fighter", "bomber", "artillery"} and not card.is_token and card.text in {"", "{}"}),
            key=lambda card: card.kredits,
        )
        deck = build_random_deck(self.cards, seed=8, main_nation="France")
        simulator = Simulator(self.cards)
        state = simulator.reset(deck, deck, nations=("France", "France"), seed=4)
        simulator.state.players["p1"].hand = [unit.id]
        simulator.state.players["p2"].hand = []
        self.assertEqual(len(state.players["p1"].hand), 4)
        while simulator.state.players["p1"].resources.kredits < unit.kredits:
            simulator.step(next(action for action in simulator.get_available_actions() if isinstance(action, PassAction)))
            simulator.step(next(action for action in simulator.get_available_actions() if isinstance(action, PassAction)))
        play = next(action for action in simulator.get_available_actions() if isinstance(action, PlayCardAction) and action.card_id == unit.id)
        simulator.step(play)
        self.assertTrue(simulator.state.players["p1"].units)
        simulator.step(next(action for action in simulator.get_available_actions() if isinstance(action, PassAction)))
        self.assertEqual(simulator.state.current_player, "p2")
        simulator.step(next(action for action in simulator.get_available_actions() if isinstance(action, PassAction)))
        attack = next(action for action in simulator.get_available_actions() if isinstance(action, AttackAction))
        simulator.step(attack)


if __name__ == "__main__":
    unittest.main()
