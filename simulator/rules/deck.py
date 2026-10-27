"""Deck legality checks suitable for deck-building agents."""
from collections import Counter
from dataclasses import dataclass
from simulator.cards.availability import is_card_available


@dataclass(frozen=True)
class DeckValidationResult:
    valid: bool
    errors: tuple[str, ...]


class DeckValidator:
    def __init__(self, cards, deck_size=39, copy_limit=4):
        # KARDS decks hold 39 cards (confirmed in game 2026-10-27; every exported
        # tournament deck decodes to 39).  The engine used to require 40.
        self.cards, self.deck_size, self.copy_limit = cards, deck_size, copy_limit

    def validate(self, deck, main_nation, ally_nation=None, ally_limit=12):
        errors = []
        if len(deck) != self.deck_size:
            errors.append("Deck must contain {0} cards".format(self.deck_size))
        counts = Counter(deck)
        if ally_nation is not None and ally_nation == main_nation:
            errors.append("Main and ally nations must differ")
        ally_count = 0
        for card_id, count in counts.items():
            if card_id not in self.cards:
                errors.append("Unknown card: {0}".format(card_id)); continue
            if not is_card_available(card_id):
                errors.append("Retired card cannot be included: {0}".format(card_id)); continue
            card = self.cards.get(card_id)
            if card.is_token:
                errors.append("OnlySpawnable card cannot be included: {0}".format(card_id))
            rarity_limit = {"standard": 4, "limited": 3, "special": 2, "elite": 1}.get(card.rarity.lower(), self.copy_limit)
            if count > min(self.copy_limit, rarity_limit):
                errors.append("Copy limit exceeded: {0}".format(card_id))
            nation = card.nation
            if nation and nation not in {main_nation, ally_nation, "Neutral"}:
                errors.append("Nation not allowed: {0}".format(card_id))
            if nation == ally_nation and nation != main_nation:
                ally_count += count
        if ally_count > ally_limit:
            errors.append("Ally nation cards cannot exceed {0}".format(ally_limit))
        return DeckValidationResult(not errors, tuple(errors))
