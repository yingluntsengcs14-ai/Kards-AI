"""Develop: choose one of three cards and add it to your hand.

Confirmed in game (2026-10-27): a developed card goes to hand, units included.
The engine used to deploy a developed unit, and to fail outright on a
developed order or countermeasure.

Three shapes appear in card text:

* ``Develop A, B or C``          -- the three named cards are offered.
* ``Develop RETRIBUTION``         -- one named card; nothing to choose.
* ``Develop a German countermeasure`` -- a description; three matching
  collectible cards are offered.

The offer for a description is drawn deterministically from the state, so the
action list (one action per offered card) and the effect agree.
"""
from __future__ import annotations

import random
import re

from simulator.rules import nations

_TYPES = {
    "unit": None, "units": None, "card": "any", "cards": "any",
    "order": {"order"}, "orders": {"order"}, "countermeasure": {"countermeasure"},
    "infantry": {"infantry"}, "tank": {"tank"}, "tanks": {"tank"}, "artillery": {"artillery"},
    "fighter": {"fighter"}, "bomber": {"bomber"}, "bombers": {"bomber"},
    "air": {"fighter", "bomber"}, "ground": {"infantry", "tank", "artillery"},
}
# Type words that are not card types: "a Navy card" (name or ability).
_TRAITS = {"navy"}
_UNIT_TYPES = {"infantry", "tank", "artillery", "fighter", "bomber"}


def parse_descriptor(phrase: str) -> dict | None:
    """Filter for "a British or US bomber", "an air unit of your main nation",
    "a British order costing 4 or less", "a British air unit with 2 or less
    attack"; None when a word is not understood."""
    text = phrase.lower().strip().rstrip(".!")
    text = re.sub(r"^(?:a|an|one)\s+", "", text)
    wanted: dict = {"nations": set(), "types": None, "main_nation": False, "max_cost": None, "max_attack": None,
                    "traits": set()}
    match = re.search(r"\s+costing\s+(\d+)\s+or\s+less$", text)
    if match:
        wanted["max_cost"] = int(match.group(1)); text = text[:match.start()]
    match = re.search(r"\s+with\s+(\d+)\s+or\s+less\s+attack$", text)
    if match:
        wanted["max_attack"] = int(match.group(1)); text = text[:match.start()]
    if text.endswith(" of your main nation"):
        wanted["main_nation"] = True; text = text[: -len(" of your main nation")]
    unit_only = False
    for word in text.replace(",", " ").split():
        if word in ("or", "and"):
            continue
        if word in nations.NATION_BY_ADJECTIVE:
            wanted["nations"].add(nations.normalize(word))
        elif word in _TRAITS:
            wanted["traits"].add(word)
        elif word in _TYPES:
            kind = _TYPES[word]
            if kind is None:
                unit_only = True
            elif kind != "any":
                wanted["types"] = set(kind) if wanted["types"] is None else wanted["types"] | set(kind)
        else:
            return None
    if wanted["types"] is None and unit_only:
        wanted["types"] = set(_UNIT_TYPES)
    return wanted


def candidates(cards, wanted: dict, main_nation: str | None) -> list[str]:
    from simulator.cards.availability import is_card_available
    pool = []
    for card in cards:
        if card.is_token or not is_card_available(card.id) or card.id.endswith("_vet"):
            continue
        if wanted["types"] is not None and card.type not in wanted["types"]:
            continue
        nation_ok = True
        if wanted["nations"]:
            nation_ok = (card.nation or "").lower() in wanted["nations"]
        if wanted["main_nation"]:
            nation_ok = nation_ok and (card.nation or "").lower() == (main_nation or "").lower()
        if not nation_ok:
            continue
        if wanted.get("traits") and not all(
                trait in card.name.lower() or any(trait in str(ability).lower() for ability in card.abilities)
                for trait in wanted["traits"]):
            continue
        if wanted["max_cost"] is not None and card.kredits > wanted["max_cost"]:
            continue
        if wanted["max_attack"] is not None and (card.attack or 0) > wanted["max_attack"]:
            continue
        pool.append(card.id)
    return sorted(pool)


def offers(state, player_id: str, cards, phrase: str, count: int = 3) -> tuple[str, ...]:
    wanted = parse_descriptor(phrase)
    if wanted is None:
        return ()
    pool = candidates(cards, wanted, state.players[player_id].nation)
    # Seeded by things that do not change between listing the actions and
    # resolving the card (the played card leaves the hand in between).
    rng = random.Random((state.rng_seed or 0) * 131 + state.turn_number * 17 + len(state.players[player_id].deck)
                        + sum(map(ord, phrase)))
    return tuple(rng.sample(pool, min(count, len(pool))))
