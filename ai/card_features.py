"""Shared card vocabulary and static card features for the v2 encoders.

2026-11-24 (observation/action encoding v2): the v1 encoders identified a
card only by a 16-bit hash squeezed into one float, so the network could not
tell cards apart beyond their raw stats, and action features carried no card
or unit information at all -- just hashes of the action's string fields.
Both encoders now share:

* a stable card vocabulary (``card_index``): every catalogue card id gets an
  integer 1..N (0 = empty slot). The network turns these into learned
  embeddings, so each of the 1,646 cards can mean something of its own.
* ``static_features``: type, stats, cost and keywords from the card text.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).resolve().parents[1]
CARD_CATALOGUE = ROOT / "data/source/kards_info_cards.json"
# Embedding table size. Must exceed the catalogue size + 1 (currently 1,646);
# kept as a fixed constant so checkpoints do not depend on the catalogue size.
CARD_VOCAB = 2048
TYPES = ("infantry", "fighter", "tank", "bomber", "artillery", "order", "countermeasure")
KEYWORDS = ("blitz", "guard", "smokescreen", "fury", "ambush", "shock", "alpine", "covert", "mobilize")
NATIONS = ("Britain", "Germany", "Soviet", "USA", "Japan", "France", "Italy", "Poland", "Finland", "Anzac", "Neutral")
# type one-hot + attack, defense, kredits, operation cost + keywords + heavy armor
STATIC_DIM = len(TYPES) + 4 + len(KEYWORDS) + 1
ATTACK_INDEX, DEFENSE_INDEX, KREDITS_INDEX = len(TYPES), len(TYPES) + 1, len(TYPES) + 2
KEYWORD_INDEX = len(TYPES) + 4


@lru_cache(maxsize=1)
def catalogue() -> CardDatabase:
    return CardDatabase.from_file(CARD_CATALOGUE)


@lru_cache(maxsize=1)
def _index_table() -> dict[str, int]:
    ids = sorted(card.id for card in catalogue())
    if len(ids) + 1 > CARD_VOCAB:
        raise RuntimeError(f"Card catalogue ({len(ids)}) outgrew CARD_VOCAB ({CARD_VOCAB})")
    return {card_id: index + 1 for index, card_id in enumerate(ids)}


def card_index(card_id: str | None) -> int:
    """Vocabulary index for a card id; 0 for empty or unknown."""
    if not card_id:
        return 0
    return _index_table().get(card_id, 0)


def card_id_of_instance(instance_id: str | None) -> str | None:
    """Unit instance ids are ``<card_id>-<n>``; recover the card id."""
    if not instance_id:
        return None
    if instance_id in _index_table():
        return instance_id
    head, _, tail = instance_id.rpartition("-")
    return head if head and tail.isdigit() and head in _index_table() else None


def _has_keyword(abilities: tuple[str, ...] | list[str], keyword: str) -> bool:
    return any(value.lower().split(":", 1)[0] == keyword for value in abilities)


def _heavy_armor(abilities) -> int:
    for value in abilities:
        value = value.lower()
        if value.startswith("heavyarmor") and value[10:].isdigit():
            return int(value[10:])
    return 0


@lru_cache(maxsize=4096)
def static_features(card_id: str | None) -> tuple[float, ...]:
    if not card_id or card_id not in catalogue():
        return (0.0,) * STATIC_DIM
    card = catalogue().get(card_id)
    abilities = tuple(card.abilities or ())
    values = [float(card.type == kind) for kind in TYPES]
    values += [(card.attack or 0) / 20.0, (card.defense or 0) / 20.0, (card.kredits or 0) / 12.0, (card.operationCost or 0) / 12.0]
    values += [float(_has_keyword(abilities, keyword)) for keyword in KEYWORDS]
    values.append(_heavy_armor(abilities) / 3.0)
    return tuple(values)
