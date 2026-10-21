"""Nation words in card text vs the catalog's nation values.

Card text uses the adjective ("your German tanks", "non-French units", "Polish
units destroyed this battle") while ``Card.nation`` holds the country
("Germany", "France", "Poland").  Comparing the two directly never matches, so
every rule that filters by nation has to normalize through here.
"""

from __future__ import annotations

NATION_BY_ADJECTIVE = {
    "british": "britain", "english": "britain",
    "german": "germany",
    "american": "usa", "us": "usa",
    "soviet": "soviet", "russian": "soviet",
    "japanese": "japan",
    "french": "france",
    "italian": "italy",
    "polish": "poland",
    "finnish": "finland",
    "anzac": "anzac",
}


def normalize(nation: str | None) -> str:
    """Return the catalog nation value (lowercased) for a nation word."""
    value = (nation or "").strip().lower()
    return NATION_BY_ADJECTIVE.get(value, value)


def matches(card_nation: str | None, wanted: str | None) -> bool:
    """Whether a card's nation matches a nation word from card text."""
    if not wanted:
        return True
    return (card_nation or "").lower() == normalize(wanted)
