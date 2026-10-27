"""Read KARDS deck codes ("%%26|0d0m...;2h2l...;1V2i...;yH").

Layout, checked against 17 exported tournament decks (2026-10-27):

* ``%%`` then two nation digits: main nation, then ally nation.
* ``|`` then four ``;``-separated groups: cards held once, twice, three
  times, four times.  Each card is its two-character ``importId``.

Every exported deck decodes to 39 cards.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from simulator.cards.loader import CardDatabase

NATION_DIGITS = {
    "1": "Germany", "2": "Britain", "3": "Japan", "4": "Soviet", "5": "USA",
    "6": "France", "7": "Italy", "8": "Poland", "9": "Finland", "a": "Anzac",
}
META_DECKS = Path(__file__).resolve().parents[2] / "data" / "decks" / "meta_decks.json"


@dataclass(frozen=True)
class Deck:
    name: str
    main_nation: str
    ally_nation: str | None
    cards: tuple[str, ...]
    tier: str = ""


def decode(code: str, cards: CardDatabase, name: str = "", tier: str = "") -> Deck:
    code = code.strip()
    if not code.startswith("%%") or "|" not in code:
        raise ValueError("Not a KARDS deck code: {0!r}".format(code))
    head, body = code[2:].split("|", 1)
    if len(head) < 2 or head[0] not in NATION_DIGITS or head[1] not in NATION_DIGITS:
        raise ValueError("Unknown nation digits {0!r} in {1!r}".format(head, code))
    by_import = {card.importId: card.id for card in cards if card.importId}
    deck: list[str] = []
    unknown: list[str] = []
    for copies, group in enumerate(body.split(";")[:4], start=1):
        if len(group) % 2:
            raise ValueError("Odd-length card group {0!r} in {1!r}".format(group, code))
        for index in range(0, len(group), 2):
            import_id = group[index:index + 2]
            if import_id not in by_import:
                unknown.append(import_id)
                continue
            deck.extend([by_import[import_id]] * copies)
    if unknown:
        raise ValueError("Deck {0!r} has cards missing from the catalog: {1}".format(name or code, ", ".join(unknown)))
    return Deck(name, NATION_DIGITS[head[0]], NATION_DIGITS[head[1]], tuple(deck), tier)


def load_meta_decks(cards: CardDatabase, path: Path = META_DECKS) -> list[Deck]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return [decode(entry["code"], cards, entry.get("name", ""), entry.get("tier", "")) for entry in document["decks"]]
