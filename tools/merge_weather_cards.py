"""Merge card sets from a kards.com GraphQL dump into the kards.info catalog.

(The filename is from this tool's first use, the weather cards; it now merges
whole sets.  --sets picks them, and the weather ids below stay as extras because
some of them live outside OCEANIA STORM.)

The bundled catalog was generated 2026-05 and predates OCEANIA STORM, so none of
the weather cards exist in it. This tool adds the weather set (the three Forecast
tokens, the nine weather cards and the cards whose text reads weather state) from
a kards.com GraphQL dump, converting the raw ``getCards`` payload into the
catalog's record schema.

It is derived-data generation, not simulator logic, and it is idempotent: an id
already present in the catalog is left untouched.

Usage (defaults assume the dumps sit next to the repository):
    python tools/merge_weather_cards.py
    python tools/merge_weather_cards.py --raw ../../kards_raw.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"
# The GraphQL dumps live in the research folder that contains the repository.
DEFAULT_RAW = ROOT.parents[1] / "kards_raw.json"
DEFAULT_EXTRA = ROOT.parents[1] / "kards_data/kards_extra_spawnable_special.json"

# The Forecast tokens and the nine weather cards they turn into.
FORECAST_TOKENS = ("blue_sky", "mist", "gale")
WEATHER_CARDS = (
    "heatwave", "jungle_fever", "scorching_sun",          # Sunny
    "deluge", "torrential_rain", "monsoon_rain",          # Rain
    "thunderstorm", "tropical_storm", "cyclone",          # Storm
)
# OCEANIA STORM cards whose text reads or writes weather state.
WEATHER_AWARE_CARDS = (
    # OVERCAST predates OCEANIA STORM but grants Forecast; the catalog holds an
    # older wording of it, so it is listed here for the text refresh.
    "overcast",
    "coastwatchers", "us_weather_bureau", "task_force_44", "cruiser_scouts",
    "duck", "kingfisher", "yokosuka_e14y", "kawanishi_h8k", "raaf_walrus",
    "22nd_pioneers", "6th_marine_regiment", "162nd_regiment",
    "16th_infantry_brigade", "16th_infantry_brigade_vet",
    "79th_infantry_regiment", "out_of_the_sun", "surprise_engagement",
)
WANTED = FORECAST_TOKENS + WEATHER_CARDS + WEATHER_AWARE_CARDS

# Whole sets merged by default.  RING OF FIRE (the 2026 autumn mini-set, 30
# cards) is tagged OceaniaStorm by the API because it joins that card pool, so
# merging OceaniaStorm brings it along.
DEFAULT_SETS = ("OceaniaStorm",)


def _english(value: object) -> str:
    """kards.com returns localized dictionaries; the catalog stores en-EN."""
    if isinstance(value, dict):
        return str(value.get("en-EN") or "")
    return str(value or "")


def _payloads(*paths: Path) -> dict[str, dict]:
    """Index every card payload in the given GraphQL dumps by card id."""
    index: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        records = document if isinstance(document, list) else document.get("cards", [])
        for record in records:
            payload = record.get("json") or record
            card_id = payload.get("id") or record.get("id")
            if card_id:
                index[card_id] = payload
    return index


def _record(card_id: str, payload: dict, veteran_bases: frozenset = frozenset()) -> dict:
    """Convert one raw getCards payload into a catalog record."""
    nation = str(payload.get("faction") or "Neutral")
    image = str(payload.get("image") or (card_id + ".avif"))
    image_path = "/images/card/v48/en-EN/" + image
    thumb_path = "/images/card/v48/en-EN/thumb/" + image
    unknown: list[str] = []
    # Preserve the Veteran link the catalog uses, derived from the _vet suffix.
    if card_id.endswith("_vet"):
        unknown.append("VeteranOf:" + card_id[: -len("_vet")])
    elif card_id in veteran_bases:
        unknown.append("BecomesVeteran:" + card_id + "_vet")
    return {
        "id": card_id,
        "importId": str(payload.get("import_id") or ""),
        "slug": nation.lower() + "-" + card_id.replace("_", "-"),
        "name": _english(payload.get("title")) or card_id.replace("_", " ").upper(),
        "nation": nation,
        "type": str(payload.get("type")),
        "rarity": str(payload.get("rarity") or "Standard"),
        "set": str(payload.get("set") or "OceaniaStorm"),
        "kredits": int(payload.get("kredits") or 0),
        "attack": payload.get("attack"),
        "defense": payload.get("defense"),
        "operationCost": payload.get("operationCost"),
        "text": _english(payload.get("text")) or "{}",
        "abilities": list(payload.get("attributes") or []),
        "abilityLabels": [],
        "unknownAttributes": unknown,
        "reserved": bool(payload.get("reserved") or False),
        "spawnable": False,
        "exile": None,
        "canCreate": None,
        "image": image,
        "imageUrl": image_path,
        "thumbUrl": thumb_path,
        "localImageUrl": "/card-images/v48/en-EN/" + image,
        "localThumbUrl": "/card-images/v48/en-EN/thumb/" + image,
    }


def _wanted_ids(payloads: dict[str, dict], sets: tuple[str, ...]) -> list[str]:
    """Explicit ids plus every card in the requested sets, plus Veteran versions.

    A Veteran counterpart is pulled in with its base card: the engine swaps the
    unit for that card, so a missing one leaves the promotion unimplementable.
    """
    wanted = list(WANTED)
    for card_id, payload in payloads.items():
        if payload.get("set") in sets:
            wanted.append(card_id)
    for card_id in list(wanted):
        veteran_id = card_id + "_vet"
        if veteran_id in payloads:
            wanted.append(veteran_id)
    # Stable order, no duplicates.
    return list(dict.fromkeys(wanted))


def merge(raw: Path, extra: Path, dry_run: bool = False,
          sets: tuple[str, ...] = DEFAULT_SETS) -> tuple[list[str], list[str]]:
    """Add every missing wanted card to the catalog. Returns (added, skipped)."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    known = {card["id"] for card in catalog["cards"]}
    payloads = _payloads(raw, extra)
    missing_payloads = [card_id for card_id in WANTED if card_id not in payloads]
    if missing_payloads:
        raise SystemExit("No GraphQL payload for: " + ", ".join(missing_payloads))
    wanted_ids = _wanted_ids(payloads, sets)
    # Bases whose Veteran counterpart exists, so the record can carry the link.
    veteran_bases = frozenset(
        card_id[: -len("_vet")] for card_id in payloads if card_id.endswith("_vet"))
    added, skipped = [], []
    by_id = {card["id"]: card for card in catalog["cards"]}
    for card_id in wanted_ids:
        if card_id in known:
            # A card can predate OCEANIA STORM and still have been reworded by
            # it: the catalog's OVERCAST text lost its "Forecast." sentence.
            # The dump is newer, so its text wins; nothing else is touched.
            current = by_id[card_id]
            fresh_text = _english(payloads[card_id].get("text")) or "{}"
            if fresh_text and fresh_text != current["text"]:
                current["text"] = fresh_text
                added.append(card_id + " (text updated)")
            else:
                skipped.append(card_id)
            continue
        catalog["cards"].append(_record(card_id, payloads[card_id], veteran_bases))
        added.append(card_id)
    catalog["metadata"]["totalCards"] = len(catalog["cards"])
    catalog["metadata"]["weatherMergedFrom"] = raw.name
    if not dry_run:
        CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    return added, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--extra", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sets", nargs="*", default=list(DEFAULT_SETS),
                        help="card sets to merge whole (default: %(default)s)")
    args = parser.parse_args()
    added, skipped = merge(args.raw, args.extra, args.dry_run, tuple(args.sets))
    print("added {0} cards, skipped {1} already present".format(len(added), len(skipped)))
    for card_id in added:
        print("  + " + card_id)
    if skipped:
        print("  (already present: " + ", ".join(skipped) + ")")


if __name__ == "__main__":
    main()
