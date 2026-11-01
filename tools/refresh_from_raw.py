"""Refresh card text and numbers from a newer kards.com GraphQL dump.

The bundled catalog is the 2026-05 kards.info export; balance patches since
then changed text and stats (HEINKEL HE 219 now reads "Deployment: Fight target
enemy unit."; HAMPSHIRE REGIMENT develops a different card set).  For every
card present in the dump, this copies over en-EN text, kredits, attack,
defense and operationCost when they differ.  Cards the dump lacks are left as
they are.

    python tools/refresh_from_raw.py --raw ../../kards_raw.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"
FIELDS = ("kredits", "attack", "defense", "operationCost")


def _norm(text: str | None) -> str:
    return " ".join((text or "").replace("\\r\\n", " ").replace("\r\n", " ").replace('\\"', '"').split())


def refresh(raw_path: Path, dry_run: bool = False) -> list[tuple[str, list[str]]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("cards", [])
    fresh = {(record.get("json") or record)["id"]: (record.get("json") or record) for record in records}
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    changes = []
    for card in catalog["cards"]:
        payload = fresh.get(card["id"])
        if payload is None:
            continue
        changed = []
        text = (payload.get("text") or {}).get("en-EN")
        if text and _norm(text) != _norm(card["text"]):
            card["text"] = text
            changed.append("text")
        for field in FIELDS:
            value = payload.get(field)
            if value is not None and value != card[field]:
                changed.append("{0} {1}->{2}".format(field, card[field], value))
                card[field] = value
        if changed:
            changes.append((card["id"], changed))
    catalog["metadata"]["refreshedFrom"] = raw_path.name
    if not dry_run:
        CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT.parents[1] / "kards_raw.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changes = refresh(args.raw, args.dry_run)
    print("{0} cards refreshed".format(len(changes)))
    for card_id, changed in changes:
        print("  {0}: {1}".format(card_id, ", ".join(changed)))


if __name__ == "__main__":
    main()
