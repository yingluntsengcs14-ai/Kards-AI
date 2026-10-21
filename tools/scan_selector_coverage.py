"""Find rules that parse and run but change nothing — silent no-ops.

The honest-status gate in ``tools/build_card_rules.py`` checks that every parsed
action *kind* has a handler.  It does not check that the action's *target*
selector resolves to anything, so a card can be marked ``implemented`` while its
effect never happens.  ``all_units`` was exactly this: no handler resolved it, so
"All units get +2 attack" silently did nothing on five cards, four of them
marked implemented.

This tool is the empirical check.  For every card it builds a synthetic board
with units, hands and decks on both sides, executes the card's rule for each of
its triggers, and compares the serialized state before and after.  A rule that
has actions but produces no state change at all is reported.

False positives are expected: a rule can legitimately do nothing on this board
(a condition that needs a specific shape, an effect that needs a choice this
tool does not make).  The value is in the aggregation: when every card using a
given (kind, target) pair changes nothing, that pair is almost certainly dead.

Usage:
    python tools/scan_selector_coverage.py              # summary
    python tools/scan_selector_coverage.py --cards       # per-card detail
    python tools/scan_selector_coverage.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine

CATALOG = ROOT / "data/source/kards_info_cards.json"

# A board with room to move, units on both sides, cards in hand and deck, and
# kredits to spend, so that as many effects as possible have something to do.
FILLER = "garrison"


def _state(cards: CardDatabase, source_card_id: str, enemy_holds_frontline: bool = False) -> GameState:
    source_card = cards.get(source_card_id)
    # The source sits in the support line and the enemy holds no frontline unit,
    # so movement effects have somewhere to go: the frontline-control rule would
    # otherwise block every move_to_frontline and look like a dead selector.
    source_unit = UnitState("src", source_card_id if source_card.is_unit else FILLER,
                            source_card.attack or 2, source_card.defense or 5, "p1", "support_line")
    state = GameState(
        current_player="p1",
        players={
            "p1": PlayerState("p1", source_card.nation, deck=[FILLER] * 12,
                              hand=[FILLER, FILLER, FILLER],
                              resources=ResourceState(12, 12),
                              units=[source_unit,
                                     UnitState("f1", FILLER, 2, 5, "p1", "support_line"),
                                     UnitState("f2", "duck", 2, 5, "p1",
                                               "support_line" if enemy_holds_frontline else "frontline")]),
            "p2": PlayerState("p2", "Soviet", deck=[FILLER] * 12,
                              hand=[FILLER, FILLER, FILLER],
                              resources=ResourceState(12, 12),
                              units=[UnitState("e1", FILLER, 2, 5, "p2",
                                                "frontline" if enemy_holds_frontline else "support_line"),
                                     UnitState("e2", "duck", 2, 5, "p2", "support_line")]),
        },
        graveyard={"p1": [], "p2": []}, rng_seed=11,
    )
    for player in state.players.values():
        for unit in player.units:
            state.battlefield.setdefault(unit.position, []).append(unit.instance_id)
    return state


def _snapshot(state: GameState) -> str:
    """Serialize everything except the event log, which always grows."""
    document = state.to_dict()
    document.pop("event_log", None)
    return json.dumps(document, sort_keys=True, default=str)


def scan(cards: CardDatabase, engine: NativeRuleEngine) -> list[dict]:
    findings = []
    for card in cards:
        rule = engine.rule_for(card.id)
        if not rule.actions:
            continue
        changed = False
        errors = []
        # Try each trigger against both an enemy and a friendly target: a rule
        # that only moves friendly units would look dead if only an enemy were
        # offered, and vice versa.
        # Two boards are tried: one where this player controls the frontline and
        # one where the enemy does.  KARDS frontline-control rules make several
        # movement effects impossible on one board but legal on the other.
        for trigger in (rule.triggers or ("on_play",)):
            for enemy_frontline in (False, True):
              for target_id in ("e1", "f1"):
                state = _state(cards, card.id, enemy_frontline)
                before = _snapshot(state)
                context = EffectContext(
                    "p1", card.id, "src", target_id, event=trigger,
                    metadata={"event_player_id": "p1", "weather": "rain",
                              "selected_option": "rain", "played_card_id": card.id,
                              "damage_amount": 3},
                )
                try:
                    engine.execute(card.id, trigger, state, context)
                except Exception as error:                  # noqa: BLE001 - diagnostic tool
                    errors.append("{0}: {1}".format(type(error).__name__, error))
                    continue
                if _snapshot(state) != before:
                    changed = True
                    break
              if changed:
                break
            if changed:
                break
        if not changed:
            findings.append({
                "card_id": card.id,
                "status": rule.status,
                "triggers": list(rule.triggers),
                "actions": [{"kind": action.kind, "target": action.target,
                             "scope": action.scope, "condition": action.condition}
                            for action in rule.actions],
                "errors": errors,
                "text": card.text,
            })
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", action="store_true", help="list every suspect card")
    parser.add_argument("--json", type=Path, help="write the full findings as JSON")
    args = parser.parse_args()

    cards = CardDatabase.from_file(CATALOG)
    engine = NativeRuleEngine(cards)
    findings = scan(cards, engine)

    total_with_actions = sum(1 for card in cards if engine.rule_for(card.id).actions)
    print("cards with actions: {0}".format(total_with_actions))
    print("changed nothing:    {0}".format(len(findings)))
    print("  of those, marked implemented: {0}".format(
        sum(1 for finding in findings if finding["status"] == "implemented")))

    # Aggregate by (kind, target): a pair whose every user is silent is dead.
    pair_total: collections.Counter = collections.Counter()
    pair_silent: collections.Counter = collections.Counter()
    for card in cards:
        rule = engine.rule_for(card.id)
        for action in rule.actions:
            pair_total[(action.kind, action.target)] += 1
    for finding in findings:
        for action in finding["actions"]:
            pair_silent[(action["kind"], action["target"])] += 1

    print("\n(kind, target) pairs where EVERY card using them changed nothing:")
    dead = [(pair, count) for pair, count in pair_silent.items() if count == pair_total[pair]]
    for (kind, target), count in sorted(dead, key=lambda item: -item[1]):
        print("  {0:<34} target={1:<20} {2} card(s)".format(kind, target, count))

    print("\ntargets used by rules but not resolvable by any handler would show up above.")
    if args.cards:
        print("\nsuspect cards:")
        for finding in sorted(findings, key=lambda item: item["status"]):
            print("  [{0}] {1}: {2}".format(finding["status"], finding["card_id"],
                                            ", ".join(a["kind"] + "->" + a["target"]
                                                      for a in finding["actions"])))
            if finding["errors"]:
                print("      errors: " + "; ".join(finding["errors"]))
    if args.json:
        args.json.write_text(json.dumps(findings, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote " + str(args.json))


if __name__ == "__main__":
    main()
