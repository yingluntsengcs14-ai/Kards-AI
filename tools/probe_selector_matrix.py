"""Measure which (kind, selector) pairs actually change state.

``tools/build_card_rules.py`` only declares a selector family for the kinds
where the target genuinely gates the outcome.  This is the measurement behind
that decision, and the way to extend the declaration safely: run it, read the
"target gates the outcome" section, and add those kinds.

For every kind in the executed whitelist and every selector any path can
resolve, the probe runs the action on a synthetic board and records whether the
serialized state changed.  Each kind then falls into one of three buckets:

  ignores target   every selector produced the same outcome -> declaring a
                   family for it would only produce false alarms
  target gates     some selectors change state and others do not -> a candidate
                   for _KIND_SELECTOR_FAMILY
  inert here       nothing changed for any selector -> the probe board does not
                   exercise it (needs a choice, an event, or a specific shape)

A kind in "inert here" is NOT evidence of a bug: see
tools/scan_selector_coverage.py for the per-card version of that question.

Usage:
    python tools/probe_selector_matrix.py
    python tools/probe_selector_matrix.py --kind damage --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules import selectors
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction

from build_card_rules import _GENUINELY_EXECUTED_KINDS

CATALOG = ROOT / "data/source/kards_info_cards.json"
FILLER = "garrison"


def _state() -> GameState:
    state = GameState(
        current_player="p1",
        players={
            "p1": PlayerState("p1", "Germany", deck=[FILLER] * 10,
                              hand=[FILLER, "radar", FILLER],
                              resources=ResourceState(8, 8),
                              units=[UnitState("src", FILLER, 2, 5, "p1", "support_line"),
                                     UnitState("f1", FILLER, 2, 5, "p1", "support_line")]),
            "p2": PlayerState("p2", "Soviet", deck=[FILLER] * 10,
                              hand=[FILLER, FILLER],
                              resources=ResourceState(8, 8),
                              units=[UnitState("e1", FILLER, 2, 5, "p2", "support_line"),
                                     UnitState("e2", "duck", 2, 5, "p2", "support_line")]),
        },
        graveyard={"p1": [], "p2": []}, rng_seed=5,
    )
    for player in state.players.values():
        for unit in player.units:
            state.battlefield["support_line"].append(unit.instance_id)
    return state


def _snapshot(state: GameState) -> str:
    document = state.to_dict()
    document.pop("event_log", None)
    return json.dumps(document, sort_keys=True, default=str)


def probe(engine: NativeRuleEngine, kind: str, target: str) -> bool:
    """Whether (kind, target) changed the board on the probe state."""
    state = _state()
    before = _snapshot(state)
    action = RuleAction(kind, target, amount=2, attack=1, defense=1)
    context = EffectContext("p1", FILLER, "src", "e1", event="on_play",
                            metadata={"event_player_id": "p1"})
    try:
        engine._execute_action(action, state, context)
    except Exception:                      # noqa: BLE001 - diagnostic tool
        return False
    return _snapshot(state) != before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", help="probe only this kind")
    parser.add_argument("--verbose", action="store_true", help="list the selectors per kind")
    args = parser.parse_args()

    cards = CardDatabase.from_file(CATALOG)
    engine = NativeRuleEngine(cards)
    targets = sorted(selectors.ALL_SELECTORS)
    kinds = [args.kind] if args.kind else sorted(_GENUINELY_EXECUTED_KINDS)

    ignores, gates, inert = [], [], []
    for kind in kinds:
        working = [target for target in targets if probe(engine, kind, target)]
        if not working:
            inert.append(kind)
        elif len(working) == len(targets):
            ignores.append(kind)
        else:
            gates.append((kind, working))

    print("probed {0} kinds x {1} selectors".format(len(kinds), len(targets)))
    print("\nignores target ({0}): declaring a family would only false-alarm".format(len(ignores)))
    if args.verbose:
        for kind in ignores:
            print("  " + kind)
    print("\ntarget gates the outcome ({0}): candidates for _KIND_SELECTOR_FAMILY".format(len(gates)))
    for kind, working in gates:
        blocked = [target for target in targets if target not in working]
        print("  {0:<34} works={1}".format(kind, ",".join(working)))
        print("  {0:<34} blocked={1}".format("", ",".join(blocked)))
    print("\ninert on this board ({0}): needs a choice, an event or a specific shape".format(len(inert)))
    if args.verbose:
        for kind in inert:
            print("  " + kind)


if __name__ == "__main__":
    main()
