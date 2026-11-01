"""Play each card on a fixed board and print what changed, for a human check.

    python tools/card_probe.py --meta            # every card in the meta decks
    python tools/card_probe.py hummel 52k        # chosen cards

For each card: its text, its parsed rule, then for every legal way to play it
(each target / option, at most --limit) the differences the play made: units
added, removed, stats, position, keywords, hands, decks, HQs, kredits, and
events that hint a rule did not run.  Listener effects ("When ...") do not show
here; the triggers are printed so they can be checked separately.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.actions.action import PlayCardAction  # noqa: E402
from simulator.cards.deck_code import load_meta_decks  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState  # noqa: E402
from simulator.rules.native import engine_for  # noqa: E402

CATALOG = ROOT / "data/source/kards_info_cards.json"
QUIET = {"native_rule_executed", "native_event_emitted", "unit_deployed", "card_played", "on_deploy",
         "kredits_spent", "order_played"}
FLAGS = ("unmodeled", "unsupported", "failed", "no_target", "unresolved", "fizzle",
         "no_longer_exists", "not_found", "skipped")


def board(card_id: str, cards) -> GameState:
    """p1 (Germany) plays; both sides hold a mixed board."""
    def u(uid, cid, owner, pos, atk=None, dfn=None):
        card = cards.get(cid)
        return UnitState(uid, cid, card.attack if atk is None else atk, card.defense if dfn is None else dfn, owner, pos)
    p1_units = [u("F_inf", "garrison", "p1", "support_line", 1, 3), u("S_tank", "panzer_iia", "p1", "support_line"),
                u("S_ftr", "gladiator_mk_i", "p1", "support_line")]
    p2_units = [u("E_inf", "garrison", "p2", "frontline", 2, 2), u("E_tank", "panzer_iia", "p2", "frontline"),
                u("E_art", "37mm_bofors_gun", "p2", "support_line"), u("E_ftr", "gladiator_mk_i", "p2", "support_line")]
    state = GameState(
        current_player="p1",
        players={
            "p1": PlayerState("p1", "Germany", deck=["garrison", "blast", "panzer_iia", "radar"] * 5,
                              hand=[card_id, "garrison", "blast"], resources=ResourceState(9, 9), units=p1_units),
            "p2": PlayerState("p2", "Britain", deck=["garrison", "radar"] * 10, hand=["garrison", "blast", "radar"],
                              resources=ResourceState(9, 9), units=p2_units),
        },
        graveyard={"p1": [], "p2": []}, rng_seed=11, turn_number=6,
    )
    state.players["p1"].hq.current_health = 18
    state.players["p2"].hq.current_health = 18
    for unit in p1_units + p2_units:
        state.battlefield[unit.position].append(unit.instance_id)
    return state


def snapshot(state, cards):
    snap = {}
    for pid, player in state.players.items():
        snap[pid] = {
            "hq": player.hq.current_health, "kredits": player.resources.kredits, "slots": player.resources.max_kredits,
            "hand": list(player.hand), "deck": len(player.deck), "deck_top": player.deck[:2],
            "units": {u.instance_id: (u.card_id, u.attack, u.defense, u.position,
                                      tuple(sorted(map(str, u.status.get("added_abilities", ())))),
                                      tuple(sorted(k for k, v in u.status.items() if v and k not in
                                                   {"added_abilities", "deployed_this_turn", "attack_count"})),
                                      tuple(m.get("type") + ":" + str(m.get("value", m.get("amount"))) for m in u.modifiers))
                      for u in player.units},
            "grave": list(state.graveyard.get(pid, [])),
        }
    return snap


def diff(before, after, cards):
    lines = []
    for pid in ("p1", "p2"):
        b, a = before[pid], after[pid]
        who = "我" if pid == "p1" else "敵"
        for key, label in (("hq", "總部"), ("slots", "kredit 槽"), ("deck", "牌庫")):
            if b[key] != a[key]:
                lines.append(f"{who}{label} {b[key]}→{a[key]}")
        if b["kredits"] - a["kredits"] and pid == "p2":
            lines.append(f"敵 kredits {b['kredits']}→{a['kredits']}")
        hb, ha = list(b["hand"]), list(a["hand"])
        if pid == "p1" and hb:
            hb = hb[1:]           # the played card
        if hb != ha:
            lines.append(f"{who}手牌 {hb}→{ha}")
        if b["deck_top"] != a["deck_top"] and b["deck"] == a["deck"]:
            lines.append(f"{who}牌庫頂 {b['deck_top']}→{a['deck_top']}")
        for uid in sorted(set(b["units"]) | set(a["units"])):
            ub, ua = b["units"].get(uid), a["units"].get(uid)
            if ub == ua:
                continue
            if ub is None:
                lines.append(f"{who}+ {ua[0]} {ua[1]}/{ua[2]} {ua[3]} {list(ua[4]) or ''} {list(ua[5]) or ''} {list(ua[6]) or ''}".rstrip())
            elif ua is None:
                lines.append(f"{who}- {uid}({ub[0]})")
            else:
                parts = []
                if ub[0] != ua[0]: parts.append(f"變成 {ua[0]}")
                if ub[1:3] != ua[1:3]: parts.append(f"{ub[1]}/{ub[2]}→{ua[1]}/{ua[2]}")
                if ub[3] != ua[3]: parts.append(f"{ub[3]}→{ua[3]}")
                if ub[4] != ua[4]: parts.append(f"能力 {list(ua[4])}")
                if ub[5] != ua[5]: parts.append(f"狀態 {sorted(set(ua[5]) - set(ub[5]))}")
                if ub[6] != ua[6]: parts.append(f"修正 {list(ua[6])}")
                lines.append(f"{who} {uid}: " + "，".join(parts))
        if b["grave"] != a["grave"]:
            lines.append(f"{who}墓地 +{a['grave'][len(b['grave']):]}")
    return lines


def probe(card_id, cards, limit):
    engine = engine_for(cards)
    card = cards.get(card_id)
    rule = engine.rule_for(card_id)
    out = [f"## {card.name} ({card_id}) {card.type} {card.kredits}K"
           + (f" {card.attack}/{card.defense}" if card.is_unit else "")
           + (f" 能力 {list(card.abilities)}" if card.abilities else ""),
           "卡面：" + card.text.replace("\r", "").replace("\n", " / "),
           f"規則：{rule.status} triggers={list(rule.triggers)}"]
    for a in rule.actions:
        bits = [a.kind, a.target]
        for f in ("amount", "attack", "defense"):
            if getattr(a, f):
                bits.append(f"{f}={getattr(a, f)}")
        for f in ("card_name", "scope", "duration", "condition"):
            if getattr(a, f):
                bits.append(f"{f}={getattr(a, f)!r}")
        if a.on:
            bits.append(f"on={list(a.on)}")
        out.append("   · " + " ".join(map(str, bits)))
    sim = Simulator(cards)
    sim.state = board(card_id, cards)
    plays = [a for a in sim.get_available_actions() if isinstance(a, PlayCardAction) and a.card_id == card_id]
    if not plays:
        out.append("   ！沒有合法的打法")
    seen = set()
    shown = 0
    for play in plays:
        key = (play.target_unit_id, play.selected_card_id, play.selected_option)
        if key in seen or (card.is_unit and play.position not in (None, "support_line") and len(plays) > 1
                           and any(p.target_unit_id == play.target_unit_id and p.position == "support_line" for p in plays)):
            continue
        seen.add(key)
        if shown >= limit:
            out.append(f"   …另有 {len([p for p in plays]) - shown} 種打法")
            break
        shown += 1
        state = copy.deepcopy(sim.state)
        before = snapshot(state, cards)
        start = len(state.event_log)
        try:
            play.execute(state, cards)
        except Exception as error:   # noqa: BLE001
            out.append(f"   [{key}] 例外 {error!r}")
            continue
        changes = diff(before, snapshot(state, cards), cards)
        flags = sorted({e["event"] for e in state.event_log[start:] if any(f in e["event"] for f in FLAGS)})
        label = "/".join(str(x) for x in key if x) or "—"
        out.append(f"   [{label}] " + ("；".join(changes) or "（沒有變化）") + (f"  ⚠{flags}" if flags else ""))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cards", nargs="*")
    parser.add_argument("--meta", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    cards = CardDatabase.from_file(CATALOG)
    ids = list(args.cards)
    if args.meta:
        users = {}
        for deck in load_meta_decks(cards):
            for cid in deck.cards:
                users.setdefault(cid, set()).add(deck.name)
        ids += sorted(users, key=lambda c: (-len(users[c]), c))
    for cid in ids:
        print(probe(cid, cards, args.limit))
        print()


if __name__ == "__main__":
    main()
