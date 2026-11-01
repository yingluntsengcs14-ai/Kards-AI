"""Check "When ..." / "At the start of ..." effects of cards already in play.

    python tools/listener_probe.py --meta
    python tools/listener_probe.py j2m_raiden shibata_regiment

The card is put on p1's board (a countermeasure: activated), then each
scenario runs twice -- with and without the card -- and only the difference is
printed, so what shows is what the card itself did.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from card_probe import CATALOG, board, snapshot, diff  # noqa: E402
from simulator.actions.action import AttackAction, MoveUnitAction, PlayCardAction  # noqa: E402
from simulator.cards.deck_code import load_meta_decks  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.turn import TurnManager  # noqa: E402
from simulator.effects import basic_effects  # noqa: E402
from simulator.rules.native import engine_for  # noqa: E402

PLAY_ONLY = {"on_play", "on_deploy"}


def _kill(state, cards, uid):
    unit = next(u for p in state.players.values() for u in p.units if u.instance_id == uid)
    start = len(state.event_log)
    basic_effects.destroy(state, unit)
    engine_for(cards).emit_deaths_since(state, start)


def _enemy_order(state, cards):
    state.current_player = "p2"
    PlayCardAction("p2", "blast", target_unit_id="F_inf").execute(state, cards)
    state.current_player = "p1"


def _attack(target):
    def run(state, cards):
        unit = next((u for u in state.players["p1"].units if u.instance_id == "PROBE"), None)
        if unit is None:
            raise RuntimeError("沒有在場上的單位")
        unit.status.pop("deployed_this_turn", None)
        if unit.position != "frontline":
            unit.position = "frontline"
            state.battlefield["support_line"].remove(unit.instance_id)
            state.battlefield["frontline"].append(unit.instance_id)
            # The enemy holds the frontline in board(); clear it for the attack.
            for enemy in list(state.players["p2"].units):
                if enemy.position == "frontline":
                    enemy.position = "support_line"
                    state.battlefield["frontline"].remove(enemy.instance_id)
                    state.battlefield["support_line"].append(enemy.instance_id)
        AttackAction("p1", "PROBE", target).execute(state, cards)
    return run


def _turn_cycle(state, cards):
    from simulator.actions.action import PassAction
    PassAction("p1").execute(state, cards)
    PassAction("p2").execute(state, cards)


def _draw(pid):
    def run(state, cards):
        start = len(state.event_log)
        TurnManager.draw_card(state, pid)
        engine_for(cards).emit_draws_since(state, start)
    return run


def _enemy_moves(state, cards):
    for enemy in list(state.players["p2"].units):
        if enemy.position == "frontline":
            enemy.position = "support_line"
            state.battlefield["frontline"].remove(enemy.instance_id)
            state.battlefield["support_line"].append(enemy.instance_id)
    state.current_player = "p2"
    MoveUnitAction("p2", "E_inf").execute(state, cards)
    state.current_player = "p1"


SCENARIOS = [
    ("敵方下指令", _enemy_order),
    ("友軍被消滅", lambda s, c: _kill(s, c, "S_tank")),
    ("敵軍被消滅", lambda s, c: _kill(s, c, "E_inf")),
    ("敵方抽牌", _draw("p2")),
    ("我方抽牌", _draw("p1")),
    ("我下指令", lambda s, c: PlayCardAction("p1", "blast", target_unit_id="E_tank").execute(s, c)),
    ("回合結束→下回合開始", _turn_cycle),
    ("本單位攻擊總部", _attack(None)),
    ("本單位攻擊敵單位", _attack("E_tank")),
    ("敵單位移進前線", _enemy_moves),
]


def place(card_id, cards, with_card=True):
    state = board("garrison", cards)
    state.players["p1"].hand = ["garrison", "blast", "blast"]
    if not with_card:
        return state
    card = cards.get(card_id)
    if card.type == "countermeasure":
        state.players["p1"].active_countermeasures.append({"card_id": card_id, "activated_turn": 5, "paid": 0})
        state.current_player = "p2"
        return state
    if card.is_unit:
        from simulator.core.state import UnitState
        unit = UnitState("PROBE", card_id, card.attack, card.defense, "p1", "support_line")
        state.players["p1"].units.append(unit)
        state.battlefield["support_line"].append("PROBE")
        engine_for(cards).execute(card_id, "on_deploy", state,
                                  __import__("simulator.effects.resolver", fromlist=["EffectContext"]).EffectContext(
                                      "p1", card_id, "PROBE", event="on_deploy"))
    else:
        return None
    return state


def probe(card_id, cards):
    engine = engine_for(cards)
    card = cards.get(card_id)
    rule = engine.rule_for(card_id)
    listens = [t for t in rule.triggers if t not in PLAY_ONLY] or ["（無觸發事件；可能是光環或常駐效果）"]
    out = [f"## {card.name} ({card_id}) {card.type}", "卡面：" + card.text.replace("\r", "").replace("\n", " / "),
           f"觸發：{listens}"]
    for label, run in SCENARIOS:
        results = []
        for with_card in (True, False):
            state = place(card_id, cards, with_card)
            if state is None:
                return None
            if card.type == "countermeasure" and with_card:
                state.current_player = "p1"
            try:
                before = snapshot(state, cards)
                run(state, cards)
                results.append((diff(before, snapshot(state, cards), cards), None))
            except Exception as error:   # noqa: BLE001
                results.append(([], repr(error)[:80]))
        (with_lines, err), (base_lines, _) = results
        if "攻擊" in label:
            base_lines = [l for l in with_lines if "frontline→support_line" in l or l.startswith("我手牌")]
        extra = [line for line in with_lines if line not in base_lines]
        missing = [line for line in base_lines if line not in with_lines]
        if err:
            if "PROBE" in label or "攻擊" in label:
                continue
            out.append(f"   {label}：例外 {err}")
        elif extra or missing:
            out.append(f"   {label}：" + "；".join(extra) + (f"  （少了：{'；'.join(missing)}）" if missing else ""))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cards", nargs="*")
    parser.add_argument("--meta", action="store_true")
    args = parser.parse_args()
    cards = CardDatabase.from_file(CATALOG)
    ids = list(args.cards)
    if args.meta:
        engine = engine_for(cards)
        seen = set()
        for deck in load_meta_decks(cards):
            for cid in deck.cards:
                if cid in seen:
                    continue
                seen.add(cid)
                card = cards.get(cid)
                rule = engine.rule_for(cid)
                if card.type == "order":
                    continue
                if card.is_unit and set(rule.triggers) <= PLAY_ONLY and "Deployment" in card.text and len(card.text) < 120 \
                        and not any(a.target == "source" or a.kind.startswith(("aura", "ground_damage")) for a in rule.actions):
                    continue
                ids.append(cid)
    for cid in ids:
        result = probe(cid, cards)
        if result:
            print(result)
            print()


if __name__ == "__main__":
    main()
