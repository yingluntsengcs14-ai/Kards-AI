"""Check the rules that tournament decks actually use.

Reads data/decks/meta_decks.json (KARDS deck codes), then:

1. Static audit -- every card in those decks whose rule is not fully
   implemented, with the reason, ranked by how many decks run it.
2. Round robin -- every deck plays every other deck (both seats), and the
   engine events that mean "a rule did not do what its card says" are tallied
   per card, with a command that replays the game.

    python tools/meta_sweep.py --games 2 --html meta.html
"""
from __future__ import annotations

import argparse
import collections
import html
import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import watch_game as wg  # noqa: E402
from simulator.cards.deck_code import load_meta_decks  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.rules.native import engine_for  # noqa: E402

e = html.escape


def static_audit(cards, decks):
    engine = engine_for(cards)
    users = collections.defaultdict(list)
    for deck in decks:
        for card_id in sorted(set(deck.cards)):
            users[card_id].append(deck.name)
    rows = []
    for card_id, deck_names in users.items():
        rule = engine.rule_for(card_id)
        if rule.status != "implemented":
            rows.append((len(deck_names), card_id, rule, deck_names))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return rows, users


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=2, help="每個先後手組合打幾局")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=600)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()

    cards = CardDatabase.from_file(wg.CARD_PATH)
    namer = wg.Namer(cards)
    decks = load_meta_decks(cards)
    partial_rows, users = static_audit(cards, decks)

    rng = random.Random(args.seed)
    totals: collections.Counter = collections.Counter()
    games_with: collections.Counter = collections.Counter()
    example: dict = {}
    crashes, problem_games = [], []
    record = collections.Counter()
    played = 0
    pairs = list(itertools.permutations(range(len(decks)), 2))
    for index, (a, b) in enumerate(pairs):
        for _ in range(args.games):
            seed = rng.randrange(2**31)
            d1, d2 = decks[a], decks[b]
            try:
                game = wg.play_game(cards, namer, seed, (d1.main_nation, d1.ally_nation), (d2.main_nation, d2.ally_nation),
                                    ("active", "active"), args.max_actions, False, decks=(d1.cards, d2.cards))
            except Exception as error:
                crashes.append((d1.name, d2.name, seed, repr(error)))
                continue
            played += 1
            if game.status == "player_one_won":
                record[(d1.name, "win")] += 1; record[(d2.name, "loss")] += 1
            elif game.status == "player_two_won":
                record[(d2.name, "win")] += 1; record[(d1.name, "loss")] += 1
            totals.update(game.suspect_cards)
            games_with.update(set(game.suspect_cards))
            for key in game.suspect_cards:
                example.setdefault(key, (d1.name, d2.name, seed))
            if game.problem_count:
                problem_games.append((d1.name, d2.name, game))
        print("\r{0}/{1}".format(index + 1, len(pairs)), end="", file=sys.stderr)
    print(file=sys.stderr)

    in_meta = set(users)
    rows = []
    for (kind, card_id), count in totals.most_common():
        d1, d2, seed = example[(kind, card_id)]
        rows.append((wg.SUSPECT_EVENTS[kind], card_id, count, games_with[(kind, card_id)],
                     len(users.get(card_id, ())), card_id in in_meta, d1, d2, seed))
    summary = ("{0} 副牌組、{1} 局｜當掉 {2} 局｜出現違規狀態的局 {3}｜牌組用到 {4} 種卡，其中 {5} 種不是完整實作｜"
               "對戰中可疑事件 {6} 次").format(len(decks), played, len(crashes), len(problem_games), len(users),
                                         len(partial_rows), sum(totals.values()))
    print(summary)
    for count, card_id, rule, names in partial_rows[:25]:
        print("  {0:2d} 副 | {1} | {2}".format(count, namer.card(card_id), "; ".join(rule.unresolved_fragments)[:110]))
    for row in rows[:25]:
        print("  {0} | {1} | {2} 次／{3} 局 | {4} 副牌組".format(row[0], namer.card(row[1]), row[2], row[3], row[4]))
    for crash in crashes[:10]:
        print("CRASH", crash)
    if not args.html:
        return

    body = "<h2>各牌組</h2><table><tr><th>牌組</th><th>分級</th><th>國家</th><th>勝／負</th><th>未完整實作的卡</th></tr>"
    engine = engine_for(cards)
    for deck in decks:
        missing = sorted({card_id for card_id in deck.cards if engine.rule_for(card_id).status != "implemented"})
        body += "<tr><td>{0}</td><td>{1}</td><td>{2}＋{3}</td><td>{4}／{5}</td><td>{6}</td></tr>".format(
            e(deck.name), e(deck.tier), e(deck.main_nation), e(deck.ally_nation or ""),
            record[(deck.name, "win")], record[(deck.name, "loss")],
            "、".join(e(namer.card(card_id)) for card_id in missing) or "—")
    body += "</table>"
    body += "<h2>牌組裡沒有完整實作的卡（依使用的牌組數）</h2><table><tr><th>卡</th><th>卡面</th><th>原因</th><th>牌組</th></tr>"
    for count, card_id, rule, names in partial_rows:
        body += "<tr><td>{0}<br><span class=meta>{1}</span></td><td class=text>{2}</td><td>{3}</td><td>{4}（{5}）</td></tr>".format(
            e(namer.card(card_id)), e(card_id), e(cards.get(card_id).text.replace("\\n", " ")),
            "<br>".join(e(fragment) for fragment in rule.unresolved_fragments), count, e("、".join(names)))
    body += "</table><h2>對戰中的可疑事件</h2><table><tr><th>類型</th><th>卡</th><th>次數／局數</th><th>牌組數</th><th>例子</th></tr>"
    for kind, card_id, count, games, n_decks, meta_card, d1, d2, seed in rows:
        body += "<tr><td>{0}</td><td>{1}{2}</td><td>{3}／{4}</td><td>{5}</td><td>{6} 對 {7}，seed {8}</td></tr>".format(
            e(kind), e(namer.card(card_id)), "" if meta_card else "<br><span class=meta>（衍生／對手產生）</span>",
            count, games, n_decks, e(d1), e(d2), seed)
    body += "</table><h2>當掉／違規</h2>"
    body += "".join("<div class=bad>當掉：{0} 對 {1}，seed {2}：{3}</div>".format(e(a), e(b), s, e(err)) for a, b, s, err in crashes)
    for d1, d2, game in problem_games[:20]:
        body += "<details><summary>{0} 對 {1}，seed {2}</summary>{3}</details>".format(
            e(d1), e(d2), game.seed, "".join(wg.step_html(step) for step in game.steps if step.problems))
    if not crashes and not problem_games:
        body += "<p>沒有。</p>"
    wg.write_html(args.html, "主流牌組規則檢查", e(summary), body)


if __name__ == "__main__":
    main()
