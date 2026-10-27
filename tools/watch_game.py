"""Play simulated games and write a human-readable log for rule review.

Two uses:

* Review one game by eye -- every action, the played card's printed text next to
  what the engine actually did (damage, deaths, stat changes, draws), and the
  board after each step::

      python tools/watch_game.py --p1 Germany:Italy --p2 Soviet:Britain --seed 7 --html game.html

* Sweep many games for things that should never happen (rule events the engine
  could not resolve, broken invariants) and list the cards involved::

      python tools/watch_game.py --sweep 200 --html sweep.html

Agents are cheap on purpose (no network): ``active`` takes a random non-pass
action 90% of the time so turns are full of plays and attacks, ``random`` is
uniform, ``rule`` is the upstream RuleBasedAgent.
"""
from __future__ import annotations

import argparse
import collections
import html
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.actions.action import (  # noqa: E402
    AttackAction, MoveUnitAction, MulliganAction, PassAction, PlayCardAction, UseAbilityAction,
)
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.core.turn import MAX_KREDITS  # noqa: E402
from simulator.rules.battlefield import FRONTLINE_UNIT_LIMIT, HAND_CARD_LIMIT, SUPPORT_UNIT_LIMIT  # noqa: E402
from simulator.testing import OFFICIAL_ALLY_NATIONS, OFFICIAL_MAIN_NATIONS, build_random_deck  # noqa: E402

CARD_PATH = ROOT / "data" / "source" / "kards_info_cards.json"

# Engine events that mean "a rule did not do what its card says".  Each is a
# lead worth reading, not automatically a bug.
SUSPECT_EVENTS = {
    "unsupported_effect": "效果種類沒有實作",
    "native_rule_unresolved_card_name": "卡名解析不到（產生／加入的卡找不到）",
    "native_rule_reference_unresolved": "數值引用解析不到",
    "native_rule_deploy_failed": "部署失敗",
    "native_rule_stale_reference": "引用的單位已不存在",
    "native_rule_retreat_failed": "撤退失敗",
    "native_rule_undamaged_destroy_failed": "「未受傷才消滅」失敗",
    "spawn_failed": "生成單位失敗",
    "custom_deploy_failed": "特殊部署失敗",
    "copy_failed": "複製失敗",
    "choice_required": "需要玩家選擇但沒有選",
    "native_rule_condition_unmodeled": "條件看不懂（當成不成立）",
}
# Shown in the per-step effect list; everything else is engine bookkeeping.
SHOWN_EVENTS = {
    "damage_dealt", "unit_died", "hq_damaged", "card_drawn", "card_discarded", "unit_deployed", "unit_repaired",
    "native_rule_executed", "native_rule_condition_false", "countermeasure_activated", "countermeasure_deactivated",
    "intel_revealed", "ambush_destroyed_attacker", "action_cancelled", "deployment_effect_cancelled",
    "deployment_effect_suppressed", "damage_ignored_immune", "covert_effect_ignored", "on_weather_played",
    "hand_cost_modified", "op_cost_rule_added", *SUSPECT_EVENTS,
}


# ── agents ──────────────────────────────────────────────────────────────────
def pick_action(kind: str, legal: list, rng: random.Random):
    if kind == "random":
        return rng.choice(legal)
    non_pass = [a for a in legal if not isinstance(a, PassAction)]
    if kind == "rule":
        hq = next((a for a in non_pass if isinstance(a, AttackAction) and a.target_unit_id is None), None)
        play = next((a for a in non_pass if isinstance(a, PlayCardAction)), None)
        return hq or play or (non_pass[0] if non_pass else legal[0])
    if non_pass and rng.random() < 0.9:
        return rng.choice(non_pass)
    return next((a for a in legal if isinstance(a, PassAction)), rng.choice(legal))


# ── description helpers ─────────────────────────────────────────────────────
class Namer:
    def __init__(self, cards: CardDatabase) -> None:
        self.cards = cards

    def card(self, card_id: str | None) -> str:
        if card_id is None:
            return "?"
        return self.cards.get(card_id).name if card_id in self.cards else card_id

    def unit(self, state, unit_id: str | None, fallback: dict | None = None) -> str:
        if unit_id is None:
            return "HQ"
        for player in state.players.values():
            for unit in player.units:
                if unit.instance_id == unit_id:
                    return f"{self.card(unit.card_id)}〔{unit.owner_id} {unit.attack}/{unit.defense}〕"
        if fallback and unit_id in fallback:
            return fallback[unit_id]
        return unit_id

    def action(self, state, action) -> str:
        if isinstance(action, PlayCardAction):
            text = f"打出 {self.card(action.card_id)}"
            if action.target_unit_id:
                text += f" → 目標 {self.unit(state, action.target_unit_id)}"
            if action.selected_option:
                text += f"（選項：{action.selected_option}）"
            if action.selected_card_id:
                text += f"（選牌：{self.card(action.selected_card_id)}）"
            if self.cards.get(action.card_id).is_unit:
                text += f"〔{ {'support_line': '支援線', 'frontline': '前線'}.get(action.position, action.position) }〕"
            return text
        if isinstance(action, AttackAction):
            return f"{self.unit(state, action.attacker_id)} 攻擊 {self.unit(state, action.target_unit_id)}"
        if isinstance(action, MoveUnitAction):
            return f"{self.unit(state, action.unit_id)} 移動到前線"
        if isinstance(action, UseAbilityAction):
            return f"{self.unit(state, action.unit_id)} 使用能力 {action.ability_id}"
        if isinstance(action, MulliganAction):
            return f"換牌 {len(action.card_ids)} 張"
        if isinstance(action, PassAction):
            return "結束回合"
        return type(action).__name__

    def event(self, event: dict, names: dict) -> str:
        kind = event.get("event")
        who = event.get("player_id", "")
        if kind == "damage_dealt":
            return f"{names.get(event.get('target_id'), event.get('target_id'))} 受到 {event.get('amount')} 傷害"
        if kind == "unit_died":
            return f"{who} 的 {self.card(event.get('card_id'))} 被消滅"
        if kind == "hq_damaged":
            amount = event.get("amount", event.get("damage", "?"))
            return f"{who} HQ 受到 {amount} 傷害"
        if kind == "card_drawn":
            return f"{who} 抽到 {self.card(event.get('card_id'))}"
        if kind == "card_discarded":
            return f"{who} 棄掉 {self.card(event.get('card_id'))}"
        if kind == "unit_deployed":
            return f"{who} 部署 {self.card(event.get('card_id'))}"
        if kind == "native_rule_executed":
            return f"規則執行：{self.card(event.get('card_id'))}（{event.get('trigger')}，{event.get('status')}）"
        if kind == "native_rule_condition_false":
            return f"條件不成立，跳過 {event.get('kind')}（{event.get('condition')}）"
        if kind in SUSPECT_EVENTS:
            source = event.get("source_card_id") or event.get("card_id")
            detail = {k: v for k, v in event.items() if k not in {"event", "source_card_id", "card_id"}}
            return f"⚠ {SUSPECT_EVENTS[kind]}：{self.card(source)} {detail or ''}"
        detail = {k: v for k, v in event.items() if k != "event"}
        return f"{kind} {detail}"


def unit_snapshot(state) -> dict[str, tuple]:
    return {u.instance_id: (u.card_id, u.owner_id, u.attack, u.defense, u.position)
            for p in state.players.values() for u in p.units}


def unit_names(state, namer: Namer) -> dict[str, str]:
    return {u.instance_id: f"{namer.card(u.card_id)}〔{u.owner_id}〕" for p in state.players.values() for u in p.units}


def stat_changes(before: dict, after: dict, namer: Namer, acting_unit_ids: set) -> list[str]:
    lines = []
    for uid, (card_id, owner, atk, dfn, pos) in after.items():
        if uid not in before:
            continue
        _, _, atk0, dfn0, pos0 = before[uid]
        parts = []
        if atk != atk0:
            parts.append(f"攻 {atk0}→{atk}")
        if dfn != dfn0:
            parts.append(f"防 {dfn0}→{dfn}")
        if pos != pos0 and uid not in acting_unit_ids:
            parts.append(f"位置 {pos0}→{pos}")
        if parts:
            lines.append(f"{namer.card(card_id)}〔{owner}〕 " + "，".join(parts))
    return lines


def invariant_problems(state) -> list[str]:
    problems = []
    for pid, player in state.players.items():
        if len(player.hand) > HAND_CARD_LIMIT:
            problems.append(f"{pid} 手牌 {len(player.hand)} 張，超過上限 {HAND_CARD_LIMIT}")
        res = player.resources
        if res.kredits < 0:
            problems.append(f"{pid} kredits 為負：{res.kredits}")
        if res.kredits > max(res.max_kredits, MAX_KREDITS) + 5:
            problems.append(f"{pid} kredits {res.kredits} 遠超過上限 {res.max_kredits}")
        if player.hq.current_health > player.hq.max_health:
            problems.append(f"{pid} HQ 血量 {player.hq.current_health} 超過上限 {player.hq.max_health}")
        support = [u for u in player.units if u.position == "support_line"]
        if len(support) > SUPPORT_UNIT_LIMIT:
            problems.append(f"{pid} 支援線 {len(support)} 個單位，超過 {SUPPORT_UNIT_LIMIT}")
        for unit in player.units:
            if unit.defense <= 0:
                problems.append(f"{pid} 的 {unit.card_id} 防禦 {unit.defense} 卻還在場上")
            if unit.attack < 0:
                problems.append(f"{pid} 的 {unit.card_id} 攻擊力為負：{unit.attack}")
    front = [u for p in state.players.values() for u in p.units if u.position == "frontline"]
    if len(front) > FRONTLINE_UNIT_LIMIT:
        problems.append(f"前線 {len(front)} 個單位，超過 {FRONTLINE_UNIT_LIMIT}")
    owners = {u.owner_id for u in front}
    if len(owners) > 1:
        problems.append("前線同時有雙方單位")
    return problems


# ── one game ────────────────────────────────────────────────────────────────
@dataclass
class Step:
    turn: int
    player: str
    action: str
    card_text: str = ""
    effects: list[str] = field(default_factory=list)
    suspects: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    board: str = ""


@dataclass
class GameResult:
    seed: int
    decks: tuple[str, str]
    status: str
    turns: int
    steps: list[Step]
    suspect_cards: collections.Counter
    problem_count: int


def board_text(state, namer: Namer) -> str:
    rows = []
    for pid, player in state.players.items():
        units = lambda pos: "、".join(f"{namer.card(u.card_id)} {u.attack}/{u.defense}" for u in player.units if u.position == pos) or "—"
        rows.append(f"{pid}｜HQ {player.hq.current_health}/{player.hq.max_health}｜K {player.resources.kredits}/{player.resources.max_kredits}"
                    f"｜手牌 {len(player.hand)}｜牌庫 {len(player.deck)}\n   前線：{units('frontline')}\n   支援：{units('support_line')}")
    return "\n".join(rows)


def play_game(cards: CardDatabase, namer: Namer, seed: int, p1: tuple, p2: tuple, agents: tuple[str, str],
              max_actions: int, keep_steps: bool) -> GameResult:
    rng = random.Random(seed)
    decks = [build_random_deck(cards, seed=rng.randrange(2**31), main_nation=nat, ally_nation=ally) for nat, ally in (p1, p2)]
    env = Simulator(cards)
    env.reset(*decks, nations=(p1[0], p2[0]), ally_nations=(p1[1], p2[1]), seed=rng.randrange(2**31))
    steps: list[Step] = []
    suspect_cards: collections.Counter = collections.Counter()
    problem_count = 0
    for _ in range(max_actions):
        if env.is_terminal():
            break
        state = env.state
        player_id = state.current_player
        legal = env.get_available_actions()
        action = pick_action(agents[0] if player_id == "p1" else agents[1], legal, rng)
        names_before = unit_names(state, namer)
        units_before = unit_snapshot(state)
        description = namer.action(state, action)
        card_text = ""
        if isinstance(action, PlayCardAction) and action.card_id in cards:
            card_text = cards.get(action.card_id).text.replace("\r\n", " ").replace("\n", " ")
        turn = state.turn_number
        log_start = len(state.event_log)
        env.step(action)
        state = env.state
        names = {**names_before, **unit_names(state, namer)}
        new_events = state.event_log[log_start:]
        effects, suspects = [], []
        for event in new_events:
            kind = event.get("event")
            if kind in SUSPECT_EVENTS:
                source = event.get("source_card_id") or event.get("card_id") or (action.card_id if isinstance(action, PlayCardAction) else "?")
                suspect_cards[(kind, source)] += 1
                suspects.append(namer.event(event, names))
            elif kind in SHOWN_EVENTS:
                effects.append(namer.event(event, names))
        acting = {getattr(action, "attacker_id", None), getattr(action, "unit_id", None)}
        effects += stat_changes(units_before, unit_snapshot(state), namer, acting)
        problems = invariant_problems(state)
        problem_count += bool(problems)
        if keep_steps or suspects or problems:
            steps.append(Step(turn, player_id, description, card_text, effects, suspects, problems,
                              board_text(state, namer) if keep_steps else ""))
    state = env.state
    return GameResult(seed, (f"{p1[0]}+{p1[1]}", f"{p2[0]}+{p2[1]}"), state.game_status.value, state.turn_number,
                      steps, suspect_cards, problem_count)


# ── output ──────────────────────────────────────────────────────────────────
CSS = """
:root{--bg:#fbfaf7;--fg:#1f2328;--muted:#6b7280;--line:#e5e2db;--card:#fff;--warn:#b45309;--warnbg:#fff7ed;--bad:#b91c1c;--badbg:#fef2f2;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--fg:#e6e6e6;--muted:#9aa1ab;--line:#2c3036;--card:#1e2126;--warn:#fbbf24;--warnbg:#2a2213;--bad:#f87171;--badbg:#2b1717;--accent:#93b4ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,"Noto Sans TC",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.meta{color:var(--muted);font-size:13px}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:6px 12px}
summary{cursor:pointer;font-weight:600}
.step{border-top:1px solid var(--line);padding:8px 0}
.step:first-of-type{border-top:0}
.act{font-weight:600}.p1{color:var(--accent)}
.text{color:var(--muted);font-size:13px;margin:2px 0 4px}
ul{margin:2px 0 2px 18px;padding:0}li{margin:1px 0}
.sus{background:var(--warnbg);color:var(--warn);border-radius:6px;padding:4px 8px;margin:4px 0}
.bad{background:var(--badbg);color:var(--bad);border-radius:6px;padding:4px 8px;margin:4px 0}
pre{font:12px/1.45 ui-monospace,Consolas,monospace;color:var(--muted);white-space:pre-wrap;margin:4px 0 0}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
"""


def e(text) -> str:
    return html.escape(str(text))


def step_html(step: Step) -> str:
    parts = [f'<div class="step"><div class="act"><span class="{step.player}">{e(step.player)}</span>　{e(step.action)}</div>']
    if step.card_text:
        parts.append(f'<div class="text">卡面：{e(step.card_text)}</div>')
    if step.effects:
        parts.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in step.effects) + "</ul>")
    parts += [f'<div class="sus">{e(x)}</div>' for x in step.suspects]
    parts += [f'<div class="bad">✖ {e(x)}</div>' for x in step.problems]
    if step.board:
        parts.append(f"<pre>{e(step.board)}</pre>")
    return "".join(parts) + "</div>"


def game_html(game: GameResult) -> str:
    by_turn: dict[int, list[Step]] = collections.defaultdict(list)
    for step in game.steps:
        by_turn[step.turn].append(step)
    blocks = []
    for turn, steps in by_turn.items():
        flag = " ⚠" if any(s.suspects for s in steps) else ""
        flag += " ✖" if any(s.problems for s in steps) else ""
        blocks.append(f"<details{' open' if turn <= 2 or flag else ''}><summary>第 {turn} 回合（{steps[0].player}）{flag}</summary>"
                      + "".join(step_html(s) for s in steps) + "</details>")
    return "".join(blocks)


def write_html(path: Path, title: str, summary: str, body: str) -> None:
    path.write_text(f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title><style>{CSS}</style></head>
<body><main><h1>{e(title)}</h1><div class="meta">{summary}</div>{body}</main></body></html>""", encoding="utf-8")


def parse_side(value: str) -> tuple[str, str | None]:
    nation, _, ally = value.partition(":")
    return nation, (ally or None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p1", default="Germany:Italy", help="主國:盟國，例如 Germany:Italy")
    parser.add_argument("--p2", default="Soviet:Britain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", default="active,active", help="p1,p2 各用 active / random / rule")
    parser.add_argument("--max-actions", type=int, default=600)
    parser.add_argument("--sweep", type=int, default=0, help="改成跑 N 局，只彙整可疑事件與違規（隨機國家組合）")
    parser.add_argument("--html", type=Path, help="輸出 HTML 檔")
    args = parser.parse_args()

    cards = CardDatabase.from_file(CARD_PATH)
    namer = Namer(cards)
    agents = tuple(args.agents.split(","))

    if not args.sweep:
        game = play_game(cards, namer, args.seed, parse_side(args.p1), parse_side(args.p2), agents, args.max_actions, True)
        summary = (f"{e(game.decks[0])}（p1）對 {e(game.decks[1])}（p2）｜seed {game.seed}｜結果 {e(game.status)}｜"
                   f"{game.turns} 回合｜可疑事件 {sum(game.suspect_cards.values())}｜違規步數 {game.problem_count}")
        if args.html:
            write_html(args.html, "KARDS 模擬對戰紀錄", summary, game_html(game))
        else:
            for step in game.steps:
                print(f"[T{step.turn} {step.player}] {step.action}")
                if step.card_text:
                    print(f"    卡面：{step.card_text}")
                for line in step.effects + [f"⚠ {s}" for s in step.suspects] + [f"✖ {p}" for p in step.problems]:
                    print(f"    · {line}")
            print(summary)
        return

    rng = random.Random(args.seed)
    totals: collections.Counter = collections.Counter()
    games_with: collections.Counter = collections.Counter()
    examples: dict = {}
    problem_games, statuses, crashes = [], collections.Counter(), []
    for index in range(args.sweep):
        nat1, nat2 = rng.sample(OFFICIAL_MAIN_NATIONS, 2)
        side1 = (nat1, rng.choice([n for n in OFFICIAL_ALLY_NATIONS if n != nat1]))
        side2 = (nat2, rng.choice([n for n in OFFICIAL_ALLY_NATIONS if n != nat2]))
        seed = rng.randrange(2**31)
        try:
            game = play_game(cards, namer, seed, side1, side2, agents, args.max_actions, False)
        except Exception as error:  # a crash is the strongest finding of all
            crashes.append((seed, side1, side2, repr(error)))
            continue
        statuses[game.status] += 1
        totals.update(game.suspect_cards)
        games_with.update(set(game.suspect_cards))
        for key in game.suspect_cards:
            examples.setdefault(key, (seed, side1, side2))
        if game.problem_count:
            problem_games.append(game)
        print(f"\r{index + 1}/{args.sweep}", end="", file=sys.stderr)
    print(file=sys.stderr)

    rows = []
    for (kind, card_id), count in totals.most_common():
        seed, s1, s2 = examples[(kind, card_id)]
        rows.append((SUSPECT_EVENTS[kind], namer.card(card_id), card_id,
                     cards.get(card_id).text.replace("\n", " ") if card_id in cards else "", count, games_with[(kind, card_id)],
                     f"--seed {seed} --p1 {s1[0]}:{s1[1]} --p2 {s2[0]}:{s2[1]}"))
    summary = (f"{args.sweep} 局｜結果 {dict(statuses)}｜當掉 {len(crashes)} 局｜出現違規的局 {len(problem_games)}｜"
               f"可疑事件種類 {len(rows)}")
    if not args.html:
        print(summary)
        for row in rows:
            print(" | ".join(map(str, row[:2] + row[4:])))
        for crash in crashes:
            print("CRASH", crash)
        for game in problem_games[:10]:
            print("PROBLEM seed", game.seed, game.decks, next((s.problems for s in game.steps if s.problems), []))
        return
    body = "<h2>當掉的局</h2>" + ("<table><tr><th>seed</th><th>雙方</th><th>錯誤</th></tr>" + "".join(
        f"<tr><td>{s}</td><td>{e(a)} / {e(b)}</td><td>{e(err)}</td></tr>" for s, a, b, err in crashes) + "</table>" if crashes else "<p>沒有。</p>")
    body += "<h2>可疑事件（依出現次數）</h2><table><tr><th>類型</th><th>卡</th><th>卡面</th><th>次數／局數</th><th>重現</th></tr>" + "".join(
        f"<tr><td>{e(r[0])}</td><td>{e(r[1])}<br><span class=meta>{e(r[2])}</span></td><td class=text>{e(r[3])}</td>"
        f"<td>{r[4]}／{r[5]}</td><td><code>{e(r[6])}</code></td></tr>" for r in rows) + "</table>"
    body += "<h2>違規（狀態不該出現）</h2>"
    if problem_games:
        for game in problem_games[:20]:
            body += f"<details><summary>seed {game.seed}｜{e(game.decks[0])} 對 {e(game.decks[1])}</summary>" + "".join(
                step_html(s) for s in game.steps if s.problems) + "</details>"
    else:
        body += "<p>沒有。</p>"
    write_html(args.html, "KARDS 批次模擬檢查", summary, body)
    print(summary)


if __name__ == "__main__":
    main()
