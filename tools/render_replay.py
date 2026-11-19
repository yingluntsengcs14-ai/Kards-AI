"""Render a visual, step-through HTML replay of one game -- for the final
project presentation, not for rule debugging (see ``tools/watch_game.py``
for the text-log debugging tool that this script deliberately does not
reuse the look of).

Plays one game with real agents -- a trained checkpoint searched with MCTS
or PIMC, the raw policy network, or the simple random/rule baselines used
elsewhere in this repo for sanity checks -- and writes a single
self-contained HTML file. A small bit of inline JS lets the viewer step
forward/back (or autoplay) through a drawn card-table board: a health bar
and kredit count per HQ, and the frontline/support-line units laid out as
card boxes with attack/defense badges. No card artwork is embedded; boxes
are drawn (name + stats + nation colour), which also sidesteps bundling the
game's copyrighted card images into the report.

Examples::

    # Two simple baselines, so the mechanism itself can be demoed without a
    # trained checkpoint yet (useful while the desktop is still training):
    python tools/render_replay.py --p1 Germany:Italy --p2 Soviet:Britain \\
        --seed 7 --agent1 rule --agent2 random --out replay.html

    # The actual trained AI (PIMC = honest hidden-information search) vs the
    # rule baseline, using a champion checkpoint from auto_train.py:
    python tools/render_replay.py --agent1 pimc --model runs/champion.pt \\
        --agent2 rule --meta-decks --seed 7 --out replay.html
"""
from __future__ import annotations

import argparse
import html
import json
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
from simulator.cards.deck_code import load_meta_decks  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.testing import build_random_deck  # noqa: E402

CARD_PATH = ROOT / "data" / "source" / "kards_info_cards.json"

# Presentation-worthy events only -- no debug/"unmodeled rule" jargon here
# (see tools/watch_game.py's SUSPECT_EVENTS for that; this tool is for an
# audience, not for finding engine bugs).
SHOWN_EVENTS = {
    "damage_dealt": "{target} 受到 {amount} 點傷害",
    "unit_died": "{owner} 的 {card} 被摧毀",
    "hq_damaged": "{owner} 的總部受到 {amount} 點傷害",
    "unit_deployed": "{owner} 部署了 {card}",
    "card_drawn": "{owner} 抽了一張牌",
    "unit_repaired": "{owner} 的 {card} 恢復了部分防禦",
    "countermeasure_activated": "{owner} 啟用了反制卡 {card}",
    "countermeasure_deactivated": "{owner} 的反制卡 {card} 解除",
    "intel_revealed": "{owner} 偷看了對手的手牌",
}

NATION_COLORS = {
    "Britain": "#1d4ed8", "USA": "#b91c1c", "Germany": "#374151", "Soviet": "#b45309",
    "Japan": "#be123c", "Italy": "#15803d", "France": "#2563eb", "Poland": "#dc2626",
    "Finland": "#0e7490", "ANZAC": "#7c3aed", None: "#6b7280",
}


# ── naming/description helpers ───────────────────────────────────────────
class Namer:
    def __init__(self, cards: CardDatabase) -> None:
        self.cards = cards

    def card(self, card_id: str | None) -> str:
        if card_id is None:
            return "?"
        return self.cards.get(card_id).name if card_id in self.cards else card_id

    def unit(self, state, unit_id: str | None) -> str:
        if unit_id is None:
            return "總部"
        for player in state.players.values():
            for unit in player.units:
                if unit.instance_id == unit_id:
                    return self.card(unit.card_id)
        return unit_id

    def action(self, state, action) -> str:
        if isinstance(action, PlayCardAction):
            text = f"打出「{self.card(action.card_id)}」"
            if action.target_unit_id:
                text += f"，目標：{self.unit(state, action.target_unit_id)}"
            return text
        if isinstance(action, AttackAction):
            return f"{self.unit(state, action.attacker_id)} 攻擊 {self.unit(state, action.target_unit_id)}"
        if isinstance(action, MoveUnitAction):
            return f"{self.unit(state, action.unit_id)} 移動到前線"
        if isinstance(action, UseAbilityAction):
            return f"{self.unit(state, action.unit_id)} 使用能力"
        if isinstance(action, MulliganAction):
            return "換牌"
        if isinstance(action, PassAction):
            return "結束回合"
        return type(action).__name__

    def event(self, event: dict) -> str | None:
        kind = event.get("event")
        template = SHOWN_EVENTS.get(kind)
        if not template:
            return None
        owner = event.get("player_id", "")
        card = self.card(event.get("card_id"))
        target = event.get("target_id", "")
        amount = event.get("amount", event.get("damage", ""))
        return template.format(owner=owner, card=card, target=target, amount=amount)


# ── structured board snapshot (consumed by the JS renderer, not text) ────
def unit_box(unit, namer: Namer) -> dict:
    return {
        "name": namer.card(unit.card_id), "atk": unit.attack, "def": unit.defense,
        "owner": unit.owner_id, "veteran": bool(unit.status.get("veteran")),
    }


def board_snapshot(state, namer: Namer) -> dict:
    players = {}
    for pid, player in state.players.items():
        nation = player.nation
        players[pid] = {
            "nation": nation, "ally_nation": player.ally_nation, "color": NATION_COLORS.get(nation, NATION_COLORS[None]),
            "hq": {"current": player.hq.current_health, "max": player.hq.max_health},
            "kredits": {"current": player.resources.kredits, "max": player.resources.max_kredits},
            "hand": len(player.hand), "deck": len(player.deck),
            "frontline": [unit_box(u, namer) for u in player.units if u.position == "frontline"],
            "support": [unit_box(u, namer) for u in player.units if u.position == "support_line"],
        }
    return {"turn": state.turn_number, "active_player": state.current_player, "status": state.game_status.value,
            "players": players}


@dataclass
class Step:
    index: int
    player: str
    action: str
    effects: list[str] = field(default_factory=list)
    board: dict = field(default_factory=dict)


# ── agent construction ────────────────────────────────────────────────────
def build_agent(kind: str, seed: int, model_path: str | None, device: str, encoder,
                 mcts_simulations: int, pimc_worlds: int):
    """Returns (agent, needs_set_state: bool). The two baseline agents follow
    the simple style already used by tools/watch_game.py; the three
    network-driven kinds reuse ai.agents/ai.pimc exactly as selfplay.py does,
    so a replay genuinely shows what training/evaluation would have seen."""
    if kind in ("random", "rule"):
        from ai.agents import RandomAgent, RuleBasedAgent
        return (RandomAgent(seed) if kind == "random" else RuleBasedAgent()), False

    import torch
    from ai.network import KARDSNet
    model = (KARDSNet.load_checkpoint(model_path, device=device) if model_path else KARDSNet()).to(device)
    if kind == "neural":
        from ai.agents import NeuralAgent
        return NeuralAgent(model, encoder, device=device), True
    if kind == "mcts":
        from ai.agents import MCTSAgent
        return MCTSAgent(model, encoder, mcts_simulations, seed), True
    if kind == "pimc":
        from ai.pimc import PIMCAgent
        return PIMCAgent(model, encoder, worlds=pimc_worlds, simulations=mcts_simulations, seed=seed), True
    raise ValueError(f"unknown agent kind: {kind}")


def select(agent, needs_set_state: bool, env, player_id: str, cards, legal):
    if needs_set_state:
        try:
            agent.set_state(env.state, player_id, cards)
        except TypeError:
            agent.set_state(env.state, player_id)
    observation = env.get_observation(player_id)
    agent.observe(observation)
    return agent.select_action(observation, legal)


def parse_side(value: str) -> tuple[str, str | None]:
    nation, _, ally = value.partition(":")
    return nation, (ally or None)


# ── play one game, recording a Step per action ────────────────────────────
def play_and_record(cards: CardDatabase, args) -> tuple[list[Step], dict]:
    from ai.observation import ObservationEncoder
    encoder = ObservationEncoder(cards)
    namer = Namer(cards)
    rng = random.Random(args.seed)

    if args.meta_decks:
        decks_catalog = load_meta_decks(cards)
        deck_a, deck_b = rng.sample(decks_catalog, 2)
        decks = [list(deck_a.cards), list(deck_b.cards)]
        nations = (deck_a.main_nation, deck_b.main_nation)
        allies = (deck_a.ally_nation, deck_b.ally_nation)
    else:
        p1_nation, p1_ally = parse_side(args.p1)
        p2_nation, p2_ally = parse_side(args.p2)
        decks = [build_random_deck(cards, seed=rng.randrange(2 ** 31), main_nation=n, ally_nation=a)
                 for n, a in ((p1_nation, p1_ally), (p2_nation, p2_ally))]
        nations, allies = (p1_nation, p2_nation), (p1_ally, p2_ally)

    env = Simulator(cards)
    env.reset(*decks, nations=nations, ally_nations=allies, seed=rng.randrange(2 ** 31))

    agent1, needs1 = build_agent(args.agent1, args.seed + 1, args.model, args.device, encoder,
                                  args.mcts_simulations, args.pimc_worlds)
    agent2, needs2 = build_agent(args.agent2, args.seed + 2, args.model, args.device, encoder,
                                  args.mcts_simulations, args.pimc_worlds)
    agent1.reset(); agent2.reset()

    steps: list[Step] = [Step(0, "-", "開局", [], board_snapshot(env.state, namer))]
    for index in range(1, args.max_actions + 1):
        if env.is_terminal():
            break
        player_id = env.state.current_player
        agent, needs_set_state = (agent1, needs1) if player_id == "p1" else (agent2, needs2)
        legal = env.get_available_actions()
        action = select(agent, needs_set_state, env, player_id, cards, legal)
        description = namer.action(env.state, action)
        log_start = len(env.state.event_log)
        env.step_fast(action)
        new_events = env.state.event_log[log_start:]
        effects = [line for line in (namer.event(event) for event in new_events) if line]
        steps.append(Step(index, player_id, description, effects, board_snapshot(env.state, namer)))

    final_state = env.state
    outcome = {
        "player_one_won": "p1 獲勝", "player_two_won": "p2 獲勝", "draw": "平手",
    }.get(final_state.game_status.value, final_state.game_status.value)
    meta = {
        "seed": args.seed, "turns": final_state.turn_number, "outcome": outcome,
        "agent1": args.agent1, "agent2": args.agent2,
        "p1": {"nation": nations[0], "ally": allies[0]}, "p2": {"nation": nations[1], "ally": allies[1]},
    }
    return steps, meta


# ── HTML/JS output ─────────────────────────────────────────────────────────
PAGE_TEMPLATE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#0f1115;--panel:#1a1d24;--card:#242832;--line:#333844;--fg:#e8e9ec;--muted:#9aa1ad;--accent:#60a5fa;--good:#34d399;--bad:#f87171}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Noto Sans TC",sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:20px 16px 48px}}
h1{{font-size:20px;margin:0 0 4px}}
.meta{{color:var(--muted);font-size:13px;margin-bottom:16px}}
.board{{background:var(--panel);border-radius:12px;padding:16px;border:1px solid var(--line)}}
.row{{display:flex;align-items:center;gap:12px;margin:6px 0;flex-wrap:wrap}}
.hqbar{{flex:0 0 auto;min-width:220px}}
.hqlabel{{font-weight:700;font-size:14px;display:flex;align-items:center;gap:6px}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.bar{{height:10px;border-radius:5px;background:#333;overflow:hidden;margin-top:4px;width:180px}}
.bar>i{{display:block;height:100%;background:var(--good);transition:width .2s}}
.stats{{color:var(--muted);font-size:12px;margin-top:2px}}
.lane{{display:flex;gap:8px;flex-wrap:wrap;min-height:64px;align-items:center}}
.lanelabel{{color:var(--muted);font-size:11px;width:56px;flex:0 0 56px}}
.unit{{width:76px;border-radius:8px;padding:5px 6px;font-size:11px;position:relative;color:#fff;box-shadow:0 1px 3px rgba(0,0,0,.4)}}
.unit .nm{{display:block;font-weight:600;line-height:1.2;max-height:2.4em;overflow:hidden}}
.unit .ad{{display:flex;justify-content:space-between;margin-top:4px;font-size:12px;font-weight:700}}
.unit .ad .a{{color:#fde68a}}.unit .ad .d{{color:#bfdbfe}}
.unit.vet::after{{content:"★";position:absolute;top:-6px;right:-4px;color:#fbbf24;font-size:13px}}
hr.sep{{border:0;border-top:1px dashed var(--line);margin:14px 0}}
.panel{{margin-top:16px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-height:90px}}
.turnline{{font-size:12px;color:var(--muted)}}
.action{{font-size:16px;font-weight:600;margin:4px 0}}
.effects{{margin:6px 0 0 0;padding:0 0 0 18px;color:var(--muted);font-size:13px}}
.controls{{display:flex;align-items:center;gap:10px;margin-top:14px;flex-wrap:wrap}}
button{{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer}}
button:hover{{border-color:var(--accent)}}
button:disabled{{opacity:.4;cursor:default}}
input[type=range]{{flex:1;min-width:140px}}
.step-count{{color:var(--muted);font-size:13px;min-width:70px;text-align:center}}
.outcome{{font-weight:700;color:var(--accent)}}
</style></head>
<body><main>
<h1>{title}</h1>
<div class="meta">{summary}</div>
<div class="board">
  <div id="p2row"></div>
  <hr class="sep">
  <div id="p1row"></div>
</div>
<div class="panel">
  <div class="turnline" id="turnline"></div>
  <div class="action" id="actiontext"></div>
  <ul class="effects" id="effectlist"></ul>
</div>
<div class="controls">
  <button id="prev">◀ 上一步</button>
  <button id="play">▶ 播放</button>
  <button id="next">下一步 ▶</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <span class="step-count" id="count"></span>
</div>
</main>
<script id="replay-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('replay-data').textContent);
const steps = DATA.steps;
let i = 0, playing = false, timer = null;

function unitEl(u) {{
  const d = document.createElement('div');
  d.className = 'unit' + (u.veteran ? ' vet' : '');
  d.style.background = DATA.meta.colors[u.owner] || '#555';
  d.innerHTML = `<span class="nm">${{u.name}}</span><span class="ad"><span class="a">${{u.atk}}</span><span class="d">${{u.def}}</span></span>`;
  return d;
}}

function lane(label, units) {{
  const wrap = document.createElement('div');
  wrap.className = 'row';
  const l = document.createElement('div'); l.className = 'lanelabel'; l.textContent = label;
  const box = document.createElement('div'); box.className = 'lane';
  units.forEach(u => box.appendChild(unitEl(u)));
  wrap.appendChild(l); wrap.appendChild(box);
  return wrap;
}}

function renderSide(container, pid, p) {{
  container.innerHTML = '';
  const hq = document.createElement('div'); hq.className = 'row';
  const pct = Math.max(0, Math.min(100, 100 * p.hq.current / Math.max(1, p.hq.max)));
  hq.innerHTML = `<div class="hqbar">
      <div class="hqlabel"><span class="dot" style="background:${{p.color}}"></span>${{pid}}｜${{p.nation || '?'}}${{p.ally_nation ? ('+' + p.ally_nation) : ''}}</div>
      <div class="bar"><i style="width:${{pct}}%"></i></div>
      <div class="stats">總部 ${{p.hq.current}}/${{p.hq.max}}　Kredits ${{p.kredits.current}}/${{p.kredits.max}}　手牌 ${{p.hand}}　牌庫 ${{p.deck}}</div>
    </div>`;
  container.appendChild(hq);
  container.appendChild(lane('前線', p.frontline));
  container.appendChild(lane('支援線', p.support));
}}

function render() {{
  const s = steps[i];
  renderSide(document.getElementById('p2row'), 'p2', s.board.players.p2);
  renderSide(document.getElementById('p1row'), 'p1', s.board.players.p1);
  document.getElementById('turnline').textContent = `第 ${{s.board.turn}} 回合 ・ 第 ${{i}} / ${{steps.length - 1}} 步`;
  document.getElementById('actiontext').innerHTML = (s.player === '-' ? '' : `<b>${{s.player}}</b>　`) + s.action;
  const list = document.getElementById('effectlist');
  list.innerHTML = '';
  s.effects.forEach(e => {{ const li = document.createElement('li'); li.textContent = e; list.appendChild(li); }});
  document.getElementById('slider').value = i;
  document.getElementById('count').textContent = `${{i}} / ${{steps.length - 1}}`;
  document.getElementById('prev').disabled = i === 0;
  document.getElementById('next').disabled = i === steps.length - 1;
}}

document.getElementById('slider').max = steps.length - 1;
document.getElementById('prev').onclick = () => {{ i = Math.max(0, i - 1); render(); }};
document.getElementById('next').onclick = () => {{ i = Math.min(steps.length - 1, i + 1); render(); }};
document.getElementById('slider').oninput = (e) => {{ i = parseInt(e.target.value, 10); render(); }};
document.getElementById('play').onclick = () => {{
  playing = !playing;
  document.getElementById('play').textContent = playing ? '⏸ 暫停' : '▶ 播放';
  if (playing) {{
    timer = setInterval(() => {{
      if (i >= steps.length - 1) {{ playing = false; document.getElementById('play').textContent = '▶ 播放'; clearInterval(timer); return; }}
      i += 1; render();
    }}, 900);
  }} else {{ clearInterval(timer); }}
}};
render();
</script>
</body></html>"""


def render_html(steps: list[Step], meta: dict, title: str) -> str:
    summary = (f"{meta['p1']['nation']}+{meta['p1']['ally']}（p1，{meta['agent1']}）　vs　"
               f"{meta['p2']['nation']}+{meta['p2']['ally']}（p2，{meta['agent2']}）　｜　"
               f"seed {meta['seed']}　｜　{meta['turns']} 回合　｜　"
               f"<span class=\"outcome\">{html.escape(meta['outcome'])}</span>")
    data = {
        "steps": [{"player": s.player, "action": s.action, "effects": s.effects, "board": s.board} for s in steps],
        "meta": {"colors": {"p1": NATION_COLORS.get(meta["p1"]["nation"], NATION_COLORS[None]),
                             "p2": NATION_COLORS.get(meta["p2"]["nation"], NATION_COLORS[None])}},
    }
    return PAGE_TEMPLATE.format(title=html.escape(title), summary=summary,
                                 data_json=json.dumps(data, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p1", default="Germany:Italy", help="主國:盟國，例如 Germany:Italy（--meta-decks 時忽略）")
    parser.add_argument("--p2", default="Soviet:Britain")
    parser.add_argument("--meta-decks", action="store_true", help="改成從 17 副主流牌組隨機抽兩副對戰")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent1", choices=("random", "rule", "neural", "mcts", "pimc"), default="mcts")
    parser.add_argument("--agent2", choices=("random", "rule", "neural", "mcts", "pimc"), default="rule")
    parser.add_argument("--model", help="checkpoint 路徑；neural/mcts/pimc 沒給就用隨機初始化的網路（純示範用）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--pimc-worlds", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("replay.html"))
    parser.add_argument("--title", default="KARDS 對戰覆盤")
    args = parser.parse_args()

    cards = CardDatabase.from_file(CARD_PATH)
    steps, meta = play_and_record(cards, args)
    args.out.write_text(render_html(steps, meta, args.title), encoding="utf-8")
    print(f"寫入 {args.out}（{len(steps)} 步，結果：{meta['outcome']}）")


if __name__ == "__main__":
    main()
