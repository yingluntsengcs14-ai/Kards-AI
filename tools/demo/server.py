"""KARDS AI 復盤／分析介面（期末成果展示用）。

在瀏覽器裡操作一局 KARDS 對局：
* 任一方都可以由人手動選動作（模擬「對手如果這樣出」），或交給 AI。
* 「AI 建議」會針對目前輪到的一方跑搜尋，列出每個合法動作的
  搜尋次數佔比、預估勝率、網路先驗與人工盤面評分，說明 AI 為什麼這樣出。
* 時間軸上每一步都可以點回去，從那一步改走別的動作（分支復盤）。
* 對局紀錄可以存成 JSON，之後再載入重看。

卡牌名稱用 kards.com 官方繁體中文（data/source/card_names_zh.json）。

執行（在 repo 根目錄）::

    python tools/demo/server.py --model runs/champion.pt

然後瀏覽器開 http://127.0.0.1:8770 。沒有 --model 時只能用不需要網路的 AI
（人工評分 MCTS、貪婪 AI、rule）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai.agents import RuleBasedAgent  # noqa: E402
from ai.heuristic import action_scores, evaluate as heuristic_value  # noqa: E402
from simulator.actions.action import (  # noqa: E402
    AttackAction, MoveUnitAction, MulliganAction, PassAction, PlayCardAction, UseAbilityAction,
)
from simulator.cards.deck_code import load_meta_decks  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402

CARDS_PATH = ROOT / "data/source/kards_info_cards.json"
NAMES_PATH = ROOT / "data/source/card_names_zh.json"
PAGE = Path(__file__).with_name("index.html")

NATION_ZH = {"Britain": "英國", "Germany": "德國", "Soviet": "蘇聯", "USA": "美國", "Japan": "日本",
             "France": "法國", "Italy": "義大利", "Poland": "波蘭", "Finland": "芬蘭", "Anzac": "澳紐",
             "ANZAC": "澳紐", "Neutral": "中立"}
TYPE_ZH = {"infantry": "步兵", "tank": "坦克", "artillery": "火砲", "fighter": "戰鬥機", "bomber": "轟炸機",
           "order": "指令", "countermeasure": "反制"}
POSITION_ZH = {"frontline": "前線", "support_line": "支援線"}

AI_KINDS = {
    "heuristic_mcts": "人工評分 MCTS",
    "net_mcts": "網路＋人工評分 MCTS",
    "net": "網路直接出牌（不搜尋）",
    "greedy": "貪婪 AI（往前看一步）",
    "rule": "規則型 AI",
}


class Catalog:
    def __init__(self) -> None:
        self.cards = CardDatabase.from_file(CARDS_PATH)
        names = json.loads(NAMES_PATH.read_text(encoding="utf-8"))["cards"] if NAMES_PATH.exists() else {}
        self.names = names

    def zh(self, card_id: str | None) -> str:
        if not card_id:
            return "?"
        entry = self.names.get(card_id)
        if entry:
            return entry["zh"]
        return self.cards.get(card_id).name if card_id in self.cards else card_id

    def info(self, card_id: str) -> dict:
        card = self.cards.get(card_id) if card_id in self.cards else None
        entry = self.names.get(card_id, {})
        return {
            "id": card_id, "zh": self.zh(card_id), "en": entry.get("en") or (card.name if card else card_id),
            "text": entry.get("text_zh") or (card.text if card else ""),
            "type": TYPE_ZH.get(card.type, card.type) if card else "", "kredits": card.kredits if card else None,
            "attack": card.attack if card else None, "defense": card.defense if card else None,
            "nation": NATION_ZH.get(card.nation, card.nation) if card else "",
        }


# ── one game session with a snapshot per step (for rewinding) ─────────────
class Session:
    def __init__(self, catalog: Catalog, deck_names: tuple[str, str], seed: int) -> None:
        self.catalog = catalog
        decks = {deck.name: deck for deck in load_meta_decks(catalog.cards)}
        self.decks = (decks[deck_names[0]], decks[deck_names[1]])
        self.seed = seed
        env = Simulator(catalog.cards, record_replay=False)
        env.reset(self.decks[0].cards, self.decks[1].cards,
                  nations=(self.decks[0].main_nation, self.decks[1].main_nation),
                  ally_nations=(self.decks[0].ally_nation, self.decks[1].ally_nation), seed=seed)
        self.env = env
        self.snapshots = [env.state.clone()]   # state before step i
        self.history: list[dict] = []          # step i: who, action text, legal index, effects

    def label(self, pid: str) -> str:
        deck = self.decks[0 if pid == "p1" else 1]
        return f"{'P1' if pid == 'p1' else 'P2'}（{deck.name}）"

    # unit / action naming
    def _unit(self, state, unit_id: str | None) -> str:
        if unit_id is None:
            return "總部"
        for player in state.players.values():
            for unit in player.units:
                if unit.instance_id == unit_id:
                    side = "P1" if unit.owner_id == "p1" else "P2"
                    return f"{side} {self.catalog.zh(unit.card_id)}（{unit.attack}/{unit.defense}）"
        return unit_id

    def describe(self, state, action) -> str:
        z = self.catalog.zh
        if isinstance(action, PlayCardAction):
            text = f"打出「{z(action.card_id)}」"
            card = self.catalog.cards.get(action.card_id) if action.card_id in self.catalog.cards else None
            if card is not None and card.type not in ("order", "countermeasure"):
                text += f" 到{POSITION_ZH.get(action.position, action.position)}"
            if action.target_unit_id:
                text += f"，目標：{self._unit(state, action.target_unit_id)}"
            if action.selected_card_id:
                text += f"，選擇：{z(action.selected_card_id)}"
            if action.selected_option:
                text += f"，選項：{action.selected_option}"
            return text
        if isinstance(action, AttackAction):
            target = "敵方總部" if action.target_unit_id is None else self._unit(state, action.target_unit_id)
            return f"{self._unit(state, action.attacker_id)} 攻擊 {target}"
        if isinstance(action, MoveUnitAction):
            return f"{self._unit(state, action.unit_id)} 移動"
        if isinstance(action, UseAbilityAction):
            return f"{self._unit(state, action.unit_id)} 使用能力"
        if isinstance(action, MulliganAction):
            return "換牌"
        if isinstance(action, PassAction):
            return "結束回合"
        return type(action).__name__

    @staticmethod
    def kind(action) -> str:
        return {PlayCardAction: "play", AttackAction: "attack", MoveUnitAction: "move",
                UseAbilityAction: "ability", PassAction: "pass"}.get(type(action), "other")

    def legal(self):
        return self.env.get_available_actions()

    def _effects(self, before, after) -> list[str]:
        lines = []
        for pid in ("p1", "p2"):
            b, a = before.players[pid], after.players[pid]
            if a.hq.current_health != b.hq.current_health:
                delta = a.hq.current_health - b.hq.current_health
                lines.append(f"{self.label(pid)} 總部 {'+' if delta > 0 else ''}{delta}（剩 {a.hq.current_health}）")
            old = {u.instance_id: u for u in b.units}; new = {u.instance_id: u for u in a.units}
            for uid, unit in old.items():
                if uid not in new:
                    lines.append(f"{self.label(pid)} 的 {self.catalog.zh(unit.card_id)} 被摧毀")
                elif (new[uid].attack, new[uid].defense) != (unit.attack, unit.defense):
                    lines.append(f"{self.catalog.zh(unit.card_id)}：{unit.attack}/{unit.defense} → {new[uid].attack}/{new[uid].defense}")
                elif new[uid].position != unit.position:
                    lines.append(f"{self.catalog.zh(unit.card_id)} 移到{POSITION_ZH.get(new[uid].position, new[uid].position)}")
            for uid, unit in new.items():
                if uid not in old:
                    lines.append(f"{self.label(pid)} 部署 {self.catalog.zh(unit.card_id)}（{unit.attack}/{unit.defense}）")
            if len(a.hand) > len(b.hand) and before.current_player == after.current_player:
                lines.append(f"{self.label(pid)} 手牌 +{len(a.hand) - len(b.hand)}")
        if after.current_player != before.current_player:
            lines.append(f"輪到 {self.label(after.current_player)}，第 {after.turn_number} 回合")
        return lines

    def play(self, index: int, by: str = "手動") -> dict:
        legal = self.legal()
        if not 0 <= index < len(legal):
            raise ValueError("動作編號超出範圍")
        action = legal[index]
        before = self.env.state.clone(); pid = before.current_player
        text = self.describe(before, action)
        self.env.step_fast(action)
        self.history.append({"player": pid, "label": self.label(pid), "action": text, "index": index, "by": by,
                             "turn": before.turn_number, "effects": self._effects(before, self.env.state)})
        self.snapshots.append(self.env.state.clone())
        return self.history[-1]

    def goto(self, step: int) -> None:
        if not 0 <= step < len(self.snapshots):
            raise ValueError("步數超出範圍")
        self.snapshots = self.snapshots[: step + 1]; self.history = self.history[:step]
        self.env = Simulator(self.catalog.cards, state=self.snapshots[-1].clone(), record_replay=False)

    # ── free play: edit a side's hand / kredits (simulate any opponent move) ──
    def edit(self, op: dict, by: str = "自由編輯") -> dict:
        """Apply a hand/kredit edit as its own timeline step.

        ``add``: put a card into ``player``'s hand (one copy is taken out of
        the deck when the deck holds it, so the deck stays consistent).
        ``remove``: take the hand card at ``slot`` back to the bottom of the deck.
        ``clear``: return the whole hand to the deck.
        ``kredits``: set current kredits (0..max 12).
        """
        state = self.env.state; pid = op["player"]; player = state.players[pid]
        kind = op["kind"]; z = self.catalog.zh
        if kind == "add":
            card_id = op["card_id"]
            if card_id not in self.catalog.cards:
                raise ValueError(f"沒有這張卡：{card_id}")
            if card_id in player.deck:
                player.deck.remove(card_id)
            player.hand.append(card_id); text = f"【自由出牌】{self.label(pid)} 手牌加入「{z(card_id)}」"
        elif kind == "remove":
            card_id = player.hand.pop(int(op["slot"])); player.deck.append(card_id)
            text = f"【自由出牌】{self.label(pid)} 手牌移除「{z(card_id)}」"
        elif kind == "clear":
            player.deck.extend(player.hand); count = len(player.hand); player.hand.clear()
            text = f"【自由出牌】{self.label(pid)} 清空手牌（{count} 張放回牌庫）"
        elif kind == "kredits":
            value = max(0, min(12, int(op["value"])))
            player.resources.kredits = value
            player.resources.max_kredits = max(player.resources.max_kredits, value)
            text = f"【自由出牌】{self.label(pid)} 指揮點設為 {value}"
        else:
            raise ValueError(f"未知的編輯：{kind}")
        self.env._action_cache = None
        self.history.append({"player": pid, "label": self.label(pid), "action": text, "index": None, "op": op,
                             "by": by, "turn": state.turn_number, "effects": []})
        self.snapshots.append(self.env.state.clone())
        return self.history[-1]

    def apply(self, entry: dict) -> None:
        if entry.get("op"):
            self.edit(entry["op"], entry.get("by", "自由編輯"))
        else:
            self.play(int(entry["index"]), entry.get("by", "手動"))

    def cards_for(self, pid: str, scope: str) -> list[dict]:
        """Cards a side could plausibly hold: its deck list, or its two nations."""
        deck = self.decks[0 if pid == "p1" else 1]
        left = self.env.state.players[pid].deck
        if scope == "deck":
            ids = sorted(set(deck.cards))
        else:
            nations = {deck.main_nation, deck.ally_nation}
            ids = sorted(c.id for c in self.catalog.cards if c.nation in nations and not getattr(c, "is_token", False)
                         and c.set not in ("OnlySpawnable",) and not c.reserved)
        rows = [{**self.catalog.info(card_id), "in_deck": left.count(card_id), "in_list": deck.cards.count(card_id)} for card_id in ids]
        return sorted(rows, key=lambda r: (r["kredits"] if r["kredits"] is not None else 99, r["zh"]))

    def record(self) -> dict:
        steps = [{"op": h["op"], "by": h["by"]} if h.get("op") else {"index": h["index"], "by": h["by"]} for h in self.history]
        return {"decks": [d.name for d in self.decks], "seed": self.seed, "steps": steps}

    def view(self, hide_hand: str | None = None) -> dict:
        state = self.env.state
        players = {}
        for pid, player in state.players.items():
            deck = self.decks[0 if pid == "p1" else 1]
            units = [{"id": u.instance_id, **self.catalog.info(u.card_id), "atk": u.attack, "def": u.defense,
                      "position": u.position, "status": {k: v for k, v in u.status.items() if isinstance(v, (bool, int, str)) and v}}
                     for u in player.units]
            players[pid] = {
                "label": self.label(pid), "deck": deck.name,
                "nation": NATION_ZH.get(deck.main_nation, deck.main_nation), "ally": NATION_ZH.get(deck.ally_nation, deck.ally_nation),
                "hq": player.hq.current_health, "hq_max": player.hq.max_health,
                "kredits": player.resources.kredits, "max_kredits": player.resources.max_kredits,
                "deck_left": len(player.deck), "hand_count": len(player.hand),
                "hand": [] if hide_hand == pid else [self.catalog.info(card_id) for card_id in player.hand],
                "units": units,
                "countermeasures": len(player.active_countermeasures),
                "heuristic": round(heuristic_value(state, pid), 3),
            }
        legal = self.legal() if not self.env.is_terminal() else []
        status = state.game_status.value
        outcome = {"player_one_won": f"{self.label('p1')} 獲勝", "player_two_won": f"{self.label('p2')} 獲勝",
                   "draw": "平手"}.get(status)
        return {
            "turn": state.turn_number, "current": state.current_player, "status": status, "outcome": outcome,
            "players": players,
            "legal": [{"index": i, "text": self.describe(state, a), "kind": self.kind(a)} for i, a in enumerate(legal)],
            "history": self.history, "step": len(self.history), "seed": self.seed,
        }


# ── AIs: choose an action and/or explain the choice ───────────────────────
class Brains:
    def __init__(self, catalog: Catalog, model_path: str | None) -> None:
        self.catalog = catalog
        from ai.observation import ObservationEncoder
        self.encoder = ObservationEncoder(catalog.cards)
        self.model = None
        if model_path and Path(model_path).exists():
            from ai.network import KARDSNet
            self.model = KARDSNet.load_checkpoint(model_path, device="cpu"); self.model.eval()
        self.model_path = model_path

    def available(self) -> dict:
        return {k: v for k, v in AI_KINDS.items() if self.model is not None or k not in ("net_mcts", "net")}

    def _net(self, state, pid, legal):
        import torch
        from ai.action_encoder import ActionEncoder
        feats, mask = ActionEncoder().encode_legal_actions(legal, pad_to_max=False)
        with torch.no_grad():
            logits, value = self.model(self.encoder.encode(state, pid), feats, mask)
        probs = torch.softmax(logits[0][: len(legal)], dim=0).tolist()
        return probs, float(value.reshape(-1)[0])

    def analyse(self, session: Session, kind: str, simulations: int = 64) -> dict:
        """Per-action explanation for the side to move."""
        state = session.env.state; pid = state.current_player; legal = session.legal()
        if not legal:
            return {"rows": [], "best": None}
        t = time.time()
        heur = action_scores(state, pid, legal, self.catalog.cards)
        rows = [{"index": i, "text": session.describe(state, a), "kind": session.kind(a),
                 "heuristic": None if h == -math.inf else round(h, 3)} for i, (a, h) in enumerate(zip(legal, heur))]
        summary = {"ai": AI_KINDS.get(kind, kind), "player": session.label(pid)}
        if kind in ("heuristic_mcts", "net_mcts"):
            from ai.mcts import MCTS
            model = self.model if kind == "net_mcts" else None
            search = MCTS(model, self.encoder, simulations=simulations, seed=session.seed + len(session.history))
            if kind == "net_mcts":
                search.heuristic_value_weight, search.heuristic_prior_weight = 0.3, 0.5
            else:  # no network: heuristic priors and values (84% vs rule at 32 simulations)
                search.heuristic_value_weight = 1.0
            action, policy = search.search(state, self.catalog.cards, pid)
            root = search.last_root
            for i, row in enumerate(rows):
                child = root.children.get(i)
                row["visits"] = policy.get(str(i), 0.0)
                row["prior"] = round(root.priors[i], 3) if i < len(root.priors) else None
                if child is not None and child.visit_count:
                    q = child.value if child.player_to_move == pid else -child.value
                    row["winrate"] = round((q + 1) / 2, 3)
            best = legal.index(action) if action in legal else max(range(len(rows)), key=lambda i: rows[i].get("visits", 0))
            summary["winrate"] = round((root.value + 1) / 2, 3) if root.visit_count else None
            summary["simulations"] = simulations
        elif kind == "net":
            probs, value = self._net(state, pid, legal)
            for row, p in zip(rows, probs):
                row["prior"] = round(p, 3)
            best = max(range(len(rows)), key=lambda i: probs[i])
            summary["winrate"] = round((value + 1) / 2, 3)
        elif kind == "greedy":
            best = max(range(len(rows)), key=lambda i: heur[i])
            summary["winrate"] = round((heuristic_value(state, pid) + 1) / 2, 3)
        else:  # rule
            choice = RuleBasedAgent().select_action({}, legal)
            best = legal.index(choice)
        summary["seconds"] = round(time.time() - t, 2)
        order = sorted(rows, key=lambda r: (-(r.get("visits") or 0), -(r.get("prior") or 0), -(r["heuristic"] if r["heuristic"] is not None else -9)))
        return {"rows": order, "best": best, "summary": summary}


# ── HTTP plumbing ─────────────────────────────────────────────────────────
class App:
    def __init__(self, model_path: str | None) -> None:
        self.catalog = Catalog(); self.brains = Brains(self.catalog, model_path)
        self.lock = threading.Lock(); self.session: Session | None = None

    def handle(self, path: str, body: dict) -> dict:
        with self.lock:
            if path == "/api/meta":
                return {"decks": [{"name": d.name, "nation": NATION_ZH.get(d.main_nation, d.main_nation),
                                   "ally": NATION_ZH.get(d.ally_nation, d.ally_nation)} for d in load_meta_decks(self.catalog.cards)],
                        "ais": self.brains.available(), "model": self.brains.model_path if self.brains.model else None,
                        "names": len(self.catalog.names)}
            if path == "/api/new":
                self.session = Session(self.catalog, (body["p1"], body["p2"]), int(body.get("seed", 1)))
                return self.session.view(body.get("hide"))
            if self.session is None:
                raise ValueError("請先開新局或載入紀錄")
            s = self.session
            if path == "/api/view":
                return s.view(body.get("hide"))
            if path == "/api/play":
                s.play(int(body["index"]), body.get("by", "手動")); return s.view(body.get("hide"))
            if path == "/api/goto":
                s.goto(int(body["step"])); return s.view(body.get("hide"))
            if path == "/api/analyse":
                return self.brains.analyse(s, body.get("ai", "heuristic_mcts"), int(body.get("simulations", 64)))
            if path == "/api/ai_step":
                # AI plays for the side to move: one action, or until its turn ends.
                kind = body.get("ai", "heuristic_mcts"); sims = int(body.get("simulations", 32))
                until_turn_end = bool(body.get("until_turn_end")); limit = int(body.get("limit", 1 if not until_turn_end else 60))
                mover = s.env.state.current_player
                for _ in range(limit):
                    if s.env.is_terminal() or s.env.state.current_player != mover:
                        break
                    result = self.brains.analyse(s, kind, sims)
                    s.play(result["best"], AI_KINDS.get(kind, kind))
                return s.view(body.get("hide"))
            if path == "/api/record":
                return s.record()
            if path == "/api/load":
                record = body["record"]
                self.session = Session(self.catalog, tuple(record["decks"]), int(record["seed"]))
                steps = record.get("steps") or [{"index": i, "by": b} for i, b in zip(record["actions"], record.get("by") or ["手動"] * len(record["actions"]))]
                for entry in steps:
                    self.session.apply(entry)
                return self.session.view(body.get("hide"))
            if path == "/api/edit":
                s.edit(body["op"]); return s.view(body.get("hide"))
            if path == "/api/apply":
                s.apply(body["entry"]); return s.view(body.get("hide"))
            if path == "/api/cards":
                return {"cards": s.cards_for(body["player"], body.get("scope", "deck"))}
            raise ValueError(f"未知的 API：{path}")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                result = app.handle(self.path, body); code = 200
            except Exception as error:  # surface engine/AI errors in the page
                result = {"error": f"{type(error).__name__}: {error}"}; code = 400
            self._send(code, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="runs/champion.pt", help="網路 checkpoint（不存在就只用不需要網路的 AI）")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = App(str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{args.port}"
    print(f"KARDS AI 復盤介面：{url}（模型：{app.brains.model_path if app.brains.model else '無'}；Ctrl+C 結束）")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
