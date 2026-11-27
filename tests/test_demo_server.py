"""Smoke tests for the replay/analysis demo (tools/demo/server.py)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _server():
    spec = importlib.util.spec_from_file_location("demo_server", ROOT / "tools/demo/server.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_every_catalogue_card_has_a_chinese_name():
    catalogue = json.loads((ROOT / "data/source/kards_info_cards.json").read_text(encoding="utf-8"))["cards"]
    names = json.loads((ROOT / "data/source/card_names_zh.json").read_text(encoding="utf-8"))["cards"]
    assert {card["id"] for card in catalogue} <= set(names)
    assert names["panzer_ivg"]["zh"] == "四號坦克 G 型"


def test_play_rewind_and_reload_a_game():
    server = _server()
    app = server.App(model_path=None)
    assert "net_mcts" not in app.handle("/api/meta", {})["ais"]
    view = app.handle("/api/new", {"p1": "日澳快攻", "p2": "德意直空", "seed": 7})
    assert view["step"] == 0 and view["legal"]
    for _ in range(4):
        view = app.handle("/api/ai_step", {"ai": "greedy", "until_turn_end": True})
    analysis = app.handle("/api/analyse", {"ai": "heuristic_mcts", "simulations": 16})
    assert analysis["best"] is not None and abs(sum(r.get("visits", 0) for r in analysis["rows"]) - 1) < 1e-6
    record = app.handle("/api/record", {})
    rewound = app.handle("/api/goto", {"step": 2})
    assert rewound["step"] == 2 and len(rewound["history"]) == 2
    reloaded = app.handle("/api/load", {"record": record})
    assert reloaded["step"] == view["step"]
    assert reloaded["players"]["p1"]["hq"] == view["players"]["p1"]["hq"]


def test_free_play_edits_are_steps_that_replay():
    server = _server()
    app = server.App(model_path=None)
    app.handle("/api/new", {"p1": "日澳快攻", "p2": "德意直空", "seed": 7})
    view = app.handle("/api/edit", {"op": {"kind": "clear", "player": "p2"}})
    assert view["players"]["p2"]["hand_count"] == 0
    app.handle("/api/ai_step", {"ai": "greedy", "until_turn_end": True})          # P1's turn
    cards = app.handle("/api/cards", {"player": "p2", "scope": "deck"})["cards"]
    pick = next(c for c in cards if c["kredits"] is not None and c["kredits"] <= 2 and c["attack"] is not None)
    app.handle("/api/edit", {"op": {"kind": "add", "player": "p2", "card_id": pick["id"]}})
    view = app.handle("/api/edit", {"op": {"kind": "kredits", "player": "p2", "value": 5}})
    play = next(a for a in view["legal"] if a["kind"] == "play" and pick["zh"] in a["text"])
    view = app.handle("/api/play", {"index": play["index"]})
    assert any(u["id"].startswith(pick["id"]) for u in view["players"]["p2"]["units"])
    assert len(app.handle("/api/cards", {"player": "p2", "scope": "nation"})["cards"]) > len(cards)
    record = app.handle("/api/record", {})
    assert any("op" in step for step in record["steps"])
    reloaded = app.handle("/api/load", {"record": record})
    assert reloaded["step"] == view["step"] and reloaded["players"]["p2"]["units"] == view["players"]["p2"]["units"]


def test_pimc_analysis_ignores_the_opponents_actual_hand():
    """With worlds > 0 the AI knows only the opponent's hand size."""
    server = _server()

    def analysis_with_p2_card(card_id):
        app = server.App(model_path=None)
        app.handle("/api/new", {"p1": "日澳快攻", "p2": "德意直空", "seed": 7})
        app.handle("/api/edit", {"op": {"kind": "clear", "player": "p2"}})
        app.handle("/api/edit", {"op": {"kind": "add", "player": "p2", "card_id": card_id}})
        return app.handle("/api/analyse", {"ai": "heuristic_mcts", "simulations": 16, "worlds": 3})

    cards = server.Session(server.Catalog(), ("日澳快攻", "德意直空"), 7).cards_for("p2", "deck")
    first, second = cards[0]["id"], cards[-1]["id"]
    a, b = analysis_with_p2_card(first), analysis_with_p2_card(second)
    assert a["summary"]["worlds"] == 3
    assert [(r["index"], r["visits"]) for r in a["rows"]] == [(r["index"], r["visits"]) for r in b["rows"]]
