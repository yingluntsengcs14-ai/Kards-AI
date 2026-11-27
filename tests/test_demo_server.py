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
