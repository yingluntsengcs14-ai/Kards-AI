"""Trained network + heuristic-blended MCTS vs an opponent, on the meta decks.

Same protocol as the paired meta evaluation (136 deck pairs x both seats =
272 games), but split into chunks run in parallel worker processes, so it
finishes on a desktop while training keeps running. Example (from repo root):

    python tools/experiments/net_mcts_eval.py --model runs/champion.pt --hw 0.3 --opponent heuristic --workers 4

--hw: heuristic share of the leaf value; --hp: heuristic share of the priors.
Result: printed, and written to runs/logs/net_mcts_<opponent>_hw<hw>_hp<hp>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SEED = 20260730


def _chunk(job):
    model_path, hw, hp, sims, opponent, seat, start, stop = job
    import torch
    torch.set_num_threads(1)
    from ai.agents import MCTSAgent, RuleBasedAgent
    from ai.heuristic import GreedyHeuristicAgent
    from ai.network import KARDSNet
    from ai.observation import ObservationEncoder
    from ai.replay_buffer import ReplayBuffer
    from ai.selfplay import SelfPlayRunner
    from simulator.cards.deck_code import load_meta_decks
    from simulator.cards.loader import CardDatabase
    cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json"); enc = ObservationEncoder(cards)
    model = KARDSNet.load_checkpoint(model_path, device="cpu"); model.eval()
    schedule = list(combinations(load_meta_decks(cards), 2))[start:stop]
    cand = MCTSAgent(model, enc, sims, 1, heuristic_value_weight=hw, heuristic_prior_weight=hp)
    opp = GreedyHeuristicAgent() if opponent == "heuristic" else RuleBasedAgent()
    runner = SelfPlayRunner(cards, enc, ReplayBuffer(), seed=SEED + start)
    runner.run(len(schedule), *((cand, opp) if seat == "p1" else (opp, cand)), deck_schedule=schedule)
    rows = [(r["decks"][0 if seat == "p1" else 1], r["winner"] == seat, r["winner"] == "draw") for r in runner.game_records]
    print(f"  chunk {seat} {start}-{stop}: {sum(w for _, w, _ in rows)}/{len(rows)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="runs/champion.pt")
    parser.add_argument("--hw", type=float, default=0.3)
    parser.add_argument("--hp", type=float, default=0.0)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--opponent", choices=("heuristic", "rule"), default="heuristic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=136, help="use only the first N deck pairs (quick check)")
    args = parser.parse_args()
    pairs = args.pairs
    size = max(8, min(pairs, -(-2 * pairs // (args.workers * 2))))
    jobs = [(args.model, args.hw, args.hp, args.simulations, args.opponent, seat, s, min(pairs, s + size))
            for seat in ("p1", "p2") for s in range(0, pairs, size)]
    t = time.time()
    with Pool(args.workers) as pool:
        rows = [row for chunk in pool.map(_chunk, jobs) for row in chunk]
    per_deck: dict[str, list[int]] = {}
    for deck, won, _ in rows:
        per_deck.setdefault(deck, [0, 0]); per_deck[deck][0] += won; per_deck[deck][1] += 1
    wins = sum(w for _, w, _ in rows); draws = sum(d for _, _, d in rows)
    result = {"model": args.model, "opponent": args.opponent, "hw": args.hw, "hp": args.hp, "simulations": args.simulations,
              "games": len(rows), "wins": wins, "draws": draws, "win_rate": wins / max(1, len(rows)),
              "seconds": round(time.time() - t), "per_deck": {k: v[0] / v[1] for k, v in sorted(per_deck.items())}}
    out = ROOT / "runs/logs" / f"net_mcts_{args.opponent}_hw{args.hw}_hp{args.hp}.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESULT vs {args.opponent} hw={args.hw} hp={args.hp}: {wins}/{len(rows)} = {result['win_rate']:.1%} ({result['seconds']}s) -> {out}")


if __name__ == "__main__":
    main()
