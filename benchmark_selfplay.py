"""Small reproducible CUDA self-play throughput benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from ai.agents import MCTSAgent
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.deck_code import load_meta_decks
from simulator.cards.loader import CardDatabase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=16)
    # 400 actions is generous headroom now that the countermeasure-replay
    # lockup (commit 780cebe) is fixed; a normal game finishes well under it.
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="cuda or cpu; use cpu on a machine with no GPU")
    parser.add_argument("--meta-decks", action="store_true", help="draw from the 17 meta decks instead of a fixed France mirror")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine -- pass --device cpu instead")
    root = Path(__file__).parent; cards = CardDatabase.from_file(root / "data/source/kards_info_cards.json")
    model = KARDSNet(hidden_dim=128).to(args.device).eval()
    meta_decks = load_meta_decks(cards) if args.meta_decks else None
    runner = SelfPlayRunner(cards, ObservationEncoder(cards), ReplayBuffer(), args.max_actions, seed=301)
    started = perf_counter()
    if args.workers == 1:
        report = runner.run(args.episodes, MCTSAgent(model, runner.encoder, args.simulations, 302),
                            MCTSAgent(model, runner.encoder, args.simulations, 303), meta_decks=meta_decks)
    else:
        report = runner.run_parallel(args.episodes, model, args.simulations, root / "data/source/kards_info_cards.json",
                                     use_meta_decks=args.meta_decks, workers=args.workers, device=args.device)
    elapsed = perf_counter() - started
    truncated = sum(1 for rec in runner.game_records if rec["winner"] == "draw")
    print(f"device={args.device} workers={args.workers} simulations={args.simulations} meta_decks={args.meta_decks}")
    print(f"elapsed={elapsed:.3f}s games_per_second={args.episodes / elapsed:.4f} "
          f"seconds_per_game={elapsed / args.episodes:.3f}")
    print(f"average_turns={report.average_turns:.1f} truncated(draw)={truncated}/{args.episodes} "
          f"examples={report.examples} examples_per_second={report.examples / elapsed:.1f}")


if __name__ == "__main__":
    main()
