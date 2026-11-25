"""Generate AlphaZero-style self-play training records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from ai.agents import MCTSAgent
from ai.config import load_config
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.pimc import PIMCAgent
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from ai.metrics import RunMetrics
from ai.runtime import runtime_path
from simulator.cards.deck_code import load_meta_decks
from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/selfplay.yaml"); parser.add_argument("--episodes", type=int); parser.add_argument("--model"); parser.add_argument("--mcts-simulations", type=int); parser.add_argument("--max-actions", type=int); parser.add_argument("--nation"); parser.add_argument("--ally-nation"); parser.add_argument("--deck-pool"); parser.add_argument("--meta-decks", action="store_true"); parser.add_argument("--replay"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--metrics"); parser.add_argument("--progress-every", type=int, default=10); parser.add_argument("--device"); parser.add_argument("--workers", type=int); parser.add_argument("--append-replay", action="store_true"); parser.add_argument("--replay-capacity", type=int); parser.add_argument("--temperature", type=float); parser.add_argument("--dirichlet-alpha", type=float); parser.add_argument("--exploration-fraction", type=float); parser.add_argument("--async-mcts", action="store_true"); parser.add_argument("--max-pending-leaves", type=int); parser.add_argument("--heuristic-value-weight", type=float, default=0.0, help="share of ai/heuristic.py board score blended into MCTS leaf values"); parser.add_argument("--heuristic-epsilon", type=float, default=0.1, help="--search heuristic: random-action rate for diverse bootstrap games"); parser.add_argument("--search", choices=("perfect", "pimc", "heuristic"), help="perfect = MCTS on the real GameState (sees both hands); pimc = PIMCAgent, honest hidden-information search"); parser.add_argument("--pimc-worlds", type=int, help="PIMCAgent: how many determinized worlds to sample and average per decision (only with --search pimc)")
    args = parser.parse_args(); cfg = load_config(args.config); cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json"); encoder = ObservationEncoder(cards)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (KARDSNet.load_checkpoint(args.model, device=device) if args.model else KARDSNet()).to(device)
    sims = args.mcts_simulations or int(cfg["mcts_simulations"])
    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 1.0))
    alpha = args.dirichlet_alpha if args.dirichlet_alpha is not None else float(cfg.get("dirichlet_alpha", 0.3))
    fraction = args.exploration_fraction if args.exploration_fraction is not None else float(cfg.get("exploration_fraction", 0.25))
    async_mcts = args.async_mcts or bool(cfg.get("async_inference", False))
    max_pending_leaves = args.max_pending_leaves or int(cfg.get("max_pending_leaves", 16))
    shared_inference = bool(cfg.get("shared_inference", True))
    search = args.search or str(cfg.get("search", "perfect"))
    pimc_worlds = args.pimc_worlds or int(cfg.get("pimc_worlds", 4))
    workers = args.workers or int(cfg.get("workers", 1))
    if search == "pimc":
        # PIMCAgent (ai/pimc.py) determinizes a hidden-information-consistent
        # world per search instead of letting MCTS read the real GameState
        # (which sees both hands -- "perfect" below). run_parallel's
        # shared/async-inference worker pool is wired specifically for
        # MCTSAgent, so pimc only runs on the single-process runner.run()
        # path; that path is also the one the desktop benchmark found
        # fastest anyway (workers=1, see configs/formal_training.yaml), so
        # this is not a real restriction in practice.
        if workers > 1:
            raise SystemExit("--search pimc only supports --workers 1 (run_parallel's worker pool is MCTSAgent-only; "
                             "workers=1 is also the measured-fastest setting on the training desktop)")
        agents = (PIMCAgent(model, encoder, worlds=pimc_worlds, simulations=sims, seed=args.seed + 1,
                            temperature=temperature, dirichlet_alpha=alpha, exploration_fraction=fraction),
                  PIMCAgent(model, encoder, worlds=pimc_worlds, simulations=sims, seed=args.seed + 2,
                            temperature=temperature, dirichlet_alpha=alpha, exploration_fraction=fraction))
    elif search == "heuristic":
        # Bootstrap data (2026-11-25): the greedy board-score agent plays both
        # sides; its softmax over action scores is the policy target and the
        # game result the value target, so a fresh network starts by imitating
        # a ~76%-vs-rule player instead of random play.
        from ai.heuristic import GreedyHeuristicAgent
        if workers > 1:
            raise SystemExit("--search heuristic only supports --workers 1")
        agents = (GreedyHeuristicAgent(epsilon=args.heuristic_epsilon, seed=args.seed + 1), GreedyHeuristicAgent(epsilon=args.heuristic_epsilon, seed=args.seed + 2))
    else:
        agents = tuple(MCTSAgent(model, encoder, sims, args.seed + offset, temperature=temperature, dirichlet_alpha=alpha, exploration_fraction=fraction,
                                 heuristic_value_weight=args.heuristic_value_weight) for offset in (1, 2))
    episodes = args.episodes or int(cfg["episodes"]); metrics = RunMetrics(runtime_path(args.metrics, "training_metrics.jsonl"), "selfplay")
    def progress(completed, totals, examples):
        if completed % max(1, args.progress_every) == 0 or completed == episodes:
            elapsed = metrics.elapsed_seconds()
            total_games = metrics.previous_episodes + completed
            metrics.display_selfplay(completed, episodes, total_games, totals["p1"], totals["p2"], totals["draw"], examples, elapsed)
            metrics.emit("selfplay_progress", completed_episodes=completed, total_episodes=total_games, requested_episodes=episodes, games_per_second=completed / max(elapsed, 1e-9), device=device, workers=workers, simulations=sims, wins=totals["p1"], losses=totals["p2"], draws=totals["draw"], examples=examples)
    replay_path = runtime_path(args.replay or cfg.get("replay_path"), "replay.pkl"); capacity = args.replay_capacity or int(cfg.get("replay_capacity", 50_000)); buffer = ReplayBuffer.load(replay_path) if args.append_replay and replay_path.exists() else ReplayBuffer(capacity=capacity, seed=args.seed); buffer.capacity = capacity; runner = SelfPlayRunner(cards, encoder, buffer, args.max_actions or int(cfg["max_actions"]), args.seed)
    options = {"temperature": temperature, "dirichlet_alpha": alpha, "exploration_fraction": fraction,
               "async_inference": async_mcts, "max_pending_leaves": max_pending_leaves, "heuristic_value_weight": args.heuristic_value_weight}
    raw_pool = json.loads(args.deck_pool) if args.deck_pool else cfg.get("deck_pool", [])
    deck_pool = [(item["main"], item.get("ally")) for item in raw_pool] or None
    use_meta_decks = args.meta_decks or bool(cfg.get("meta_decks", False))
    meta_decks = load_meta_decks(cards) if use_meta_decks else None
    report = runner.run_parallel(episodes, model, sims, ROOT / "data/source/kards_info_cards.json", nation=args.nation or str(cfg["nation"]), ally_nation=args.ally_nation, deck_pool=deck_pool, use_meta_decks=use_meta_decks, workers=workers, device=device, on_episode_complete=progress, mcts_options=options, shared_inference=shared_inference) if workers > 1 else runner.run(episodes, *agents, nation=args.nation or str(cfg["nation"]), ally_nation=args.ally_nation, deck_pool=deck_pool, meta_decks=meta_decks, on_episode_complete=progress)
    buffer = runner.buffer
    buffer.save(replay_path)
    games_path = runtime_path(None, "games.jsonl"); games_path.parent.mkdir(parents=True, exist_ok=True)
    with games_path.open("a", encoding="utf-8") as handle:
        for index, game in enumerate(runner.game_records, start=metrics.previous_episodes + 1):
            handle.write(json.dumps({"game": index, **game}, ensure_ascii=False) + "\n")
    metrics.emit("selfplay_complete", report=report, total_episodes=metrics.previous_episodes + report.episodes,
                 inference=runner.last_inference_stats, async_inference=runner.last_async_stats,
                 replay_path=str(args.replay or cfg["replay_path"])); print(report)
if __name__ == "__main__": main()
