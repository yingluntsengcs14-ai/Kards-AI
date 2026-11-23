"""Evaluate a checkpoint against RandomAgent or RuleBasedAgent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ai.agents import NeuralAgent, RandomAgent, RuleBasedAgent
from ai.evaluator import Evaluator
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from simulator.cards.deck_code import load_meta_decks
from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--games", type=int, default=20); parser.add_argument("--opponent", choices=("random", "rule", "model"), default="random"); parser.add_argument("--opponent-model"); parser.add_argument("--deck-pool", help="JSON list of {main, ally} evaluation decks"); parser.add_argument("--seed", type=int, default=20260730); parser.add_argument("--report")
    parser.add_argument("--meta-decks", action="store_true", help="paired evaluation over every pair of the 17 meta decks (ignores --games/--deck-pool)")
    parser.add_argument("--games-per-pair", type=int, default=2, choices=(2, 4))
    args = parser.parse_args(); cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json"); encoder = ObservationEncoder(cards)
    candidate = NeuralAgent(KARDSNet.load_checkpoint(args.model), encoder)
    if args.opponent == "random": opponent = RandomAgent()
    elif args.opponent == "rule": opponent = RuleBasedAgent()
    else:
        if not args.opponent_model: parser.error("--opponent-model is required when --opponent model")
        opponent = NeuralAgent(KARDSNet.load_checkpoint(args.opponent_model), encoder)
    raw_pool = json.loads(args.deck_pool) if args.deck_pool else []
    deck_pool = [(item["main"], item.get("ally")) for item in raw_pool] or None
    evaluator = Evaluator(cards, encoder, seed=args.seed)
    if args.meta_decks: report = evaluator.evaluate_meta(candidate, opponent, load_meta_decks(cards), args.games_per_pair)
    else: report = evaluator.evaluate(candidate, opponent, args.games, deck_pool=deck_pool)
    if args.report: evaluator.save_report(report, args.report)
    print(report)
if __name__ == "__main__": main()
