import sys, json
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from dataclasses import replace
from ai.agents import RuleBasedAgent
from ai.heuristic import GreedyHeuristicAgent, HeuristicWeights
from ai.evaluator import Evaluator
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(str(ROOT / "data/source/kards_info_cards.json")); enc = ObservationEncoder(cards); decks = load_meta_decks(cards)
w = replace(HeuristicWeights(), **json.load(open(sys.argv[1])))
out = {}
for seed in (11, 12):
    out[f"rule_new_{seed}"] = Evaluator(cards, enc, seed=seed).evaluate_meta(GreedyHeuristicAgent(w), RuleBasedAgent(), decks, 4).win_rate
    out[f"rule_old_{seed}"] = Evaluator(cards, enc, seed=seed).evaluate_meta(GreedyHeuristicAgent(), RuleBasedAgent(), decks, 4).win_rate
    out[f"h2h_{seed}"] = Evaluator(cards, enc, seed=seed).evaluate_meta(GreedyHeuristicAgent(w), GreedyHeuristicAgent(), decks, 4).win_rate
    print(out, flush=True)
json.dump(out, open(sys.argv[2], "w"))
