import sys, json
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from dataclasses import replace
from ai.agents import RuleBasedAgent, RandomAgent
from ai.heuristic import GreedyHeuristicAgent, HeuristicWeights
from ai.evaluator import Evaluator
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(str(ROOT / "data/source/kards_info_cards.json")); enc = ObservationEncoder(cards); decks = load_meta_decks(cards)
w = replace(HeuristicWeights(), **json.load(open(sys.argv[1])))
for name, opp in (("rule", RuleBasedAgent()), ("hand-set greedy", GreedyHeuristicAgent())):
    r = Evaluator(cards, enc, seed=20260730).evaluate_meta(GreedyHeuristicAgent(w), opp, decks, 2)
    print(sys.argv[1].split("/")[-1], "vs", name, round(r.win_rate, 3), flush=True)
