import sys, time, json, itertools
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ai.agents import RuleBasedAgent, RandomAgent
from ai.heuristic import GreedyHeuristicAgent
from ai.evaluator import Evaluator
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(str(ROOT / "data/source/kards_info_cards.json")); enc = ObservationEncoder(cards)
decks = load_meta_decks(cards)
if len(sys.argv) > 2: decks = decks[:int(sys.argv[2])]
opp = {"rule": RuleBasedAgent(), "random": RandomAgent(7)}[sys.argv[1]]
t = time.time(); r = Evaluator(cards, enc, seed=20260730).evaluate_meta(GreedyHeuristicAgent(), opp, decks, 2)
print(sys.argv[1], r.games, round(r.win_rate, 3), f"{(time.time()-t)/r.games:.2f}s/game", {k: round(v["win_rate"], 2) for k, v in r.per_deck.items()}, flush=True)
