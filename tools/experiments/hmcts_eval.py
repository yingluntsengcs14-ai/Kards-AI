import sys, json, time
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from dataclasses import asdict
from ai.agents import MCTSAgent, RuleBasedAgent
from ai.evaluator import Evaluator
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(str(ROOT / "data/source/kards_info_cards.json")); enc = ObservationEncoder(cards)
sims = int(sys.argv[1]); t = time.time()
agent = MCTSAgent(None, enc, sims, 1, heuristic_value_weight=1.0)
r = Evaluator(cards, enc, seed=20260730).evaluate_meta(agent, RuleBasedAgent(), load_meta_decks(cards), 2)
json.dump({**asdict(r), "seconds": time.time() - t}, open(sys.argv[2], "w"), ensure_ascii=False)
print("done", sims, r.win_rate, round(time.time() - t), flush=True)
