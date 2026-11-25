"""Evolutionary weight tuning for the heuristic (paper-style, but on weights).
Fitness = mean of paired meta-deck win rate vs rule and vs the incumbent best."""
import sys, json, random, time, math
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from dataclasses import replace
from ai.agents import RuleBasedAgent
from ai.heuristic import GreedyHeuristicAgent, HeuristicWeights
from ai.evaluator import Evaluator
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(str(ROOT / "data/source/kards_info_cards.json")); enc = ObservationEncoder(cards)
decks = load_meta_decks(cards); KEYS = ("hq", "board", "units", "hand", "frontline", "lethal", "unspent")
start = json.load(open(sys.argv[1])); out = sys.argv[2]; gens = int(sys.argv[3]); pop_n = 8
rng = random.Random(1125)
def mk(d): return replace(HeuristicWeights(), **{k: float(v) for k, v in d.items() if k in KEYS})
def wr(a, b, seed): return Evaluator(cards, enc, seed=seed).evaluate_meta(GreedyHeuristicAgent(a), b, decks, 2).win_rate
def mutate(d): return {k: max(0.0, v * math.exp(rng.gauss(0, 0.35)) if v else rng.uniform(0, 0.5)) for k, v in d.items()}
def cross(a, b): return {k: (a if rng.random() < .5 else b)[k] for k in KEYS}
best = {k: start.get(k, getattr(HeuristicWeights(), k)) for k in KEYS}
gen_seed = 20261125
best_rule = wr(mk(best), RuleBasedAgent(), gen_seed)
log = [{"gen": 0, "best": best, "vs_rule": best_rule}]
print("gen0", best_rule, best, flush=True)
pop = [mutate(best) for _ in range(pop_n)]
for g in range(1, gens + 1):
    t = time.time(); scored = []
    for d in pop:
        r = wr(mk(d), RuleBasedAgent(), gen_seed + g)
        h = wr(mk(d), GreedyHeuristicAgent(mk(best)), gen_seed + g)
        scored.append((0.5 * r + 0.5 * h, r, h, d))
    scored.sort(key=lambda s: -s[0])
    top = scored[0]
    if top[2] > 0.5 and top[1] >= best_rule - 0.02:   # beats incumbent head-to-head without losing to rule
        best, best_rule = top[3], top[1]
    log.append({"gen": g, "best": best, "vs_rule": best_rule, "top": [[round(s[0], 3), round(s[1], 3), round(s[2], 3), s[3]] for s in scored[:3]], "minutes": round((time.time() - t) / 60, 1)})
    json.dump(log, open(out, "w"), ensure_ascii=False, indent=1)
    print("gen", g, round(top[0], 3), "rule", round(top[1], 3), "h2h", round(top[2], 3), "| incumbent rule", round(best_rule, 3), flush=True)
    elite = [s[3] for s in scored[:2]] + [best]
    pop = [mutate(cross(rng.choice(elite), rng.choice(elite))) for _ in range(pop_n)]
