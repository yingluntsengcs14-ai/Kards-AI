"""Self-play with epsilon-greedy heuristic agents; record end-of-turn feature
differences (mine - theirs) and the final result, for fitting weights."""
import sys, json, random
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ai.heuristic import GreedyHeuristicAgent, features, FEATURES
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.actions.action import PassAction
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(str(ROOT / "data/source/kards_info_cards.json"))
class Rec(GreedyHeuristicAgent):
    def __init__(s, eps, seed): super().__init__(); s.rng = random.Random(seed); s.eps = eps; s.rows = []
    def select_action(s, obs, legal):
        a = s.rng.choice(legal) if s.rng.random() < s.eps else super().select_action(obs, legal)
        if isinstance(a, PassAction):
            me = s.state.players[s.player_id]; them = s.state.players["p2" if s.player_id == "p1" else "p1"]
            f, g = features(me, them), features(them, me)
            s.rows.append([s.player_id, s.state.turn_number, [f[k] - g[k] for k in FEATURES], me.resources.kredits])
        return a
games, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
with open(out, "w") as fh:
    for g in range(games):
        a, b = Rec(0.15, seed * 1000 + g), Rec(0.15, seed * 1000 + g + 500)
        r = SelfPlayRunner(cards, ObservationEncoder(cards), ReplayBuffer(), seed=seed * 7919 + g)
        r.run(1, a, b, meta_decks=load_meta_decks(cards))
        if not r.game_records: continue
        w = r.game_records[0]["winner"]
        for agent in (a, b):
            for pid, turn, diff, kr in agent.rows:
                fh.write(json.dumps({"y": 1 if w == pid else 0 if w != "draw" else 0.5, "turn": turn, "x": diff, "unspent": kr}) + "\n")
print("done", games)
