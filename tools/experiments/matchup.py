import sys, json, itertools
sys.path.insert(0, sys.argv[1])
from ai.agents import NeuralAgent
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase
from simulator.cards.deck_code import load_meta_decks
cards = CardDatabase.from_file(sys.argv[1] + "/data/source/kards_info_cards.json"); enc = ObservationEncoder(cards)
model = KARDSNet.load_checkpoint(sys.argv[2], device="cpu")
decks = load_meta_decks(cards); per_seat = int(sys.argv[4])
schedule = [p for a, b in itertools.combinations(decks, 2) for p in [(a, b)] * per_seat + [(b, a)] * per_seat]
runner = SelfPlayRunner(cards, enc, ReplayBuffer(), seed=20261125)
runner.run(len(schedule), NeuralAgent(model, enc), NeuralAgent(model, enc), deck_schedule=schedule)
cells = {}
for rec in runner.game_records:
    a, b = rec["decks"]
    for me, opp, seat in ((a, b, "p1"), (b, a, "p2")):
        c = cells.setdefault(me, {}).setdefault(opp, [0, 0])
        c[1] += 1; c[0] += rec["winner"] == seat
json.dump({"decks": [d.name for d in decks], "cells": cells, "errors": len(runner.engine_errors)}, open(sys.argv[3], "w"), ensure_ascii=False)
print("done", len(runner.game_records))
