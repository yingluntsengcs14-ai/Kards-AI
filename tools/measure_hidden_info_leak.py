"""Measure how much the perfect-information search leaks, and that PIMC does not.

``ai/mcts.py`` searches the real state, so it can plan against the opponent's
hand.  This tool quantifies that: it takes a position, re-splits the OPPONENT's
hand and deck (the union is unchanged, so nothing the acting player can observe
differs), and counts how often each agent changes its decision.

A change means the decision depended on information the player cannot see.

    perfect-information MCTS : should change sometimes -> that is the leak
    PIMC                     : should never change     -> the property we want

Note on the baseline: with ``model=None`` the MCTS value is a constant 0.0, so
the search is insensitive to everything and reports no leak.  The measurement
therefore runs with a network (untrained is enough — only sensitivity to the
input matters, not strength).

Usage:
    python tools/measure_hidden_info_leak.py
    python tools/measure_hidden_info_leak.py --simulations 240 --worlds 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ai.action_encoder import action_key
from ai.agents import MCTSAgent
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.pimc import PIMCAgent
from simulator.actions.action import PassAction
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck

CATALOG = ROOT / "data/source/kards_info_cards.json"


def rich_position(cards: CardDatabase, seed: int):
    """Advance past the opening turns until the board offers a real choice."""
    simulator = Simulator(cards=cards, record_replay=False)
    deck = build_random_deck(cards, seed=seed, main_nation="Germany")
    simulator.reset(list(deck), list(deck), nations=("Germany", "Germany"), seed=seed)
    actions = simulator.get_available_actions()
    for _ in range(14):
        if len(actions) > 3:
            break
        simulator.step(PassAction(simulator.state.current_player))
        actions = simulator.get_available_actions()
    return simulator, actions


def reshuffle_opponent(state, observer: str, seed: int):
    """Re-split the opponent's hidden zones; the union and everything else stays."""
    variant = state.clone()
    opponent = next(pid for pid in variant.players if pid != observer)
    pool = list(variant.players[opponent].hand) + list(variant.players[opponent].deck)
    random.Random(seed).shuffle(pool)
    hand_size = len(variant.players[opponent].hand)
    variant.players[opponent].hand = pool[:hand_size]
    variant.players[opponent].deck = pool[hand_size:]
    return variant, opponent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=120, help="MCTS simulations per decision")
    parser.add_argument("--worlds", type=int, default=4, help="PIMC sampled worlds")
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--torch-seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    cards = CardDatabase.from_file(CATALOG)
    encoder = ObservationEncoder(cards)
    model = KARDSNet()

    seeds = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53][: args.positions]
    cheat_changed = pimc_changed = checked = 0
    for seed in seeds:
        simulator, actions = rich_position(cards, seed)
        if len(actions) <= 3:
            continue
        observer = simulator.state.current_player
        variant, opponent = reshuffle_opponent(simulator.state, observer, seed * 7)
        if simulator.state.players[opponent].hand == variant.players[opponent].hand:
            continue
        checked += 1

        first = MCTSAgent(model, encoder, simulations=args.simulations, seed=5)
        first.set_state(simulator.state, observer, cards)
        second = MCTSAgent(model, encoder, simulations=args.simulations, seed=5)
        second.set_state(variant, observer, cards)
        if action_key(first.select_action({}, actions)) != action_key(second.select_action({}, actions)):
            cheat_changed += 1

        third = PIMCAgent(model, encoder, worlds=args.worlds,
                          simulations=max(1, args.simulations // args.worlds), seed=5)
        third.set_state(simulator.state, observer, cards)
        fourth = PIMCAgent(model, encoder, worlds=args.worlds,
                           simulations=max(1, args.simulations // args.worlds), seed=5)
        fourth.set_state(variant, observer, cards)
        if action_key(third.select_action({}, actions)) != action_key(fourth.select_action({}, actions)):
            pimc_changed += 1

    print("positions compared: {0}".format(checked))
    print("  perfect-information MCTS changed its decision: {0}/{1}".format(cheat_changed, checked))
    print("  PIMC changed its decision:                     {0}/{1}".format(pimc_changed, checked))
    if checked and cheat_changed == 0:
        print("\nNo leak observed in the baseline: raise --simulations, or the network")
        print("is too flat to act on the hidden cards at this depth.")


if __name__ == "__main__":
    main()
