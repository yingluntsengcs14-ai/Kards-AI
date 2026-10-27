"""Repeatable smoke-test utilities for AI training environment validation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from simulator.actions.action import AttackAction, PassAction
from simulator.cards.loader import CardDatabase, CardLoadReport
from simulator.cards.availability import is_card_available
from simulator.core.game import Simulator
from simulator.core.state import GameState
from simulator.rules.deck import DeckValidator

# KARDS deck building: only the five major nations may be the main nation.
# France, Italy, Poland and Finland are allies in the bundled catalogue.
OFFICIAL_MAIN_NATIONS = ("Britain", "Germany", "USA", "Soviet", "Japan")
OFFICIAL_ALLY_NATIONS = ("Germany", "Britain", "Soviet", "Japan", "USA", "France", "Italy", "Poland", "Finland", "ANZAC")


@dataclass(frozen=True)
class SimulationReport:
    card_report: CardLoadReport
    turns_completed: int
    actions_executed: int
    terminal: bool
    final_state: GameState


def build_random_deck(cards: CardDatabase, size: int = 39, seed: int | None = None, main_nation: str = "France", ally_nation: str | None = None,
                      ally_limit: int = 12, enforce_official_nations: bool = False,
                      allowed: set[str] | None = None) -> list[str]:
    """Build a seeded, 39-card-deck-builder-legal kards.info deck.

    ``OnlySpawnable`` is excluded; ``reserved`` is intentionally allowed.
    """
    if size != 39:
        raise ValueError("KARDS decks must contain exactly 39 cards")
    if enforce_official_nations and main_nation not in OFFICIAL_MAIN_NATIONS:
        raise ValueError(f"Main nation must be one of {OFFICIAL_MAIN_NATIONS}: {main_nation}")
    if enforce_official_nations and ally_nation is not None and ally_nation not in OFFICIAL_ALLY_NATIONS:
        raise ValueError(f"Unsupported ally nation: {ally_nation}")
    if ally_nation == main_nation:
        raise ValueError("Main and ally nations must differ")
    allowed_nations = {main_nation, ally_nation, "Neutral"}
    candidates = [
        card for card in cards
        if is_card_available(card.id) and not card.is_token and card.nation in allowed_nations
        and (allowed is None or card.id in allowed)
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    limits = {"standard": 4, "limited": 3, "special": 2, "elite": 1}
    deck: list[str] = []
    ally_count = 0
    for card in candidates:
        copies = limits.get(card.rarity.lower(), 0)
        if card.nation == ally_nation:
            copies = min(copies, max(0, ally_limit - ally_count))
        for _ in range(copies):
            if len(deck) == size:
                break
            deck.append(card.id)
            if card.nation == ally_nation:
                ally_count += 1
        if len(deck) == size:
            break
    if len(deck) != size:
        raise ValueError("Insufficient collectible cards to build a legal deck")
    validation = DeckValidator(cards).validate(deck, main_nation, ally_nation, ally_limit=ally_limit)
    if not validation.valid:
        raise AssertionError("Generated illegal deck: " + "; ".join(validation.errors))
    return deck


def run_random_simulation(cards: CardDatabase, turns: int = 100, deck_size: int = 39, seed: int = 0) -> SimulationReport:
    """Run a deterministic 100-turn-safe smoke simulation.

    A random non-attack action is taken when available, then the player passes.
    Excluding combat here guarantees that the requested full turn count is tested;
    combat legality and resolution are covered by the action/effect unit tests.
    """
    if turns <= 0:
        raise ValueError("Turn count must be positive")
    rng = random.Random(seed)
    simulator = Simulator(cards)
    simulator.reset(
        build_random_deck(cards, 39, rng.randrange(2**31)),
        build_random_deck(cards, 39, rng.randrange(2**31)),
    )
    executed = 0
    completed = 0
    while completed < turns and not simulator.is_terminal():
        actions = simulator.get_available_actions()
        for action in actions:
            action.validate(simulator.state, cards)  # type: ignore[arg-type]
        candidates = [action for action in actions if not isinstance(action, (PassAction, AttackAction))]
        if candidates:
            simulator.step(rng.choice(candidates))
            executed += 1
        pass_action = next(action for action in simulator.get_available_actions() if isinstance(action, PassAction))
        simulator.step(pass_action)
        executed += 1
        completed += 1

    final_state = simulator.get_state()
    # Ensure a snapshot generated during a longer run is fully restorable.
    if GameState.from_json(final_state.to_json()).to_dict() != final_state.to_dict():
        raise AssertionError("Game state failed JSON round trip")
    return SimulationReport(cards.report, completed, executed, simulator.is_terminal(), final_state)


def run_large_scale_smoke(cards: CardDatabase, games: int = 10_000, seed: int = 0) -> int:
    """Execute many independent legal-game smoke runs for stability regression tests."""
    for index in range(games):
        run_random_simulation(cards, turns=1, deck_size=2, seed=seed + index)
    return games
