"""Deterministic admission checks to run before self-play training.

This module does not import an ML framework or update a model.  It validates
that a selected legal card pool can produce reproducible, executable episodes
without falling through to an unsupported runtime effect.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from simulator.actions.action import Action
from simulator.cards.availability import is_card_available
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.rules.native import engine_for
from simulator.testing import build_random_deck


ROOT = Path(__file__).parents[1]


@dataclass(frozen=True)
class PreflightReport:
    """Evidence generated before a training run is allowed to begin."""

    episodes: int
    actions_executed: int
    terminal_episodes: int
    eligible_card_count: int


def eligible_card_ids(cards: CardDatabase) -> set[str]:
    """Return active cards whose native rule is marked fully implemented.

    Deck construction still enforces the KARDS nation, rarity, token, and copy
    rules.  The status filter makes a future non-retired partial card fail
    closed instead of silently entering a training trajectory.
    """
    rules = engine_for(cards)
    return {
        card.id for card in cards
        if is_card_available(card.id) and rules.rule_for(card.id).status == "implemented"
    }


def _build_eligible_deck(cards: CardDatabase, seed: int, nation: str) -> list[str]:
    """Build a legal deck and assert the status filter did not leak."""
    allowed = eligible_card_ids(cards)
    deck = build_random_deck(cards, seed=seed, main_nation=nation, allowed=allowed)
    rejected = set(deck) - allowed
    if rejected:
        raise AssertionError("Training deck contains non-eligible cards: " + ", ".join(sorted(rejected)))
    return deck


def _assert_supported_events(events: Iterable[dict]) -> None:
    unsupported = [event for event in events if event.get("event") == "unsupported_effect"]
    if unsupported:
        raise AssertionError("Unsupported effects reached a training trajectory: " + repr(unsupported))


def run_preflight(episodes: int = 32, max_actions: int = 300, seed: int = 20260724,
                  nation: str = "France") -> PreflightReport:
    """Exercise seeded, legal self-play trajectories without training a model."""
    if episodes <= 0 or max_actions <= 0:
        raise ValueError("episodes and max_actions must be positive")

    cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
    rng = random.Random(seed)
    actions_executed = 0
    terminal_episodes = 0
    for _ in range(episodes):
        simulator = Simulator(cards)
        first_deck = _build_eligible_deck(cards, rng.randrange(2**31), nation)
        second_deck = _build_eligible_deck(cards, rng.randrange(2**31), nation)
        simulator.reset(first_deck, second_deck, nations=(nation, nation), seed=rng.randrange(2**31))
        for _ in range(max_actions):
            if simulator.is_terminal():
                terminal_episodes += 1
                break
            candidates: list[Action] = simulator.get_available_actions()
            if not candidates:
                raise AssertionError("Non-terminal state has no legal actions")
            action = rng.choice(candidates)
            action.validate(simulator.state, cards)  # type: ignore[arg-type]
            simulator.step(action)
            actions_executed += 1
            _assert_supported_events(simulator.state.event_log)  # type: ignore[union-attr]

    return PreflightReport(episodes, actions_executed, terminal_episodes, len(eligible_card_ids(cards)))


if __name__ == "__main__":
    report = run_preflight()
    print(
        "preflight passed: episodes={0.episodes} actions={0.actions_executed} "
        "terminal={0.terminal_episodes} eligible_cards={0.eligible_card_count}".format(report)
    )
