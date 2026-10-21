"""Hidden-information search: PIMC over observation-consistent worlds.

``ai/mcts.py`` searches the real ``GameState``, so every node can read the
opponent's hand and both deck orders.  That makes the搜尋 a *perfect
information* search: it plans against cards it should not know about, and its
strength does not transfer to a real match where that information is hidden.

Perfect Information Monte Carlo (PIMC) keeps the existing search untouched and
fixes the input instead:

1. ``determinize`` samples a complete state that is consistent with everything
   the acting player can actually observe, and re-rolls everything they cannot.
2. The ordinary MCTS runs on each sampled world.
3. ``PIMCAgent`` averages the per-world visit distributions and plays the action
   that is best across worlds, not the one that is best in the single world the
   dealer happened to deal.

What is observable (never resampled)
    both battlefields, both HQs, kredits and kredit slots, graveyards, removed
    cards, turn state and every counter, the observer's own hand, the SIZE of
    the opponent's hand and deck, and enemy cards revealed by Intel.

What is hidden (resampled)
    the contents of the opponent's hand, the split of their unseen cards between
    hand and deck, the order of both decks, and the identity of the opponent's
    face-down countermeasures.

Known limitation: the sampling universe is the opponent's actual deck list,
which a real opponent's deck would not hand you.  That is the usual compromise
for self-play, and it is the one place a stronger belief model would plug in.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random
from typing import Sequence

from simulator.actions.action import Action
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState

from ai.action_encoder import action_key
from ai.agents import BaseAgent
from ai.observation import ObservationEncoder

KNOWN_ENEMY_HAND = "known_enemy_hand"


def _opponent_of(state: GameState, player_id: str) -> str:
    return next(candidate for candidate in state.players if candidate != player_id)


def observable_signature(state: GameState, observer_id: str) -> tuple:
    """Everything ``observer_id`` can see, in a comparable form.

    Determinization must not change this.  The test suite compares the signature
    before and after sampling, which is what makes "no leak" checkable rather
    than merely intended.
    """
    opponent_id = _opponent_of(state, observer_id)
    observer, opponent = state.players[observer_id], state.players[opponent_id]
    units = tuple(
        (unit.owner_id, unit.instance_id, unit.card_id, unit.attack, unit.defense, unit.position)
        for player in state.players.values() for unit in player.units
    )
    return (
        state.current_player, state.turn_number, state.game_status.value, units,
        tuple(sorted(observer.hand)),                  # own hand: known exactly
        len(opponent.hand), len(opponent.deck),        # enemy zones: sizes only
        tuple(sorted(Counter(observer.deck).items())), # own deck: contents, not order
        observer.hq.current_health, opponent.hq.current_health,
        observer.resources.kredits, observer.resources.max_kredits,
        opponent.resources.kredits, opponent.resources.max_kredits,
        tuple(sorted(state.graveyard.get(observer_id, ()))),
        tuple(sorted(state.graveyard.get(opponent_id, ()))),
        len(opponent.active_countermeasures),
        tuple(sorted(observer.status.get(KNOWN_ENEMY_HAND, ()))),
    )


def _resample_countermeasures(world: GameState, opponent_id: str, pool: list[str],
                              cards: CardDatabase, rng: random.Random) -> None:
    """Swap each face-down countermeasure for another of the same cost.

    A countermeasure is played face down: the observer sees that one is active
    and sees the kredits it cost, but not which card it is.  Keeping the cost
    fixed keeps the world consistent with what was spent.
    """
    opponent = world.players[opponent_id]
    for entry in opponent.active_countermeasures:
        current_id = entry.get("card_id")
        if current_id not in cards:
            continue
        cost = cards.get(current_id).kredits
        candidates = [card_id for card_id in pool
                      if card_id in cards
                      and cards.get(card_id).type == "countermeasure"
                      and cards.get(card_id).kredits == cost]
        if not candidates:
            continue
        replacement = rng.choice(candidates)
        pool.remove(replacement)
        pool.append(current_id)
        entry["card_id"] = replacement


def determinize(state: GameState, observer_id: str, cards: CardDatabase,
                rng: random.Random) -> GameState:
    """Sample a complete world consistent with what ``observer_id`` observes."""
    world = state.clone()
    opponent_id = _opponent_of(world, observer_id)
    observer, opponent = world.players[observer_id], world.players[opponent_id]

    # The observer knows what is in their own deck but not the order.  The list
    # is sorted before shuffling: shuffling with a fixed seed is sensitive to the
    # INPUT order, and that order is the hidden truth.  Without the sort, two
    # states that differ only in hidden information sample different worlds, and
    # the hidden information leaks back into the decision through the sampler.
    observer.deck.sort()
    rng.shuffle(observer.deck)

    # The opponent's unseen cards: hand contents and deck order are both unknown,
    # so they are pooled and re-dealt.  Intel-revealed cards are known and stay.
    known = list(observer.status.get(KNOWN_ENEMY_HAND, ()) or ())
    pool = list(opponent.hand) + list(opponent.deck)
    retained: list[str] = []
    for card_id in known:
        if card_id in pool:
            pool.remove(card_id)
            retained.append(card_id)
    # Same reason as above: sample from the belief (a multiset), not from the
    # order the真實 split happens to have.
    pool.sort()
    retained.sort()
    rng.shuffle(pool)

    hand_size = len(opponent.hand)
    take = max(0, hand_size - len(retained))
    opponent.hand = retained[:hand_size] + pool[:take]
    opponent.deck = pool[take:]

    _resample_countermeasures(world, opponent_id, opponent.deck, cards, rng)
    return world


@dataclass
class PIMCStats:
    """Per-decision diagnostics, so a run can be audited after the fact."""

    worlds: int = 0
    skipped_worlds: int = 0
    votes: list[float] = field(default_factory=list)


class PIMCAgent(BaseAgent):
    """Plays by averaging an ordinary MCTS over sampled worlds.

    The per-world searches are the unmodified ``MCTS``; only the states they are
    handed differ.  ``last_policy`` is the averaged visit distribution, so a
    self-play target recorded from this agent is a hidden-information target.
    """

    def __init__(self, model, encoder: ObservationEncoder, *, worlds: int = 8,
                 simulations: int = 32, seed: int = 0, inference=None,
                 temperature: float = 0.0, dirichlet_alpha: float = 0.0,
                 exploration_fraction: float = 0.0) -> None:
        from ai.mcts import MCTS
        self.worlds = max(1, int(worlds))
        self.rng = random.Random(seed)
        self.searchers = []
        for index in range(self.worlds):
            searcher = MCTS(model, encoder, simulations=simulations, seed=seed + index * 7919,
                            inference=inference)
            searcher.set_training_policy(temperature=temperature, dirichlet_alpha=dirichlet_alpha,
                                         exploration_fraction=exploration_fraction)
            self.searchers.append(searcher)
        self.state: GameState | None = None
        self.player_id: str | None = None
        self.cards: CardDatabase | None = None
        self.stats = PIMCStats()

    def set_state(self, state: GameState, player_id: str, cards: CardDatabase) -> None:
        self.state, self.player_id, self.cards = state, player_id, cards

    def reset(self) -> None:
        super().reset()
        self.stats = PIMCStats()

    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action:
        if self.state is None or self.cards is None or self.player_id is None:
            raise RuntimeError("PIMCAgent needs set_state from the runner")
        if len(legal_actions) == 1:
            self.last_policy = [1.0]
            self.stats = PIMCStats(worlds=0, votes=[1.0])
            return legal_actions[0]

        slot_of = {action_key(action): index for index, action in enumerate(legal_actions)}
        votes = [0.0] * len(legal_actions)
        used = skipped = 0
        for searcher in self.searchers:
            world = determinize(self.state, self.player_id, self.cards, self.rng)
            world_actions = Simulator(self.cards, state=world).get_available_actions()
            if not world_actions:
                skipped += 1
                continue
            _, policy = searcher.search(world, self.cards, self.player_id)
            counted = False
            for index_text, share in policy.items():
                index = int(index_text)
                if index >= len(world_actions):
                    continue
                slot = slot_of.get(action_key(world_actions[index]))
                # A sampled world can only differ from the observer's view in
                # hidden information, so its legal actions should match.  An
                # action that does not map is dropped rather than mis-credited.
                if slot is not None:
                    votes[slot] += share
                    counted = True
            used += 1 if counted else 0
            skipped += 0 if counted else 1

        total = sum(votes)
        if total <= 0:
            # No world produced a usable distribution; fall back to the first
            # legal action instead of pretending to have an opinion.
            self.last_policy = [1.0 / len(legal_actions)] * len(legal_actions)
            self.stats = PIMCStats(worlds=used, skipped_worlds=skipped, votes=list(votes))
            return legal_actions[0]

        self.last_policy = [vote / total for vote in votes]
        self.stats = PIMCStats(worlds=used, skipped_worlds=skipped, votes=list(votes))
        best = max(range(len(votes)), key=votes.__getitem__)
        return legal_actions[best]
