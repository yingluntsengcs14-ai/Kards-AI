"""Hand-designed board evaluation and a greedy agent built on it (2026-11-25).

Inspired by the board score of Piotrowski & Mittenentzwei used in the NTNU
Hearthstone thesis (GP + MCTS, 2020): score each side's resources, take
"mine minus theirs", squash to [-1, 1]. Uses:

* ``GreedyHeuristicAgent``: tries every legal action on a cloned state and
  plays the one whose resulting board scores best -- a fast, training-free
  baseline that understands what an action does to the board.
* ``evaluate`` can later be blended into MCTS leaf values.

Weights are hand-set starting values; they are meant to be refit (logistic
regression on self-play outcomes, or an evolutionary search on win rate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ai.agents import BaseAgent
from ai.card_features import KEYWORDS, KEYWORD_INDEX, static_features
from simulator.actions.action import Action, PassAction
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState, PlayerState, UnitState

FEATURES = ("hq", "board", "units", "hand", "frontline", "lethal", "resources", "deck")
_GUARD, _BLITZ = KEYWORD_INDEX + KEYWORDS.index("guard"), KEYWORD_INDEX + KEYWORDS.index("blitz")


@dataclass
class HeuristicWeights:
    # 2026-11-25: fitted by logistic regression on 19k end-of-turn boards
    # from epsilon-greedy heuristic self-play (17 meta decks), outcome = win.
    # Hand-set v0 (hq 3, board 1, units 0.5, hand 0.6, frontline 2,
    # lethal 15): 69.1% vs rule. Fitted: 76.1% vs rule, 59.6% vs v0.
    hq: float = 2.419        # per sqrt(HQ health): losing the last points hurts most
    board: float = 0.044     # per point of attack + defense on the battlefield
    units: float = 0.332     # per unit (board presence beyond raw stats)
    hand: float = 0.198      # per card in hand, capped at HAND_CAP
    frontline: float = 3.474 # holding the frontline
    lethal: float = 2.435    # my on-board attack >= their HQ health
    # 2026-11-26 candidates (user suggestion: more cards = more resources);
    # 0 = off until fitted. resources = hand (up to the 9-card limit) + units,
    # so playing a card no longer looks like a loss; deck = cards left / 10.
    resources: float = 0.0
    deck: float = 0.0
    unspent: float = 0.4     # per kredit left unused when ending the turn
    scale: float = 12.0      # tanh scale for the [-1, 1] value
    hand_cap: int = 6
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in (*FEATURES, "unspent", "scale")}


def _unit_value(unit: UnitState) -> float:
    static = static_features(unit.card_id)
    value = max(0, unit.attack) + max(0, unit.defense)
    value += static[-1] * 3.0 * 1.5          # heavy armor level (stored /3)
    value += 1.0 * static[_GUARD] + 0.5 * static[_BLITZ]
    if unit.status.get("pinned") or unit.status.get("suppressed"):
        value *= 0.7
    return value


def features(player: PlayerState, enemy: PlayerState, hand_cap: int = 6) -> dict[str, float]:
    attack = sum(max(0, unit.attack) for unit in player.units)
    return {
        "hq": math.sqrt(max(0, player.hq.current_health)),
        "board": sum(_unit_value(unit) for unit in player.units),
        "units": float(len(player.units)),
        "hand": float(min(len(player.hand), hand_cap)),
        "frontline": float(any(unit.position == "frontline" for unit in player.units)),
        "lethal": float(attack >= max(1, enemy.hq.current_health)),
        "resources": float(min(len(player.hand), 9) + len(player.units)),
        "deck": len(player.deck) / 10.0,
    }


def evaluate(state: GameState, player_id: str, weights: HeuristicWeights | None = None) -> float:
    """Value in [-1, 1] from ``player_id``'s point of view."""
    w = weights or HeuristicWeights()
    status = state.game_status.value
    if status != "in_progress":
        if status == "draw": return 0.0
        return 1.0 if (status == "player_one_won") == (player_id == "p1") else -1.0
    enemy_id = "p2" if player_id == "p1" else "p1"
    me, them = state.players[player_id], state.players[enemy_id]
    mine, theirs = features(me, them, w.hand_cap), features(them, me, w.hand_cap)
    score = sum(getattr(w, name) * (mine[name] - theirs[name]) for name in FEATURES)
    return math.tanh(score / w.scale)


class GreedyHeuristicAgent(BaseAgent):
    """One-step lookahead: play the action whose resulting board scores best.

    Ending the turn is scored on the current board minus a penalty for each
    unspent kredit, so the agent keeps acting while that improves its board.
    """
    def __init__(self, weights: HeuristicWeights | None = None, *, epsilon: float = 0.0, temperature: float = 0.05, seed: int = 0) -> None:
        # epsilon: chance of a uniformly random legal action (diverse bootstrap
        # games). temperature: softmax temperature of ``last_policy``, which
        # SelfPlayRunner stores as the policy target when imitating this agent.
        self.weights = weights or HeuristicWeights(); self.state = None; self.player_id = None; self.cards = None
        self.epsilon, self.temperature = epsilon, temperature
        import random
        self.rng = random.Random(seed)

    def set_state(self, state, player_id: str, cards: CardDatabase | None = None) -> None:
        self.state, self.player_id, self.cards = state, player_id, cards

    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action:
        if self.state is None or self.cards is None:
            raise RuntimeError("GreedyHeuristicAgent needs set_state(state, player_id, cards)")
        me = self.player_id
        best, best_score = None, -math.inf
        scores: list[float] = []
        for action in legal_actions:
            if isinstance(action, PassAction):
                kredits = self.state.players[me].resources.kredits
                score = evaluate(self.state, me, self.weights) - math.tanh(self.weights.unspent * kredits / self.weights.scale)
            else:
                clone = self.state.clone_for_search()
                env = Simulator(self.cards, state=clone)
                try:
                    env.step_fast(action)
                except Exception:  # an engine error on one branch must not lose the game
                    scores.append(-math.inf); continue
                score = evaluate(clone, me, self.weights)
            scores.append(score)
            if score > best_score:
                best, best_score = action, score
        best = best if best is not None else legal_actions[0]
        finite = [x for x in scores if x != -math.inf]
        top = max(finite) if finite else 0.0
        weights = [math.exp((x - top) / self.temperature) if x != -math.inf else 0.0 for x in scores]
        total = sum(weights) or 1.0
        self.last_policy = [x / total for x in weights]
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(list(legal_actions))
        return best
