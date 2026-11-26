"""Unified agents operating only through simulator observations/actions."""
from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Sequence

import torch

from ai.action_encoder import ActionEncoder
from ai.observation import ObservationEncoder
from ai.network import KARDSNet
from simulator.actions.action import Action, AttackAction, PassAction, PlayCardAction
from simulator.core.game import Simulator


class BaseAgent(ABC):
    last_policy: list[float] | None = None
    def reset(self) -> None: self.last_policy = None
    def observe(self, observation: dict) -> None: pass
    @abstractmethod
    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action: ...


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0) -> None: self.rng = random.Random(seed)
    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action:
        choice = self.rng.randrange(len(legal_actions)); self.last_policy = [float(i == choice) for i in range(len(legal_actions))]
        return legal_actions[choice]


class RuleBasedAgent(BaseAgent):
    """Small deterministic baseline: lethal/direct attacks, then plays, then pass."""
    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action:
        preferred = next((a for a in legal_actions if isinstance(a, AttackAction) and a.target_unit_id is None), None)
        preferred = preferred or next((a for a in legal_actions if isinstance(a, PlayCardAction)), None)
        preferred = preferred or next((a for a in legal_actions if not isinstance(a, PassAction)), legal_actions[0])
        self.last_policy = [float(a == preferred) for a in legal_actions]; return preferred


class NeuralAgent(BaseAgent):
    def __init__(self, model: KARDSNet, encoder: ObservationEncoder, device: str = "cpu", temperature: float = 0.0) -> None:
        self.model, self.encoder, self.codec, self.device, self.temperature = model.to(device).eval(), encoder, ActionEncoder(), device, temperature
        self.state = None; self.player_id = None
    def set_state(self, state, player_id: str) -> None: self.state, self.player_id = state, player_id
    @torch.no_grad()
    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action:
        if self.state is None or self.player_id is None: raise RuntimeError("NeuralAgent needs set_state from SelfPlayRunner")
        feats, mask = self.codec.encode_legal_actions(legal_actions, pad_to_max=False)
        logits, _ = self.model(self.encoder.encode(self.state, self.player_id).to(self.device), feats.to(self.device), mask.to(self.device))
        probs = torch.softmax(logits[0], dim=0).cpu(); index = int(torch.argmax(probs)) if self.temperature <= 0 else int(torch.multinomial(probs, 1))
        self.last_policy = probs[:len(legal_actions)].tolist(); return self.codec.action_at(legal_actions, index)


class MCTSAgent(BaseAgent):
    def __init__(self, model: KARDSNet | None, encoder: ObservationEncoder, simulations: int = 64, seed: int = 0, inference=None,
                 *, temperature: float = 0.0, dirichlet_alpha: float = 0.0, exploration_fraction: float = 0.0,
                 async_inference: bool = False, max_pending_leaves: int = 16, heuristic_value_weight: float = 0.0, heuristic_prior_weight: float = 0.0) -> None:
        from ai.mcts import MCTS
        self.searcher = MCTS(model, encoder, simulations=simulations, seed=seed, inference=inference,
                             async_inference=async_inference, max_pending_leaves=max_pending_leaves)
        self.searcher.set_training_policy(temperature=temperature, dirichlet_alpha=dirichlet_alpha, exploration_fraction=exploration_fraction)
        self.searcher.heuristic_value_weight = heuristic_value_weight
        self.searcher.heuristic_prior_weight = heuristic_prior_weight
        self.state = None; self.player_id = None; self.cards = None
    def set_state(self, state, player_id: str, cards) -> None: self.state, self.player_id, self.cards = state, player_id, cards
    def select_action(self, observation: dict, legal_actions: Sequence[Action]) -> Action:
        if self.state is None or self.cards is None or self.player_id is None: raise RuntimeError("MCTSAgent needs set_state from SelfPlayRunner")
        action, policy = self.searcher.search(self.state, self.cards, self.player_id)
        self.last_policy = [policy.get(str(i), 0.0) for i in range(len(legal_actions))]
        # The root obtains legal actions from the same cloned state; match by equality.
        return next(a for a in legal_actions if a == action)
