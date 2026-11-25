"""PUCT Monte-Carlo Tree Search with optional KARDSNet priors/value."""
from __future__ import annotations

from dataclasses import dataclass, field
import math, random
from concurrent.futures import Future
from threading import RLock
from time import perf_counter
from typing import Any
import torch

from ai.action_encoder import ActionEncoder
from ai.observation import ObservationEncoder
from ai.network import KARDSNet
from simulator.actions.action import Action
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState


@dataclass
class MCTSNode:
    state: GameState
    player_to_move: str
    parent: "MCTSNode | None" = None
    action: Action | None = None
    prior_probability: float = 1.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)
    priors: list[float] = field(default_factory=list)
    prior_order: list[int] = field(default_factory=list)
    virtual_visits: int = 0
    action_features: torch.Tensor | None = None
    legal_mask: torch.Tensor | None = None
    observations: dict[str, torch.Tensor] = field(default_factory=dict)
    evaluations: dict[str, tuple[torch.Tensor, float]] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    pending_evaluations: int = 0
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)
    @property
    def value(self) -> float: return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, model: KARDSNet | None, encoder: ObservationEncoder, simulations: int = 64, c_puct: float = 1.5, seed: int = 0, inference: Any | None = None,
                 *, initial_expansion: int | None = 16, widening_factor: float = 2.0, virtual_loss: float = 1.0,
                 async_inference: bool = False, max_pending_leaves: int = 16) -> None:
        self.model, self.encoder, self.simulations, self.c_puct, self.rng, self.codec = model.eval() if model else None, encoder, simulations, c_puct, random.Random(seed), ActionEncoder()
        self.device = next(model.parameters()).device if model else torch.device("cpu")
        self.inference = inference
        # Share of the hand-designed board score (ai/heuristic.py) blended into
        # every fresh leaf value: v = (1 - w) * network + w * heuristic. Lets a
        # young network lean on board knowledge; decayed by auto_train.py.
        self.heuristic_value_weight = 0.0
        self.initial_expansion = initial_expansion
        self.widening_factor = widening_factor
        self.virtual_loss = virtual_loss
        self.temperature = 0.0
        self.dirichlet_alpha = 0.0
        self.exploration_fraction = 0.0
        self.async_inference = async_inference and inference is not None
        self.max_pending_leaves = max_pending_leaves
        self.last_async_stats: dict[str, float] = {}
        self.async_totals: dict[str, float] = {"searches": 0.0, "submitted_leaves": 0.0,
                                               "completed_leaves": 0.0, "latency_sum_ms": 0.0,
                                               "pending_queue_peak": 0.0}

    def set_training_policy(self, *, temperature: float = 0.0, dirichlet_alpha: float = 0.0,
                            exploration_fraction: float = 0.0) -> None:
        self.temperature = max(0.0, temperature)
        self.dirichlet_alpha = max(0.0, dirichlet_alpha)
        self.exploration_fraction = min(1.0, max(0.0, exploration_fraction))

    def search(self, state: GameState, cards: CardDatabase, root_player: str) -> tuple[Action, dict[str, float]]:
        if self.async_inference:
            return self._search_async(state, cards, root_player)
        root = MCTSNode(state.clone_for_search(), state.current_player); self._expand(root, cards)
        self._apply_root_noise(root)
        for _ in range(self.simulations):
            node = root; path = [node]
            # Nodes retain the complete legal-action list, but successor
            # states are materialised only when PUCT selects that action.
            # This keeps the search policy identical while avoiding full
            # state clones for actions that the allotted simulations never
            # visit.
            while node.expanded and node.actions:
                node = self._select(node, cards); path.append(node)
                node.virtual_visits += 1
            value = self._evaluate(node, cards, root_player)
            self._backpropagate(path, value, root_player)
        if not root.children: return root.actions[0], {"0": 1.0}
        visits = {index: child.visit_count for index, child in root.children.items()}
        total = sum(visits.values()) or 1
        best = self._sample_visit(visits)
        return root.actions[best], {str(index): count / total for index, count in visits.items()}

    def _search_async(self, state: GameState, cards: CardDatabase, root_player: str) -> tuple[Action, dict[str, float]]:
        """Pipeline leaf evaluation while the same tree keeps selecting branches.

        Tree mutation happens on this worker thread only.  Futures are polled
        here rather than invoking callbacks from the inference thread/process,
        so Simulator states and MCTS nodes never cross thread boundaries.
        """
        from ai.inference import AsyncInferenceQueue
        root = MCTSNode(state.clone_for_search(), state.current_player)
        self._prepare_node(root, cards)
        if not root.actions:
            # Simulator normally always provides PassAction here.  Keeping the
            # explicit guard makes the async path fail in the same controlled
            # way as the synchronous path if a malformed terminal state leaks
            # into the public MCTS API.
            raise RuntimeError("MCTS root has no legal actions")
        queue = AsyncInferenceQueue(self.inference)
        root_future = queue.submit(self._observation(root, root.player_to_move), root.action_features, root.legal_mask)  # type: ignore[arg-type]
        while not root_future.done(): queue.wait([root_future], timeout=0.005)
        self._apply_evaluation(root, queue.resolved(root_future))
        # 2026-11-22: the async path never applied root Dirichlet noise, so the
        # formal run (which always uses --async-mcts) self-played with zero
        # exploration; with 16 simulations the visit counts then mostly copy
        # the network's own prior and the policy drifted (end-turn share rose
        # from 31% to 45%, cards played halved over 45k steps).
        self._apply_root_noise(root)
        pending: list[tuple[Future, MCTSNode, list[MCTSNode], float]] = []
        submitted = completed = 0; pending_peak = 0; latencies: list[float] = []
        while completed < self.simulations:
            while submitted < self.simulations and len(pending) < self.max_pending_leaves:
                node, path = root, [root]
                while node.expanded and node.actions:
                    node = self._select(node, cards); path.append(node)
                status = node.state.game_status.value
                if status != "in_progress":
                    self._backpropagate(path, self._terminal_value(status, root_player), root_player); submitted += 1; completed += 1; continue
                self._prepare_node(node, cards)
                for item in path: item.virtual_visits += 1
                with node.lock:
                    node.pending_evaluations += 1
                future = queue.submit(self._observation(node, node.player_to_move), node.action_features, node.legal_mask)  # type: ignore[arg-type]
                pending.append((future, node, path, perf_counter())); submitted += 1; pending_peak = max(pending_peak, len(pending))
            queue.poll(); resolved = [item for item in pending if item[0].done()]
            if not resolved:
                queue.wait([item[0] for item in pending], timeout=0.005); continue
            for future, node, path, started in resolved:
                pending.remove((future, node, path, started))
                with node.lock:
                    node.pending_evaluations -= 1
                value = self._apply_evaluation(node, queue.resolved(future))[1]
                root_value = value if node.player_to_move == root_player else -value
                self._backpropagate(path, root_value, root_player); completed += 1; latencies.append(perf_counter() - started)
        visits = {index: child.visit_count for index, child in root.children.items()}
        total = sum(visits.values()) or 1; best = self._sample_visit(visits) if visits else 0
        self.last_async_stats = {"submitted_leaves": float(submitted), "completed_leaves": float(completed),
                                 "pending_queue_peak": float(pending_peak), "inference_latency_ms": 1000.0 * sum(latencies) / len(latencies) if latencies else 0.0}
        self.async_totals["searches"] += 1
        self.async_totals["submitted_leaves"] += submitted
        self.async_totals["completed_leaves"] += completed
        self.async_totals["latency_sum_ms"] += sum(latencies) * 1000.0
        self.async_totals["pending_queue_peak"] = max(self.async_totals["pending_queue_peak"], pending_peak)
        return root.actions[best], {str(index): count / total for index, count in visits.items()}

    def _apply_root_noise(self, root: MCTSNode) -> None:
        if not (self.exploration_fraction and self.dirichlet_alpha and root.priors):
            return
        noise = self._dirichlet(len(root.priors), self.dirichlet_alpha)
        root.priors = [(1.0 - self.exploration_fraction) * prior + self.exploration_fraction * sample
                       for prior, sample in zip(root.priors, noise)]
        root.prior_order = sorted(range(len(root.actions)), key=root.priors.__getitem__, reverse=True)

    def _blend(self, node: MCTSNode, player_id: str, value: float) -> float:
        weight = self.heuristic_value_weight
        if not weight:
            return value
        from ai.heuristic import evaluate as heuristic_value
        return (1.0 - weight) * value + weight * heuristic_value(node.state, player_id)

    @staticmethod
    def _terminal_value(status: str, root_player: str) -> float:
        return 1.0 if (status == "player_one_won") == (root_player == "p1") else -1.0 if status != "draw" else 0.0

    def _prepare_node(self, node: MCTSNode, cards: CardDatabase) -> None:
        with node.lock:
            if node.actions or node.state.game_status.value != "in_progress": return
            env = Simulator(cards, state=node.state); node.actions = env.get_available_actions()
            if node.actions: node.action_features, node.legal_mask = self.codec.encode_legal_actions(node.actions, pad_to_max=False)

    def _apply_evaluation(self, node: MCTSNode, result: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, float]:
        logits, value = result; number = self._blend(node, node.player_to_move, float(value.item()))
        with node.lock:
            node.evaluations[node.player_to_move] = (logits, number)
            node.priors = torch.softmax(logits, dim=0)[:len(node.actions)].tolist()
            node.prior_order = sorted(range(len(node.actions)), key=node.priors.__getitem__, reverse=True)
            node.expanded = True
        return logits, number

    def _sample_visit(self, visits: dict[int, int]) -> int:
        if self.temperature <= 0.0:
            return max(visits, key=visits.get)
        weights = [count ** (1.0 / self.temperature) for count in visits.values()]
        threshold = self.rng.random() * sum(weights); cumulative = 0.0
        for index, weight in zip(visits, weights):
            cumulative += weight
            if cumulative >= threshold:
                return index
        return next(reversed(visits))

    def _dirichlet(self, size: int, alpha: float) -> list[float]:
        samples = [self.rng.gammavariate(alpha, 1.0) for _ in range(size)]
        total = sum(samples)
        return [sample / total for sample in samples] if total else [1.0 / size] * size

    def _expand(self, node: MCTSNode, cards: CardDatabase) -> None:
        if node.expanded:
            return
        if node.state.game_status.value != "in_progress": return
        # Legal-action generation is read-only. Branch states are still
        # cloned below before executing an action, but cloning merely to ask
        # the validator for candidates is redundant.
        env = Simulator(cards, state=node.state); node.actions = env.get_available_actions()
        if not node.actions: return
        # Candidate rows are scored independently. Padding to MAX_ACTIONS is
        # required in replay/training tensors, but would only run masked GPU
        # work during tree search.
        node.action_features, node.legal_mask = self.codec.encode_legal_actions(node.actions, pad_to_max=False)
        logits, _ = self._network(node, node.player_to_move)
        node.priors = torch.softmax(logits, dim=0)[:len(node.actions)].tolist()
        # Rule legality is never pruned.  This ordering only controls which
        # legal successor is materialised first; progressive widening admits
        # every candidate as search visits grow.
        node.prior_order = sorted(range(len(node.actions)), key=node.priors.__getitem__, reverse=True)
        node.expanded = True

    def _observation(self, node: MCTSNode, player_id: str) -> torch.Tensor:
        if player_id not in node.observations:
            node.observations[player_id] = self.encoder.encode(node.state, player_id)
        return node.observations[player_id]

    def _network(self, node: MCTSNode, player_id: str) -> tuple[torch.Tensor, float]:
        """Return policy and value from one shared model forward per view."""
        cached = node.evaluations.get(player_id)
        if cached is not None:
            return cached
        assert node.action_features is not None and node.legal_mask is not None
        if self.inference is not None:
            logits, value = self.inference.evaluate(self._observation(node, player_id), node.action_features, node.legal_mask)
            result = (logits, float(value.item()))
        elif self.model is None:
            result = (torch.zeros(len(node.actions)), 0.0)
        else:
            with torch.inference_mode():
                logits, value = self.model(self._observation(node, player_id).to(self.device), node.action_features.to(self.device), node.legal_mask.to(self.device))
                result = (logits[0].cpu(), float(value.item()))
        if self.heuristic_value_weight:
            result = (result[0], self._blend(node, player_id, result[1]))
        node.evaluations[player_id] = result
        return result

    def _select(self, node: MCTSNode, cards: CardDatabase) -> MCTSNode:
        with node.lock:
            candidates = self._candidate_indices(node)
            scale = math.sqrt(node.visit_count + node.virtual_visits + 1)
            # child.value is stored from the perspective of the player to move
            # at the child. Across a turn boundary (end turn) that is the
            # opponent, so flip it to this node's mover before maximising.
            # Without the flip end turn was chosen when it helped the opponent
            # (2026-11-22; see tests/test_mcts_turn_boundary_sign.py).
            def q(child: MCTSNode) -> float:
                return child.value if child.player_to_move == node.player_to_move else -child.value
            best_index = max(
                candidates,
                key=lambda index: (
                    (q(node.children[index]) - self.virtual_loss * node.children[index].virtual_visits)
                    + self.c_puct * node.children[index].prior_probability * scale / (1 + node.children[index].visit_count + node.children[index].virtual_visits)
                    if index in node.children
                    else self.c_puct * node.priors[index] * scale
                ),
            )
            child = node.children.get(best_index)
            if child is not None:
                return child

            # Apply the exact same action implementation as before, but only for
            # the branch PUCT has chosen.  `clone_for_search` keeps every mutable
            # game component isolated while sharing immutable history entries.
            child_state = node.state.clone_for_search()
            action = node.actions[best_index]
            action.execute(child_state, cards)
            child = MCTSNode(child_state, child_state.current_player, node, action, node.priors[best_index])
            node.children[best_index] = child
            return child

    def _candidate_indices(self, node: MCTSNode) -> list[int]:
        """Top-prior legal actions, widened as a node receives visits.

        ``initial_expansion=None`` is the exact full-action-space mode used
        by compatibility tests and debugging.  The training default limits
        early branching but never changes simulator legality or action rules.
        """
        if self.initial_expansion is None:
            return node.prior_order
        width = max(self.initial_expansion, int(self.widening_factor * math.sqrt(node.visit_count + node.virtual_visits + 1)))
        return node.prior_order[:min(len(node.prior_order), width)]

    def _evaluate(self, node: MCTSNode, cards: CardDatabase, root_player: str) -> float:
        status = node.state.game_status.value
        if status != "in_progress": return 1.0 if (status == "player_one_won") == (root_player == "p1") else -1.0 if status != "draw" else 0.0
        self._expand(node, cards)
        return self._network(node, root_player)[1]

    @staticmethod
    def _backpropagate(path: list[MCTSNode], value: float, root_player: str) -> None:
        for node in path:
            if node.virtual_visits:
                node.virtual_visits -= 1
            node.visit_count += 1; node.value_sum += value if node.player_to_move == root_player else -value
