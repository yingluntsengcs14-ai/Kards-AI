"""End-to-end smoke tests for the separate AI training layer."""
from pathlib import Path
import random
import tempfile
import unittest
import multiprocessing as mp

import numpy as np
import torch

from ai.action_encoder import ACTION_FEATURE_DIM, MAX_ACTIONS, ActionEncoder
from ai.agents import MCTSAgent, RandomAgent, RuleBasedAgent
from ai.mcts import MCTS
from ai.metrics import RunMetrics
from ai.inference import BatchedInference, ProcessInferenceService, RemoteInferenceClient, SharedInferenceBuffers
from ai.evaluator import Evaluator
from ai.network import KARDSNet
from ai.observation import ObservationEncoder, STATE_DIM
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from ai.trainer import Trainer
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck

ROOT = Path(__file__).parents[1]


class AITrainingFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.encoder = ObservationEncoder(cls.cards)

    def _environment(self) -> Simulator:
        deck = build_random_deck(self.cards, seed=91)
        env = Simulator(self.cards); env.reset(deck, deck, nations=("France", "France")); return env

    def test_environment_random_agent_contract(self) -> None:
        env = self._environment(); agent = RandomAgent(1); legal = env.get_available_actions()
        self.assertNotIn("ConcedeAction", {type(action).__name__ for action in legal})
        action = agent.select_action(env.get_observation("p1"), legal)
        env.step(action); self.assertIsNotNone(env.state)

    def test_evaluator_supports_official_main_and_ally_deck_pool(self) -> None:
        report = Evaluator(self.cards, self.encoder, seed=73).evaluate(
            RandomAgent(1), RandomAgent(2), games=2, deck_pool=[("Britain", "France")])
        self.assertEqual(report.games, 2)

    def test_fixed_observation_and_action_mask(self) -> None:
        env = self._environment(); vector = self.encoder.encode(env.get_state(), "p1")
        features, mask = ActionEncoder().encode_legal_actions(env.get_available_actions())
        self.assertEqual(tuple(vector.shape), (STATE_DIM,)); self.assertEqual(features.shape[0], mask.shape[0]); self.assertTrue(mask.any())

    def test_network_forward_and_checkpoint(self) -> None:
        env = self._environment(); features, mask = ActionEncoder().encode_legal_actions(env.get_available_actions()); model = KARDSNet(hidden_dim=32)
        logits, value = model(self.encoder.encode(env.get_state(), "p1"), features, mask)
        self.assertEqual(logits.shape[-1], ActionEncoder().max_actions); self.assertEqual(tuple(value.shape), (1,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"; model.save_checkpoint(path); restored = KARDSNet.load_checkpoint(path)
            self.assertIsInstance(restored, KARDSNet)

    def test_forward_returns_policy_and_value(self) -> None:
        torch.manual_seed(7)
        model = KARDSNet(hidden_dim=32).eval()
        states = torch.randn(2, STATE_DIM)
        actions = torch.randn(2, 5, ACTION_FEATURE_DIM)
        mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])
        logits, values = model(states, actions, mask)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(values.shape), (2,))

    def test_compact_legal_action_inference_matches_padded_slots(self) -> None:
        env = self._environment(); model = KARDSNet(hidden_dim=32).eval()
        actions = env.get_available_actions(); codec = ActionEncoder()
        padded_features, padded_mask = codec.encode_legal_actions(actions)
        compact_features, compact_mask = codec.encode_legal_actions(actions, pad_to_max=False)
        state = self.encoder.encode(env.get_state(), "p1")
        padded_logits = model(state, padded_features, padded_mask)[0][0, :len(actions)]
        compact_logits = model(state, compact_features, compact_mask)[0][0]
        torch.testing.assert_close(compact_logits, padded_logits)

    def test_compact_action_encoding_allows_more_than_shared_capacity(self) -> None:
        action = self._environment().get_available_actions()[0]
        actions = [action] * (MAX_ACTIONS + 1)
        features, mask = ActionEncoder().encode_legal_actions(actions, pad_to_max=False)
        self.assertEqual(features.shape[0], MAX_ACTIONS + 1)
        self.assertTrue(mask.all())

    def test_action_feature_cache_preserves_encoded_values(self) -> None:
        env = self._environment(); action = env.get_available_actions()[0]; codec = ActionEncoder()
        first = codec.encode(action)
        second = codec.encode(action)
        self.assertIs(first, second)
        torch.testing.assert_close(first, ActionEncoder().encode(action))

    def test_batched_inference_matches_local_network(self) -> None:
        torch.manual_seed(13)
        model = KARDSNet(hidden_dim=32).eval()
        states = [torch.randn(STATE_DIM), torch.randn(STATE_DIM)]
        features = [torch.randn(3, ACTION_FEATURE_DIM), torch.randn(5, ACTION_FEATURE_DIM)]
        masks = [torch.tensor([True, True, False]), torch.tensor([True, False, True, True, False])]
        with BatchedInference(model, max_batch_size=4, max_wait_ms=0) as inference:
            for state, action_features, mask in zip(states, features, masks):
                logits, value = inference.evaluate(state, action_features, mask)
                expected_logits, expected_value = model(state, action_features, mask)
                torch.testing.assert_close(logits, expected_logits[0]); torch.testing.assert_close(value, expected_value[0])

    def test_process_inference_matches_local_network(self) -> None:
        model = KARDSNet(hidden_dim=32).eval()
        manager = mp.Manager(); requests = manager.Queue(); replies = manager.Queue()
        service = ProcessInferenceService(model, requests, [replies], max_wait_ms=0)
        try:
            client = RemoteInferenceClient(requests, replies)
            state = torch.randn(STATE_DIM); features = torch.randn(3, ACTION_FEATURE_DIM)
            mask = torch.tensor([True, True, False])
            logits, value = client.evaluate(state, features, mask)
            expected_logits, expected_value = model(state, features, mask)
            torch.testing.assert_close(logits, expected_logits[0]); torch.testing.assert_close(value, expected_value[0])
        finally:
            service.close(); manager.shutdown()

    def test_shared_process_inference_matches_local_network(self) -> None:
        model = KARDSNet(hidden_dim=32).eval()
        manager = mp.Manager(); requests = manager.Queue(); replies = manager.Queue()
        buffers = SharedInferenceBuffers.create(1, slots_per_worker=2)
        service = ProcessInferenceService(model, requests, [replies], max_wait_ms=0, shared_spec=buffers.spec)
        try:
            client = RemoteInferenceClient(requests, replies, shared_spec=buffers.spec)
            state = torch.randn(STATE_DIM); features = torch.randn(3, ACTION_FEATURE_DIM)
            mask = torch.tensor([True, True, False])
            logits, value = client.evaluate(state, features, mask)
            expected_logits, expected_value = model(state, features, mask)
            torch.testing.assert_close(logits, expected_logits[0]); torch.testing.assert_close(value, expected_value[0])
            self.assertGreater(client.stats()["requests"], 0)
            client.close()
        finally:
            service.close(); buffers.unlink(); manager.shutdown()

    def test_shared_inference_falls_back_for_large_legal_action_sets(self) -> None:
        model = KARDSNet(hidden_dim=32).eval()
        manager = mp.Manager(); requests = manager.Queue(); replies = manager.Queue()
        buffers = SharedInferenceBuffers.create(1, slots_per_worker=2)
        service = ProcessInferenceService(model, requests, [replies], max_wait_ms=0, shared_spec=buffers.spec)
        try:
            client = RemoteInferenceClient(requests, replies, shared_spec=buffers.spec)
            state = torch.randn(STATE_DIM); features = torch.randn(MAX_ACTIONS + 1, ACTION_FEATURE_DIM)
            mask = torch.ones(MAX_ACTIONS + 1, dtype=torch.bool)
            logits, value = client.evaluate(state, features, mask)
            expected_logits, expected_value = model(state, features, mask)
            torch.testing.assert_close(logits, expected_logits[0]); torch.testing.assert_close(value, expected_value[0])
            self.assertEqual(client.stats()["shared_fallback_requests"], 1.0)
            client.close()
        finally:
            service.close(); buffers.unlink(); manager.shutdown()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_parallel_selfplay_uses_process_workers(self) -> None:
        runner = SelfPlayRunner(self.cards, self.encoder, ReplayBuffer(), max_actions=1, seed=71)
        report = runner.run_parallel(2, KARDSNet(hidden_dim=32).to("cuda"), simulations=1,
                                     card_path=ROOT / "data/source/kards_info_cards.json", workers=2, device="cuda")
        self.assertEqual(report.episodes, 2)
        self.assertEqual(report.examples, 2)

    def test_batched_mcts_matches_local_search(self) -> None:
        env = self._environment(); model = KARDSNet(hidden_dim=32).eval(); state = env.get_state()
        local_action, local_policy = MCTS(model, self.encoder, simulations=3, seed=29).search(state, self.cards, "p1")
        with BatchedInference(model, max_wait_ms=0) as inference:
            batched_action, batched_policy = MCTS(model, self.encoder, simulations=3, seed=29, inference=inference).search(state, self.cards, "p1")
        self.assertEqual(batched_action, local_action)
        self.assertEqual(batched_policy, local_policy)

    def test_async_batched_mcts_produces_legal_policy(self) -> None:
        env = self._environment(); model = KARDSNet(hidden_dim=32).eval(); state = env.get_state()
        with BatchedInference(model, max_wait_ms=0) as inference:
            action, policy = MCTS(model, self.encoder, simulations=3, seed=29, inference=inference,
                                  async_inference=True, max_pending_leaves=3).search(state, self.cards, "p1")
        self.assertIn(action, env.get_available_actions())
        self.assertAlmostEqual(sum(policy.values()), 1.0)

    def test_async_mcts_applies_root_dirichlet_noise(self) -> None:
        """Regression (2026-11-22): the async path skipped exploration noise,
        so formal self-play (always --async-mcts) never explored."""
        from unittest import mock
        env = self._environment(); model = KARDSNet(hidden_dim=32).eval(); state = env.get_state()
        with BatchedInference(model, max_wait_ms=0) as inference:
            search = MCTS(model, self.encoder, simulations=3, seed=29, inference=inference,
                          async_inference=True, max_pending_leaves=3)
            search.set_training_policy(temperature=1.0, dirichlet_alpha=0.3, exploration_fraction=0.25)
            with mock.patch.object(search, "_dirichlet", wraps=search._dirichlet) as noise:
                search.search(state, self.cards, "p1")
            noise.assert_called_once()

    def test_mcts_agent_runs(self) -> None:
        env = self._environment(); agent = MCTSAgent(None, self.encoder, simulations=2); agent.set_state(env.get_state(), "p1", self.cards)
        action = agent.select_action(env.get_observation("p1"), env.get_available_actions()); env.step(action)

    def test_selfplay_replay_and_one_training_batch(self) -> None:
        buffer = ReplayBuffer(); runner = SelfPlayRunner(self.cards, self.encoder, buffer, max_actions=30, seed=3)
        report = runner.run(episodes=1, player_one=RandomAgent(1), player_two=RuleBasedAgent())
        self.assertGreater(report.examples, 0)
        metrics = Trainer(KARDSNet(hidden_dim=32), device="cpu").train_batch(buffer, 4)
        self.assertIn("loss", metrics)

    def test_trainer_checkpoint_restores_optimizer_step_and_rng(self) -> None:
        buffer = ReplayBuffer(); runner = SelfPlayRunner(self.cards, self.encoder, buffer, max_actions=1, seed=44)
        runner.run(1, RandomAgent(1), RandomAgent(2))
        trainer = Trainer(KARDSNet(hidden_dim=32), device="cpu")
        trainer.train_batch(buffer, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.pt"; trainer.save_checkpoint(path, replay_path="replay.pkl")
            expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
            restored = Trainer(KARDSNet(hidden_dim=32), device="cpu"); restored.load_checkpoint(path)
            actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
            self.assertEqual(restored.training_step, trainer.training_step)
            self.assertTrue(restored.optimizer.state_dict()["state"])
            self.assertEqual(actual, expected)

    def test_random_agent_can_start_one_thousand_games(self) -> None:
        """Required scale check: agent/environment contract survives 1,000 resets."""
        buffer = ReplayBuffer()
        report = SelfPlayRunner(self.cards, self.encoder, buffer, max_actions=1, seed=31).run(
            episodes=1000, player_one=RandomAgent(1), player_two=RandomAgent(2))
        self.assertEqual(report.episodes, 1000)
        self.assertEqual(report.examples, 1000)
        self.assertGreaterEqual(report.average_turns, 1.0)

    def test_metrics_persist_cumulative_selfplay_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            first = RunMetrics(path, "selfplay"); first.emit("selfplay_complete", total_episodes=3)
            second = RunMetrics(path, "selfplay")
            self.assertEqual(second.previous_episodes, 3)


if __name__ == "__main__": unittest.main()
