# -*- coding: utf-8 -*-
"""Observation / action encoding v2 (2026-11-24): countermeasure, weather and
unit-status state are visible, cards are identified by vocabulary index for
the network's embedding, and hidden enemy information still does not leak."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.action_encoder import ACTION_FEATURE_DIM, ACTION_NUMERIC_DIM, ActionEncoder  # noqa: E402
from ai.card_features import card_index  # noqa: E402
from ai.network import KARDSNet  # noqa: E402
from ai.observation import CARD_SLOTS, NUMERIC_DIM, STATE_DIM, ObservationEncoder  # noqa: E402
from simulator.actions.action import AttackAction, PlayCardAction  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402
from simulator.core.game import Simulator  # noqa: E402
from simulator.testing import build_random_deck  # noqa: E402

CARDS = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
ENCODER = ObservationEncoder(CARDS)


def fresh_state():
    deck = build_random_deck(CARDS, seed=5, main_nation="Germany", ally_nation="Italy")
    env = Simulator(CARDS); env.reset(deck, deck, nations=("Germany", "Germany"), ally_nations=("Italy", "Italy"), seed=1)
    return env.get_state()


class ObservationV2Tests(unittest.TestCase):
    def test_enemy_hand_and_countermeasures_do_not_leak(self) -> None:
        state = fresh_state()
        base = ENCODER.encode(state, "p1")
        enemy = state.players["p2"]
        enemy.hand = ["blast"] * len(enemy.hand)
        enemy.active_countermeasures = []
        hidden_a = ENCODER.encode(state, "p1")
        enemy.active_countermeasures = [{"card_id": "entrapment"}]
        hidden_b = ENCODER.encode(state, "p1")
        self.assertTrue(torch.equal(base, hidden_a))
        # Only the *count* of enemy countermeasures changes, not which one.
        enemy.active_countermeasures = [{"card_id": "blast"}]
        self.assertTrue(torch.equal(hidden_b, ENCODER.encode(state, "p1")))
        self.assertFalse(torch.equal(hidden_a, hidden_b))

    def test_own_countermeasure_forecast_and_weather_are_visible(self) -> None:
        state = fresh_state()
        before = ENCODER.encode(state, "p1")
        own = state.players["p1"]
        own.active_countermeasures = [{"card_id": "entrapment"}]
        with_counter = ENCODER.encode(state, "p1")
        self.assertEqual(with_counter[-CARD_SLOTS:].tolist().count(float(card_index("entrapment"))), 1)
        own.status["pending_forecast"] = "mist"
        with_forecast = ENCODER.encode(state, "p1")
        own.status["weather_played_this_turn"] = ["deluge"]
        with_weather = ENCODER.encode(state, "p1")
        self.assertEqual(len({tuple(t.tolist()) for t in (before, with_counter, with_forecast, with_weather)}), 4)
        # The opponent sees the public forecast too.
        enemy_view = ENCODER.encode(state, "p2")
        own.status.pop("pending_forecast")
        self.assertFalse(torch.equal(enemy_view, ENCODER.encode(state, "p2")))

    def test_shape_and_card_slots(self) -> None:
        vector = ENCODER.encode(fresh_state(), "p1")
        self.assertEqual(tuple(vector.shape), (STATE_DIM,))
        hand_ids = vector[NUMERIC_DIM + 16:NUMERIC_DIM + 28]
        self.assertTrue(all(float(v).is_integer() for v in hand_ids))
        self.assertGreater(int((hand_ids > 0).sum()), 0)


class ActionV2Tests(unittest.TestCase):
    def test_actions_carry_card_semantics(self) -> None:
        codec = ActionEncoder()
        tank = codec.encode(AttackAction("p1", "panzer_ivg-1", None))
        infantry = codec.encode(AttackAction("p1", "1st_airborne-2", "panzer_ivg-3"))
        self.assertEqual(tank.shape[0], ACTION_FEATURE_DIM)
        self.assertEqual(float(tank[ACTION_NUMERIC_DIM - 2]), 1.0)  # aimed at the HQ
        self.assertEqual(float(infantry[ACTION_NUMERIC_DIM - 2]), 0.0)
        self.assertFalse(torch.equal(tank[28:49], infantry[28:49]))
        self.assertGreater(card_index("panzer_ivg"), 0)
        self.assertEqual(int(tank[ACTION_NUMERIC_DIM]), card_index("panzer_ivg"))
        self.assertEqual(int(infantry[ACTION_NUMERIC_DIM]), card_index("1st_airborne"))
        self.assertEqual(int(infantry[ACTION_NUMERIC_DIM + 1]), card_index("panzer_ivg"))
        play = codec.encode(PlayCardAction("p1", "blast", target_unit_id="panzer_ivg-3"))
        self.assertEqual(int(play[ACTION_NUMERIC_DIM]), card_index("blast"))


class NetworkV2Tests(unittest.TestCase):
    def test_card_embedding_changes_output_and_checkpoint_round_trips(self) -> None:
        torch.manual_seed(0)
        model = KARDSNet(hidden_dim=32).eval()
        states = torch.zeros(2, STATE_DIM); states[1, -1] = float(card_index("blast"))
        actions = torch.zeros(2, 3, ACTION_FEATURE_DIM); mask = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad(): _, value = model(states, actions, mask)
        self.assertNotAlmostEqual(float(value[0]), float(value[1]), places=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"; model.save_checkpoint(path)
            loaded = KARDSNet.load_checkpoint(path).eval()
            with torch.no_grad(): self.assertTrue(torch.allclose(loaded(states, actions, mask)[1], value))

    def test_v1_checkpoint_still_loads(self) -> None:
        old = KARDSNet(state_dim=362, action_dim=28, hidden_dim=16, layout_version=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"state_dict": old.state_dict(), "architecture": {"state_dim": 362, "action_dim": 28, "hidden_dim": 16}}, path)
            loaded = KARDSNet.load_checkpoint(path)
            self.assertEqual(loaded.architecture["layout_version"], 1)


if __name__ == "__main__":
    unittest.main()
