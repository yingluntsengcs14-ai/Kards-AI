# -*- coding: utf-8 -*-
"""Regression test (2026-11-24): restoring the optimizer state from a
checkpoint also restored its old learning rate, overriding --learning-rate,
so the per-cycle schedule in auto_train.py never took effect."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.network import KARDSNet  # noqa: E402
from ai.trainer import Trainer  # noqa: E402


class TrainerLearningRateTests(unittest.TestCase):
    def test_requested_learning_rate_survives_checkpoint_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            Trainer(KARDSNet(hidden_dim=16), 1e-4, "cpu").save_checkpoint(path)
            trainer = Trainer(KARDSNet(hidden_dim=16), 2e-5, "cpu")
            trainer.load_checkpoint(path)
            self.assertEqual([g["lr"] for g in trainer.optimizer.param_groups], [2e-5] * len(trainer.optimizer.param_groups))


if __name__ == "__main__":
    unittest.main()
