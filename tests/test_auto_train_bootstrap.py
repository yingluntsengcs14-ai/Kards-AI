# -*- coding: utf-8 -*-
"""Regression test for auto_train.py's first-run checkpoint bootstrap, and
for the --device passthrough that makes this testable at all (2026-11-19).

Context: starting formal training for the first time against a brand-new,
empty ``runs/`` directory crashed immediately with ``RuntimeError: No
checkpoint is available to initialise the Champion model`` -- discovered
live while walking the user through their actual first launch on the
Windows/GPU training desktop. ``auto_train.py`` needed a pre-existing
checkpoint to seed the local self-play "champion" opponent, but nothing in
the documented start-up path (``start_training.py`` / this project's own
instructions) ever created one, so every fresh ``runs/`` directory hit this
on its very first launch.

Fixed two things:
1. Build a freshly-initialised ``KARDSNet`` (using the config's
   ``model.hidden_dim``) as the champion when none exists yet, instead of
   raising -- training from a random network is the normal AlphaZero
   starting point, not an error condition.
2. ``auto_train.py`` previously hardcoded ``--device cuda`` for both of its
   ``selfplay.py``/``train.py`` subprocess calls (every other entry point in
   this repo already exposes ``--device``), which also made the whole loop
   untestable anywhere without a GPU. Added a passthrough ``--device`` flag
   (default ``cuda``, so the real desktop's behaviour is unchanged).

This test runs one full, real cycle (selfplay -> train -> evaluate -> gate
-> checkpoint) on CPU with a tiny config, starting from a completely empty
runs/ directory -- exactly the situation that crashed.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]

TINY_CONFIG = """\
self_play:
  episodes_per_cycle: 1
  workers: 1
  simulations: 2
  search: perfect
  pimc_worlds: 2
  async_mcts: true
  shared_memory: true
  max_pending_leaves: 4
  max_actions: 60
  meta_decks: true
  deck_pool: []
  replay_capacity: 500
  replay_recent_fraction: 0.6
model:
  hidden_dim: 16
training:
  batch_size: 8
  learning_rate: 0.001
  min_learning_rate: 0.0001
  lr_decay_every_steps: 1000
  lr_decay: 0.9
  updates_per_cycle: 2
checkpoint:
  keep_last: 2
  interval_cycles: 1
evaluation:
  seed: 1
  interval_games: 1
  games_per_opponent: 1
  champion_games: 1
  promotion_win_rate: 0.0
  external_tolerance: 1.0
  opponents: [random]
logging:
  metrics_path: training_metrics.jsonl
  replay_path: replay.pkl
  latest_checkpoint: latest.pt
  history_dir: checkpoints
  evaluation_history: evaluation_history.jsonl
"""


class AutoTrainBootstrapTests(unittest.TestCase):
    def test_brand_new_runs_dir_completes_a_full_cpu_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"  # deliberately does not exist yet
            config = Path(tmp) / "tiny_config.yaml"
            config.write_text(TINY_CONFIG, encoding="utf-8")

            env = os.environ.copy()
            env["KARDS_RUNS_DIR"] = str(runs)
            result = subprocess.run(
                [sys.executable, "auto_train.py", "--config", str(config),
                 "--cycles", "1", "--device", "cpu"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            for name in ("champion.pt", "best_external.pt", "latest.pt", "manifest.json",
                         "training_metrics.jsonl", "evaluation_history.jsonl", "games.jsonl"):
                self.assertTrue((runs / name).exists(), f"missing {name}")

            from ai.network import KARDSNet
            model = KARDSNet.load_checkpoint(runs / "champion.pt", device="cpu")
            self.assertEqual(model.architecture["hidden_dim"], 16)

    def test_continuous_training_accumulates_steps_without_promotion(self) -> None:
        """Regression (2026-11-21): a non-promoting cycle used to discard its
        updates (training restarted from Champion every cycle), freezing the
        formal run at training_step 1000 for 90+ cycles."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            config = Path(tmp) / "tiny_config.yaml"
            # Impossible gate: nothing can ever promote.
            config.write_text(TINY_CONFIG.replace("promotion_win_rate: 0.0", "promotion_win_rate: 2.0"), encoding="utf-8")
            env = os.environ.copy(); env["KARDS_RUNS_DIR"] = str(runs)
            result = subprocess.run(
                [sys.executable, "auto_train.py", "--config", str(config), "--cycles", "2", "--device", "cpu"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            trainee = torch.load(runs / "trainee.pt", map_location="cpu", weights_only=False)
            self.assertEqual(int(trainee["training_step"]), 4)  # 2 cycles x 2 updates
            champion = torch.load(runs / "champion.pt", map_location="cpu", weights_only=False)
            self.assertEqual(int(champion.get("training_step", 0)), 0)  # never promoted


if __name__ == "__main__":
    unittest.main()
