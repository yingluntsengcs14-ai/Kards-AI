# -*- coding: utf-8 -*-
"""2026-11-25: train.py --max-replay-passes caps updates to N passes over the
replay that exists, so a fresh run's small first replays are not memorised."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.agents import RandomAgent  # noqa: E402
from ai.observation import ObservationEncoder  # noqa: E402
from ai.replay_buffer import ReplayBuffer  # noqa: E402
from ai.selfplay import SelfPlayRunner  # noqa: E402
from simulator.cards.loader import CardDatabase  # noqa: E402

CARDS = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")


class ReplayPassCapTests(unittest.TestCase):
    def test_updates_are_capped_by_replay_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            buffer = ReplayBuffer()
            SelfPlayRunner(CARDS, ObservationEncoder(CARDS), buffer, max_actions=30, seed=1).run(1, RandomAgent(1), RandomAgent(2), deck_pool=[("Germany", "Italy")])
            buffer.save(tmp / "replay.pkl")
            examples = len(buffer)
            metrics = tmp / "metrics.jsonl"
            subprocess.run([sys.executable, "train.py", "--config", "configs/training.yaml", "--replay", str(tmp / "replay.pkl"), "--updates", "500",
                            "--batch-size", "4", "--checkpoint", str(tmp / "m.pt"), "--metrics", str(metrics), "--device", "cpu",
                            "--max-replay-passes", "2"], cwd=ROOT, check=True, capture_output=True)
            done = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if '"train_complete"' in line][-1]
            self.assertEqual(done["training_step"], -(-2 * examples // 4))
            self.assertLess(done["training_step"], 500)


if __name__ == "__main__":
    unittest.main()
