"""Train KARDSNet on saved self-play data."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from ai.config import load_config
from ai.network import KARDSNet
from ai.replay_buffer import ReplayBuffer
from ai.trainer import Trainer
from ai.metrics import RunMetrics
from ai.runtime import runtime_path

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/training.yaml"); parser.add_argument("--replay"); parser.add_argument("--model"); parser.add_argument("--batch-size", type=int); parser.add_argument("--updates", type=int); parser.add_argument("--learning-rate", type=float); parser.add_argument("--checkpoint"); parser.add_argument("--device"); parser.add_argument("--metrics"); parser.add_argument("--progress-every", type=int, default=10)
    # 2026-11-24: cap updates at N passes over the replay actually present. The
    # first v2 cycle drew 2000 x 256 samples from ~33k examples (~15 passes)
    # and the value loss fell to 1e-4 -- it memorised each game's result.
    parser.add_argument("--max-replay-passes", type=float)
    args = parser.parse_args(); cfg = load_config(args.config); replay_path = runtime_path(args.replay or cfg.get("replay_path"), "replay.pkl"); buffer = ReplayBuffer.load(replay_path); model = KARDSNet.load_checkpoint(args.model) if args.model else KARDSNet()
    checkpoint = runtime_path(args.checkpoint or cfg.get("checkpoint"), "kardsnet.pt"); trainer = Trainer(model, args.learning_rate or float(cfg["learning_rate"]), args.device)
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
    latest = {}; updates = args.updates or int(cfg["updates"])
    batch_size = args.batch_size or int(cfg["batch_size"])
    if args.max_replay_passes:
        capped = max(1, math.ceil(args.max_replay_passes * len(buffer) / batch_size))
        if capped < updates:
            print(f"[train] {updates} -> {capped} updates ({args.max_replay_passes:g} passes over {len(buffer)} examples)", flush=True)
            updates = capped
    metrics = RunMetrics(runtime_path(args.metrics, "training_metrics.jsonl"), "train")
    for update in range(1, updates + 1):
        latest = trainer.train_batch(buffer, batch_size)
        if update % max(1, args.progress_every) == 0 or update == updates:
            elapsed = metrics.elapsed_seconds(); metrics.display_train(update, updates, latest, elapsed, trainer.device)
            metrics.emit("train_progress", completed_updates=update, requested_updates=updates, updates_per_second=update / max(elapsed, 1e-9), device=trainer.device, replay_examples=len(buffer), **latest)
    trainer.save_checkpoint(checkpoint, replay_path=str(replay_path), device=trainer.device, **latest)
    metrics.emit("train_complete", checkpoint=str(checkpoint), training_step=trainer.training_step, device=trainer.device, replay_examples=len(buffer), **latest); print(latest)
if __name__ == "__main__": main()
