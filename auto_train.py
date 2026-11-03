"""Continuous self-play, training, and evaluation loop with graceful stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from ai.runtime import runs_dir
from ai.config import load_config
import shutil
import subprocess as sp

ROOT = Path(__file__).parent
RUNS = runs_dir()


def execute(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def wait_for_resume(stop_file: Path, pause_file: Path) -> bool:
    """Safe boundary control: never terminate an active self-play/optimizer job."""
    while pause_file.exists():
        if stop_file.exists():
            return False
        time.sleep(2)
    return not stop_file.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/formal_training.yaml")
    parser.add_argument("--cycles", type=int, help="0 means run until STOP is created")
    parser.add_argument("--episodes", type=int); parser.add_argument("--mcts-simulations", type=int); parser.add_argument("--workers", type=int)
    parser.add_argument("--updates", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--evaluation-games", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); spcfg=cfg["self_play"]; tcfg=cfg["training"]; ecfg=cfg["evaluation"]; ccfg=cfg["checkpoint"]; lcfg=cfg["logging"]
    cycles = 0 if args.cycles is None else args.cycles; episodes=args.episodes or int(spcfg["episodes_per_cycle"]); sims=args.mcts_simulations or int(spcfg["simulations"]); workers=args.workers or int(spcfg["workers"]); updates=args.updates or int(tcfg["updates_per_cycle"]); batch=args.batch_size or int(tcfg["batch_size"]); eval_games=args.evaluation_games or int(ecfg["games_per_opponent"])
    RUNS.mkdir(parents=True, exist_ok=True); stop_file = RUNS / "STOP"; stop_file.unlink(missing_ok=True); pause_file = RUNS / "PAUSE"
    checkpoint, replay, metrics = RUNS / lcfg["latest_checkpoint"], RUNS / lcfg["replay_path"], RUNS / lcfg["metrics_path"]
    champion, candidate = RUNS / "champion.pt", RUNS / "candidate.pt"
    best_external, best_metrics = RUNS / "best_external.pt", RUNS / "best_external_metrics.json"
    checkpoints = RUNS / lcfg["history_dir"]; checkpoints.mkdir(exist_ok=True); (RUNS / "config_snapshot.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    if not champion.exists() and checkpoint.exists():
        shutil.copy2(checkpoint, champion)
    if not champion.exists():
        raise RuntimeError("No checkpoint is available to initialise the Champion model")
    # `latest.pt` is the historical best generalisation model, while Champion
    # remains the local self-play opponent.  Neither is overwritten by a
    # candidate that merely exploits the immediately preceding Champion.
    if not best_external.exists(): shutil.copy2(champion, best_external)
    shutil.copy2(best_external, checkpoint)
    iteration = 0
    while cycles == 0 or iteration < cycles:
        if not wait_for_resume(stop_file, pause_file): break
        iteration += 1
        selfplay_args = ["selfplay.py", "--episodes", str(episodes), "--mcts-simulations", str(sims), "--workers", str(workers), "--max-actions", str(spcfg["max_actions"]), "--device", "cuda", "--replay", str(replay), "--append-replay", "--replay-capacity", str(spcfg["replay_capacity"]), "--metrics", str(metrics), "--progress-every", "1", "--async-mcts", "--max-pending-leaves", str(spcfg["max_pending_leaves"]), "--model", str(champion)]
        if spcfg.get("meta_decks"):
            selfplay_args += ["--meta-decks"]
        else:
            selfplay_args += ["--deck-pool", json.dumps(spcfg["deck_pool"])]
        execute(selfplay_args)
        if stop_file.exists(): break

        if not wait_for_resume(stop_file, pause_file): break
        shutil.copy2(champion, candidate)
        payload_before = __import__("torch").load(candidate, map_location="cpu", weights_only=False)
        prior_step = int(payload_before.get("training_step", 0)); decay_every = int(tcfg.get("lr_decay_every_steps", 1000))
        learning_rate = max(float(tcfg.get("min_learning_rate", tcfg["learning_rate"])),
                            float(tcfg["learning_rate"]) * float(tcfg.get("lr_decay", 1.0)) ** (prior_step // max(1, decay_every)))
        execute(["train.py", "--replay", str(replay), "--updates", str(updates), "--batch-size", str(batch), "--learning-rate", str(learning_rate), "--checkpoint", str(candidate), "--model", str(candidate), "--metrics", str(metrics), "--device", "cuda"])
        if stop_file.exists(): break
        torch = __import__("torch"); payload = torch.load(candidate, map_location="cpu", weights_only=False)
        payload.setdefault("metadata", {}).update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "config_snapshot": cfg, "learning_rate": learning_rate,
                                                     "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        torch.save(payload, candidate); step=int(payload.get("training_step", 0))
        for old in sorted(checkpoints.glob("checkpoint_step_*.pt"))[:-int(ccfg["keep_last"])]: old.unlink()
        (RUNS / "SAVE_CHECKPOINT").unlink(missing_ok=True)
        if not wait_for_resume(stop_file, pause_file): break
        reports=[]; candidate_scores = {}; champion_scores = {}
        for opponent in ecfg["opponents"]:
            report = RUNS / "evaluations" / f"cycle-{iteration:05d}-candidate-{opponent}.json"; report.parent.mkdir(exist_ok=True)
            evaluation_args = ["--games", str(eval_games), "--deck-pool", json.dumps(spcfg["deck_pool"]), "--seed", str(ecfg.get("seed", 20260730))]
            execute(["evaluate.py", "--model", str(candidate), *evaluation_args, "--opponent", opponent, "--report", str(report)])
            candidate_scores[opponent] = json.loads(report.read_text(encoding="utf-8")); reports.append(("candidate", report))
            baseline = RUNS / "evaluations" / f"cycle-{iteration:05d}-champion-{opponent}.json"
            execute(["evaluate.py", "--model", str(champion), *evaluation_args, "--opponent", opponent, "--report", str(baseline)])
            champion_scores[opponent] = json.loads(baseline.read_text(encoding="utf-8")); reports.append(("champion", baseline))
        gate_report = RUNS / "evaluations" / f"cycle-{iteration:05d}-champion.json"
        gate_args = ["--games", str(ecfg.get("champion_games", eval_games)), "--deck-pool", json.dumps(spcfg["deck_pool"]), "--seed", str(ecfg.get("seed", 20260730))]
        execute(["evaluate.py", "--model", str(candidate), *gate_args, "--opponent", "model", "--opponent-model", str(champion), "--report", str(gate_report)])
        gate = json.loads(gate_report.read_text(encoding="utf-8"))
        tolerance = float(ecfg.get("external_tolerance", 0.0))
        external_ok = all(candidate_scores[name]["win_rate"] + tolerance >= champion_scores[name]["win_rate"] for name in ecfg["opponents"])
        promoted = float(gate["win_rate"]) >= float(ecfg.get("promotion_win_rate", 0.55)) and external_ok
        if promoted: shutil.copy2(candidate, champion)
        candidate_score = sum(item["win_rate"] for item in candidate_scores.values()) / len(candidate_scores)
        best_score = float(json.loads(best_metrics.read_text(encoding="utf-8")).get("score", -1.0)) if best_metrics.exists() else -1.0
        best_updated = promoted and candidate_score >= best_score
        if best_updated:
            shutil.copy2(candidate, best_external)
            best_metrics.write_text(json.dumps({"score": candidate_score, "training_step": step, "cycle": iteration, "scores": candidate_scores}, indent=2), encoding="utf-8")
        elif not best_metrics.exists():
            champion_score = sum(item["win_rate"] for item in champion_scores.values()) / len(champion_scores)
            best_metrics.write_text(json.dumps({"score": champion_score, "training_step": prior_step, "cycle": iteration, "scores": champion_scores}, indent=2), encoding="utf-8")
        shutil.copy2(best_external, checkpoint)
        history=checkpoints / f"checkpoint_step_{step}.pt"; shutil.copy2(checkpoint, history)
        reports.append(("gate", gate_report))
        with (RUNS / lcfg["evaluation_history"]).open("a", encoding="utf-8") as handle:
            for role, report in reports: handle.write(json.dumps({"cycle": iteration, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "training_step": step, "report": str(report), "role": role, "candidate_vs_champion": role == "gate", "promoted": promoted if role == "gate" else None, "external_gate_passed": external_ok if role == "gate" else None, "best_external_updated": best_updated if role == "gate" else None, **json.loads(report.read_text(encoding="utf-8"))}, ensure_ascii=False) + "\n")
        if stop_file.exists(): break

    (RUNS / "training.lock").unlink(missing_ok=True); (RUNS / "training.pid").unlink(missing_ok=True)


if __name__ == "__main__": main()
