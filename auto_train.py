"""Continuous self-play, training, and evaluation loop with graceful stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from ai.runtime import runs_dir, write_run_manifest
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
    parser.add_argument("--device", default="cuda", help="passed through to the selfplay.py/train.py subprocesses (default: cuda, matching the formal training desktop)")
    args = parser.parse_args(); cfg = load_config(args.config); spcfg=cfg["self_play"]; tcfg=cfg["training"]; ecfg=cfg["evaluation"]; ccfg=cfg["checkpoint"]; lcfg=cfg["logging"]
    cycles = 0 if args.cycles is None else args.cycles; episodes=args.episodes or int(spcfg["episodes_per_cycle"]); sims=args.mcts_simulations or int(spcfg["simulations"]); workers=args.workers or int(spcfg["workers"]); updates=args.updates or int(tcfg["updates_per_cycle"]); batch=args.batch_size or int(tcfg["batch_size"]); eval_games=args.evaluation_games or int(ecfg["games_per_opponent"])
    RUNS.mkdir(parents=True, exist_ok=True); stop_file = RUNS / "STOP"; stop_file.unlink(missing_ok=True); pause_file = RUNS / "PAUSE"
    checkpoint, replay, metrics = RUNS / lcfg["latest_checkpoint"], RUNS / lcfg["replay_path"], RUNS / lcfg["metrics_path"]
    champion, candidate = RUNS / "champion.pt", RUNS / "candidate.pt"
    best_external, best_metrics = RUNS / "best_external.pt", RUNS / "best_external_metrics.json"
    checkpoints = RUNS / lcfg["history_dir"]; checkpoints.mkdir(exist_ok=True); (RUNS / "config_snapshot.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    write_run_manifest(RUNS, args.config)
    if not champion.exists() and checkpoint.exists():
        shutil.copy2(checkpoint, champion)
    if not champion.exists():
        # First-ever run in this runs/ directory: nothing to seed the local
        # self-play opponent from yet. Previously this raised and made the
        # operator manually bootstrap a checkpoint by hand before every fresh
        # runs/ directory (discovered the hard way: a brand-new formal run
        # crashed immediately with "No checkpoint is available"). Build a
        # freshly-initialised network with the configured architecture
        # instead -- training from a random network is the normal AlphaZero
        # starting point, not an error condition.
        from ai.network import KARDSNet
        hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 128))
        KARDSNet(hidden_dim=hidden_dim).save_checkpoint(champion)
    # `latest.pt` is the historical best generalisation model, while Champion
    # remains the local self-play opponent.  Neither is overwritten by a
    # candidate that merely exploits the immediately preceding Champion.
    if not best_external.exists(): shutil.copy2(champion, best_external)
    shutil.copy2(best_external, checkpoint)
    # Continuous training (2026-11-21, AlphaZero-style): one network keeps
    # learning across cycles (`trainee.pt`) and generates self-play data,
    # whether or not it beats Champion. Previously every cycle restarted from a
    # fresh copy of Champion and self-played with Champion, so a cycle that did
    # not promote threw its 500 updates away -- the run sat at training_step
    # 1000 for 90+ cycles. Champion is now only the "best model so far" record
    # used for gating/evaluation/latest.pt. Set training.continuous: false to
    # get the old behaviour back.
    continuous = bool(tcfg.get("continuous", True))
    trainee = RUNS / "trainee.pt"
    if continuous and not trainee.exists(): shutil.copy2(champion, trainee)
    learner = trainee if continuous else champion
    iteration = 0
    while cycles == 0 or iteration < cycles:
        if not wait_for_resume(stop_file, pause_file): break
        iteration += 1
        write_run_manifest(RUNS, args.config, extra={"last_cycle": iteration})
        selfplay_args = ["selfplay.py", "--episodes", str(episodes), "--mcts-simulations", str(sims), "--workers", str(workers), "--max-actions", str(spcfg["max_actions"]), "--device", args.device, "--replay", str(replay), "--append-replay", "--replay-capacity", str(spcfg["replay_capacity"]), "--metrics", str(metrics), "--progress-every", "1", "--async-mcts", "--max-pending-leaves", str(spcfg["max_pending_leaves"]), "--model", str(learner)]
        # Fresh self-play seed every cycle (previously always the default 0, so
        # every cycle replayed the same deck pairings/RNG stream). Derived from
        # the learner's training_step so it also keeps changing across restarts.
        learner_step = int(__import__("torch").load(learner, map_location="cpu", weights_only=False).get("training_step", 0))
        selfplay_args += ["--seed", str(int(spcfg.get("seed", 20261122)) + learner_step + iteration)]
        if spcfg.get("search"):
            selfplay_args += ["--search", str(spcfg["search"]), "--pimc-worlds", str(spcfg.get("pimc_worlds", 4))]
        if spcfg.get("meta_decks"):
            selfplay_args += ["--meta-decks"]
        else:
            selfplay_args += ["--deck-pool", json.dumps(spcfg["deck_pool"])]
        execute(selfplay_args)
        if stop_file.exists(): break

        if not wait_for_resume(stop_file, pause_file): break
        shutil.copy2(learner, candidate)
        payload_before = __import__("torch").load(candidate, map_location="cpu", weights_only=False)
        prior_step = int(payload_before.get("training_step", 0)); decay_every = int(tcfg.get("lr_decay_every_steps", 1000))
        learning_rate = max(float(tcfg.get("min_learning_rate", tcfg["learning_rate"])),
                            float(tcfg["learning_rate"]) * float(tcfg.get("lr_decay", 1.0)) ** (prior_step // max(1, decay_every)))
        execute(["train.py", "--replay", str(replay), "--updates", str(updates), "--batch-size", str(batch), "--learning-rate", str(learning_rate), "--checkpoint", str(candidate), "--model", str(candidate), "--metrics", str(metrics), "--device", args.device])
        if stop_file.exists(): break
        torch = __import__("torch"); payload = torch.load(candidate, map_location="cpu", weights_only=False)
        payload.setdefault("metadata", {}).update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "config_snapshot": cfg, "learning_rate": learning_rate,
                                                     "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        torch.save(payload, candidate); step=int(payload.get("training_step", 0))
        if continuous: shutil.copy2(candidate, trainee)  # keep this cycle's learning regardless of the gate
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
        # Anchor the external check to the best external scores ever recorded
        # (best_external_metrics.json), not to the current Champion: comparing
        # against a moving Champion let every promotion be a little worse than
        # the last, and 80 cycles of that slid Champion from 87%/57% to
        # 50%/34% vs random/rule (2026-11-22).
        anchor = json.loads(best_metrics.read_text(encoding="utf-8")).get("scores", {}) if best_metrics.exists() else {}
        external_ok = all(candidate_scores[name]["win_rate"] + tolerance >= anchor.get(name, champion_scores[name])["win_rate"] for name in ecfg["opponents"])
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
