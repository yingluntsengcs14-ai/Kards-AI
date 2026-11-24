"""Unattended training supervisor: restarts crashed runs and applies fixes.

Run once on the training desktop, from the repo root, and leave it open:

    python tools/supervisor.py

Every ``--interval`` seconds it:

1. Restarts training (via ``start_training.py``) if the training process is
   gone without anyone asking it to stop -- e.g. an overnight crash. At most
   ``--max-restarts`` times, so a deterministic crash does not loop forever.
2. Runs ``git fetch``. If ``origin/<branch>`` has new commits and one of their
   messages contains ``[auto-restart]``, it stops the running training (the
   whole process tree), ``git pull --ff-only``, and starts training again on
   the new code. A commit that also contains ``[fresh-runs]`` makes it rename
   ``runs/`` to ``runs_<timestamp>/`` first (for changes that are not
   checkpoint-compatible). Commits without the marker are left alone: pulling
   under a running cycle would make its next selfplay/train subprocess run
   different code from the one that started the cycle.

Nothing is deleted: old runs are renamed. Creating ``runs/STOP`` (normal
graceful stop) or ``SUPERVISOR_STOP`` in the repo root stops supervision;
with STOP present the supervisor will not restart training.

Log: ``supervisor.log`` in the repo root, mirrored to ``runs/logs/supervisor.log``.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from start_training import _process_alive  # noqa: E402

RESTART_MARKER = "[auto-restart]"
FRESH_MARKER = "[fresh-runs]"
LOG = ROOT / "supervisor.log"


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    # Also mirrored into runs/logs/, the folder remote checks can read.
    mirror = ROOT / "runs" / "logs" / "supervisor.log"
    for path in (LOG, mirror):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check, timeout=300)


def training_pid(runs: Path) -> int | None:
    try:
        return int((runs / "training.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def training_alive(runs: Path) -> bool:
    pid = training_pid(runs)
    return pid is not None and _process_alive(pid)


def stop_training(runs: Path) -> None:
    pid = training_pid(runs)
    if pid is None or not _process_alive(pid):
        return
    log(f"stopping training process tree (pid {pid})")
    if sys.platform == "win32":
        # auto_train.py runs selfplay/train/evaluate as child processes; /T
        # takes the whole tree down so no orphan keeps the GPU busy.
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
    else:
        try: os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError: pass
    for _ in range(60):
        if not _process_alive(pid):
            break
        time.sleep(2)


def start_training(config: str) -> bool:
    result = subprocess.run([sys.executable, "start_training.py", "--config", config], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"start_training failed: {(result.stderr or result.stdout).strip()[-800:]}")
        return False
    log("training started")
    return True


def pending_commits(branch: str) -> list[str]:
    """Full messages of commits on origin/<branch> not yet in HEAD."""
    fetched = git("fetch", "--quiet", "origin", branch)
    if fetched.returncode != 0:
        log(f"git fetch failed: {fetched.stderr.strip()[-300:]}")
        return []
    out = git("log", f"HEAD..origin/{branch}", "--format=%H %B%x00").stdout
    return [entry.strip() for entry in out.split("\x00") if entry.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/formal_training.yaml")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--interval", type=int, default=300, help="seconds between checks")
    parser.add_argument("--max-restarts", type=int, default=5, help="crash restarts allowed (fix-driven restarts do not count)")
    args = parser.parse_args()
    runs = ROOT / "runs"
    crash_restarts = 0
    log(f"supervisor started at {git('rev-parse', '--short', 'HEAD').stdout.strip()} (interval {args.interval}s)")
    if not training_alive(runs) and not (runs / "STOP").exists():
        start_training(args.config)
    while True:
        if (ROOT / "SUPERVISOR_STOP").exists():
            log("SUPERVISOR_STOP found; supervisor exits (training left as is)")
            return
        commits = pending_commits(args.branch)
        if any(RESTART_MARKER in message for message in commits):
            fresh = any(FRESH_MARKER in message for message in commits)
            log(f"{len(commits)} new commit(s) with {RESTART_MARKER}: {[m.split(chr(10))[0][:80] for m in commits]}")
            stop_training(runs)
            pulled = git("pull", "--ff-only", "origin", args.branch)
            if pulled.returncode != 0:
                log(f"git pull failed, restarting on the current code: {pulled.stderr.strip()[-500:]}")
            else:
                log(f"pulled; now at {git('rev-parse', '--short', 'HEAD').stdout.strip()}")
                if fresh and runs.exists():
                    target = ROOT / f"runs_{datetime.now():%Y%m%d_%H%M}"
                    runs.rename(target)
                    log(f"{FRESH_MARKER}: renamed runs/ to {target.name}/")
            (runs / "STOP").unlink(missing_ok=True)
            start_training(args.config)
        elif not training_alive(runs):
            if (runs / "STOP").exists():
                log("training stopped via runs/STOP; not restarting")
            elif crash_restarts < args.max_restarts:
                crash_restarts += 1
                log(f"training process is gone; restart {crash_restarts}/{args.max_restarts}")
                start_training(args.config)
            else:
                log("training keeps dying; restart limit reached, waiting for a fix commit")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
