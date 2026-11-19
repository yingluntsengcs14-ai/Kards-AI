"""Formal unattended-training launcher with environment and lock checks."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from ai.config import load_config
from ai.runtime import runs_dir

ROOT = Path(__file__).parent


def _process_alive(target_pid: int) -> bool:
    """Existence-only check -- never signals/kills ``target_pid``.

    ``os.kill(pid, 0)`` is a safe no-op existence probe on POSIX, but on
    Windows ``os.kill`` has no signal-0 special case: it calls
    ``TerminateProcess(handle, 0)``, which actually **terminates** whatever
    process currently holds that PID. Combined with how aggressively Windows
    recycles PIDs, calling it here could both (a) silently kill an unrelated
    process that happens to have reused the dead training run's old PID, and
    (b) leave the stale lock in place anyway (no exception is raised, so the
    cleanup branch below never runs) -- which is exactly what happened the
    first time this ran: the training.lock/training.pid from a crashed
    first-ever run were never cleared, blocking every later launch attempt
    with "Training lock already exists" even though nothing was actually
    running. Query-only on Windows (``tasklist``), real existence-check
    (``os.kill(pid, 0)``) on POSIX.
    """
    if sys.platform == "win32":
        try:
            output = subprocess.run(["tasklist", "/FI", f"PID eq {target_pid}", "/NH"],
                                     capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return True  # can't tell -- assume alive rather than risk clearing a live lock
        return str(target_pid) in output
    try:
        os.kill(target_pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default=ROOT / "configs/formal_training.yaml"); p.add_argument("--runs"); p.add_argument("--foreground", action="store_true")
    a = p.parse_args(); config = Path(a.config).resolve(); cfg = load_config(config)
    if not torch.cuda.is_available(): raise RuntimeError("Formal training requires CUDA")
    if cfg["self_play"]["workers"] < 1 or cfg["self_play"]["simulations"] < 1: raise ValueError("Invalid self-play configuration")
    runs = Path(a.runs) if a.runs else runs_dir(); runs.mkdir(parents=True, exist_ok=True)
    lock = runs / "training.lock"; pid = runs / "training.pid"
    if pid.exists():
        try:
            existing_pid = int(pid.read_text(encoding="utf-8"))
        except ValueError:
            existing_pid = None
        if existing_pid is None or not _process_alive(existing_pid):
            pid.unlink(missing_ok=True); lock.unlink(missing_ok=True)
    if lock.exists() or pid.exists(): raise RuntimeError(f"Training lock already exists: {lock}")
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "cuda": torch.cuda.get_device_name(0), "config": str(config), "config_snapshot": cfg}
    (runs / "startup.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    env = os.environ.copy(); env["KARDS_RUNS_DIR"] = str(runs)
    cmd = [sys.executable, "auto_train.py", "--config", str(config)]
    if a.foreground:
        lock.touch(); pid.write_text(str(os.getpid()), encoding="utf-8")
        try: subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        finally: lock.unlink(missing_ok=True); pid.unlink(missing_ok=True)
    else:
        (runs / "logs").mkdir(exist_ok=True)
        with (runs / "logs" / "training.log").open("a", encoding="utf-8") as out:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        lock.touch(); pid.write_text(str(proc.pid), encoding="utf-8"); print(json.dumps({**info, "pid": proc.pid, "runs": str(runs)}))
if __name__ == "__main__": main()
