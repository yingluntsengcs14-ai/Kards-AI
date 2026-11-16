"""Locations for mutable training artifacts.

Runtime data is deliberately kept off the repository (and, by default, off
the system drive) so an unattended training run does not churn OneDrive/C:.
Set KARDS_RUNS_DIR to override the location for another machine or disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runs_dir() -> Path:
    configured = os.environ.get("KARDS_RUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    d_drive = Path("D:/")
    return d_drive / "KardsAI" / "runs" if d_drive.exists() else ROOT / "runs"


def runtime_path(value: str | Path | None, default_name: str) -> Path:
    """Resolve legacy ``runs/...`` defaults into the selected runtime folder."""
    path = Path(value) if value else Path(default_name)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "runs":
        path = Path(*path.parts[1:])
    return runs_dir() / path


def _git_info() -> dict:
    """Best-effort git provenance: commit hash + whether the tree is dirty.

    A dirty tree is only recorded, never fatal -- refusing to start an
    overnight run because of an unrelated unstaged file would cost more
    (a missed training night) than it protects against. Whoever reads the
    manifest later can judge whether the dirty state matters for that run.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - no git / not a repo
        return {"commit": None, "dirty": None, "error": str(exc)}


def _rule_stats() -> dict:
    """Implemented/partial/unresolved counts + a short hash of the rule
    store that actually drives the simulator at runtime (``NativeRuleEngine``
    loads ``data/rules/card_rules.json`` in preference to live parser
    output), so a run's manifest records exactly which rule generation
    produced its results."""
    path = ROOT / "data/rules/card_rules.json"
    try:
        raw = path.read_bytes()
        rules = json.loads(raw).get("rules", [])
        counts: dict[str, int] = {}
        for rule in rules:
            status = rule.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {"file": "data/rules/card_rules.json", "sha256_12": hashlib.sha256(raw).hexdigest()[:12],
                "total": len(rules), "by_status": counts}
    except Exception as exc:  # pragma: no cover - missing/unreadable file
        return {"file": "data/rules/card_rules.json", "error": str(exc)}


def _environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - torch always present in practice
        info["torch_error"] = str(exc)
    return info


def write_run_manifest(run_dir: Path, config_path: str | Path, seed: int | None = None,
                        extra: dict | None = None) -> Path:
    """Write/refresh ``manifest.json`` at the top of a run directory.

    Captures what ``claude/訓練工作流.md``'s run-folder design calls for:
    git commit + dirty flag, the rule generation (hash + implemented/
    partial/unresolved counts) that produced this run's self-play data,
    the config path, the environment (Python/PyTorch/CUDA/GPU), and a
    seed if the caller has one. ``run_id`` and ``start_time_utc`` are
    fixed the first time this is called for a given ``run_dir`` and kept
    on every later refresh (e.g. once per ``auto_train.py`` cycle), so the
    manifest always reflects the *current* git/rule state while still
    recording when the run itself began.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": existing.get("run_id") or run_dir.name,
        "start_time_utc": existing.get("start_time_utc") or now,
        "last_updated_utc": now,
        "config": str(config_path),
        "seed": seed if seed is not None else existing.get("seed"),
        "git": _git_info(),
        "rules": _rule_stats(),
        "environment": _environment_info(),
    }
    if extra:
        manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
