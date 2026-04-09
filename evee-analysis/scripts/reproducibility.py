"""Shared reproducibility infrastructure for analysis pipelines.

Provides:
  - RNG seed enforcement (numpy + stdlib random + PYTHONHASHSEED)
  - run_config.json serialization
  - environment.txt snapshot (Python version, pip freeze)
  - run_manifest.txt (command, timestamp, git hash)
  - Output checksums for determinism verification
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def enforce_seeds(seed: int) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"  Seeds enforced: random={seed}, numpy={seed}, PYTHONHASHSEED={seed}")


def save_run_config(out_dir: Path, config: dict) -> Path:
    path = out_dir / "run_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    log.info(f"  Saved {path.name}")
    return path


def save_environment(out_dir: Path) -> Path:
    path = out_dir / "environment.txt"
    lines = [
        f"python_version: {sys.version}",
        f"platform: {platform.platform()}",
        f"architecture: {platform.machine()}",
        f"numpy: {np.__version__}",
    ]
    try:
        import polars
        lines.append(f"polars: {polars.__version__}")
    except ImportError:
        pass
    try:
        import duckdb
        lines.append(f"duckdb: {duckdb.__version__}")
    except ImportError:
        pass
    try:
        import matplotlib
        lines.append(f"matplotlib: {matplotlib.__version__}")
    except ImportError:
        pass

    lines.append("")
    lines.append("--- pip freeze ---")
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        lines.append(freeze.strip())
    except Exception as e:
        lines.append(f"(pip freeze failed: {e})")

    path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {path.name}")
    return path


def save_run_manifest(out_dir: Path) -> Path:
    path = out_dir / "run_manifest.txt"
    lines = [
        f"timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"command: {' '.join(sys.argv)}",
        f"script: {os.path.abspath(sys.argv[0])}",
        f"cwd: {os.getcwd()}",
        f"python: {sys.executable}",
    ]

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
        lines.append(f"git_commit: {git_hash}")
    except Exception:
        lines.append("git_commit: N/A")

    try:
        git_dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
        lines.append(f"git_dirty: {bool(git_dirty)}")
    except Exception:
        lines.append("git_dirty: N/A")

    path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {path.name}")
    return path


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checksums(out_dir: Path, fig_dir: Path | None = None) -> Path:
    """Write SHA-256 checksums of all output artifacts."""
    path = out_dir / "checksums.sha256"
    lines = []
    dirs = [out_dir]
    if fig_dir is not None:
        dirs.append(fig_dir)
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in (".parquet", ".json", ".md", ".txt", ".png"):
                if f.name == "checksums.sha256":
                    continue
                cs = checksum_file(f)
                rel = f.relative_to(d.parent)
                lines.append(f"{cs}  {rel}")
    path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {path.name} ({len(lines)} files)")
    return path


def verify_checksums(checksums_a: Path, checksums_b: Path) -> list[dict]:
    """Compare two checksum manifests. Returns list of comparison dicts."""
    def parse(path: Path) -> dict[str, str]:
        entries = {}
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) == 2:
                entries[parts[1]] = parts[0]
        return entries

    a = parse(checksums_a)
    b = parse(checksums_b)
    all_files = sorted(set(a.keys()) | set(b.keys()))

    results = []
    for f in all_files:
        ha = a.get(f)
        hb = b.get(f)
        if ha is None:
            results.append({"file": f, "match": False, "reason": "only in run B"})
        elif hb is None:
            results.append({"file": f, "match": False, "reason": "only in run A"})
        else:
            results.append({"file": f, "match": ha == hb, "hash_a": ha[:16], "hash_b": hb[:16]})
    return results
