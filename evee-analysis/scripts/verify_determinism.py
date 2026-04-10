#!/usr/bin/env python3
"""Verify deterministic outputs by running an analysis script twice and comparing checksums.

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/verify_determinism.py demeter2
    uv run python evee-analysis/scripts/verify_determinism.py chronos
    uv run python evee-analysis/scripts/verify_determinism.py both
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
SCRIPTS_DIR = EVEE_ROOT / "scripts"
OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"

SCRIPTS = {
    "demeter2": SCRIPTS_DIR / "run_neighbor_depmap_analysis.py",
    "chronos": SCRIPTS_DIR / "run_neighbor_chronos_analysis.py",
}


def collect_outputs(dataset: str) -> dict[str, Path]:
    """Gather all output files for a dataset."""
    prefix = "chronos_" if dataset == "chronos" else ""
    files = {}
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if dataset == "chronos" and p.name.startswith("chronos_"):
            files[p.name] = p
        elif dataset == "demeter2" and not p.name.startswith("chronos_"):
            if p.suffix in (".parquet", ".json", ".md", ".txt"):
                files[p.name] = p
    for p in sorted(FIG_DIR.iterdir()):
        if not p.is_file():
            continue
        if dataset == "chronos" and p.name.startswith("chronos_"):
            files[p.name] = p
        elif dataset == "demeter2" and not p.name.startswith("chronos_"):
            if p.suffix == ".png":
                files[p.name] = p
    return files


def backup_outputs(files: dict[str, Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, backup_dir / name)


def run_script(dataset: str) -> int:
    script = SCRIPTS[dataset]
    env_cmd = f"MPLCONFIGDIR=/tmp/mpl_config PYTHONHASHSEED=42"
    cmd = f"{env_cmd} {sys.executable} {script}"
    print(f"\n{'=' * 60}")
    print(f"Running {script.name} ...")
    print(f"{'=' * 60}")
    result = subprocess.run(
        cmd, shell=True, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        print(f"FAILED (exit code {result.returncode})")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return result.returncode


def compare_files(dir_a: Path, dir_b: Path) -> list[dict]:
    """Byte-level comparison of all files in two directories."""
    import hashlib

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    files_a = {f.name: f for f in sorted(dir_a.iterdir()) if f.is_file()}
    files_b = {f.name: f for f in sorted(dir_b.iterdir()) if f.is_file()}

    skip = {"checksums.sha256", "run_manifest.txt", "environment.txt"}

    results = []
    for name in sorted(set(files_a.keys()) | set(files_b.keys())):
        if name in skip:
            results.append({"file": name, "match": "skipped", "reason": "non-deterministic metadata"})
            continue

        fa = files_a.get(name)
        fb = files_b.get(name)
        if fa is None:
            results.append({"file": name, "match": False, "reason": "only in run 2"})
        elif fb is None:
            results.append({"file": name, "match": False, "reason": "only in run 1"})
        else:
            ha, hb = sha256(fa), sha256(fb)
            results.append({
                "file": name,
                "match": ha == hb,
                "hash_run1": ha[:16],
                "hash_run2": hb[:16],
            })
    return results


def verify_one(dataset: str) -> bool:
    print(f"\n{'#' * 60}")
    print(f"# Verifying determinism: {dataset.upper()}")
    print(f"{'#' * 60}")

    # Run 1
    rc = run_script(dataset)
    if rc != 0:
        print(f"\nRun 1 FAILED for {dataset}. Aborting verification.")
        return False

    run1_files = collect_outputs(dataset)
    tmpdir = Path(tempfile.mkdtemp(prefix=f"verify_{dataset}_run1_"))
    backup_outputs(run1_files, tmpdir)
    print(f"\nRun 1 complete. Backed up {len(run1_files)} files to {tmpdir}")

    # Run 2
    rc = run_script(dataset)
    if rc != 0:
        print(f"\nRun 2 FAILED for {dataset}. Aborting verification.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False

    run2_files = collect_outputs(dataset)
    tmpdir2 = Path(tempfile.mkdtemp(prefix=f"verify_{dataset}_run2_"))
    backup_outputs(run2_files, tmpdir2)
    print(f"\nRun 2 complete. Backed up {len(run2_files)} files to {tmpdir2}")

    # Compare
    results = compare_files(tmpdir, tmpdir2)
    print(f"\n{'=' * 60}")
    print(f"DETERMINISM RESULTS: {dataset.upper()}")
    print(f"{'=' * 60}")

    all_pass = True
    for r in results:
        status = r["match"]
        if status == "skipped":
            symbol = "~"
        elif status:
            symbol = "+"
        else:
            symbol = "X"
            all_pass = False
        detail = ""
        if "reason" in r:
            detail = f"  ({r['reason']})"
        elif not status and "hash_run1" in r:
            detail = f"  (run1={r['hash_run1']}... run2={r['hash_run2']}...)"
        print(f"  [{symbol}] {r['file']}{detail}")

    if all_pass:
        print(f"\nPASSED: All outputs are deterministic for {dataset}.")
    else:
        print(f"\nFAILED: Some outputs differ between runs for {dataset}.")

    shutil.rmtree(tmpdir, ignore_errors=True)
    shutil.rmtree(tmpdir2, ignore_errors=True)
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify analysis pipeline determinism")
    parser.add_argument("dataset", choices=["demeter2", "chronos", "both"],
                        help="Which pipeline to verify")
    args = parser.parse_args()

    datasets = ["demeter2", "chronos"] if args.dataset == "both" else [args.dataset]
    results = {}
    for ds in datasets:
        results[ds] = verify_one(ds)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    for ds, passed in results.items():
        print(f"  {ds:12s} {'PASSED' if passed else 'FAILED'}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
