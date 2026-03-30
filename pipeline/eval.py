"""Evaluate probe predictions against ground truth annotations.

Reads scores.feather + split.feather from a probe directory, computes per-head
AUC (binary), Pearson correlation (continuous), and accuracy (categorical)
on the test set. Writes eval.json alongside the probe.

Usage:
    uv run python pipeline/eval.py --probe-dir $ACTS/probe_v9
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from paths import MAYO_DATA


def main(
    probe_dir: Path = typer.Option(..., help="Probe directory with scores.feather + split.feather"),
    preset: str = typer.Option("deconfounded-full", help="Annotation preset"),
    min_samples: int = typer.Option(20, help="Minimum samples for a head to be evaluated"),
) -> None:
    scores = pl.read_ipc(str(probe_dir / "scores.feather"))
    split = pl.read_ipc(str(probe_dir / "split.feather"))
    test_ids = set(split.filter(pl.col("split") == "test")["variant_id"].to_list())

    gt = pl.read_ipc(str(MAYO_DATA / "clinvar" / preset / "annotations.feather"))
    meta = pl.read_ipc(str(MAYO_DATA / "clinvar" / preset / "metadata.feather")).select("variant_id", "label")

    df = (
        scores
        .join(gt, on="variant_id", how="left")
        .join(meta, on="variant_id", how="left")
        .filter(pl.col("variant_id").is_in(list(test_ids)))
    )
    logger.info(f"Test set: {df.height:,} variants")

    eval_results: dict[str, dict] = {}

    # Effect heads: score_* columns
    for col in [c for c in df.columns if c.startswith("score_") and c != "score_pathogenic"]:
        head = col[6:]
        if head not in gt.columns:
            continue
        pred = df[col].to_numpy().astype(np.float64)
        truth = df[head].to_numpy().astype(np.float64)
        valid = ~(np.isnan(pred) | np.isnan(truth))
        if valid.sum() < min_samples:
            continue
        p, t = pred[valid], truth[valid]
        if len(np.unique(t)) == 2:
            eval_results[head] = {"kind": "binary", "auc": round(float(roc_auc_score(t, p)), 4), "n": int(valid.sum())}
        else:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                eval_results[head] = {"kind": "continuous", "correlation": round(float(r), 4), "n": int(valid.sum())}

    # Disruption heads: ref_score_* columns (use ref view as baseline)
    for col in [c for c in df.columns if c.startswith("ref_score_")]:
        head = col[10:]
        if head not in gt.columns or head in eval_results:
            continue
        pred = df[col].to_numpy().astype(np.float64)
        truth = df[head].to_numpy().astype(np.float64)
        valid = ~(np.isnan(pred) | np.isnan(truth))
        if valid.sum() < min_samples:
            continue
        p, t = pred[valid], truth[valid]
        if len(np.unique(t)) == 2:
            eval_results[head] = {"kind": "binary", "auc": round(float(roc_auc_score(t, p)), 4), "n": int(valid.sum())}
        else:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                eval_results[head] = {"kind": "continuous", "correlation": round(float(r), 4), "n": int(valid.sum())}

    # Pathogenicity (always evaluated from clinical labels)
    df_path = df.filter(pl.col("label").is_in(["benign", "pathogenic"]))
    y_true = (df_path["label"] == "pathogenic").to_numpy().astype(int)
    y_score = df_path["score_pathogenic"].to_numpy()
    valid = ~np.isnan(y_score)
    if valid.sum() >= min_samples:
        eval_results["pathogenic"] = {
            "kind": "binary",
            "auc": round(float(roc_auc_score(y_true[valid], y_score[valid])), 4),
            "n": int(valid.sum()),
        }

    # Write
    out_path = probe_dir / "eval.json"
    out_path.write_text(json.dumps(eval_results, indent=2))

    aucs = [v["auc"] for v in eval_results.values() if "auc" in v]
    corrs = [v["correlation"] for v in eval_results.values() if "correlation" in v]
    logger.info(
        f"eval.json: {len(eval_results)} heads"
        f" | pathogenic AUC={eval_results.get('pathogenic', {}).get('auc', '?')}"
        f" | mean AUC={sum(aucs) / len(aucs):.3f}" if aucs else ""
        f" | mean corr={sum(corrs) / len(corrs):.3f}" if corrs else ""
    )


if __name__ == "__main__":
    typer.run(main)
