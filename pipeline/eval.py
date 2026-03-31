"""Evaluate probe predictions against ground truth annotations.

Reads scores.feather + split.feather from a probe directory, computes per-head
metrics on the test set. Writes eval.json alongside the probe.

Metrics per head type:
  Binary:     AUC, AUPRC, nAUPRC, MCC, accuracy, F1, prevalence
  Continuous: Pearson r, Spearman rho, R², MAE
  Categorical: accuracy, top-1 accuracy

Usage:
    uv run python pipeline/eval.py --probe-dir $ACTS/probe_v12
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from loaders import load_variants


def _eval_binary(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Compute all binary classification metrics."""
    prevalence = truth.mean()
    auc = float(roc_auc_score(truth, pred))
    auprc = float(average_precision_score(truth, pred))
    # Normalized AUPRC: (AUPRC - prevalence) / (1 - prevalence)
    # 0 = random, 1 = perfect, negative = worse than random
    nauprc = (auprc - prevalence) / (1 - prevalence) if prevalence < 1 else 0.0
    # Threshold at 0.5 for discrete metrics
    pred_binary = (pred >= 0.5).astype(int)
    mcc = float(matthews_corrcoef(truth, pred_binary))
    acc = float(accuracy_score(truth, pred_binary))
    f1 = float(f1_score(truth, pred_binary, zero_division=0))
    return {
        "kind": "binary",
        "auc": round(auc, 4),
        "auprc": round(auprc, 4),
        "nauprc": round(nauprc, 4),
        "mcc": round(mcc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "prevalence": round(float(prevalence), 4),
    }


def _eval_continuous(pred: np.ndarray, truth: np.ndarray) -> dict | None:
    """Compute all continuous regression metrics. Returns None if degenerate."""
    if np.std(pred) < 1e-10 or np.std(truth) < 1e-10:
        return None
    r, _ = pearsonr(pred, truth)
    rho, _ = spearmanr(pred, truth)
    if np.isnan(r):
        return None
    r2 = float(r2_score(truth, pred))
    mae = float(mean_absolute_error(truth, pred))
    return {
        "kind": "continuous",
        "correlation": round(float(r), 4),
        "spearman": round(float(rho), 4),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
    }


def main(
    probe_dir: Path = typer.Option(..., help="Probe directory with scores.feather + split.feather"),
    min_samples: int = typer.Option(20, help="Minimum samples for a head to be evaluated"),
) -> None:
    scores = pl.read_ipc(str(probe_dir / "scores.feather"))
    split = pl.read_ipc(str(probe_dir / "split.feather"))
    test_ids = set(split.filter(pl.col("split") == "test")["variant_id"].to_list())

    variants = load_variants()
    df = (
        scores
        .join(variants, on="variant_id", how="left")
        .filter(pl.col("variant_id").is_in(list(test_ids)))
    )
    logger.info(f"Test set: {df.height:,} variants")

    eval_results: dict[str, dict] = {}

    def evaluate_head(head: str, pred: np.ndarray, truth: np.ndarray):
        valid = ~(np.isnan(pred) | np.isnan(truth))
        n = int(valid.sum())
        if n < min_samples:
            return
        p, t = pred[valid], truth[valid]
        unique = len(np.unique(t))
        if unique == 2:
            result = _eval_binary(p, t)
        elif unique > 2 and np.all(t == t.astype(int)) and t.max() < 100:
            # Categorical (integer labels, few unique values)
            result = {"kind": "categorical", "accuracy": round(float(accuracy_score(t, np.round(p).clip(0, t.max()))), 4)}
        else:
            result = _eval_continuous(p, t)
        if result is None:
            return
        result["n"] = n
        eval_results[head] = result

    # Effect heads: score_* columns
    for col in [c for c in df.columns if c.startswith("score_") and c != "score_pathogenic"]:
        head = col[6:]
        if head not in df.columns:
            continue
        evaluate_head(head, df[col].to_numpy().astype(np.float64), df[head].to_numpy().astype(np.float64))

    # Disruption heads: ref_score_* columns
    for col in [c for c in df.columns if c.startswith("ref_score_")]:
        head = col[10:]
        if head not in df.columns or head in eval_results:
            continue
        evaluate_head(head, df[col].to_numpy().astype(np.float64), df[head].to_numpy().astype(np.float64))

    # Pathogenicity (from clinical labels, not annotation column)
    df_path = df.filter(pl.col("label").is_in(["benign", "pathogenic"]))
    y_true = (df_path["label"] == "pathogenic").to_numpy().astype(int)
    y_score = df_path["score_pathogenic"].to_numpy()
    valid = ~np.isnan(y_score)
    if valid.sum() >= min_samples:
        eval_results["pathogenic"] = _eval_binary(y_score[valid], y_true[valid])
        eval_results["pathogenic"]["n"] = int(valid.sum())

    # Write
    out_path = probe_dir / "eval.json"
    out_path.write_text(json.dumps(eval_results, indent=2))

    # Summary
    binary = [v for v in eval_results.values() if v["kind"] == "binary"]
    continuous = [v for v in eval_results.values() if v["kind"] == "continuous"]
    aucs = [v["auc"] for v in binary]
    nauprcs = [v["nauprc"] for v in binary]
    corrs = [v["correlation"] for v in continuous]
    rhos = [v["spearman"] for v in continuous]

    logger.info(
        f"eval.json: {len(eval_results)} heads ({len(binary)} binary, {len(continuous)} continuous)"
    )
    if "pathogenic" in eval_results:
        p = eval_results["pathogenic"]
        logger.info(f"  pathogenic: AUC={p['auc']}, AUPRC={p['auprc']}, nAUPRC={p['nauprc']}, MCC={p['mcc']}")
    if aucs:
        logger.info(f"  binary mean: AUC={sum(aucs)/len(aucs):.4f}, nAUPRC={sum(nauprcs)/len(nauprcs):.4f}")
    if corrs:
        logger.info(f"  continuous mean: r={sum(corrs)/len(corrs):.4f}, rho={sum(rhos)/len(rhos):.4f}")


if __name__ == "__main__":
    typer.run(main)
