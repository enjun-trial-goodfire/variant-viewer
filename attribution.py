"""Per-variant attribution for pathogenicity predictions.

Two-tier approach:
1. Ridge regression: y = beta^T scores + intercept (excluding clinical predictors)
2. Per variant: rank heads by |beta_h * score_ih| (contribution to prediction)
3. Split into "baseline" (always important) and "specific" (unique to this variant)

    >>> model = AttributionModel.fit(scores_path, split_path)
    >>> model.save(probe_dir / "attribution.pt")
    >>> attr = model.attribute(scores_df)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger

CLINICAL_PREDICTORS = frozenset({
    "alphamissense_c", "bayesdel_c", "cadd_c", "cadd_wg_c", "clinpred_c",
    "deogen2_c", "eve_c", "mcap_c", "metalr_c", "mpc_c", "mutpred_c",
    "mvp_c", "polyphen_c", "primateai_c", "revel_c", "sift_c", "vest4_c",
    "spliceai_ag_c", "spliceai_al_c", "spliceai_dg_c", "spliceai_dl_c",
    "spliceai_max_c",
})


def _discover_heads(scores: pl.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (diff_heads, ref_heads) excluding clinical predictors."""
    diff = tuple(
        c.removeprefix("score_") for c in scores.columns
        if c.startswith("score_") and c != "score_pathogenic"
        and not c.startswith("ref_score_") and not c.startswith("var_score_")
        and c.removeprefix("score_") not in CLINICAL_PREDICTORS
    )
    ref = tuple(
        c.removeprefix("ref_score_") for c in scores.columns
        if c.startswith("ref_score_")
        and f"var_score_{c.removeprefix('ref_score_')}" in scores.columns
        and c.removeprefix("ref_score_") not in CLINICAL_PREDICTORS
    )
    return diff, ref


def _feature_matrix(df: pl.DataFrame, diff: tuple[str, ...], ref: tuple[str, ...]) -> np.ndarray:
    """[N, p] float32 matrix: diff scores + ref-var deltas."""
    parts = [np.nan_to_num(df.select([f"score_{h}" for h in diff]).to_numpy(), nan=0.5)]
    if ref:
        r = np.nan_to_num(df.select([f"ref_score_{h}" for h in ref]).to_numpy(), nan=0.5)
        v = np.nan_to_num(df.select([f"var_score_{h}" for h in ref]).to_numpy(), nan=0.5)
        parts.append(v - r)
    return np.hstack(parts).astype(np.float32)


class AttributionModel:
    """Two-tier per-variant attribution.

    Baseline heads appear in the top-k for most variants (always-on signal).
    Variant-specific heads capture what's unique about each variant.
    """

    def __init__(
        self,
        beta: torch.Tensor,
        intercept: float,
        head_names: tuple[tuple[str, str], ...],
        diff_heads: tuple[str, ...],
        ref_heads: tuple[str, ...],
        baseline_heads: tuple[int, ...],
    ):
        self.beta = beta
        self.intercept = intercept
        self.head_names = head_names
        self.diff_heads = diff_heads
        self.ref_heads = ref_heads
        self.baseline_heads = set(baseline_heads)

    @classmethod
    def fit(
        cls,
        scores_path: Path,
        split_path: Path,
        k_baseline: int = 10,
        baseline_threshold: float = 0.5,
        device: str = "cuda",
    ) -> AttributionModel:
        """Fit ridge + identify baseline heads by frequency."""
        scores = pl.read_ipc(str(scores_path))
        split = pl.read_ipc(str(split_path))
        joined = scores.join(split, on="variant_id")
        diff, ref = _discover_heads(scores)
        names = tuple([(h, "effect") for h in diff] + [(h, "disruption") for h in ref])
        logger.info(f"Attribution: {len(diff)} effect + {len(ref)} disruption heads")

        features = _feature_matrix(joined, diff, ref)
        raw_score = np.clip(joined["score_pathogenic"].to_numpy(), 1e-6, 1 - 1e-6)
        target = np.log(raw_score / (1 - raw_score)).astype(np.float32)
        train = (joined["split"] == "train").to_numpy()

        x = torch.from_numpy(features).to(device)
        y = torch.from_numpy(target).to(device)
        p = x.size(1)
        x_tr, y_tr = x[train], y[train]
        beta = torch.linalg.solve(
            x_tr.T @ x_tr + torch.eye(p, device=device),
            x_tr.T @ (y_tr - y_tr.mean()),
        )
        intercept = (y_tr.mean() - x_tr.mean(0) @ beta).item()

        # Baseline: heads in top-k for >threshold of training variants
        top_k = torch.topk((x_tr * beta[None, :]).abs(), k_baseline, dim=1).indices
        freq = torch.zeros(p, device=device)
        for j in range(k_baseline):
            freq.scatter_add_(0, top_k[:, j], torch.ones(x_tr.size(0), device=device))
        freq /= x_tr.size(0)

        baseline = tuple(int(i) for i in torch.where(freq > baseline_threshold)[0])
        test_r2 = 1 - ((y[~train] - x[~train] @ beta - intercept) ** 2).sum() / ((y[~train] - y[~train].mean()) ** 2).sum()
        logger.info(f"Ridge R²={test_r2:.4f}, {len(baseline)} baseline heads (>{baseline_threshold:.0%})")
        for i in baseline:
            logger.info(f"  {names[i][0]:35s} freq={freq[i]:.1%}")

        return cls(
            beta=beta.cpu(), intercept=intercept, head_names=names,
            diff_heads=diff, ref_heads=ref, baseline_heads=baseline,
        )

    def attribute(self, scores: pl.DataFrame, n_specific: int = 10) -> pl.DataFrame:
        """Per-variant two-tier attribution.

        Returns DataFrame with variant_id, attribution_logit, attribution_json.
        JSON schema: {"baseline": [...], "specific": [...]} where each entry has
        name, kind, score, contribution.
        """
        features = _feature_matrix(scores, self.diff_heads, self.ref_heads)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.from_numpy(features).to(device)
        beta_d = self.beta.to(device)
        contrib = x * beta_d[None, :]
        logits = (contrib.sum(1) + self.intercept).cpu().numpy().round(4)

        # Mask baseline heads, get top-n_specific from the rest
        baseline_idx = sorted(self.baseline_heads)
        non_baseline = contrib.clone()
        non_baseline[:, baseline_idx] = 0
        topk_specific = torch.topk(non_baseline.abs(), n_specific, dim=1).indices

        contrib_cpu = contrib.cpu().numpy()
        topk_cpu = topk_specific.cpu().numpy()
        names = self.head_names

        rows = []
        for i in range(x.size(0)):
            baseline = [
                {"name": names[h][0], "kind": names[h][1],
                 "score": round(float(features[i, h]), 4),
                 "contribution": round(float(contrib_cpu[i, h]), 4)}
                for h in baseline_idx
            ]
            specific = [
                {"name": names[h][0], "kind": names[h][1],
                 "score": round(float(features[i, h]), 4),
                 "contribution": round(float(contrib_cpu[i, h]), 4)}
                for h in topk_cpu[i]
                if abs(contrib_cpu[i, h]) > 1e-4
            ]
            rows.append(json.dumps({"baseline": baseline, "specific": specific}))

        return pl.DataFrame({
            "variant_id": scores["variant_id"],
            "attribution_logit": logits,
            "attribution_json": rows,
        })

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "beta": self.beta, "intercept": self.intercept,
            "head_names": self.head_names,
            "diff_heads": self.diff_heads, "ref_heads": self.ref_heads,
            "baseline_heads": tuple(self.baseline_heads),
        }, path)
        logger.info(f"Saved attribution: {path}")

    @classmethod
    def load(cls, path: Path) -> AttributionModel:
        return cls(**torch.load(path, map_location="cpu", weights_only=False))
