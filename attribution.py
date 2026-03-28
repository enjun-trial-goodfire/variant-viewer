"""Per-variant attribution for pathogenicity predictions.

Ridge regression on biological heads (excluding clinical predictors), then
show the top-k heads by |beta_h * score_ih| per variant, split into effect
and disruption groups.

    >>> model = AttributionModel.fit(scores_path, split_path)
    >>> model.save(probe_dir / "attribution.pt")
    >>> attr = model.attribute(scores_df)
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _feature_matrix(df: pl.DataFrame, diff: tuple[str, ...], ref: tuple[str, ...]) -> torch.Tensor:
    """[N, p] float32 tensor: diff scores + ref-var deltas."""
    parts = [torch.nan_to_num(torch.from_numpy(df.select([f"score_{h}" for h in diff]).to_numpy()), nan=0.5)]
    if ref:
        r = torch.nan_to_num(torch.from_numpy(df.select([f"ref_score_{h}" for h in ref]).to_numpy()), nan=0.5)
        v = torch.nan_to_num(torch.from_numpy(df.select([f"var_score_{h}" for h in ref]).to_numpy()), nan=0.5)
        parts.append(v - r)
    return torch.cat(parts, dim=1).float()


class AttributionModel:
    """Per-variant attribution: top-k heads by |beta * score|."""

    def __init__(
        self,
        beta: torch.Tensor,
        intercept: float,
        head_names: tuple[tuple[str, str], ...],
        diff_heads: tuple[str, ...],
        ref_heads: tuple[str, ...],
    ):
        self.beta = beta
        self.intercept = intercept
        self.head_names = head_names
        self.diff_heads = diff_heads
        self.ref_heads = ref_heads

    @classmethod
    def fit(
        cls,
        scores_path: Path,
        split_path: Path,
        device: str = "cuda",
    ) -> AttributionModel:
        """Fit ridge on biological heads."""
        scores = pl.read_ipc(str(scores_path))
        split = pl.read_ipc(str(split_path))
        joined = scores.join(split, on="variant_id")
        diff, ref = _discover_heads(scores)
        names = tuple([(h, "effect") for h in diff] + [(h, "disruption") for h in ref])
        logger.info(f"Attribution: {len(diff)} effect + {len(ref)} disruption heads")

        features = _feature_matrix(joined, diff, ref)
        raw_score = torch.clamp(torch.tensor(joined["score_pathogenic"].to_list(), dtype=torch.float32), 1e-6, 1 - 1e-6)
        target = torch.log(raw_score / (1 - raw_score))
        train_mask = torch.tensor((joined["split"] == "train").to_list(), dtype=torch.bool)

        x = features.to(device)
        y = target.to(device)
        p = x.size(1)
        x_tr, y_tr = x[train_mask], y[train_mask]
        beta = torch.linalg.solve(
            x_tr.T @ x_tr + torch.eye(p, device=device),
            x_tr.T @ (y_tr - y_tr.mean()),
        )
        intercept = (y_tr.mean() - x_tr.mean(0) @ beta).item()
        test_mask = ~train_mask
        test_r2 = 1 - ((y[test_mask] - x[test_mask] @ beta - intercept) ** 2).sum() / ((y[test_mask] - y[test_mask].mean()) ** 2).sum()
        logger.info(f"Ridge R²={test_r2:.4f}")

        return cls(beta=beta.cpu(), intercept=intercept, head_names=names,
                   diff_heads=diff, ref_heads=ref)

    def attribute(
        self, scores: pl.DataFrame, k_effect: int = 8, k_disruption: int = 7,
    ) -> pl.DataFrame:
        """Top-k heads per variant, split by effect/disruption with separate quotas.

        JSON schema: {"effect": [...], "disruption": [...]}
        Each entry: {name, score, contribution}
        Sorted by |contribution| descending within each group.
        """
        features = _feature_matrix(scores, self.diff_heads, self.ref_heads)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = features.to(device)
        beta_d = self.beta.to(device)
        contrib = x * beta_d[None, :]
        logits = (contrib.sum(1) + self.intercept).cpu()

        n_eff = len(self.diff_heads)
        n_dis = contrib.size(1) - n_eff
        k_e = min(k_effect, n_eff)
        k_d = min(k_disruption, n_dis)

        eff_idx = torch.topk(contrib[:, :n_eff].abs(), k_e, dim=1).indices.cpu()
        dis_idx = torch.topk(contrib[:, n_eff:].abs(), k_d, dim=1).indices.cpu()

        contrib_cpu = contrib.cpu()
        features_cpu = features

        # Build name lookup: index → head name
        names = tuple(n[0] for n in self.head_names)

        # Gather top-k values: [N, k]
        eff_contrib = torch.gather(contrib_cpu[:, :n_eff], 1, eff_idx)
        eff_scores = torch.gather(features_cpu[:, :n_eff], 1, eff_idx)
        dis_contrib = torch.gather(contrib_cpu[:, n_eff:], 1, dis_idx)
        dis_scores = torch.gather(features_cpu[:, n_eff:], 1, dis_idx)

        # Threshold mask: only include heads with |contribution| > 1e-4
        eff_mask = eff_contrib.abs() > 1e-4
        dis_mask = dis_contrib.abs() > 1e-4

        # Build JSON strings vectorized per row
        eff_idx_list = eff_idx.tolist()
        dis_idx_list = dis_idx.tolist()
        eff_contrib_list = eff_contrib.tolist()
        dis_contrib_list = dis_contrib.tolist()
        eff_scores_list = eff_scores.tolist()
        dis_scores_list = dis_scores.tolist()
        eff_mask_list = eff_mask.tolist()
        dis_mask_list = dis_mask.tolist()

        rows = [
            json.dumps({
                "effect": [
                    {"name": names[eff_idx_list[i][j]], "score": round(eff_scores_list[i][j], 4),
                     "contribution": round(eff_contrib_list[i][j], 4)}
                    for j in range(k_e) if eff_mask_list[i][j]
                ],
                "disruption": [
                    {"name": names[n_eff + dis_idx_list[i][j]], "score": round(dis_scores_list[i][j], 4),
                     "contribution": round(dis_contrib_list[i][j], 4)}
                    for j in range(k_d) if dis_mask_list[i][j]
                ],
            })
            for i in range(x.size(0))
        ]

        return pl.DataFrame({
            "variant_id": scores["variant_id"],
            "attribution_logit": logits.round(decimals=4).tolist(),
            "attribution_json": rows,
        })

    def save(self, path: Path) -> None:
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "beta": self.beta.tolist(),
            "intercept": self.intercept,
            "head_names": self.head_names,
            "diff_heads": self.diff_heads,
            "ref_heads": self.ref_heads,
        }
        path.write_text(json.dumps(data))
        logger.info(f"Saved attribution: {path}")

    @classmethod
    def load(cls, path: Path) -> AttributionModel:
        data = json.loads(Path(path).read_text())
        data["beta"] = torch.tensor(data["beta"], dtype=torch.float32)
        data["head_names"] = tuple(tuple(x) for x in data["head_names"])
        data["diff_heads"] = tuple(data["diff_heads"])
        data["ref_heads"] = tuple(data["ref_heads"])
        data.pop("baseline_heads", None)
        return cls(**data)
