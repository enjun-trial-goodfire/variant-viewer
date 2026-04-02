"""Build DuckDB from precomputed artifacts (output of transform.py).

No computation. Reads clean parquet + JSON side artifacts, loads into DuckDB
with indexes. Optionally adds precomputed neighbors and UMAP.

Usage:
    uv run vv build [--parquet builds/clean.parquet] [--neighbors] [--umap] [--probe probe_v12]
"""

import json
import time
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import torch
from goodfire_core.storage import FilesystemStorage
from loguru import logger
from sklearn.decomposition import PCA
from umap import UMAP

from constants import LABEL_TO_IDX, PROBE_NAME
from paths import ARTIFACTS

LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"
K_NEIGHBORS = 10


# ── Embeddings + neighbors + UMAP ────────────────────────────────────

def _load_emb(path: Path, probe: str, d_hidden: int) -> tuple[torch.Tensor, list[str]]:
    storage = FilesystemStorage(path / probe)
    from training import local_activation_dataset
    dataset = local_activation_dataset(storage, "embeddings", batch_size=4096, include_provenance=True)
    embeddings, ids = [], []
    d_h2 = d_hidden ** 2
    for batch in dataset.training_iterator(device="cpu", n_epochs=1, shuffle=False, drop_last=False):
        embeddings.append(batch.acts.flatten(1)[:, :d_h2])
        ids.extend(batch.sequence_ids)
    return torch.cat(embeddings), ids


def load_embeddings(cfg: dict, probe: str) -> tuple[torch.Tensor, list[str]]:
    emb_l, ids_l = _load_emb(LABELED, probe, cfg["d_hidden"])
    emb_v, ids_v = _load_emb(VUS, probe, cfg["d_hidden"])
    return torch.nn.functional.normalize(torch.cat([emb_l, emb_v]).float(), dim=1), ids_l + ids_v


SIMILARITY_BATCH = 4096  # rows per batch for pairwise cosine similarity


def compute_neighbors(emb: torch.Tensor, emb_ids: list[str], df: pl.DataFrame, k: int = 10) -> dict[str, list]:
    n = len(emb_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_gpu = emb.to(device, non_blocking=True)

    topk_indices, topk_values = [], []
    for start in range(0, n, SIMILARITY_BATCH):
        end = min(start + SIMILARITY_BATCH, n)
        sim = emb_gpu[start:end] @ emb_gpu.T
        sim[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -1
        topk = sim.topk(k, dim=1)
        topk_indices.append(topk.indices.cpu())
        topk_values.append(topk.values.cpu())
    topk_i = torch.cat(topk_indices).numpy()
    topk_v = torch.cat(topk_values).numpy()
    del emb_gpu

    emb_df = (
        pl.DataFrame({"emb_i": range(n), "variant_id": emb_ids})
        .join(df.select(
            "variant_id",
            pl.col("gene_name").alias("gene"),
            "consequence_display", "label", "label_display",
            pl.col("pathogenicity").alias("score"),
        ), on="variant_id", how="left")
        .with_columns(pl.col("gene").fill_null("?"), pl.col("consequence_display").fill_null("?"),
                      pl.col("label").fill_null("?"), pl.col("label_display").fill_null("?"),
                      pl.col("score").fill_null(0.0))
    )

    edges = pl.DataFrame({
        "src_i": torch.arange(n).repeat_interleave(k).to(torch.int32).numpy(),
        "dst_i": topk_i.ravel().astype(np.int32),
        "similarity": topk_v.ravel().round(4).astype(np.float32),
    })
    nb = (edges
          .join(emb_df.select(pl.col("emb_i").alias("dst_i"),
                              pl.col("variant_id").alias("id"),
                              "gene", "consequence_display", "label", "label_display", "score"),
                on="dst_i", how="left")
          .join(emb_df.select(pl.col("emb_i").alias("src_i"), pl.col("variant_id").alias("src_id")),
                on="src_i", how="left")
          .drop("src_i", "dst_i"))

    grouped = nb.group_by("src_id").agg(
        pl.struct("id", "gene", "consequence_display", "label", "label_display", "score", "similarity").alias("neighbors"))
    return dict(zip(grouped["src_id"].to_list(), grouped["neighbors"].to_list(), strict=True))


def compute_umap(emb: torch.Tensor, emb_ids: list[str], df: pl.DataFrame, n_sample: int = 30_000) -> dict:
    n = len(emb_ids)
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(n, min(n_sample, n), replace=False))

    pca = PCA(n_components=50, random_state=42).fit_transform(emb[idx].numpy())
    coords = UMAP(n_components=2, n_neighbors=30, min_dist=0.05, spread=10.0,
                   metric="correlation", random_state=42).fit_transform(pca)

    sub = (
        pl.DataFrame({"emb_i": range(n), "variant_id": emb_ids})
        .join(df.select(
            "variant_id",
            pl.col("gene_name").alias("gene"),
            "label",
            pl.col("pathogenicity").alias("score"),
        ), on="variant_id", how="left")
        .with_columns(pl.col("gene").fill_null("?"), pl.col("label").fill_null("?"),
                      pl.col("score").fill_null(0.0).round(2))
    )[idx.tolist()]

    gene_list = sorted(sub["gene"].unique().to_list())
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    return {
        "x": np.round(coords[:, 0], 2).tolist(),
        "y": np.round(coords[:, 1], 2).tolist(),
        "score": sub["score"].to_list(),
        "ids": sub["variant_id"].to_list(),
        "genes": [gene_to_idx[g] for g in sub["gene"].to_list()],
        "labels": [LABEL_TO_IDX.get(lab, 2) for lab in sub["label"].to_list()],
        "gene_list": gene_list,
    }


# ── Main ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path("builds/variants.duckdb")


def main(
    parquet: Path = Path("builds/clean.parquet"),
    db_path: Path = DEFAULT_DB_PATH,
    umap: bool = False,
    neighbors: bool = False,
    probe: str = PROBE_NAME,
) -> Path:
    from db import create_db

    t0 = time.time()
    def _t(msg: str) -> None:
        logger.info(f"[{time.time() - t0:.1f}s] {msg}")

    # 1. Read clean parquet
    _t(f"Reading {parquet}")
    df = pl.read_parquet(parquet)
    _t(f"  {df.height:,} variants, {df.width} columns")

    # 2. Optional GPU steps
    nb_map: dict[str, list] = {}
    umap_data: dict | None = None

    if neighbors or umap:
        cfg = json.loads((LABELED / probe / "config.json").read_text())
        _t("Loading embeddings...")
        emb, emb_ids = load_embeddings(cfg, probe)

        # Filter embeddings to variants in df (important for --dev subsets)
        df_ids = set(df["variant_id"].to_list())
        n_total = len(emb_ids)
        mask = [i for i, eid in enumerate(emb_ids) if eid in df_ids]
        if len(mask) < n_total:
            emb = emb[mask]
            emb_ids = [emb_ids[i] for i in mask]
            _t(f"  Filtered embeddings: {n_total} → {len(emb_ids)} (matching df)")

        if neighbors:
            _t("Computing neighbors...")
            nb_map = compute_neighbors(emb, emb_ids, df, k=K_NEIGHBORS)
        if umap:
            _t("Computing UMAP...")
            umap_data = compute_umap(emb, emb_ids, df)

    # 3. Add neighbors column
    if nb_map:
        ids = df["variant_id"].to_list()
        df = df.with_columns(
            pl.Series("neighbors", [orjson.dumps(nb_map.get(v, [])).decode() for v in ids], dtype=pl.Utf8)
        )

    # 4. Insert into DuckDB
    conn = create_db(db_path)
    _t(f"Inserting {df.height:,} variants...")
    arrow_table = df.to_arrow()
    conn.execute("CREATE TABLE variants AS SELECT * FROM arrow_table")
    conn.execute("CREATE INDEX idx_id ON variants(variant_id)")
    conn.execute("CREATE INDEX idx_gene ON variants(gene_name)")

    # 5. Load precomputed JSON artifacts into global_config
    stats_path = parquet.parent / "statistics.json"
    heads_path = parquet.parent / "heads.json"

    if heads_path.exists():
        conn.execute("INSERT INTO global_config VALUES (?, ?)", ["heads", heads_path.read_text()])
        _t(f"Loaded {heads_path}")

    if stats_path.exists():
        conn.execute("INSERT INTO global_config VALUES (?, ?)", ["distributions", stats_path.read_text()])
        _t(f"Loaded {stats_path}")

    if umap_data:
        conn.execute("INSERT INTO global_config VALUES (?, ?)", ["umap", orjson.dumps(umap_data).decode()])

    conn.close()
    _t(f"Done. {df.height:,} variants in {db_path}")
    return db_path


if __name__ == "__main__":
    main()
