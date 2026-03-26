"""Build variant viewer static site from probe scores.

Writes per-variant JSONs + global.json + search.json to /tmp staging,
then optionally rsyncs to the output path. See SCHEMA.md for the data contract.

Two score types:
  - **disruption**: ref_score/var_score from the ref-view probe. Shows what changed
    between reference and variant allele. Stored as [ref, var] tuples per head.
  - **effect**: score from the diff-view probe. Predicts variant-level properties
    (clinical predictors, consequence, domain effects). Stored as scalars per head.

Timings (H200, 232K variants, probe_v7):
  Load 3s | GPU similarity 10s | Neighbors 3s | UMAP 40s | Write 70s | Total ~2min
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import torch
from loguru import logger
from sklearn.decomposition import PCA
from tqdm import tqdm
from umap import UMAP

from display import auto_group, display_name
from goodfire_core.storage import ActivationDataset, FilesystemStorage

# ── Inlined constants (from src.datasets.clinvar.main) ────────────────────
_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SWAP_CLASSES = tuple(f"{a}>{b}" for a in _AA for b in _AA if a != b)

CONSEQUENCE_CLASSES = (
    "missense_variant", "intron_variant", "synonymous_variant", "nonsense",
    "frameshift_variant", "non-coding_transcript_variant", "splice_donor_variant",
    "splice_acceptor_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
    "splice_region_variant", "start_lost", "inframe_deletion", "inframe_insertion",
    "inframe_indel", "stop_lost", "genic_downstream_transcript_variant",
    "genic_upstream_transcript_variant", "no_sequence_alteration",
    "initiator_codon_variant",
)

# ── Constants ────────────────────────────────────────────────────────────
K_NEIGHBORS = 10
LABEL_TO_IDX = {"benign": 0, "pathogenic": 1, "VUS": 2}
EVAL_KEYS = (("correlation", "r"), ("auc", "AUC"), ("accuracy", "acc"))
VEP_COLS = (
    "variant_id", "vep_hgvsc", "vep_hgvsp", "vep_impact",
    "vep_exon", "vep_transcript_id", "vep_protein_id", "vep_swissprot",
    "vep_domains", "vep_loeuf",
    "vep_gnomade", "vep_gnomade_afr", "vep_gnomade_amr", "vep_gnomade_asj",
    "vep_gnomade_eas", "vep_gnomade_fin", "vep_gnomade_nfe", "vep_gnomade_sas",
)

def sanitize_vid(v: str) -> str:
    """Make a variant ID safe for use as a filename.

    Long variant IDs (indels) are hashed to avoid filesystem limits.
    Uses FNV-1a so the JS frontend can compute the same hash synchronously.
    """
    s = v.replace(":", "_").replace("/", "_")
    if len(s) <= 200:
        return s
    # FNV-1a 64-bit, matching the JS implementation
    h = 0xCBF29CE484222325
    for b in v.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{s[:60]}_{h:016x}"


def prebin(values: torch.Tensor, n_bins: int = 40) -> list[int]:
    """Histogram of [0, 1) values into n_bins bins."""
    v = values[~values.isnan()]
    if v.numel() == 0:
        return [0] * n_bins
    return torch.bincount(
        torch.clamp((v * n_bins).long(), 0, n_bins - 1), minlength=n_bins
    ).tolist()


# ── Main build ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build variant viewer static site from probe scores.")
    parser.add_argument("--labeled", required=True, type=Path, help="Labeled activations dir")
    parser.add_argument("--vus", type=Path, help="VUS activations dir")
    parser.add_argument("--probe", default="probe_v8", help="Probe name")
    parser.add_argument("--vep", type=Path, help="VEP annotation parquets dir")
    parser.add_argument("--metadata", type=Path, help="Metadata feather (if not baked into scores)")
    parser.add_argument("--annotations", type=Path, help="Annotations feather with gt_ columns")
    parser.add_argument("--output", type=Path, default=Path("build/"))
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    labeled: Path = args.labeled
    probe: str = args.probe

    t0 = time.time()

    def _t(msg: str) -> None:
        logger.info(f"[{time.time() - t0:.1f}s] {msg}")

    # ── Load ─────────────────────────────────────────────────────────────
    _t("Loading...")

    scores_l = pl.read_ipc(str(labeled / probe / "scores.feather"))

    # If scores already contain metadata columns, no separate load needed.
    # Otherwise join from --metadata feather.
    META_COLS = (
        "variant_id", "gene_name", "label", "consequence", "clinical_significance",
        "stars", "disease_name", "chrom", "pos", "ref", "alt", "rs_id", "allele_id", "gene_id",
    )
    missing_meta = [c for c in META_COLS if c not in scores_l.columns]
    if missing_meta and args.metadata:
        meta = pl.read_ipc(str(args.metadata))
        available = [c for c in META_COLS if c in meta.columns]
        scores_l = scores_l.join(meta.select(available), on="variant_id", how="left")

    # VUS scores (optional)
    if args.vus:
        scores_v = pl.read_ipc(str(args.vus / probe / "scores.feather"))
        missing_meta_v = [c for c in META_COLS if c not in scores_v.columns]
        if missing_meta_v and args.metadata:
            meta_v = pl.read_ipc(str(args.metadata))
            available_v = [c for c in META_COLS if c in meta_v.columns]
            scores_v = scores_v.join(meta_v.select(available_v), on="variant_id", how="left")

        df = pl.concat([
            scores_l.with_columns(
                pl.col("stars").cast(pl.Int32) if "stars" in scores_l.columns else pl.lit(0).cast(pl.Int32).alias("stars"),
            ),
            scores_v.with_columns(
                pl.lit("VUS").alias("label"), pl.lit(0).cast(pl.Int32).alias("stars"),
            ),
        ], how="diagonal")
    else:
        df = scores_l.with_columns(
            pl.col("stars").cast(pl.Int32) if "stars" in scores_l.columns else pl.lit(0).cast(pl.Int32).alias("stars"),
        )

    df = df.with_columns(
        pl.col("pred_consequence").replace_strict(dict(enumerate(CONSEQUENCE_CLASSES)), default="unknown").alias("consequence"),
        pl.col("pred_aa_swap").replace_strict(dict(enumerate(AA_SWAP_CLASSES)), default=None).alias("substitution"),
    )

    # Ground-truth annotations (gt_ columns) from --annotations or from scores
    if args.annotations:
        gt = pl.read_ipc(str(args.annotations))
        df = df.join(
            gt.rename({c: f"gt_{c}" for c in gt.columns if c != "variant_id"}),
            on="variant_id", how="left",
        )

    # VEP annotations (HGVS, gnomAD frequencies, domains, etc.)
    vep_dir = args.vep
    if vep_dir and vep_dir.exists():
        _t("Loading VEP annotations...")
        vep_string_cols = {"variant_id", "vep_hgvsc", "vep_hgvsp", "vep_impact",
                            "vep_exon", "vep_transcript_id", "vep_protein_id", "vep_swissprot", "vep_domains"}
        vep_dfs = []
        for chrom_file in sorted(vep_dir.glob("variant_annotations_chr*.parquet")):
            schema = pl.read_parquet_schema(chrom_file)
            available = [c for c in VEP_COLS if c in schema]
            chunk = pl.read_parquet(chrom_file, columns=available)
            # Normalize mixed String/Float columns across chromosomes (single pass)
            float_casts = [c for c in chunk.columns if c not in vep_string_cols and chunk[c].dtype in (pl.Utf8, pl.String)]
            if float_casts:
                chunk = chunk.with_columns(*(pl.col(c).cast(pl.Float64, strict=False) for c in float_casts))
            vep_dfs.append(chunk)
        vep_all = pl.concat(vep_dfs, how="diagonal")
        df = df.join(vep_all, on="variant_id", how="left")

    _t(f"{df.height:,} variants loaded")

    # ── Head classification ──────────────────────────────────────────────
    cfg = json.loads((labeled / probe / "config.json").read_text())
    # Support both old (ref_heads/diff_heads) and new (disruption_heads/effect_heads) config keys
    disruption_set = set(cfg.get("disruption_heads", cfg.get("ref_heads", ())))
    effect_set = set(cfg.get("effect_heads", cfg.get("diff_heads", ())))

    ref_cols = sorted(c for c in df.columns if c.startswith("ref_score_") and c[10:] in disruption_set)
    var_cols = sorted(c for c in df.columns if c.startswith("var_score_") and c[10:] in disruption_set)
    eff_cols = sorted(c for c in df.columns if c.startswith("score_") and c != "score_pathogenic" and c[6:] in effect_set)
    gt_cols = sorted(c for c in df.columns if c.startswith("gt_") and df[c].dtype in (pl.Float32, pl.Float64))

    disruption_heads = tuple(c[10:] for c in ref_cols)
    effect_heads = tuple(c[6:] for c in eff_cols)
    _t(f"{len(disruption_heads)} disruption + {len(effect_heads)} effect heads")

    # ── Prepare data (round, fill nulls, NaN→null) ──────────────────────
    score_cols = ref_cols + var_cols + eff_cols + gt_cols
    float_score_cols = [c for c in score_cols if df[c].dtype in (pl.Float32, pl.Float64)]

    df = df.with_columns(
        *(pl.col(c).round(4).fill_nan(None) for c in float_score_cols),
        pl.col("gene_name").fill_null("?"),
        pl.col("consequence").fill_null("unknown"),
        pl.col("label").fill_null("?"),
        pl.col("clinical_significance").fill_null(""),
        pl.col("stars").fill_null(0),
        pl.col("disease_name").fill_null(""),
        pl.col("score_pathogenic").fill_null(0.0).round(4),
    )

    # ── Embeddings + neighbors ───────────────────────────────────────────
    _t("Loading embeddings...")

    def load_emb(path: Path) -> tuple[torch.Tensor, list[str]]:
        storage = FilesystemStorage(path / probe)
        dataset = ActivationDataset(storage, "embeddings", batch_size=4096, include_provenance=True)
        embeddings, ids = [], []
        for batch in dataset.training_iterator(device="cpu", n_epochs=1, shuffle=False, drop_last=False):
            # Extract diff-view embedding (first d_h^2 elements) for UMAP/neighbors
            flat = batch.acts.flatten(1)
            d_h2 = cfg["d_hidden"] ** 2
            embeddings.append(flat[:, :d_h2])
            ids.extend(batch.sequence_ids)
        return torch.cat(embeddings), ids

    emb_parts = [load_emb(labeled)]
    if args.vus:
        emb_parts.append(load_emb(args.vus))
    emb = torch.nn.functional.normalize(
        torch.cat([e for e, _ in emb_parts]).float(), dim=1,
    )
    emb_ids = [vid for _, ids in emb_parts for vid in ids]
    n_emb = len(emb_ids)

    _t("GPU cosine similarity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_gpu = emb.to(device, non_blocking=True)
    topk_indices, topk_values = [], []
    for start in range(0, n_emb, 4096):
        end = min(start + 4096, n_emb)
        similarity = emb_gpu[start:end] @ emb_gpu.T
        similarity[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -1
        topk = similarity.topk(K_NEIGHBORS, dim=1)
        topk_indices.append(topk.indices.cpu())
        topk_values.append(topk.values.cpu())
    topk_i = torch.cat(topk_indices).numpy()
    topk_v = torch.cat(topk_values).numpy()
    del emb_gpu

    _t("Building neighbor table...")
    emb_df = (
        pl.DataFrame({"emb_i": range(n_emb), "variant_id": emb_ids})
        .join(df.select(
            "variant_id", pl.col("gene_name").alias("gene"),
            "consequence", "label", pl.col("score_pathogenic").alias("score"),
        ), on="variant_id", how="left")
        .with_columns(
            pl.col("gene").fill_null("?"),
            pl.col("consequence").fill_null("?"),
            pl.col("label").fill_null("?"),
            pl.col("score").fill_null(0.0),
        )
    )

    edges = pl.DataFrame({
        "src_i": torch.arange(n_emb).repeat_interleave(K_NEIGHBORS).to(torch.int32).numpy(),
        "dst_i": topk_i.ravel().astype(np.int32),
        "similarity": topk_v.ravel().round(4).astype(np.float32),
    })
    nb = (edges
          .join(emb_df.select(
              pl.col("emb_i").alias("dst_i"), pl.col("variant_id").alias("id"),
              "gene", "consequence", "label", "score",
          ), on="dst_i", how="left")
          .join(emb_df.select(
              pl.col("emb_i").alias("src_i"), pl.col("variant_id").alias("src_vid"),
          ), on="src_i", how="left")
          .drop("src_i", "dst_i"))

    nb_grouped = nb.group_by("src_vid").agg(
        pl.struct("id", "gene", "consequence", "label", "score", "similarity").alias("neighbors"))
    nb_map = dict(zip(nb_grouped["src_vid"].to_list(), nb_grouped["neighbors"].to_list(), strict=True))
    _t(f"Neighbors: {nb.height:,} edges → {nb_grouped.height:,} variants")

    # ── UMAP ─────────────────────────────────────────────────────────────
    _t("UMAP...")
    rng = np.random.RandomState(42)
    n_sample = min(30_000, n_emb)
    umap_idx = np.sort(rng.choice(n_emb, n_sample, replace=False))

    pca = PCA(n_components=50, random_state=42).fit_transform(emb[umap_idx].numpy())
    umap_coords = UMAP(
        n_components=2, n_neighbors=30, min_dist=0.05, spread=10.0,
        metric="correlation", random_state=42,
    ).fit_transform(pca)

    # Compact UMAP metadata: int labels, gene index, 2dp coords
    umap_sub = emb_df.select(
        "variant_id", "gene", "label", pl.col("score").round(2),
    )[umap_idx.tolist()]
    gene_list = sorted(umap_sub["gene"].unique().to_list())
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # ── Attribution ──────────────────────────────────────────────────────
    attribution_path = labeled / probe / "attribution.pt"
    if attribution_path.exists():
        # Lazy import: attribution module may live in a sibling repo
        try:
            from src.attribution import AttributionModel
        except ImportError:
            gfm_gen_root = Path(__file__).parent.parent
            sys.path.insert(0, str(gfm_gen_root))
            from src.attribution import AttributionModel
        attr_model = AttributionModel.load(attribution_path)
        attr_df = attr_model.attribute(df, k=5, n_heads=8)
        attr_by_vid = dict(zip(attr_df["variant_id"].to_list(), attr_df["attribution_json"].to_list(), strict=True))
        _t(f"Attribution loaded for {len(attr_by_vid):,} variants")
    else:
        attr_by_vid = {}
        logger.warning(f"No attribution model at {attribution_path}, skipping")

    # ── Write to staging ─────────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="variant_viewer_"))
    staging_vdir = staging / "variants"
    staging_vdir.mkdir()

    # ── Variant JSONs ────────────────────────────────────────────────────
    _t("Writing variant JSONs...")
    meta_fields = (
        "variant_id", "gene_name", "chrom", "pos", "ref", "alt", "consequence",
        "substitution", "label", "clinical_significance", "stars", "disease_name",
        "score_pathogenic", "rs_id", "allele_id", "gene_id",
    )
    vep_fields = [c for c in VEP_COLS if c != "variant_id" and c in df.columns]
    all_cols = [c for c in (*meta_fields, *vep_fields, *ref_cols, *var_cols, *eff_cols, *gt_cols) if c in df.columns]
    col_data = df.select(all_cols).to_dict(as_series=False)

    ref_d = {h: col_data[c] for c, h in zip(ref_cols, disruption_heads, strict=True)}
    var_d = {h: col_data[c] for c, h in zip(var_cols, disruption_heads, strict=True)}
    eff_d = {h: col_data[c] for c, h in zip(eff_cols, effect_heads, strict=True)}
    gt_d = {c[3:]: col_data[c] for c in gt_cols}

    # Pre-resolve VEP columns once (avoids [None]*n allocation per iteration)
    n = df.height
    _none = [None] * n
    vep = {
        k: col_data.get(f"vep_{k}", _none)
        for k in ("hgvsc", "hgvsp", "impact", "exon",
                  "transcript_id", "swissprot", "domains", "loeuf", "gnomade")
    }
    gnomad_pops = {
        k: col_data.get(f"vep_gnomade_{k}", _none)
        for k in ("afr", "amr", "asj", "eas", "fin", "nfe", "sas")
    }

    for i in tqdm(range(n), desc="write", mininterval=5):
        vid = col_data["variant_id"][i]

        # Flat value dicts (schema v2: values only, no display names or grouping)
        disruption = {}
        for h in disruption_heads:
            r = ref_d[h][i]
            if r is not None:
                va = var_d[h][i] if var_d[h][i] is not None else r
                disruption[h] = [r, va]

        effect = {}
        for h in effect_heads:
            ev = eff_d[h][i]
            if ev is not None:
                effect[h] = ev

        gt = {h: v for h, col in gt_d.items() if (v := col[i]) is not None and v > 0}

        neighbors = nb_map.get(vid, [])
        n_p = sum(1 for nb in neighbors if "pathogenic" in nb.get("label", ""))
        n_b = sum(1 for nb in neighbors if "benign" in nb.get("label", ""))

        data = {
            "id": vid,
            "gene": col_data["gene_name"][i],
            "chrom": col_data["chrom"][i], "pos": col_data["pos"][i],
            "ref": col_data["ref"][i], "alt": col_data["alt"][i],
            "consequence": col_data["consequence"][i],
            "substitution": col_data["substitution"][i],
            "label": col_data["label"][i],
            "significance": col_data["clinical_significance"][i],
            "stars": col_data["stars"][i],
            "disease": col_data["disease_name"][i],
            "score": col_data["score_pathogenic"][i],
            "rs_id": col_data["rs_id"][i],
            "allele_id": col_data["allele_id"][i],
            "gene_id": col_data["gene_id"][i],
            "hgvsc": vep["hgvsc"][i],
            "hgvsp": vep["hgvsp"][i],
            "impact": vep["impact"][i],
            "exon": vep["exon"][i],
            "transcript": vep["transcript_id"][i],
            "swissprot": vep["swissprot"][i],
            "domains": vep["domains"][i],
            "loeuf": vep["loeuf"][i],
            "gnomad": vep["gnomade"][i],
            "gnomad_pop": {k: v for k, col in gnomad_pops.items() if (v := col[i]) is not None and v > 0},
            "disruption": disruption,
            "effect": effect,
            "gt": gt,
            "attribution": orjson.loads(attr_by_vid[vid]) if vid in attr_by_vid else [],
            "neighbors": neighbors,
            "nP": n_p, "nB": n_b, "nV": len(neighbors) - n_p - n_b,
        }
        (staging_vdir / f"{sanitize_vid(vid)}.json").write_bytes(orjson.dumps(data))

    _t(f"Wrote {n:,} variants to staging")

    # ── global.json ──────────────────────────────────────────────────────
    _t("global.json...")
    all_head_names = disruption_heads + effect_heads
    scores_t = torch.tensor(col_data["score_pathogenic"], dtype=torch.float32)
    ben_mask = torch.from_numpy((df["label"] == "benign").to_numpy())
    path_mask = torch.from_numpy((df["label"] == "pathogenic").to_numpy())

    # Stack all head scores into a matrix for vectorized histogram computation
    score_matrix = torch.from_numpy(
        df.select(ref_cols + eff_cols).to_numpy(allow_copy=True).T
    ).float()  # (n_heads, n_variants)

    hh = {}
    for j, head_name in enumerate(all_head_names):
        hh[head_name] = {
            "benign": prebin(score_matrix[j][ben_mask], 40),
            "pathogenic": prebin(score_matrix[j][path_mask], 40),
            "bins": 40,
        }

    eval_metrics = {}
    eval_path = labeled / probe / "eval.json"
    if eval_path.exists():
        for h, info in json.loads(eval_path.read_text()).items():
            for key, label in EVAL_KEYS:
                if key in info:
                    eval_metrics[h] = {"metric": label, "value": round(info[key], 2)}
                    break

    decomp_path = labeled / probe / "score_decomposition.json"
    (staging / "global.json").write_bytes(orjson.dumps({
        "umap": {
            "x": np.round(umap_coords[:, 0], 2).tolist(),
            "y": np.round(umap_coords[:, 1], 2).tolist(),
            "score": umap_sub["score"].to_list(),
            "ids": umap_sub["variant_id"].to_list(),
            "genes": [gene_to_idx[g] for g in umap_sub["gene"].to_list()],
            "labels": [LABEL_TO_IDX.get(lab, 2) for lab in umap_sub["label"].to_list()],
            "gene_list": gene_list,
        },
        "distribution": {
            "benign": prebin(scores_t[ben_mask], 80),
            "pathogenic": prebin(scores_t[path_mask], 80),
            "bins": 80,
        },
        "head_distributions": hh,
        "eval": eval_metrics,
        "heads": {"disruption": auto_group(set(disruption_heads)), "effect": auto_group(set(effect_heads))},
        "display": {h: display_name(h) for h in all_head_names},
        "decomposition": json.loads(decomp_path.read_text()) if decomp_path.exists() else None,
    }))

    # ── search.json ──────────────────────────────────────────────────────
    _t("search.json...")
    search_df = (df
        .select("variant_id", "gene_name", "label", "score_pathogenic", "consequence")
        .filter(pl.col("gene_name") != "?")
        .sort("score_pathogenic", descending=True)
        .with_columns(pl.col("gene_name").str.to_uppercase().alias("gene_upper")))

    search_agg = search_df.group_by("gene_upper").agg(
        pl.struct(
            pl.col("variant_id").alias("v"),
            pl.col("label").alias("l"),
            pl.col("score_pathogenic").alias("s"),
            pl.col("consequence").alias("c"),
        ).alias("variants"))

    search_idx = dict(zip(search_agg["gene_upper"].to_list(), search_agg["variants"].to_list(), strict=True))
    (staging / "search.json").write_bytes(orjson.dumps(search_idx))

    # ── Copy static files ────────────────────────────────────────────────
    shutil.copy2(Path(__file__).parent / "index.html", staging / "index.html")
    if args.vus:
        interp_src = args.vus / "interpretations"
        if interp_src.exists():
            interp_dst = staging / "interpretations"
            interp_dst.mkdir(parents=True, exist_ok=True)
            for f in interp_src.glob("*.json"):
                shutil.copy2(f, interp_dst / f.name)

    # ── Sync or serve from staging ───────────────────────────────────────
    out = args.output
    no_sync = args.no_sync

    if no_sync:
        _t(f"Done. {n:,} variants in {staging} (--no-sync, skipping rsync)")
        print(f"STAGING_DIR={staging}")
    else:
        _t(f"Syncing to {out}...")
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{staging}/", f"{out}/"], check=True)
        shutil.rmtree(staging)
        _t(f"Done. {n:,} variants.")


if __name__ == "__main__":
    main()
