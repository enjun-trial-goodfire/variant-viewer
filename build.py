"""Build variant viewer static site from probe scores.

Writes per-variant JSONs + global.json + search.json to /tmp staging,
then optionally rsyncs to the output path. See SCHEMA.md for the data contract.

Two score types:
  - **disruption**: ref_score/var_score from the ref-view probe. Shows what changed
    between reference and variant allele. Stored as delta scalars per head.
  - **effect**: score from the diff-view probe. Predicts variant-level properties
    (clinical predictors, consequence, domain effects). Stored as scalars per head.

Timings (H200, 232K variants, probe_v9):
  Load 3s | GPU similarity 10s | Neighbors 3s | UMAP 40s | Write 70s | Total ~2min
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import torch
from loguru import logger
from sklearn.decomposition import PCA
from rich.progress import track

from attribution import AttributionModel
from umap import UMAP

from goodfire_core.storage import ActivationDataset, FilesystemStorage

from constants import AA_SWAP_CLASSES, CONSEQUENCE_CLASSES, LABEL_TO_IDX
from display import auto_group, display_name
from paths import (
    ARTIFACTS, MAYO_DATA, sanitize_vid,
    load_vep, load_domain_names, resolve_domains, load_metadata,
)

ANNOTATIONS = MAYO_DATA / "clinvar" / "deconfounded-full" / "annotations.feather"
LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"
PROBE = "probe_v9"
K_NEIGHBORS = 10
EVAL_KEYS = (("correlation", "r"), ("auc", "AUC"), ("accuracy", "acc"))
VARIANT_ANN_DIR = ARTIFACTS / "clinvar_evo2_labeled" / "variant_annotations"
VEP_COLS = (
    "variant_id", "vep_hgvsc", "vep_hgvsp", "vep_impact",
    "vep_exon", "vep_transcript_id", "vep_protein_id", "vep_swissprot",
    "vep_domains", "vep_loeuf",
    "vep_gnomade", "vep_gnomade_afr", "vep_gnomade_amr", "vep_gnomade_asj",
    "vep_gnomade_eas", "vep_gnomade_fin", "vep_gnomade_nfe", "vep_gnomade_sas",
)

def _parse_attribution(json_str: str | None) -> dict:
    """Parse attribution JSON."""
    if not json_str:
        return {}
    return orjson.loads(json_str)


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
    import argparse
    parser = argparse.ArgumentParser(description="Build variant viewer static site")
    parser.add_argument("output", nargs="?", default=None, help="Output directory (default: /tmp staging)")
    parser.add_argument("--sync", action="store_true", help="Rsync staging to output dir (default: write to /tmp only)")
    parser.add_argument("--umap", action="store_true", help="Compute UMAP embedding (slow, ~40s)")
    parser.add_argument("--neighbors", action="store_true", help="Compute nearest neighbors (requires GPU)")
    args = parser.parse_args()

    t0 = time.time()

    def _t(msg: str) -> None:
        logger.info(f"[{time.time() - t0:.1f}s] {msg}")

    # ── Load ─────────────────────────────────────────────────────────────
    _t("Loading...")

    scores_l = pl.read_ipc(str(LABELED / PROBE / "scores.feather"))
    scores_v = pl.read_ipc(str(VUS / PROBE / "scores.feather"))
    meta_l = load_metadata("deconfounded-full").select(
        "variant_id", "label", "consequence", "gene_name",
        "clinical_significance", "stars", "disease_name",
        "chrom", "pos", "ref", "alt", "rs_id", "allele_id", "gene_id", "gene_strand")
    meta_v = load_metadata("vus").select(
        "variant_id", "consequence", "gene_name",
        "clinical_significance", "disease_name",
        "chrom", "pos", "ref", "alt", "rs_id", "allele_id", "gene_id", "gene_strand")

    df = pl.concat([
        scores_l.join(meta_l, on="variant_id", how="left").with_columns(pl.col("stars").cast(pl.Int32)),
        scores_v.join(meta_v, on="variant_id", how="left").with_columns(
            pl.lit("VUS").alias("label"), pl.lit(0).cast(pl.Int32).alias("stars")),
    ], how="diagonal")

    # ref/alt in the raw ClinVar source are already VCF forward-strand alleles
    # (ref matches FASTA at 0-based pos). No strand complementing needed.
    # Only need 0-based → 1-based position conversion for external links.
    df = df.with_columns(
        (pl.col("pos") + 1).alias("vcf_pos"),  # 0-based → 1-based
    )

    # Decode integer predictions → strings (backward compat for older probes)
    if "pred_consequence" in df.columns and df["pred_consequence"].dtype in (pl.Int32, pl.Int64, pl.UInt32):
        df = df.with_columns(
            pl.col("pred_consequence").replace_strict(dict(enumerate(CONSEQUENCE_CLASSES)), default="unknown").alias("consequence"),
            pl.col("pred_aa_swap").replace_strict(dict(enumerate(AA_SWAP_CLASSES)), default=None).alias("substitution"),
        )
    elif "pred_consequence" in df.columns:
        # Already strings (new extract.py writes labels directly)
        df = df.rename({"pred_consequence": "consequence", "pred_aa_swap": "substitution"})

    gt = pl.read_ipc(ANNOTATIONS)
    df = df.join(
        gt.rename({c: f"gt_{c}" for c in gt.columns if c != "variant_id"}),
        on="variant_id", how="left",
    )

    # VEP annotations (HGVS, gnomAD frequencies, domains, etc.)
    _t("Loading VEP annotations...")
    vep = load_vep(VARIANT_ANN_DIR, VEP_COLS)
    df = df.join(vep, on="variant_id", how="left")

    # ClinVar submission data (ACMG codes, submitters, cytogenetic, etc.)
    submissions_path = Path("data/clinvar/submissions.feather")
    if submissions_path.exists():
        subs = pl.read_ipc(submissions_path).select(
            "allele_id", "variation_id", "cytogenetic", "review_status",
            "acmg_codes", "submitters", "clinical_features",
            "n_submissions", "last_evaluated", "origin",
        )
        df = df.join(subs, on="allele_id", how="left")
        _t(f"Joined ClinVar submissions ({subs.height:,} records)")
    else:
        logger.warning(f"No submissions data at {submissions_path}, run clinvar_submissions.py first")

    domain_cache = load_domain_names()
    _t(f"{df.height:,} variants loaded, {len(domain_cache):,} domain names cached")

    # ── Head classification ──────────────────────────────────────────────
    cfg = json.loads((LABELED / PROBE / "config.json").read_text())
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
        storage = FilesystemStorage(path / PROBE)
        dataset = ActivationDataset(storage, "embeddings", batch_size=4096, include_provenance=True)
        embeddings, ids = [], []
        for batch in dataset.training_iterator(device="cpu", n_epochs=1, shuffle=False, drop_last=False):
            # Extract diff-view embedding (first d_h^2 elements) for UMAP/neighbors
            flat = batch.acts.flatten(1)
            d_h2 = cfg["d_hidden"] ** 2
            embeddings.append(flat[:, :d_h2])
            ids.extend(batch.sequence_ids)
        return torch.cat(embeddings), ids

    if not args.neighbors and not args.umap:
        _t("Skipping embeddings (use --neighbors or --umap to enable)")
        nb_map = {}
        emb_df = None
    else:
        emb_l, ids_l = load_emb(LABELED)
        emb_v, ids_v = load_emb(VUS)
        emb = torch.nn.functional.normalize(torch.cat([emb_l, emb_v]).float(), dim=1)
        emb_ids = ids_l + ids_v
        n_emb = len(emb_ids)

    if not args.neighbors:
        _t("Skipping neighbors (use --neighbors to enable)")
        nb_map = {}
    else:
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

    if args.neighbors:
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
    if not args.umap:
        _t("Skipping UMAP (use --umap to enable)")
        umap_coords = None
        umap_sub = None
        gene_list = []
        gene_to_idx = {}
    else:
        _t("UMAP...")
        rng = np.random.RandomState(42)
        n_sample = min(30_000, n_emb)
        umap_idx = np.sort(rng.choice(n_emb, n_sample, replace=False))

        pca = PCA(n_components=50, random_state=42).fit_transform(emb[umap_idx].numpy())
        umap_coords = UMAP(
            n_components=2, n_neighbors=30, min_dist=0.05, spread=10.0,
            metric="correlation", random_state=42,
        ).fit_transform(pca)

        umap_sub = emb_df.select(
            "variant_id", "gene", "label", pl.col("score").round(2),
        )[umap_idx.tolist()]
        gene_list = sorted(umap_sub["gene"].unique().to_list())
        gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # ── Attribution ──────────────────────────────────────────────────────
    attribution_path = LABELED / PROBE / "attribution.json"
    if attribution_path.exists():
        attr_model = AttributionModel.load(attribution_path)
        attr_df = attr_model.attribute(df, k_effect=10, k_disruption=10)
        attr_by_vid = dict(zip(attr_df["variant_id"].to_list(), attr_df["attribution_json"].to_list(), strict=True))
        attr_logits = dict(zip(attr_df["variant_id"].to_list(), attr_df["attribution_logit"].tolist(), strict=True))
        _t(f"Attribution loaded for {len(attr_by_vid):,} variants")
    else:
        attr_by_vid = {}
        attr_logits = {}
        logger.warning(f"No attribution model at {attribution_path}, skipping")

    # ── Write to staging ─────────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="variant_viewer_"))
    staging_vdir = staging / "variants"
    staging_vdir.mkdir()

    # ── Variant JSONs ────────────────────────────────────────────────────
    _t("Writing variant JSONs...")
    meta_fields = (
        "variant_id", "gene_name", "chrom", "pos", "ref", "alt",
        "vcf_pos", "gene_strand",
        "consequence", "substitution", "label", "clinical_significance",
        "stars", "disease_name", "score_pathogenic", "rs_id", "allele_id", "gene_id",
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
    # ClinVar submission fields (may not exist if submissions.feather not generated)
    clinvar = {
        k: col_data.get(k, _none)
        for k in ("variation_id", "cytogenetic", "review_status", "acmg_codes",
                  "submitters", "clinical_features", "n_submissions",
                  "last_evaluated", "origin")
    }

    for i in track(range(n), description="Writing variants..."):
        vid = col_data["variant_id"][i]

        # Only include heads with meaningful delta
        disruption = {}
        for h in disruption_heads:
            r = ref_d[h][i]
            if r is None:
                continue
            va = var_d[h][i] if var_d[h][i] is not None else r
            delta = round(va - r, 4)
            if abs(delta) > 0.01:
                disruption[h] = delta

        effect = {h: eff_d[h][i] for h in effect_heads if eff_d[h][i] is not None}
        gt = {h: v for h, col in gt_d.items() if (v := col[i]) is not None and v > 0}

        neighbors = nb_map.get(vid, [])
        n_p = sum(1 for nb in neighbors if "pathogenic" in nb.get("label", ""))
        n_b = sum(1 for nb in neighbors if "benign" in nb.get("label", ""))

        data = {
            "id": vid,
            "gene": col_data["gene_name"][i],
            "chrom": col_data["chrom"][i], "pos": col_data["pos"][i],
            "ref": col_data["ref"][i], "alt": col_data["alt"][i],
            "vcf_pos": col_data["vcf_pos"][i],
            "gene_strand": col_data["gene_strand"][i],
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
            "domains": resolve_domains(vep["domains"][i], domain_cache),
            "loeuf": vep["loeuf"][i],
            "gnomad": vep["gnomade"][i],
            "gnomad_pop": {k: v for k, col in gnomad_pops.items() if (v := col[i]) is not None and v > 0},
            "variation_id": clinvar["variation_id"][i],
            "cytogenetic": clinvar["cytogenetic"][i],
            "review_status": clinvar["review_status"][i],
            "acmg": [c for c in (clinvar["acmg_codes"][i] or "").split(";") if c],
            "n_submissions": clinvar["n_submissions"][i],
            "submitters": [s for s in (clinvar["submitters"][i] or "").split(";") if s],
            "last_evaluated": clinvar["last_evaluated"][i],
            "clinical_features": [f for f in (clinvar["clinical_features"][i] or "").split(";") if f and f != "not provided"],
            "origin": clinvar["origin"][i],
            "disruption": disruption,
            "effect": effect,
            "gt": gt,
            "attribution": _parse_attribution(attr_by_vid.get(vid)),
            "attr_logit": attr_logits.get(vid),
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

    # Disruption histograms use delta (var - ref), effect histograms use raw scores
    delta_exprs = [(pl.col(vc) - pl.col(rc)).alias(f"delta_{h}")
                   for rc, vc, h in zip(ref_cols, var_cols, disruption_heads)]
    delta_cols = [f"delta_{h}" for h in disruption_heads]
    df_hist = df.with_columns(delta_exprs)
    score_matrix = torch.from_numpy(
        df_hist.select(delta_cols + eff_cols).to_numpy(allow_copy=True).T
    ).float()  # (n_heads, n_variants)

    n_disruption = len(disruption_heads)
    # Remap delta heads from [-1,1] → [0,1] for histogram binning
    score_matrix[:n_disruption] = (score_matrix[:n_disruption] + 1) / 2

    hh = {}
    for j, head_name in enumerate(all_head_names):
        hh[head_name] = {
            "benign": prebin(score_matrix[j][ben_mask], 40),
            "pathogenic": prebin(score_matrix[j][path_mask], 40),
            "bins": 40,
            "range": [-1, 1] if j < n_disruption else [0, 1],
        }

    eval_metrics = {}
    eval_path = LABELED / PROBE / "eval.json"
    if eval_path.exists():
        for h, info in json.loads(eval_path.read_text()).items():
            for key, label in EVAL_KEYS:
                if key in info:
                    eval_metrics[h] = {"metric": label, "value": round(info[key], 2)}
                    break

    decomp_path = LABELED / PROBE / "score_decomposition.json"
    (staging / "global.json").write_bytes(orjson.dumps({
        "umap": {
            "x": np.round(umap_coords[:, 0], 2).tolist(),
            "y": np.round(umap_coords[:, 1], 2).tolist(),
            "score": umap_sub["score"].to_list(),
            "ids": umap_sub["variant_id"].to_list(),
            "genes": [gene_to_idx[g] for g in umap_sub["gene"].to_list()],
            "labels": [LABEL_TO_IDX.get(lab, 2) for lab in umap_sub["label"].to_list()],
            "gene_list": gene_list,
        } if umap_coords is not None else None,
        "distributions": {
            "pathogenic": {
                "benign": prebin(scores_t[ben_mask], 80),
                "pathogenic": prebin(scores_t[path_mask], 80),
                "bins": 80,
            },
            **hh,
        },
        "eval": eval_metrics,
        "heads": {"disruption": auto_group(set(disruption_heads)), "effect": auto_group(set(effect_heads))},
        "display": {h: display_name(h, domain_cache) for h in all_head_names},
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
    interp_src = VUS / "interpretations"
    if interp_src.exists():
        interp_dst = staging / "interpretations"
        interp_dst.mkdir(parents=True, exist_ok=True)
        for f in interp_src.glob("*.json"):
            shutil.copy2(f, interp_dst / f.name)

    # ── Sync or serve from staging ───────────────────────────────────────
    if args.sync and args.output:
        out = Path(args.output)
        _t(f"Syncing to {out}...")
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{staging}/", f"{out}/"], check=True)
        shutil.rmtree(staging)
        _t(f"Done. {n:,} variants in {out}")
    else:
        _t(f"Done. {n:,} variants in {staging}")
        print(f"STAGING_DIR={staging}")


if __name__ == "__main__":
    main()
