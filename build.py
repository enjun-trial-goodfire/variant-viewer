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

from attribution import AttributionModel
from umap import UMAP

from goodfire_core.storage import ActivationDataset, FilesystemStorage

# ── Constants ────────────────────────────────────────────────────────────
MAYO_DATA = Path(__file__).parent.parent / "data"
ARTIFACTS = Path("/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian")

# Consequence → integer encoding (deterministic order by frequency in deconfounded-full)
CONSEQUENCE_CLASSES = (
    "missense_variant", "intron_variant", "synonymous_variant", "nonsense",
    "frameshift_variant", "non-coding_transcript_variant", "splice_donor_variant",
    "splice_acceptor_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
    "splice_region_variant", "start_lost", "inframe_deletion", "inframe_insertion",
    "inframe_indel", "stop_lost", "genic_downstream_transcript_variant",
    "genic_upstream_transcript_variant", "no_sequence_alteration",
    "initiator_codon_variant",
)

_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SWAP_CLASSES = tuple(f"{a}>{b}" for a in _AA for b in _AA if a != b)

ANNOTATIONS = MAYO_DATA / "clinvar" / "deconfounded-full" / "annotations_v8.feather"
LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"
PROBE = "probe_v8"
K_NEIGHBORS = 10
LABEL_TO_IDX = {"benign": 0, "pathogenic": 1, "VUS": 2}
EVAL_KEYS = (("correlation", "r"), ("auc", "AUC"), ("accuracy", "acc"))
VARIANT_ANN_DIR = ARTIFACTS / "clinvar_evo2_labeled" / "variant_annotations"
VEP_DOMAIN_CACHE = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/annotations/"
    "sources/cache/vep_domain_lookup.json"
)
VEP_COLS = (
    "variant_id", "vep_hgvsc", "vep_hgvsp", "vep_impact",
    "vep_exon", "vep_transcript_id", "vep_protein_id", "vep_swissprot",
    "vep_domains", "vep_loeuf",
    "vep_gnomade", "vep_gnomade_afr", "vep_gnomade_amr", "vep_gnomade_asj",
    "vep_gnomade_eas", "vep_gnomade_fin", "vep_gnomade_nfe", "vep_gnomade_sas",
)

# ── Display names ────────────────────────────────────────────────────────
_DISPLAY_OVERRIDES: dict[str, str] = {
    "cadd_c": "CADD", "revel_c": "REVEL", "alphamissense_c": "AlphaMissense",
    "sift_c": "SIFT", "polyphen_c": "PolyPhen-2", "eve_c": "EVE",
    "spliceai_max_c": "SpliceAI", "gnomad_af_c": "gnomAD AF",
    "spliceai_ag_c": "SpliceAI AG", "spliceai_al_c": "SpliceAI AL",
    "spliceai_dg_c": "SpliceAI DG", "spliceai_dl_c": "SpliceAI DL",
    "mpc_c": "MPC", "mvp_c": "MVP", "mcap_c": "M-CAP",
    "metalr_c": "MetaLR", "vest4_c": "VEST4", "primateai_c": "PrimateAI",
    "mutpred_c": "MutPred", "clinpred_c": "ClinPred", "deogen2_c": "DEOGEN2",
    "bayesdel_c": "BayesDel", "remm_c": "ReMM", "regulomedb_c": "RegulomeDB",
    "phylop_c": "PhyloP", "phastcons_c": "PhastCons", "gerp_c": "GERP",
    "phylop_100way": "PhyloP 100-way", "phastcons_100way": "PhastCons 100-way",
    "blosum62_c": "BLOSUM62", "grantham_c": "Grantham",
    "hydrophobicity_c": "Hydrophobicity", "volume_c": "Volume", "mw_c": "Mol. Weight",
    "loeuf_c": "LOEUF",
    "gtex_max_tpm_c": "GTEx Max TPM", "gtex_n_tissues_c": "GTEx Tissues",
    "gtex_brain_max_c": "GTEx Brain Max",
    "in_domain": "In Domain", "is_exonic": "Exonic",
    "splice_disrupting": "Splice Disrupting", "charge_altering": "Charge Altering",
    "pathogenic": "Pathogenicity", "consequence": "Consequence", "impact": "Impact",
    "aa_swap": "AA Substitution",
    # AlphaFold structure features (avoid confusion with splicing)
    "psi": "AlphaFold Psi Angle", "phi": "AlphaFold Phi Angle",
    "plddt": "AlphaFold Confidence (pLDDT)", "rsa": "Relative Solvent Accessibility",
    "sasa": "Solvent Accessible Surface Area",
    "secondary_structure_H": "Alpha Helix", "secondary_structure_E": "Beta Strand",
    "secondary_structure_C": "Coil/Loop",
    "is_disordered": "Intrinsically Disordered", "is_buried": "Buried Residue",
    "n_contacts": "Residue Contacts", "residue_number": "Residue Number",
    "gc_content": "GC Content", "cpg_density": "CpG Density",
    "recomb_rate": "Recombination Rate", "trinuc_mutation_rate": "Trinucleotide Mutation Rate",
    "codon_position": "Codon Position", "syn_potential": "Synonymous Potential",
    "exon_number": "Exon Number", "n_transcripts_with_exon": "Transcripts with Exon",
    "disorder_content_fraction": "Disorder Content",
    "ppi_partner_count": "PPI Partners", "is_ppi_interface": "PPI Interface",
    "is_splice_donor": "Splice Donor Site", "is_splice_acceptor": "Splice Acceptor Site",
    "is_branchpoint_region": "Branchpoint Region", "is_polypyrimidine_tract": "Polypyrimidine Tract",
    "is_exon_to_intron": "Exon-Intron Boundary", "is_intron_to_exon": "Intron-Exon Boundary",
    "cadd_wg_c": "CADD (whole-genome)",
}

_ACRONYMS = {"chipseq", "atacseq", "chromhmm", "fstack", "ptm", "elm"}
_GROUP_PREFIXES = (
    "interpro_", "pfam_", "amino_acid_", "elm_", "gtex_",
    "chipseq_", "atacseq_", "chromhmm_", "fstack_", "ptm_",
)

# ── Head grouping (reverse map: group → prefixes) ───────────────────────
_PREFIX_TO_GROUP: dict[str, str] = {}
for _group, _prefixes in {
    "Conservation": ("phylop", "phastcons", "gerp"),
    "Protein features": ("secondary", "disorder", "plddt", "rsa", "sasa", "phi", "psi", "ppi", "residue"),
    "Structure": ("in", "is", "has"),
    "InterPro domains": ("interpro",),
    "Pfam domains": ("pfam",),
    "ELM motifs": ("elm",),
    "ChIP-seq": ("chipseq",),
    "ATAC-seq": ("atacseq",),
    "Chromatin": ("chromhmm", "fstack"),
    "Regulatory": ("ccre", "dna"),
    "Amino acid": ("amino",),
    "Sequence context": ("codon", "trinuc", "gc", "cpg", "syn"),
    "Substitution": ("aa", "blosum62", "grantham", "hydrophobicity", "volume", "mw"),
    "Modifications": ("ptm",),
    "Expression": ("gtex",),
    "Region": ("region", "exon", "n", "trf", "segdup", "recomb"),
    "Constraint": ("loeuf",),
    "Clinical": (
        "cadd", "revel", "alphamissense", "sift", "polyphen", "eve", "bayesdel",
        "clinpred", "deogen2", "mcap", "metalr", "mpc", "mutpred", "mvp",
        "primateai", "vest4", "remm", "regulomedb",
    ),
    "Splice": ("spliceai",),
    "Variant effects": ("splice", "charge", "consequence", "impact", "csq"),
    "Population": ("gnomad",),
    "Pathogenicity": ("pathogenic",),
}.items():
    _PREFIX_TO_GROUP.update({p: _group for p in _prefixes})


def load_domain_names() -> dict[str, str]:
    """Load VEP domain ID→name cache (e.g. 'Pfam:PF00079' → 'Serpin')."""
    if VEP_DOMAIN_CACHE.exists():
        with open(VEP_DOMAIN_CACHE) as f:
            return json.load(f)
    return {}


def resolve_domains(raw: str | None, cache: dict[str, str]) -> list[dict] | None:
    """Convert 'Pfam:PF00079;CDD:cd02056;...' to [{db, id, name?}, ...]."""
    if not raw:
        return None
    result = []
    for entry in raw.split(";"):
        parts = entry.split(":", 1)
        if len(parts) != 2:
            continue
        db, did = parts
        key = f"{db}:{did}"
        name = cache.get(key)
        d = {"db": db, "id": did}
        if name:
            d["name"] = name
        result.append(d)
    return result or None


def display_name(h: str, domain_cache: dict[str, str] | None = None) -> str:
    """Human-readable head name, with group prefix stripped."""
    if h in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[h]
    base = h.removesuffix("_c")
    stripped_prefix = ""
    for prefix in _GROUP_PREFIXES:
        if base.startswith(prefix):
            stripped_prefix = prefix.rstrip("_").replace("_", " ").title()
            base = base[len(prefix):]
            break
    base = base.replace("_", " ")
    # Keep prefix for very short names (e.g., amino_acid_L → "Amino Acid L")
    if len(base.strip()) <= 2 and stripped_prefix:
        return f"{stripped_prefix} {base.strip().upper()}"
    if h.split("_")[0] in _ACRONYMS:
        return base.upper()
    if base.startswith("PF"):
        name = (domain_cache or {}).get(f"Pfam:{base}")
        return f"{name} ({base})" if name else base
    return base.title()


def auto_group(heads: set[str]) -> dict[str, list[str]]:
    """Group heads by first-token prefix lookup."""
    g: dict[str, list[str]] = {}
    for h in sorted(heads):
        g.setdefault(_PREFIX_TO_GROUP.get(h.split("_")[0], "Other"), []).append(h)
    return g


def _parse_attribution(json_str: str | None) -> dict:
    """Parse attribution JSON. Expects {effect: [...], disruption: [...]}."""
    if not json_str:
        return {}
    return orjson.loads(json_str)


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


def load_metadata(preset: str) -> pl.DataFrame:
    """Load ClinVar metadata and enrich with gene_id/gene_strand from GENCODE."""
    meta = pl.read_ipc(MAYO_DATA / "clinvar" / preset / "metadata.feather")
    genes = (
        pl.read_ipc(MAYO_DATA / "gencode" / "genes.feather")
        .select("gene_name", "gene_id", "strand")
        .unique(subset=["gene_name"])
    )
    return (
        meta.join(genes, on="gene_name", how="inner")
        .unique(subset=["variant_id"])
        .rename({"strand": "gene_strand"})
    )


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
    parser.add_argument("output", nargs="?", default="webapp/build", help="Output directory")
    parser.add_argument("--no-sync", action="store_true", help="Write to /tmp, skip rsync")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP computation (reuse from previous build)")
    parser.add_argument("--skip-neighbors", action="store_true", help="Skip neighbor computation")
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

    df = df.with_columns(
        pl.col("pred_consequence").replace_strict(dict(enumerate(CONSEQUENCE_CLASSES)), default="unknown").alias("consequence"),
        pl.col("pred_aa_swap").replace_strict(dict(enumerate(AA_SWAP_CLASSES)), default=None).alias("substitution"),
    )

    gt = pl.read_ipc(ANNOTATIONS)
    df = df.join(
        gt.rename({c: f"gt_{c}" for c in gt.columns if c != "variant_id"}),
        on="variant_id", how="left",
    )

    # VEP annotations (HGVS, gnomAD frequencies, domains, etc.)
    _t("Loading VEP annotations...")
    vep_string_cols = {"variant_id", "vep_hgvsc", "vep_hgvsp", "vep_impact",
                        "vep_exon", "vep_transcript_id", "vep_protein_id", "vep_swissprot", "vep_domains"}
    vep_dfs = []
    for chrom_file in sorted(VARIANT_ANN_DIR.glob("variant_annotations_chr*.parquet")):
        schema = pl.read_parquet_schema(chrom_file)
        available = [c for c in VEP_COLS if c in schema]
        chunk = pl.read_parquet(chrom_file, columns=available)
        # Normalize mixed String/Float columns across chromosomes (single pass)
        float_casts = [c for c in chunk.columns if c not in vep_string_cols and chunk[c].dtype in (pl.Utf8, pl.String)]
        if float_casts:
            chunk = chunk.with_columns(*(pl.col(c).cast(pl.Float64, strict=False) for c in float_casts))
        vep_dfs.append(chunk)
    vep = pl.concat(vep_dfs, how="diagonal")
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

    if args.skip_neighbors and args.skip_umap:
        _t("Skipping embeddings (--skip-neighbors --skip-umap)")
        nb_map = {}
        emb_df = None
    else:
        emb_l, ids_l = load_emb(LABELED)
        emb_v, ids_v = load_emb(VUS)
        emb = torch.nn.functional.normalize(torch.cat([emb_l, emb_v]).float(), dim=1)
        emb_ids = ids_l + ids_v
        n_emb = len(emb_ids)

    if args.skip_neighbors:
        _t("Skipping neighbors (--skip-neighbors)")
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
    if args.skip_umap:
        _t("Skipping UMAP (--skip-umap)")
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
        "distribution": {
            "benign": prebin(scores_t[ben_mask], 80),
            "pathogenic": prebin(scores_t[path_mask], 80),
            "bins": 80,
        },
        "head_distributions": hh,
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
    out = Path(args.output)

    if args.no_sync:
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
