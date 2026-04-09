-- Phase 2 — Clean coding base + exploded neighbors + pairwise joins
-- Grounded in variant-viewer build.py / transform.py; validate on your DuckDB version.
--
-- Optional session attach:
--   ATTACH '/mnt/polished-lake/home/enjunyang/variant-viewer/builds/variants.duckdb' AS vv (READ_ONLY);
--   Then use vv.variants instead of variants, or USE vv;

-- ═══════════════════════════════════════════════════════════════════════════
-- Block 1a — Conservative coding base
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW coding_variants_base AS
SELECT
  variant_id,
  gene_name,
  gene_id,
  chrom,
  pos,
  ref,
  alt,
  consequence,
  consequence_display,
  vep_impact,
  hgvsc,
  hgvsp,
  hgvsc_short,
  hgvsp_short,
  domains,
  vep_transcript_id,
  vep_protein_id,
  pathogenicity,
  label,
  significance,
  label_display,
  substitution,
  stars,
  disease,
  rs_id,
  allele_id,
  review_status,
  acmg,
  loeuf,
  gnomad
FROM variants
WHERE variant_id IS NOT NULL
  AND TRIM(CAST(variant_id AS VARCHAR)) <> ''
  AND gene_name IS NOT NULL
  AND TRIM(CAST(gene_name AS VARCHAR)) <> ''
  AND gene_id IS NOT NULL
  AND TRIM(CAST(gene_id AS VARCHAR)) <> ''
  AND (
    vep_impact IN ('HIGH', 'MODERATE')
    OR (
      COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%missense%'
      OR COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%frameshift%'
      OR COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%stop%'
      OR COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%splice%'
      OR COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%inframe%'
      OR COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%start_lost%'
      OR COALESCE(LOWER(CAST(consequence AS VARCHAR)), '') LIKE '%stop_lost%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%missense%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%frameshift%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%stop%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%splice%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%inframe%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%start lost%'
      OR COALESCE(LOWER(CAST(consequence_display AS VARCHAR)), '') LIKE '%stop lost%'
    )
    -- Optional CDS probe (uncomment if ref_region_CDS / var_region_CDS exist):
    -- OR (try_cast(ref_region_CDS AS DOUBLE) >= 0.5 OR try_cast(var_region_CDS AS DOUBLE) >= 0.5)
  );

-- ═══════════════════════════════════════════════════════════════════════════
-- Block 1b — Stricter protein / domains subset
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW coding_variants_protein AS
SELECT *
FROM coding_variants_base
WHERE hgvsp IS NOT NULL
  AND TRIM(CAST(hgvsp AS VARCHAR)) <> ''
  AND TRIM(CAST(hgvsp AS VARCHAR)) <> '?'
  AND (
    domains IS NOT NULL
    AND TRIM(CAST(domains AS VARCHAR)) NOT IN ('', '[]')
  );

-- ═══════════════════════════════════════════════════════════════════════════
-- Block 2 — Explode neighbors (preferred: from_json + generate_series + list_extract)
-- If (n).id fails, use struct_extract fallback in Block 2b below.
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW neighbor_pairs_raw AS
WITH neighbor_json AS (
  SELECT
    variant_id AS source_variant_id,
    TRIM(CAST(neighbors AS VARCHAR)) AS nb_text,
    neighbors AS neighbors_raw_column
  FROM variants
  WHERE neighbors IS NOT NULL
    AND TRIM(CAST(neighbors AS VARCHAR)) <> ''
    AND TRIM(CAST(neighbors AS VARCHAR)) <> '[]'
),
parsed AS (
  SELECT
    source_variant_id,
    neighbors_raw_column,
    nb_text,
    from_json(
      nb_text,
      'STRUCT(
        id VARCHAR,
        gene VARCHAR,
        consequence_display VARCHAR,
        label VARCHAR,
        label_display VARCHAR,
        score DOUBLE,
        similarity DOUBLE
      )[]'
    ) AS arr
  FROM neighbor_json
),
expanded AS (
  SELECT
    p.source_variant_id,
    p.neighbors_raw_column,
    p.nb_text,
    idx AS neighbor_rank,
    list_extract(p.arr, idx) AS n
  FROM parsed p
  CROSS JOIN generate_series(1, len(p.arr)) AS t(idx)
)
SELECT
  source_variant_id,
  (n).id AS neighbor_variant_id,
  neighbor_rank,
  (n).gene AS neighbor_gene_json,
  (n).consequence_display AS neighbor_consequence_display_json,
  (n).label AS neighbor_label_json,
  (n).label_display AS neighbor_label_display_json,
  (n).score AS neighbor_pathogenicity_snapshot_json,
  (n).similarity AS neighbor_embedding_cosine_similarity,
  nb_text AS neighbors_json_debug,
  CAST(n AS VARCHAR) AS neighbor_struct_debug
FROM expanded
WHERE n IS NOT NULL
  AND (n).id IS NOT NULL
  AND TRIM(CAST((n).id AS VARCHAR)) <> '';

-- ═══════════════════════════════════════════════════════════════════════════
-- Block 2b — FALLBACK: replace final SELECT if struct dot-syntax fails
-- ═══════════════════════════════════════════════════════════════════════════
-- SELECT
--   source_variant_id,
--   struct_extract(n, 'id') AS neighbor_variant_id,
--   neighbor_rank,
--   struct_extract(n, 'gene') AS neighbor_gene_json,
--   struct_extract(n, 'consequence_display') AS neighbor_consequence_display_json,
--   struct_extract(n, 'label') AS neighbor_label_json,
--   struct_extract(n, 'label_display') AS neighbor_label_display_json,
--   struct_extract(n, 'score') AS neighbor_pathogenicity_snapshot_json,
--   struct_extract(n, 'similarity') AS neighbor_embedding_cosine_similarity,
--   nb_text AS neighbors_json_debug,
--   CAST(n AS VARCHAR) AS neighbor_struct_debug
-- FROM expanded
-- WHERE n IS NOT NULL
--   AND struct_extract(n, 'id') IS NOT NULL
--   AND TRIM(CAST(struct_extract(n, 'id') AS VARCHAR)) <> '';

-- ═══════════════════════════════════════════════════════════════════════════
-- Block 3 — Pairwise join: source = coding base, neighbor = full variants
-- For coding-only neighbors: INNER JOIN coding_variants_base n instead of variants n
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW neighbor_pairs_joined AS
SELECT
  s.variant_id AS source_variant_id,
  n.variant_id AS neighbor_variant_id,
  r.neighbor_rank,
  r.neighbor_embedding_cosine_similarity,
  r.neighbors_json_debug,

  s.gene_name AS source_gene_name,
  n.gene_name AS neighbor_gene_name,
  CAST(s.gene_id AS VARCHAR) AS source_gene_id,
  CAST(n.gene_id AS VARCHAR) AS neighbor_gene_id,

  s.consequence AS source_consequence,
  n.consequence AS neighbor_consequence,
  s.consequence_display AS source_consequence_display,
  n.consequence_display AS neighbor_consequence_display,

  s.domains AS source_domains,
  n.domains AS neighbor_domains,

  s.pathogenicity AS source_pathogenicity,
  n.pathogenicity AS neighbor_pathogenicity,

  s.hgvsp AS source_hgvsp,
  n.hgvsp AS neighbor_hgvsp,

  (CAST(s.gene_name AS VARCHAR) = CAST(n.gene_name AS VARCHAR)) AS same_gene,
  (CAST(s.gene_id AS VARCHAR) = CAST(n.gene_id AS VARCHAR)) AS same_gene_id,
  (
    COALESCE(CAST(s.consequence_display AS VARCHAR), CAST(s.consequence AS VARCHAR), '')
    = COALESCE(CAST(n.consequence_display AS VARCHAR), CAST(n.consequence AS VARCHAR), '')
  ) AS same_consequence,
  (COALESCE(CAST(s.domains AS VARCHAR), '') = COALESCE(CAST(n.domains AS VARCHAR), '')) AS same_domain_string

FROM neighbor_pairs_raw r
INNER JOIN coding_variants_base s
  ON s.variant_id = r.source_variant_id
INNER JOIN variants n
  ON n.variant_id = r.neighbor_variant_id;

-- ═══════════════════════════════════════════════════════════════════════════
-- Block 4 — Validation
-- ═══════════════════════════════════════════════════════════════════════════
-- Row counts
-- SELECT 'coding_variants_base' AS name, COUNT(*) AS n FROM coding_variants_base
-- UNION ALL
-- SELECT 'coding_variants_protein', COUNT(*) FROM coding_variants_protein
-- UNION ALL
-- SELECT 'neighbor_pairs_raw', COUNT(*) FROM neighbor_pairs_raw
-- UNION ALL
-- SELECT 'neighbor_pairs_joined', COUNT(*) FROM neighbor_pairs_joined;

-- Neighbor ID resolvability vs full variants
-- SELECT
--   COUNT(*) AS n_pairs,
--   SUM(CASE WHEN n.variant_id IS NULL THEN 1 ELSE 0 END) AS neighbor_missing_in_variants,
--   100.0 * SUM(CASE WHEN n.variant_id IS NOT NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS pct_neighbor_in_variants
-- FROM neighbor_pairs_raw r
-- LEFT JOIN variants n ON n.variant_id = r.neighbor_variant_id;

-- Neighbor also in coding base (optional)
-- SELECT
--   COUNT(*) AS n_pairs,
--   100.0 * SUM(CASE WHEN c.variant_id IS NOT NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS pct_neighbor_in_coding_base
-- FROM neighbor_pairs_raw r
-- LEFT JOIN coding_variants_base c ON c.variant_id = r.neighbor_variant_id;

-- Same-gene vs cross-gene
-- SELECT same_gene, COUNT(*) AS n_pairs, 100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct
-- FROM neighbor_pairs_joined GROUP BY same_gene ORDER BY same_gene;

-- Same consequence
-- SELECT same_consequence, COUNT(*) AS n_pairs, 100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct
-- FROM neighbor_pairs_joined GROUP BY same_consequence ORDER BY same_consequence;

-- Sample
-- SELECT * FROM neighbor_pairs_joined ORDER BY source_variant_id, neighbor_rank LIMIT 20;

-- ═══════════════════════════════════════════════════════════════════════════
-- Optional materialization (in-database)
-- ═══════════════════════════════════════════════════════════════════════════
-- Tables are created by scripts/run_phase2.py as phase2_* to avoid name clashes
-- with the views above. Exports go to data/intermediate/*.parquet (or .csv).

-- CREATE OR REPLACE TABLE phase2_coding_variants_base AS SELECT * FROM coding_variants_base;
-- CREATE OR REPLACE TABLE phase2_neighbor_pairs_raw AS SELECT * FROM neighbor_pairs_raw;
-- CREATE OR REPLACE TABLE phase2_neighbor_pairs_joined AS SELECT * FROM neighbor_pairs_joined;
