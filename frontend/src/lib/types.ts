export interface Domain {
  db: string;
  id: string;
  name?: string;
}

export interface AttributionHead {
  name: string;
  kind: string;
  score: number;
  contribution: number;
}

export interface Attribution {
  id: number;
  contribution: number;
  direction: string;
  z: number;
  heads: AttributionHead[];
  name: string;
}

export interface Neighbor {
  id: string;
  gene: string;
  consequence_display: string;
  label: string;
  label_display: string;
  score: number;
  similarity: number;
}

export interface Variant {
  variant_id: string;
  gene_name: string;
  chrom: string;
  pos: number;
  ref: string;
  alt: string;
  vcf_pos: number;
  gene_strand: string;
  consequence: string;
  consequence_display: string;
  substitution: string;
  label: string;
  label_display: string;
  significance: string;
  stars: number;
  disease: string;
  score_pathogenic: number;
  rs_id: string;
  allele_id: number | null;
  gene_id: string;
  hgvsc: string;
  hgvsc_short: string;
  hgvsp: string;
  hgvsp_short: string;
  vep_impact: string;
  exon: string;
  vep_transcript_id: string;
  vep_protein_id: string;
  domains: Domain[];
  loeuf: number | null;
  loeuf_label: string | null;
  gnomad: number | null;
  gnomad_display: string;
  gnomad_label: string | null;
  gnomad_pop: Record<string, number>;
  variation_id: string;
  cytogenetic: string;
  review_status: string;
  acmg: string[];
  n_submissions: number | null;
  submitters: string[];
  last_evaluated: string | null;
  clinical_features: string[];
  origin: string;
  disruption: Record<string, { ref: number; var: number; z: number; ref_lr: number; var_lr: number }>;
  effect: Record<string, { value: number; lr: number }>;
  gt: Record<string, number>;
  attribution: Attribution[];
  neighbors: Neighbor[];
}

export interface SearchResult {
  v: string;
  l: string;
  s: number;
  c: string;
}

export interface DistributionData {
  benign: number[];
  pathogenic: number[];
  bins: number;
  range: [number, number];
  _bTotal: number;
  _pTotal: number;
}

export interface HeatmapData {
  data: [number, number, number, number][];
  ref_range: [number, number];
  var_range: [number, number];
  bins: number;
}

export interface HeadDistribution {
  delta?: DistributionData;
  ref?: DistributionData;
  heatmap?: HeatmapData;
  benign?: number[];
  pathogenic?: number[];
  bins?: number;
  range?: [number, number];
}

export interface HeadStat {
  mean: number;
  std: number;
}

export interface EvalMetric {
  metric: string;
  value: number;
}

export interface HeadConfig {
  display?: string;
  category: 'disruption' | 'effect';
  group?: string;
  eval?: EvalMetric;
  mean?: number;
  std?: number;
  description?: string;
  quality?: string;
  predictor?: { order: number; threshold: number; invert?: boolean; display?: string };
  exclude_from_attribution?: boolean;
  exclude_from_effect_expansion?: boolean;
}

export interface GlobalData {
  distributions: Record<string, HeadDistribution | DistributionData>;
  heads: {
    _meta: Record<string, any>;
    heads: Record<string, HeadConfig>;
  };
}

export interface UmapData {
  x: number[];
  y: number[];
  score: number[];
  ids: string[];
  genes: number[];
  labels: number[];
  gene_list: string[];
}

export interface Interpretation {
  status: string;
  variant_id: string;
  summary: string;
  mechanism: string;
  confidence: 'high' | 'medium' | 'low';
  key_evidence: string[];
  model: string;
  generated_at: number;
}
