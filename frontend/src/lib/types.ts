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
  consequence: string;
  label: string;
  score: number;
  similarity: number;
}

export interface Variant {
  id: string;
  gene: string;
  chrom: string;
  pos: number;
  ref: string;
  alt: string;
  vcf_pos: number;
  gene_strand: string;
  consequence: string;
  substitution: string;
  label: string;
  significance: string;
  stars: number;
  disease: string;
  score: number;
  rs_id: string;
  allele_id: number | null;
  gene_id: string;
  hgvsc: string;
  hgvsp: string;
  impact: string;
  exon: string;
  transcript: string;
  swissprot: string;
  domains: Domain[];
  loeuf: number | null;
  gnomad: number | null;
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
  disruption: Record<string, [number, number]>;
  effect: Record<string, number>;
  gt: Record<string, number>;
  attribution: Attribution[];
  neighbors: Neighbor[];
  nP: number;
  nB: number;
  nV: number;
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
  range?: [number, number];
}

export interface HeadDistribution {
  delta?: DistributionData;
  ref?: DistributionData;
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

export interface GlobalData {
  distributions: Record<string, HeadDistribution | DistributionData>;
  eval: Record<string, EvalMetric>;
  heads: {
    disruption: Record<string, string[]>;
    effect: Record<string, string[]>;
  };
  display: Record<string, string>;
  head_stats: Record<string, HeadStat>;
  descriptions: Record<string, string>;
  decomposition: any;
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
