/**
 * View model for multi-lane local locus visualization (sequence + annotation + nearby variants).
 * V1: annotation intervals are mocked. Nearby variants come from the served DuckDB subset via
 * GET /api/variants/nearby-locus (see fetchNearbyLocusVariants in api.ts).
 */

import type { Variant } from './types';

export const LOCUS_PAD_BP = 3000;
/** Reserved (e.g. future zoomed sequence); reference lane uses full locus [locusStart, locusEnd]. */
export const SEQUENCE_PAD_BP = 20;

export type AnnotationCategory = 'coding' | 'utr' | 'intron' | 'regulatory' | 'other';

export interface AnnotationSpan {
  start: number;
  end: number;
  category: AnnotationCategory;
  label?: string;
}

/**
 * Point variant in the locus lane. interactionTerm drives marker color (interactionTermColor);
 * until a dedicated field exists, the API uses pathogenicity clamped to [0, 1] as a placeholder.
 */
export interface NearbyVariantPoint {
  genomicPos: number;
  variantId: string;
  interactionTerm: number;
  /** From dataset (optional tooltip). */
  labelDisplay?: string;
}

export interface LocusFocal {
  vcfPos: number;
  ref: string;
  alt: string;
}

/**
 * Single source of truth for the locus SVG x-axis: closed genomic interval [locusStart, locusEnd].
 * Reference sequence is fetched for [seqStart, seqEnd]; V1 sets seqStart/seqEnd equal to the locus (full displayed window).
 */
export interface LocusTrackViewModel {
  chrom: string;
  ensemblChrom: string;
  locusStart: number;
  locusEnd: number;
  seqStart: number;
  seqEnd: number;
  focal: LocusFocal;
  /** Uppercase bases without whitespace; from Ensembl REST for [seqStart, seqEnd]; expect length seqEnd - seqStart + 1. */
  sequence: string;
  annotations: AnnotationSpan[];
  nearbyVariants: NearbyVariantPoint[];
  /** Set when annotation lane uses placeholder data (V1). */
  mockAnnotations: boolean;
}

/** Inclusive locus length in bp. */
export function locusBaseCount(locusStart: number, locusEnd: number): number {
  return Math.max(1, locusEnd - locusStart + 1);
}

/**
 * Linear band: each integer 1-based position maps to a column of width widthBp.
 * left(p) = left edge of base p; interval [a,b] inclusive has width (b-a+1)*widthBp.
 * Sequence character i (0-based) at genomic seqStart+i uses center(seqStart+i) — same scale as center(vcfPos).
 */
export function genomicBand(locusStart: number, locusEnd: number, innerWidth: number): {
  widthBp: number;
  left: (pos: number) => number;
  center: (pos: number) => number;
  intervalWidth: (start: number, end: number) => number;
} {
  const n = locusBaseCount(locusStart, locusEnd);
  const widthBp = innerWidth / n;
  return {
    widthBp,
    left: (pos: number) => (pos - locusStart) * widthBp,
    center: (pos: number) => (pos - locusStart + 0.5) * widthBp,
    intervalWidth: (start: number, end: number) => (end - start + 1) * widthBp,
  };
}

function stripChr(chrom: string): string {
  const c = chrom.replace(/^chr/i, '');
  return c === 'M' ? 'MT' : c;
}

/** ±LOCUS_PAD_BP window and Ensembl chromosome name (REST / Location/View). */
export function locusRegionFromVariant(v: Variant): {
  locusStart: number;
  locusEnd: number;
  ensemblChrom: string;
} {
  return {
    locusStart: Math.max(1, v.vcf_pos - LOCUS_PAD_BP),
    locusEnd: v.vcf_pos + LOCUS_PAD_BP,
    ensemblChrom: stripChr(v.chrom),
  };
}

/** Deterministic PRNG for stable mocks per variant. */
function rngForVariant(v: Variant): () => number {
  let h = 0;
  for (let i = 0; i < v.variant_id.length; i++) h = (Math.imul(31, h) + v.variant_id.charCodeAt(i)) | 0;
  let s = (h ^ v.vcf_pos) >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function clampInterval(
  start: number,
  end: number,
  winStart: number,
  winEnd: number,
): { start: number; end: number } | null {
  const a = Math.max(start, winStart);
  const b = Math.min(end, winEnd);
  if (a > b) return null;
  return { start: a, end: b };
}

/** V1 mock: GENCODE-style categorical intervals overlapping the locus. */
export function mockAnnotations(
  locusStart: number,
  locusEnd: number,
  focalPos: number,
  rnd: () => number,
): AnnotationSpan[] {
  const spans: AnnotationSpan[] = [];
  const w = locusEnd - locusStart;
  const cats: AnnotationCategory[] = ['coding', 'utr', 'intron', 'regulatory', 'other'];

  // Gene-like body around focal
  const geneStart = focalPos - Math.floor(400 + rnd() * 200);
  const geneEnd = focalPos + Math.floor(500 + rnd() * 300);
  const g0 = clampInterval(geneStart, geneEnd, locusStart, locusEnd);
  if (g0) {
    spans.push({ ...g0, category: 'intron', label: 'mock intron (gene body)' });
    const cds0 = clampInterval(focalPos - 80, focalPos + 120, locusStart, locusEnd);
    if (cds0) spans.push({ ...cds0, category: 'coding', label: 'mock CDS' });
    const utr5 = clampInterval(geneStart, Math.min(geneStart + 120, focalPos - 81), locusStart, locusEnd);
    if (utr5 && utr5.start <= utr5.end) spans.push({ ...utr5, category: 'utr', label: "mock 5' UTR" });
    const reg = clampInterval(focalPos + 200, focalPos + 900, locusStart, locusEnd);
    if (reg) spans.push({ ...reg, category: 'regulatory', label: 'mock regulatory' });
  }

  // Extra scattered regulatory / other
  for (let k = 0; k < 3; k++) {
    const len = 80 + Math.floor(rnd() * 400);
    const s = locusStart + Math.floor(rnd() * Math.max(1, w - len));
    const e = s + len;
    const c = cats[Math.floor(rnd() * cats.length)];
    const iv = clampInterval(s, e, locusStart, locusEnd);
    if (iv) spans.push({ ...iv, category: c, label: `mock ${c}` });
  }

  return spans;
}

export function buildLocusTrackViewModel(
  v: Variant,
  sequenceRaw: string,
  nearbyVariants: NearbyVariantPoint[],
): LocusTrackViewModel {
  const { locusStart, locusEnd, ensemblChrom } = locusRegionFromVariant(v);
  const seqStart = locusStart;
  const seqEnd = locusEnd;

  const rnd = rngForVariant(v);
  const sequence = sequenceRaw.replace(/\s+/g, '').toUpperCase();

  return {
    chrom: v.chrom,
    ensemblChrom,
    locusStart,
    locusEnd,
    seqStart,
    seqEnd,
    focal: { vcfPos: v.vcf_pos, ref: v.ref ?? '', alt: v.alt ?? '' },
    sequence,
    annotations: mockAnnotations(locusStart, locusEnd, v.vcf_pos, rnd),
    nearbyVariants,
    mockAnnotations: true,
  };
}

/** HSL color for interactionTerm in [0,1] (low = cool/quiet, high = warm/strong). */
export function interactionTermColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  const h = 220 - x * 200;
  const s = 42 + x * 28;
  const l = 38 + x * 18;
  return `hsl(${h.toFixed(0)}, ${s.toFixed(0)}%, ${l.toFixed(0)}%)`;
}

export const ANNOTATION_COLORS: Record<AnnotationCategory, string> = {
  coding: '#2d8a54',
  utr: '#3b7dd6',
  intron: '#9aa3ad',
  regulatory: '#c1782c',
  other: '#b8b0a8',
};
