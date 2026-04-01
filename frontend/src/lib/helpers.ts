export const LABEL_DISPLAY: Record<string, string> = {
  pathogenic: 'Pathogenic',
  likely_pathogenic: 'Likely Pathogenic',
  benign: 'Benign',
  likely_benign: 'Likely Benign',
  VUS: 'VUS',
};

export const CSQ_DISPLAY: Record<string, string> = {
  missense_variant: 'Missense',
  synonymous_variant: 'Synonymous',
  frameshift_variant: 'Frameshift',
  nonsense: 'Nonsense',
  stop_gained: 'Stop Gained',
  splice_donor_variant: 'Splice Donor',
  splice_acceptor_variant: 'Splice Acceptor',
  splice_region_variant: 'Splice Region',
  intron_variant: 'Intronic',
  'non-coding_transcript_variant': 'Non-coding',
  start_lost: 'Start Lost',
  stop_lost: 'Stop Lost',
  inframe_deletion: 'In-frame Deletion',
  inframe_insertion: 'In-frame Insertion',
  inframe_indel: 'In-frame Indel',
  '5_prime_UTR_variant': "5' UTR",
  '3_prime_UTR_variant': "3' UTR",
  genic_downstream_transcript_variant: 'Downstream',
  genic_upstream_transcript_variant: 'Upstream',
  initiator_codon_variant: 'Initiator Codon',
  no_sequence_alteration: 'No Change',
};

export function humanConsequence(c: string): string {
  if (!c) return '';
  return CSQ_DISPLAY[c] || c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export function truncate(s: string, n = 30): string {
  return s.length > n ? s.slice(0, n - 2) + '\u2026' : s;
}

export function escAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function formatAF(af: number | null): string {
  if (af == null || af === 0) return 'Not observed';
  if (af < 0.0001) return af.toExponential(1) + ` (1 / ${Math.round(1 / af).toLocaleString()})`;
  return af.toFixed(4);
}

export function afNote(af: number | null): string {
  if (af == null || af === 0) return 'Absent from gnomAD';
  if (af > 0.05) return 'Common (BA1)';
  if (af > 0.01) return 'Polymorphism (BS1)';
  if (af < 0.0001) return 'Rare (supports PM2)';
  return '';
}

export function loeufInterpretation(loeuf: number | null): string | null {
  if (loeuf == null) return null;
  if (loeuf < 0.35) return 'highly constrained';
  if (loeuf < 0.6) return 'constrained';
  if (loeuf < 1.0) return 'tolerant';
  return 'unconstrained';
}

export function extractDelta(rawD: [number, number] | number | null): { ref: number; var: number; delta: number } {
  if (rawD == null) return { ref: 0, var: 0, delta: 0 };
  if (Array.isArray(rawD)) return { ref: rawD[0], var: rawD[1], delta: rawD[1] - rawD[0] };
  return { ref: 0, var: 0, delta: rawD };
}

export const LABEL_NAMES = ['benign', 'pathogenic', 'VUS'];
