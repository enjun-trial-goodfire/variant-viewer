<script lang="ts">
  import { scoreColor } from '../../lib/colors';
  import { labelClass, navigate } from '../../lib/helpers';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  // All display strings come precomputed from the transform step.
  // The frontend just assembles them.
  const context = $derived(
    [
      v.exon ? `exon ${v.exon}` : '',
      (v.domains || []).slice(0, 3).map(d => d.name || d.id || '?').join(', '),
      v.loeuf != null ? `LOEUF: ${v.loeuf.toFixed(2)} (${v.loeuf_label})` : '',
    ].filter(Boolean).join(' \u00a0|\u00a0 ')
  );

  const idLine = $derived(
    [
      `${v.chrom}:${v.vcf_pos} ${v.ref}>${v.alt}`,
      v.hgvsc_short,
      v.hgvsp_short,
      v.consequence_display,
    ].filter(Boolean).join(' \u00a0|\u00a0 ')
  );

  const stars = $derived(
    v.stars > 0 ? ' ' + '\u2605'.repeat(v.stars) + '\u2606'.repeat(Math.max(0, 4 - v.stars)) : ''
  );

  const reviewParts = $derived(
    [
      v.review_status ? `Review: ${v.review_status}` : '',
      v.n_submissions ? `${v.n_submissions} submitter${v.n_submissions > 1 ? 's' : ''}` : '',
      v.last_evaluated ? `evaluated ${v.last_evaluated}` : '',
    ].filter(Boolean).join(' \u00b7 ')
  );

  const links = $derived(
    [
      v.allele_id ? { label: 'ClinVar', href: `https://www.ncbi.nlm.nih.gov/clinvar/?term=${v.allele_id}[alleleid]` } : null,
      { label: 'gnomAD', href: `https://gnomad.broadinstitute.org/variant/${v.chrom}-${v.vcf_pos}-${v.ref}-${v.alt}?dataset=gnomad_r4` },
      v.rs_id ? { label: 'dbSNP', href: `https://www.ncbi.nlm.nih.gov/snp/${v.rs_id}` } : null,
      { label: 'UCSC', href: `https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=${v.chrom}:${v.vcf_pos}` },
      { label: 'UniProt', href: `https://www.uniprot.org/uniprotkb?query=gene:${v.gene_name}+organism_id:9606` },
      v.gene_id ? { label: 'Ensembl', href: `https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=${v.gene_id}` } : null,
      { label: 'OMIM', href: `https://omim.org/search?search=${v.gene_name}` },
      { label: 'GeneCards', href: `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${v.gene_name}` },
    ].filter(Boolean) as Array<{ label: string; href: string }>
  );

  function acmgColor(code: string) {
    if (code.startsWith('PVS') || code.startsWith('PS')) return '#c55';
    if (code.startsWith('PM')) return '#c93';
    if (code.startsWith('BA') || code.startsWith('BS')) return '#27a';
    return '#888';
  }
</script>

<div class="card">
  <div class="verdict">
    <div class="verdict-info">
      <div class="verdict-gene">
        {v.gene_name}
        <span style="font-size:14px;font-weight:400;color:var(--text-secondary)">{context}</span>
      </div>
      <div class="verdict-id" title={v.variant_id}>{idLine}</div>
      <div class="verdict-meta">
        <span class="label-badge {labelClass(v.label)}">{v.label_display}</span>
        {#if stars}<span style="color:var(--accent)">{stars}</span>{/if}
        {#each v.acmg || [] as code}
          <span style="font-size:10px;font-weight:600;color:{acmgColor(code)};background:color-mix(in srgb, {acmgColor(code)} 12%, transparent);padding:1px 5px;border-radius:4px">{code}</span>
        {/each}
        {#if v.disease} · {v.disease}{/if}
        {#if v.origin} · {v.origin}{/if}
        · gnomAD AF: <b>{v.gnomad_display}</b>
        {#if v.gnomad_label}
          <span style="font-size:10px;padding:1px 5px;border-radius:4px;background:var(--bg-track)">{v.gnomad_label}</span>
        {/if}
      </div>
      {#if reviewParts}
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px">{reviewParts}</div>
      {/if}
      {#if v.clinical_features?.length}
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px">
          {#each v.clinical_features as f}
            <span style="background:var(--bg-track);padding:1px 6px;border-radius:4px;margin-right:3px">{f}</span>
          {/each}
        </div>
      {/if}
      <div class="verdict-links">
        {#each links as link}
          <a href={link.href} target="_blank" rel="noopener">{link.label}</a>
        {/each}
      </div>
    </div>
    <div class="verdict-score">
      <div class="score-label">Predicted<br>Pathogenicity</div>
      <div class="number" style="color:{scoreColor(v.score_pathogenic)}">{(v.score_pathogenic * 100).toFixed(0)}%</div>
    </div>
  </div>
</div>
