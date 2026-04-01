<script lang="ts">
  import { scoreColor } from '../../lib/colors';
  import { humanConsequence, LABEL_DISPLAY, formatAF, afNote, loeufInterpretation } from '../../lib/helpers';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  let showHelp = $state(false);

  const hgvsp = $derived(v.hgvsp ? v.hgvsp.split(':').pop() : null);
  const hgvsc = $derived(v.hgvsc ? v.hgvsc.split(':').pop() : null);
  const exonStr = $derived(v.exon ? `exon ${v.exon}` : '');
  const domainStr = $derived(
    Array.isArray(v.domains) ? v.domains
      .filter(d => !['PDB-ENSP_mappings','AFDB-ENSP_mappings','ENSP_mappings','Gene3D'].includes(d.db))
      .slice(0, 3).map(d => d.name || (d.id?.split(',')[0] || '?')).join(', ') : ''
  );
  const contextParts = $derived([exonStr, domainStr].filter(Boolean));
  const stars = $derived(v.stars > 0 ? ' ' + '\u2605'.repeat(v.stars) + '\u2606'.repeat(Math.max(0, 4 - v.stars)) : '');
  const loeuf = $derived(loeufInterpretation(v.loeuf));
  const af = $derived(formatAF(v.gnomad));
  const afNoteStr = $derived(afNote(v.gnomad));

  function labelClass(label: string) {
    if (label.includes('pathogenic')) return 'label-pathogenic';
    if (label.includes('benign')) return 'label-benign';
    return 'label-vus';
  }

  function acmgColor(code: string) {
    if (code.startsWith('PVS') || code.startsWith('PS')) return '#c55';
    if (code.startsWith('PM')) return '#c93';
    if (code.startsWith('BA') || code.startsWith('BS')) return '#27a';
    return '#888';
  }
</script>

<div class="card">
  <button class="card-help-btn" onclick={() => showHelp = !showHelp}>?</button>
  {#if showHelp}
    <div class="card-help open">
      <div class="card-help-inner">
        <b>Variant Summary.</b> Shows gene, coordinates, ClinVar metadata, and predicted pathogenicity score.
      </div>
    </div>
  {/if}

  <div class="verdict">
    <div class="verdict-info">
      <div class="verdict-gene">
        {v.gene}
        <span style="font-size:14px;font-weight:400;color:var(--text-secondary)">
          {contextParts.join(' \u00a0|\u00a0 ')}
          {#if v.loeuf != null}
            {contextParts.length ? ' \u00a0|\u00a0 ' : ''}LOEUF: {v.loeuf.toFixed(2)} ({loeuf})
          {/if}
        </span>
      </div>
      <div class="verdict-id" title={v.id}>
        {v.chrom}:{v.vcf_pos ?? (v.pos+1)} {v.ref}>{v.alt}
        {#if hgvsc} &nbsp;|&nbsp; {hgvsc}{/if}
        {#if hgvsp} &nbsp;|&nbsp; {hgvsp}{/if}
        {#if v.consequence} &nbsp;|&nbsp; {humanConsequence(v.consequence)}{/if}
      </div>
      <div class="verdict-meta">
        <span class="label-badge {labelClass(v.label)}">{LABEL_DISPLAY[v.label] || v.label.replace(/_/g, ' ')}</span>
        {#if stars}<span style="color:var(--accent)">{stars}</span>{/if}
        {#each v.acmg || [] as code}
          <span style="font-size:10px;font-weight:600;color:{acmgColor(code)};background:color-mix(in srgb, {acmgColor(code)} 12%, transparent);padding:1px 5px;border-radius:4px">{code}</span>
        {/each}
        {#if v.disease} · {v.disease}{/if}
        {#if v.origin} · {v.origin}{/if}
        · gnomAD AF: <b>{af}</b>
        {#if v.gnomad != null && v.gnomad > 0 && v.gnomad < 0.0001}
          <span style="font-size:10px;padding:1px 5px;border-radius:4px;background:#dfd;color:#384">Rare (supports PM2)</span>
        {:else if v.gnomad != null && v.gnomad > 0.01}
          <span style="font-size:10px;padding:1px 5px;border-radius:4px;background:{v.gnomad > 0.05 ? '#fdd' : '#fed'};color:{v.gnomad > 0.05 ? '#944' : '#974'}">{v.gnomad > 0.05 ? 'Common (BA1)' : 'Polymorphism (BS1)'}</span>
        {/if}
      </div>
      {#if v.review_status || v.n_submissions || v.last_evaluated}
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px">
          {v.review_status ? `Review: ${v.review_status}` : ''}
          {v.n_submissions ? ` · ${v.n_submissions} submitter${v.n_submissions > 1 ? 's' : ''}` : ''}
          {v.last_evaluated ? ` · evaluated ${v.last_evaluated}` : ''}
        </div>
      {/if}
      {#if v.clinical_features?.length}
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px">
          {#each v.clinical_features as f}
            <span style="background:#f0ebe4;padding:1px 6px;border-radius:4px;margin-right:3px">{f}</span>
          {/each}
        </div>
      {/if}
      <div class="verdict-links">
        {#if v.allele_id}<a href="https://www.ncbi.nlm.nih.gov/clinvar/?term={v.allele_id}[alleleid]" target="_blank" rel="noopener">ClinVar</a>{/if}
        <a href="https://gnomad.broadinstitute.org/variant/{v.chrom}-{v.vcf_pos ?? (v.pos+1)}-{v.ref}-{v.alt}?dataset=gnomad_r4" target="_blank" rel="noopener">gnomAD</a>
        {#if v.rs_id}<a href="https://www.ncbi.nlm.nih.gov/snp/{v.rs_id}" target="_blank" rel="noopener">dbSNP</a>{/if}
        <a href="https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position={v.chrom}:{v.vcf_pos ?? (v.pos+1)}" target="_blank" rel="noopener">UCSC</a>
        {#if v.swissprot}
          <a href="https://www.uniprot.org/uniprotkb/{v.swissprot.split('.')[0]}" target="_blank" rel="noopener">UniProt</a>
        {:else}
          <a href="https://www.uniprot.org/uniprotkb?query=gene:{v.gene}+organism_id:9606" target="_blank" rel="noopener">UniProt</a>
        {/if}
        {#if v.gene_id}<a href="https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={v.gene_id}" target="_blank" rel="noopener">Ensembl</a>{/if}
        <a href="https://omim.org/search?search={v.gene}" target="_blank" rel="noopener">OMIM</a>
        <a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene={v.gene}" target="_blank" rel="noopener">GeneCards</a>
      </div>
    </div>
    <div class="verdict-score">
      <div class="score-label">Predicted<br>Pathogenicity</div>
      <div class="number" style="color:{scoreColor(v.score)}">{(v.score * 100).toFixed(0)}%</div>
    </div>
  </div>
</div>
