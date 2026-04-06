<script lang="ts">
  import { buildUcscHg38ContextUrl } from '../../lib/ucscHg38ContextUrl';
  import type { Variant } from '../../lib/types';

  interface Props {
    variant: Variant;
  }
  let { variant: v }: Props = $props();

  const PAD = 3000;

  const start = $derived(Math.max(1, v.vcf_pos - PAD));
  const end = $derived(v.vcf_pos + PAD);
  const rangeLabel = $derived(`${v.chrom}:${start}-${end}`);

  // UCSC: programmatic / embedded hgTracks loads need apiKey=... to bypass CAPTCHA
  // (https://genome.ucsc.edu/FAQ/FAQdownloads.html#download29). Keys are per-user;
  // do not commit. Optional VITE_UCSC_API_KEY for private builds only — it ships in the JS bundle.
  const ucscApiKey = import.meta.env.VITE_UCSC_API_KEY?.trim();

  const ucscUrl = $derived.by(() =>
    buildUcscHg38ContextUrl(v.chrom, start, end, ucscApiKey || undefined),
  );
</script>

<div class="card ucsc-context-card">
  <div class="section-title" style="margin-bottom:8px">Genomic context (±3 kb, hg38)</div>
  <div class="ucsc-range">{rangeLabel}</div>
  <div class="verdict-links" style="margin-top:8px;margin-bottom:12px">
    <a href={ucscUrl} target="_blank" rel="noopener">Open in UCSC</a>
  </div>
  {#if ucscApiKey}
    <iframe
      class="ucsc-frame"
      src={ucscUrl}
      title={`UCSC Genome Browser — ${rangeLabel}`}
      width="100%"
      height="400"
      loading="lazy"
      sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
    ></iframe>
  {:else}
    <p class="ucsc-embed-note">
      Embedded browser is off: UCSC requires an
      <a href="https://genome.ucsc.edu/FAQ/FAQdownloads.html#download29" target="_blank" rel="noopener">API key</a>
      on programmatic requests (CAPTCHA). Use <b>Open in UCSC</b>, or set <code>VITE_UCSC_API_KEY</code>
      at build time for a private deployment (key is visible in the client bundle).
    </p>
  {/if}
</div>

<style>
  .ucsc-range {
    font-family: 'SF Mono', ui-monospace, monospace;
    font-size: 12px;
    color: var(--text-secondary);
  }
  .ucsc-frame {
    display: block;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-track);
  }
  .ucsc-embed-note {
    font-size: 12px;
    line-height: 1.5;
    color: var(--text-muted);
    margin: 0;
    padding: 12px;
    background: var(--bg-track);
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .ucsc-embed-note a {
    color: var(--accent);
  }
  .ucsc-embed-note code {
    font-family: ui-monospace, monospace;
    font-size: 11px;
  }
</style>
