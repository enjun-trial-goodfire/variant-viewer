<script lang="ts">
  import { tick } from 'svelte';
  import {
    ANNOTATION_COLORS,
    LOCUS_PAD_BP,
    buildLocusTrackViewModel,
    genomicBand,
    interactionTermColor,
    locusRegionFromVariant,
    type NearbyVariantPoint,
  } from '../../lib/locusTrackModel';
  import { fetchNearbyLocusVariants } from '../../lib/api';
  import type { Variant } from '../../lib/types';

  interface Props {
    variant: Variant;
  }
  let { variant: v }: Props = $props();

  /** Preset labels: each maps to px/bp via REF_VIEWPORT_PX / target (~bases in a reference-width view). */
  const ZOOM_TARGETS = [21, 51, 101, 201] as const;
  const DEFAULT_ZOOM_IDX = 1;
  const REF_VIEWPORT_PX = 640;

  const region = $derived(locusRegionFromVariant(v));
  const regionPath = $derived(`${region.ensemblChrom}:${region.locusStart}-${region.locusEnd}`);
  const restUrl = $derived(
    `https://rest.ensembl.org/sequence/region/human/${regionPath}:1?content-type=text/plain`,
  );
  const ensemblViewUrl = $derived(
    `https://www.ensembl.org/Homo_sapiens/Location/View?r=${encodeURIComponent(regionPath)}`,
  );

  /** Index into ZOOM_TARGETS — zoom = visual density (px per bp), not genomic slice. */
  let zoomIdx = $state(DEFAULT_ZOOM_IDX);

  $effect(() => {
    void v.variant_id;
    zoomIdx = DEFAULT_ZOOM_IDX;
  });

  let sequence = $state('');
  let loading = $state(false);
  let fetchError = $state('');

  /** Scrollport for the wide track (browser clips; inner SVG is full locus width). */
  let scrollEl: HTMLDivElement | undefined = $state();

  let nearbyVariants = $state<NearbyVariantPoint[]>([]);
  /** True after the nearby-locus API call settles (success, empty, or failure). */
  let nearbyReady = $state(false);

  $effect(() => {
    const vid = v.variant_id;
    const chrom = v.chrom;
    const pos = v.vcf_pos;
    const ac = new AbortController();
    nearbyReady = false;
    nearbyVariants = [];
    fetchNearbyLocusVariants(chrom, pos, vid, { pad: LOCUS_PAD_BP, signal: ac.signal })
      .then((rows) => {
        if (ac.signal.aborted) return;
        nearbyVariants = rows;
        nearbyReady = true;
      })
      .catch(() => {
        if (ac.signal.aborted) return;
        nearbyVariants = [];
        nearbyReady = true;
      });
    return () => ac.abort();
  });

  const model = $derived(buildLocusTrackViewModel(v, sequence, nearbyVariants));
  const refLen = $derived(Math.max(v.ref?.length ?? 1, 1));

  const locusBases = $derived(model.sequence.replace(/\s+/g, '').toUpperCase().split(''));
  const nLocus = $derived(region.locusEnd - region.locusStart + 1);
  const locusLenOk = $derived(locusBases.length > 0 && locusBases.length === nLocus);

  const pxPerBase = $derived(
    Math.max(2, Math.min(36, Math.round(REF_VIEWPORT_PX / ZOOM_TARGETS[zoomIdx]))),
  );

  const LABEL_W = 86;
  const INNER_W = $derived(nLocus * pxPerBase);
  const SVG_W = $derived(LABEL_W + INNER_W);
  const seqFontPx = $derived(Math.min(14, Math.max(5, pxPerBase * 0.88)));

  const H_ANN = 22;
  const H_NEAR = 26;
  const H_SEQ = 40;
  const H_TOTAL = H_ANN + H_NEAR + H_SEQ + 8;

  const band = $derived(genomicBand(region.locusStart, region.locusEnd, INNER_W));
  const focalX = $derived(LABEL_W + band.center(model.focal.vcfPos));

  const focalIndex0 = $derived(v.vcf_pos - region.locusStart);
  const indexOk = $derived(
    locusLenOk && focalIndex0 >= 0 && focalIndex0 + refLen <= locusBases.length,
  );

  function centerScrollOnFocal() {
    const el = scrollEl;
    if (!el || !locusLenOk) return;
    const innerW = nLocus * pxPerBase;
    const b = genomicBand(region.locusStart, region.locusEnd, innerW);
    const x = LABEL_W + b.center(v.vcf_pos);
    const half = el.clientWidth / 2;
    const maxSl = Math.max(0, el.scrollWidth - el.clientWidth);
    el.scrollLeft = Math.max(0, Math.min(x - half, maxSl));
  }

  async function recenter() {
    await tick();
    requestAnimationFrame(() => centerScrollOnFocal());
  }

  $effect(() => {
    void v.variant_id;
    void sequence;
    void pxPerBase;
    void scrollEl;
    if (!scrollEl || loading || fetchError || !locusLenOk) return;
    tick().then(() => requestAnimationFrame(() => centerScrollOnFocal()));
  });

  $effect(() => {
    const url = restUrl;
    const ac = new AbortController();
    loading = true;
    fetchError = '';
    sequence = '';

    fetch(url, {
      signal: ac.signal,
      headers: { Accept: 'text/plain' },
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.text();
      })
      .then((t) => {
        if (ac.signal.aborted) return;
        sequence = t;
      })
      .catch((e: Error) => {
        if (ac.signal.aborted || e.name === 'AbortError') return;
        fetchError = e.message || 'Failed to load sequence';
      })
      .finally(() => {
        if (!ac.signal.aborted) loading = false;
      });

    return () => ac.abort();
  });
</script>

<div class="card seq-strip-card">
  <div class="section-title" style="margin-bottom:6px">Local locus track</div>
  <p class="seq-sub">
    Full ±{LOCUS_PAD_BP.toLocaleString()} bp region ({nLocus.toLocaleString()} bp) — scroll horizontally;
    zoom changes width per base. Annotation lane is mocked; nearby dots are from the served variant subset.
  </p>
  <div class="seq-meta">
    <span class="seq-range">{model.ensemblChrom}:{region.locusStart.toLocaleString()}–{region.locusEnd.toLocaleString()}</span>
    <span class="verdict-links seq-links">
      <a href={ensemblViewUrl} target="_blank" rel="noopener">Ensembl</a>
      <a href={restUrl} target="_blank" rel="noopener">REST</a>
    </span>
  </div>

  <div class="track-controls">
    <span class="ctrl-label">Zoom</span>
    {#each ZOOM_TARGETS as z, zi (z)}
      <button
        type="button"
        class="zoom-btn"
        class:active={zoomIdx === zi}
        onclick={() => {
          zoomIdx = zi;
        }}>~{z} bp</button>
    {/each}
    <button type="button" class="recenter-btn" onclick={recenter}>Recenter</button>
  </div>

  {#if loading}
    <div class="seq-status">Loading sequence…</div>
  {:else if fetchError}
    <div class="seq-status seq-error">{fetchError}</div>
    <p class="seq-fallback">Open region in Ensembl:</p>
    <div class="verdict-links">
      <a href={ensemblViewUrl} target="_blank" rel="noopener">{regionPath}</a>
    </div>
  {/if}

  <div class="track-scroll" bind:this={scrollEl}>
    <svg
      class="locus-svg"
      width={SVG_W}
      height={H_TOTAL}
      viewBox="0 0 {SVG_W} {H_TOTAL}"
      aria-label="Local locus: annotation, nearby variants, reference sequence"
    >
      <rect x={LABEL_W} y="0" width={INNER_W} height={H_ANN - 2} fill="var(--bg-track)" rx="3" />
      <rect x={LABEL_W} y={H_ANN} width={INNER_W} height={H_NEAR - 2} fill="color-mix(in srgb, var(--bg-track) 85%, #fff)" rx="3" />
      <rect x={LABEL_W} y={H_ANN + H_NEAR} width={INNER_W} height={H_SEQ} fill="var(--bg-track)" rx="3" />

      <text x="4" y="14" class="lane-lbl">Annotation</text>
      <text x="4" y={H_ANN + 14} class="lane-lbl">Nearby</text>
      <text x="4" y={H_ANN + H_NEAR + 14} class="lane-lbl">Reference</text>

      <line x1={focalX} y1="0" x2={focalX} y2={H_TOTAL} class="focal-line" />

      {#each model.annotations as ann, i (i)}
        {@const x0 = LABEL_W + band.left(ann.start)}
        {@const w = band.intervalWidth(ann.start, ann.end)}
        <rect
          x={x0}
          y="4"
          width={Math.max(w, 1)}
          height="14"
          fill={ANNOTATION_COLORS[ann.category]}
          opacity="0.85"
          rx="2"
        >
          <title>{ann.label ?? ann.category} ({ann.start}–{ann.end})</title>
        </rect>
      {/each}

      {#if !nearbyReady}
        <text x={LABEL_W + 8} y={H_ANN + 14} class="lane-empty-svg">Loading nearby variants…</text>
      {:else if model.nearbyVariants.length === 0}
        <text x={LABEL_W + 8} y={H_ANN + 14} class="lane-empty-svg">
          No other variants in this dataset within ±{LOCUS_PAD_BP.toLocaleString()} bp.
        </text>
      {:else}
        {#each model.nearbyVariants as nb (nb.variantId)}
          {@const cx = LABEL_W + band.center(nb.genomicPos)}
          {@const cy = H_ANN + 12}
          <polygon
            points={`${cx},${cy - 7} ${cx - 6},${cy + 5} ${cx + 6},${cy + 5}`}
            fill={interactionTermColor(nb.interactionTerm)}
            stroke="#fff"
            stroke-width="1"
          >
            <title>
              {nb.variantId} @ {nb.genomicPos}{#if nb.labelDisplay} · {nb.labelDisplay}{/if} · pathogenicity (color)={nb.interactionTerm.toFixed(2)}
            </title>
          </polygon>
        {/each}
      {/if}

      <polygon
        points={`${focalX},${H_ANN + 3} ${focalX - 8},${H_ANN + 17} ${focalX + 8},${H_ANN + 17}`}
        class="focal-marker"
      >
        <title>Focal variant {v.variant_id}</title>
      </polygon>

      {#if locusBases.length > 0}
        {#if !indexOk}
          <text x={LABEL_W + 4} y={H_ANN + H_NEAR + 26} class="seq-warn-svg">Sequence length / focal index mismatch.</text>
        {:else}
          <text
            x={focalX}
            y={H_ANN + H_NEAR + 11}
            text-anchor="middle"
            class="focal-alleles"
          >REF {model.focal.ref || '—'} · ALT {model.focal.alt || '—'}</text>

          <rect
            x={LABEL_W + band.left(v.vcf_pos)}
            y={H_ANN + H_NEAR + 14}
            width={Math.max(band.intervalWidth(v.vcf_pos, v.vcf_pos + refLen - 1), 2)}
            height={22}
            class="focal-cell-bg"
            rx="3"
          />

          {#each locusBases as base, i (region.locusStart + i)}
            {@const pos = region.locusStart + i}
            {@const cx = LABEL_W + band.center(pos)}
            {@const inFocal = i >= focalIndex0 && i < focalIndex0 + refLen}
            <text
              x={cx}
              y={H_ANN + H_NEAR + 30}
              text-anchor="middle"
              class="seq-char"
              class:seq-char-focal={inFocal}
              font-size={inFocal ? Math.min(15, seqFontPx + 3) : seqFontPx}
            >{base}</text>
          {/each}
        {/if}
      {:else if !loading && !fetchError}
        <text x={LABEL_W + 8} y={H_ANN + H_NEAR + 26} class="seq-muted-svg">No sequence</text>
      {/if}
    </svg>
  </div>

  <div class="legend">
    <span class="legend-mock">
      V1: annotation lane is mocked. Nearby: same chrom, ±{LOCUS_PAD_BP.toLocaleString()} bp in the ClinVar subset (<code>/api/variants/nearby-locus</code>); marker color uses pathogenicity as a stand-in for interactionTerm.
    </span>
    <div class="legend-ann">
      {#each Object.entries(ANNOTATION_COLORS) as [cat, color] (cat)}
        <span class="leg-item"><i style:background={color}></i>{cat}</span>
      {/each}
    </div>
    <div class="legend-scale">
      <span>pathogenicity → color (placeholder)</span>
      <span class="grad"></span>
      <span>low</span><span>high</span>
    </div>
  </div>

  <p class="seq-note">
    One Ensembl fetch for the full buffer; all lanes use <code>genomicBand(locusStart, locusEnd, nBp × pxPerBase)</code>. Strand <code>:1</code>.
  </p>
</div>

<style>
  .seq-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0 0 10px;
    line-height: 1.4;
  }
  .seq-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }
  .seq-range {
    font-family: ui-monospace, monospace;
    font-size: 11px;
    color: var(--text-secondary);
  }
  .seq-links {
    margin: 0;
  }
  .track-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 8px;
    margin-bottom: 10px;
    font-size: 12px;
  }
  .ctrl-label {
    color: var(--text-muted);
    font-weight: 600;
    margin-right: 4px;
  }
  .zoom-btn,
  .recenter-btn {
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
  }
  .zoom-btn:hover,
  .recenter-btn:hover {
    background: var(--bg-hover);
    color: var(--accent);
  }
  .zoom-btn.active {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 14%, transparent);
    color: var(--text);
    font-weight: 600;
  }
  .recenter-btn {
    margin-left: 8px;
  }
  .seq-status {
    font-size: 13px;
    color: var(--text-muted);
    padding: 8px 0;
  }
  .seq-error {
    color: var(--pathogenic);
  }
  .seq-fallback {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 0 0 8px;
  }
  .track-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-card);
    margin-bottom: 8px;
    max-width: 100%;
  }
  .locus-svg {
    display: block;
  }
  .lane-lbl {
    font-size: 10px;
    font-weight: 600;
    fill: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .lane-empty-svg {
    font-size: 9px;
    font-weight: 500;
    fill: var(--text-muted);
  }
  .focal-line {
    stroke: var(--accent);
    stroke-width: 3;
    opacity: 0.9;
    pointer-events: none;
  }
  .focal-marker {
    fill: var(--accent);
    stroke: #fff;
    stroke-width: 1.5;
    filter: drop-shadow(0 0 4px color-mix(in srgb, var(--accent) 70%, transparent));
  }
  .focal-cell-bg {
    fill: color-mix(in srgb, var(--accent) 42%, transparent);
    stroke: color-mix(in srgb, var(--accent) 55%, transparent);
    stroke-width: 1;
  }
  .seq-char {
    font-family: ui-monospace, monospace;
    font-size: 13px;
    font-weight: 600;
    fill: var(--text);
  }
  .seq-char-focal {
    fill: var(--pathogenic);
    font-weight: 800;
  }
  .focal-alleles {
    font-size: 10px;
    font-weight: 700;
    fill: var(--text);
  }
  .seq-warn-svg {
    font-size: 10px;
    fill: #974;
  }
  .seq-muted-svg {
    font-size: 11px;
    fill: var(--text-muted);
  }
  .legend {
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.6;
    margin-bottom: 6px;
  }
  .legend-mock {
    display: block;
    margin-bottom: 6px;
  }
  .legend-ann {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 14px;
    margin-bottom: 6px;
  }
  .leg-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    text-transform: capitalize;
  }
  .leg-item i {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
  .legend-scale {
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 4px 8px;
    align-items: center;
    max-width: 320px;
  }
  .legend-scale .grad {
    grid-column: 2;
    height: 8px;
    border-radius: 2px;
    background: linear-gradient(90deg, hsl(220, 42%, 38%), hsl(20, 70%, 56%));
  }
  .seq-note {
    font-size: 10px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.4;
  }
</style>
