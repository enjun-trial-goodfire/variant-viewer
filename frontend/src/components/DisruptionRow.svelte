<script lang="ts">
  import { lrColor } from '../lib/colors';
  import HeadHeatmap from './HeadHeatmap.svelte';
  import ScoreHistogram from './ScoreHistogram.svelte';

  interface Props {
    name: string;
    ref: number;
    var: number;
    z: number;
    head: string;
    evalStr?: string;
    description?: string;
    distributions?: Record<string, any>;
    dist?: number;
    spread?: number;
  }
  let { name, ref, var: va, z, head, evalStr, description, distributions, dist, spread }: Props = $props();

  const localityLabel = $derived.by(() => {
    if (dist == null) return null;
    const absDist = Math.abs(dist);
    const direction = dist > 0 ? 'downstream' : dist < 0 ? 'upstream' : 'at variant';
    const distStr = absDist === 0 ? 'at variant'
      : absDist < 100 ? `${absDist}bp ${direction}`
      : absDist < 1000 ? `${(absDist / 1000).toFixed(1)}kb ${direction}`
      : `${(absDist / 1000).toFixed(0)}kb ${direction}`;
    const spreadStr = spread != null
      ? spread <= 5 ? 'point effect' : spread <= 50 ? 'local' : 'domain-wide'
      : null;
    return { distStr, spreadStr, spread: spread ?? 0 };
  });

  let expanded = $state(false);
  let showHelp = $state(false);

  const delta = $derived(va - ref);

  // Delta bar: width proportional to |delta|, range -1 to 1 → 0% to 50%
  const deltaBarW = $derived(Math.abs(delta) * 50);

  // Color: pathogenicity fraction at this variant's ref/var position on the heatmap.
  // The heatmap bins ref (rows) × var (cols) and stores % pathogenic per cell.
  const deltaColor = $derived.by(() => {
    const hm = distributions?.[head];
    if (!hm?.data || !hm.bins) return lrColor(0.5);
    const bins = hm.bins;
    const rb = Math.min(bins - 1, Math.max(0, Math.floor(ref * bins)));
    const vb = Math.min(bins - 1, Math.max(0, Math.floor(va * bins)));
    const cell = hm.data.find((c: number[]) => c[0] === rb && c[1] === vb);
    // cell = [refBin, varBin, pathPct, count]
    const fracPath = cell ? cell[2] / 100 : 0.5;
    return lrColor(fracPath);
  });

  const heatmapData = $derived(distributions?.[head]?.data ? distributions[head] : null);
  const refDist = $derived(distributions?.[head]?.ref ?? null);
  const deltaDist = $derived(distributions?.[head]?.delta ?? null);
  const hasHelp = $derived(!!(evalStr || description));

  function toggle() { expanded = !expanded; }
  function toggleHelp(e: MouseEvent) { e.stopPropagation(); showHelp = !showHelp; }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div class="row" onclick={toggle}>
  <div class="col-label">
    {name}
    {#if hasHelp}
      <button class="help-btn" onclick={toggleHelp}>?</button>
    {/if}
  </div>
  <div class="col-refvar">
    <div class="bar-ref" style="width:{ref > 0 ? Math.max(2, ref * 100) : 0}%"></div>
    <div class="bar-var" style="width:{va > 0 ? Math.max(2, va * 100) : 0}%"></div>
  </div>
  <div class="col-delta">
    <div class="delta-center"></div>
    <div class="delta-fill" style="{delta >= 0 ? `left:50%;width:${deltaBarW}%` : `right:50%;width:${deltaBarW}%`};background:{deltaColor}"></div>
  </div>
  <div class="col-values">
    <span style="color:{deltaColor};font-weight:600">{delta > 0 ? '+' : ''}{delta.toFixed(2)}</span>
    <span class="sigma" style="color:{deltaColor}">{z > 0 ? `${z.toFixed(1)}\u03C3` : ''}</span>
  </div>
</div>

{#if showHelp && hasHelp}
  <div class="help-panel open">
    <div class="help-panel-inner">
      {#if evalStr}<span style="font-weight:600">{evalStr}</span>{/if}
      {#if evalStr && description} · {/if}
      {#if description}{description}{/if}
    </div>
  </div>
{/if}

{#if expanded}
  {#if localityLabel}
    <div class="locality-bar">
      <span class="locality-dist">{localityLabel.distStr}</span>
      {#if localityLabel.spreadStr}
        <span class="locality-badge" class:point={localityLabel.spread <= 5} class:local={localityLabel.spread > 5 && localityLabel.spread <= 50} class:domain={localityLabel.spread > 50}>
          {localityLabel.spreadStr} · {localityLabel.spread} positions
        </span>
      {/if}
    </div>
  {/if}
  <div class="expand-row">
    <div class="expand-cell">
      {#if heatmapData}
        <HeadHeatmap heatmap={heatmapData} variantRef={ref} variantVar={va} headName={name} />
      {/if}
    </div>
    <div class="expand-cell">
      {#if refDist}
        <ScoreHistogram histogram={refDist} markers={[{value: ref, color: '#333', label: 'ref'}, {value: va, color: '#c55', label: 'var'}]} title="Score Distribution" />
      {/if}
    </div>
    <div class="expand-cell">
      {#if deltaDist}
        <ScoreHistogram histogram={deltaDist} markers={[{value: delta, color: '#333', label: 'Δ'}]} title="Delta Distribution" />
      {/if}
    </div>
    <div></div>
  </div>
{/if}

<style>
  .row {
    display: grid;
    grid-template-columns: var(--disruption-grid);
    gap: 6px;
    align-items: center;
    padding: 3px 0;
    font-size: 12px;
    cursor: pointer;
    border-radius: 3px;
  }
  .row:hover { background: var(--bg-hover); }

  .col-label {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: var(--text-secondary);
    display: flex; align-items: center; gap: 4px;
  }
  .col-refvar {
    display: flex; flex-direction: column; gap: 1px;
    background: var(--bg-track); border-radius: 2px; padding: 1px;
  }
  .bar-ref { height: 7px; border-radius: 2px; background: #ccc; }
  .bar-var { height: 7px; border-radius: 2px; background: #999; }

  .col-delta {
    height: 16px; position: relative;
    background: var(--bg-track); border-radius: 3px; overflow: hidden;
  }
  .delta-center {
    position: absolute; left: 50%; top: 0; width: 1px; height: 100%;
    background: var(--border);
  }
  .delta-fill {
    position: absolute; top: 3px; height: 10px; border-radius: 2px; z-index: 1;
  }

  .col-values {
    display: flex; gap: 4px; align-items: center; justify-content: flex-end;
    font-family: "SF Mono", monospace; font-size: 11px;
  }
  .sigma { font-size: 10px; min-width: 36px; text-align: right; }

  .expand-row {
    display: grid;
    grid-template-columns: var(--disruption-grid);
    gap: 6px;
    padding: 8px 0;
  }
  .expand-cell {
    min-width: 0;
    padding: 4px;
  }

  .locality-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 10px;
    margin: 4px 0;
    background: var(--bg-hover);
    border-radius: 4px;
    font-size: 12px;
  }
  .locality-dist {
    font-family: "SF Mono", monospace;
    font-weight: 600;
    color: var(--text);
  }
  .locality-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
  }
  .locality-badge.point { background: #d4edda; color: #1b5e20; }
  .locality-badge.local { background: #d6e4f0; color: #1a3a5c; }
  .locality-badge.domain { background: #f0d4d4; color: #5c1a1a; }
</style>
