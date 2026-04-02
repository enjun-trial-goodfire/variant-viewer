<script lang="ts">
  import { lrColor } from '../lib/colors';
  import HeadHeatmap from './HeadHeatmap.svelte';
  import ScoreHistogram from './ScoreHistogram.svelte';

  interface Props {
    name: string;
    ref: number;
    var: number;
    z: number;
    ref_lr: number;
    var_lr: number;
    head: string;
    evalStr?: string;
    description?: string;
    distributions?: Record<string, any>;
  }
  let { name, ref, var: va, z, ref_lr, var_lr, head, evalStr, description, distributions }: Props = $props();

  let expanded = $state(false);
  let showHelp = $state(false);

  const delta = $derived(va - ref);
  const deltaBarW = $derived(Math.min(Math.abs(z) / 4 * 50, 50));
  const deltaFracPath = $derived.by(() => {
    const dd = distributions?.[head]?.delta;
    if (!dd) return 0.5;
    const { benign, pathogenic, bins, range: [lo, hi] } = dd;
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((delta - lo) / (hi - lo) * bins)));
    const b = (benign[idx] || 0) * dd._bTotal;
    const p = (pathogenic[idx] || 0) * dd._pTotal;
    return (b + p) > 0 ? p / (b + p) : 0.5;
  });
  const deltaBarColor = $derived(lrColor(deltaFracPath));
  const zLabel = $derived(z > 0 ? `${z.toFixed(1)}\u03C3` : '');
  const heatmapData = $derived(distributions?.[head]?.data ? distributions[head] : null);
  const refDist = $derived(distributions?.[head]?.ref ?? null);
  const deltaDist = $derived(distributions?.[head]?.delta ?? null);
  const hasHelp = $derived(!!(evalStr || description));

  function toggle() {
    expanded = !expanded;
  }

  function toggleHelp(e: MouseEvent) {
    e.stopPropagation();
    showHelp = !showHelp;
  }
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
    <div class="bar-ref" style="width:{ref > 0 ? Math.max(2, ref * 100) : 0}%;background:#bbb"></div>
    <div class="bar-var" style="width:{va > 0 ? Math.max(2, va * 100) : 0}%;background:#999"></div>
  </div>
  <div class="col-delta">
    <div class="delta-center"></div>
    <div class="delta-fill" style="{delta >= 0 ? `left:50%;width:${deltaBarW}%` : `right:50%;width:${deltaBarW}%`};background:{deltaBarColor}"></div>
  </div>
  <div class="col-values">
    <span style="color:{deltaBarColor};font-weight:600">{delta > 0 ? '+' : ''}{delta.toFixed(3)}</span>
    <span class="sigma" style="color:{deltaBarColor}">{zLabel}</span>
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
  <div class="expand-row">
    <div class="expand-cell">
      {#if heatmapData}
        <HeadHeatmap heatmap={heatmapData} variantRef={ref} variantVar={va} headName={name} />
      {/if}
    </div>
    <div class="expand-cell">
      {#if refDist}
        <ScoreHistogram histogram={refDist} markers={[{value: ref, color: '#333', dash: [2,2], label: 'ref'}, {value: va, color: '#c55', dash: [6,3], label: 'var'}]} title="Score Distribution (··· ref, --- var)" />
      {/if}
    </div>
    <div class="expand-cell">
      {#if deltaDist}
        <ScoreHistogram histogram={deltaDist} markers={[{value: delta, color: '#333', dash: [6,3], label: 'Δ'}]} title="Delta Distribution (var − ref)" />
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
  .bar-ref { height: 7px; border-radius: 2px; opacity: 0.5; }
  .bar-var { height: 7px; border-radius: 2px; }

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
</style>
