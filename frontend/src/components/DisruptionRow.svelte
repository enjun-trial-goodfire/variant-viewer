<script lang="ts">
  import { barColor, deltaColor } from '../lib/colors';
  import HeadHeatmap from './HeadHeatmap.svelte';

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

  const delta = $derived(va - ref);
  const sign = $derived(delta < 0 ? -1 : 1);
  const dc = $derived(deltaColor(z, sign));
  const deltaBarW = $derived(Math.abs(delta) * 50);
  const refColor = $derived(barColor(ref_lr));
  const varColor = $derived(barColor(var_lr));
  const zLabel = $derived(z > 0 ? `${z.toFixed(1)}\u03C3` : '');
  const heatmapData = $derived(distributions?.[head]?.data ? distributions[head] : null);

  function toggle() {
    expanded = !expanded;
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div class="row" onclick={toggle}>
  <div class="col-label">{name}</div>
  <div class="col-eval">{evalStr ?? ''}</div>
  <div class="col-refvar">
    <div class="bar-ref" style="width:{ref > 0 ? Math.max(2, ref * 100) : 0}%;background:{refColor}"></div>
    <div class="bar-var" style="width:{va > 0 ? Math.max(2, va * 100) : 0}%;background:{varColor}"></div>
  </div>
  <div class="col-delta">
    <div class="delta-center"></div>
    <div class="delta-fill" style="{delta >= 0 ? `left:50%;width:${deltaBarW}%` : `right:50%;width:${deltaBarW}%`};background:{dc.bar}"></div>
  </div>
  <div class="col-values">
    <span style="color:{dc.text};font-weight:600">{delta > 0 ? '+' : ''}{delta.toFixed(3)}</span>
    <span class="sigma" style="color:{dc.text}">{zLabel}</span>
  </div>
</div>

{#if expanded}
  <div class="expand-row">
    <div class="col-label"></div>
    <div class="col-eval"></div>
    <div class="col-expand">
      {#if description}
        <div class="head-desc">{description}</div>
      {/if}
      {#if heatmapData}
        <HeadHeatmap heatmap={heatmapData} variantRef={ref} variantVar={va} headName={name} />
      {/if}
    </div>
  </div>
{/if}

<style>
  /* 6-column grid: label | eval | ref/var bars | delta bar | values | pad */
  /* This grid is shared with the legend in DisruptionCard via CSS custom property */
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
  }
  .col-eval {
    font-size: 9px; color: var(--text-muted); font-family: "SF Mono", monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
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
    position: absolute; top: 3px; height: 10px; border-radius: 2px;
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
    padding: 4px 0;
  }
  .col-expand {
    grid-column: 3 / 5;
  }
  .head-desc {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.5;
    padding: 6px 10px;
    margin-bottom: 6px;
    background: var(--bg-hover);
    border-left: 2px solid var(--accent);
    border-radius: 0 4px 4px 0;
  }
</style>
