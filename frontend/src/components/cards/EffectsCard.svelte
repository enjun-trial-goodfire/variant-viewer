<script lang="ts">
  import EffectRow from '../EffectRow.svelte';
  import Swatch from '../Swatch.svelte';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  let showAll = $state(false);
  let showHelp = $state(false);

  const headsConfig = $derived(g.heads?.heads ?? {});
  const TOP_N = 5;

  const items = $derived.by(() => {
    return Object.entries(v.effect ?? {})
      .map(([head, d]) => {
        const info = headsConfig[head];
        if (!info || info.quality === 'removed' || info.exclude_from_effect_expansion || info.predictor) return null;
        return { key: head, value: d.value, lr: d.lr, display: info.display ?? head };
      })
      .filter(Boolean)
      .sort((a, b) => Math.abs(b!.value) - Math.abs(a!.value)) as Array<{key: string; value: number; lr: number; display: string}>;
  });

  const visible = $derived(showAll ? items : items.slice(0, TOP_N));
</script>

{#if items.length}
  <div class="card">
    <div class="header-row">
      <div class="title-cell">
        <span class="section-title" style="margin:0">Variant Effects</span>
        <button class="help-btn" onclick={() => showHelp = !showHelp}>?</button>
      </div>
      <div class="legend-cell">
        <span class="ll">benign</span>
        <Swatch color="#27a" label="Benign" /><Swatch color="#6ac" label="Leaning benign" /><Swatch color="#bbb" label="Neutral" /><Swatch color="#d88" label="Leaning pathogenic" /><Swatch color="#c55" label="Pathogenic" />
        <span class="ll">pathogenic</span>
      </div>
      <div></div>
    </div>

    <div class="help-panel" class:open={showHelp}>
      <div class="help-panel-inner">
        Predicted variant-level properties. Color shows likelihood ratio (benign to pathogenic). Click any row to see the population distribution.
      </div>
    </div>

    {#each visible as item}
      <EffectRow head={item.key} display={item.display} value={item.value} lr={item.lr} description={headsConfig[item.key]?.description} distributions={g.distributions} />
    {/each}

    {#if items.length > TOP_N}
      <button class="show-more" onclick={() => showAll = !showAll}>
        {showAll ? 'Show less' : `Show all ${items.length} effects`}
      </button>
    {/if}
  </div>
{/if}

<style>
  /* Header grid matches ScoreRow / EffectRow: label | bar | value */
  .header-row {
    display: grid;
    grid-template-columns: minmax(100px, 160px) 1fr 45px;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
  }
  .title-cell {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    position: relative;
    z-index: 1;
  }
  .legend-cell {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
  }
  .ll { font-size: 9px; color: var(--text-muted); }
</style>
