<script lang="ts">
  import DisruptionRow from '../DisruptionRow.svelte';
  import PathogenicityLegend from '../PathogenicityLegend.svelte';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  let showAll = $state(false);
  let showHelp = $state(false);

  const headsConfig = $derived(g.heads?.heads ?? {});

  function evalStr(head: string): string | undefined {
    const e = headsConfig[head]?.eval;
    return e ? `${e.metric}=${e.value}` : undefined;
  }

  // Use Ryo's curated group order from heads.json _meta
  const curatedGroups = $derived(g.heads?._meta?.curated_disruption_groups ?? []);

  // Rank by |delta| / σ — combines magnitude with statistical significance
  const topItems = $derived.by(() => {
    return Object.entries(v.disruption ?? {})
      .map(([head, d]) => {
        const delta = d.var - d.ref;
        const std = headsConfig[head]?.std ?? 1;
        const score = std > 0 ? Math.abs(delta) / std : 0;
        return { head, display: headsConfig[head]?.display ?? head, eval: evalStr(head), score, delta, ...d };
      })
      .filter(d => {
        if (d.score < 1) return false;
        const info = headsConfig[d.head];
        if (!info || info.quality === 'removed') return false;
        const group = info.group ?? 'Other';
        return !curatedGroups.length || curatedGroups.includes(group);
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 15);
  });

  const allGroups = $derived.by(() => {
    const groups: Record<string, typeof topItems> = {};
    for (const [head, d] of Object.entries(v.disruption ?? {})) {
      const info = headsConfig[head];
      if (!info || info.quality === 'removed') continue;
      const group = info.group ?? 'Other';
      // Only include groups in the curated list
      if (curatedGroups.length && !curatedGroups.includes(group)) continue;
      const delta = d.var - d.ref;
      const std = info.std ?? 1;
      const score = std > 0 ? Math.abs(delta) / std : 0;
      (groups[group] ??= []).push({ head, display: info.display ?? head, eval: evalStr(head), score, delta, ...d });
    }
    for (const items of Object.values(groups)) items.sort((a, b) => b.score - a.score);
    // Return in curated order
    if (curatedGroups.length) {
      return curatedGroups
        .filter((g: string) => groups[g]?.length)
        .map((g: string) => [g, groups[g]] as [string, typeof topItems]);
    }
    return Object.entries(groups).filter(([, items]) => items.length > 0);
  });
</script>

<div class="card disruption-grid">
  <!-- Header row: same grid as data rows, legends in bar columns -->
  <div class="header-row">
    <div class="title-cell">
      <span class="section-title" style="margin:0">Significant Disruptions</span>
      <button class="help-btn" onclick={() => showHelp = !showHelp}>?</button>
    </div>
    <div class="legend-cell">
      <span class="ll">ref</span><span class="swatch-inline" style="background:#ccc"></span><span class="swatch-inline" style="background:#999"></span><span class="ll">var</span>
    </div>
    <div class="legend-cell"><PathogenicityLegend /></div>
    <div></div>
  </div>

  <div class="help-panel" class:open={showHelp}>
    <div class="help-panel-inner">
      Each row shows a biological feature at the most disrupted position. Left: ref and var probe predictions (0&ndash;1). Right: &Delta; (var &minus; ref), colored by pathogenicity likelihood. Click a row for heatmap and locality.
    </div>
  </div>

  {#if topItems.length === 0}
    <div style="padding:12px 0;color:var(--text-muted);font-size:13px">No significant disruption</div>
  {:else}
    {#each topItems as item}
      <DisruptionRow name={item.display} ref={item.ref} var={item.var} z={item.z} head={item.head} evalStr={item.eval} description={headsConfig[item.head]?.description} distributions={g.distributions} dist={item.dist} spread={item.spread} />
    {/each}
  {/if}

  {#if showAll}
    <hr style="border:none;border-top:1px solid var(--border);margin:16px 0 12px" />
    <div class="section-title">All Disruptions</div>
    {#each allGroups as [group, items]}
      <div class="profile-group">
        <div class="profile-group-title">{group}</div>
        {#each items as item}
          <DisruptionRow name={item.display} ref={item.ref} var={item.var} z={item.z} head={item.head} evalStr={item.eval} description={headsConfig[item.head]?.description} distributions={g.distributions} dist={item.dist} spread={item.spread} />
        {/each}
      </div>
    {/each}
  {/if}

  <button class="show-more" onclick={() => showAll = !showAll}>
    {showAll ? 'Hide all disruptions' : 'Show all disruptions'}
  </button>
</div>

<style>
  .disruption-grid {
    --disruption-grid: 1fr 1fr 1fr 90px;
  }

  .header-row {
    display: grid;
    grid-template-columns: var(--disruption-grid);
    gap: 6px;
    align-items: center;
    margin-bottom: 6px;
  }
  .title-cell {
    display: flex;
    align-items: center;
    gap: 6px;
    grid-column: 1;
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
  .swatch-inline { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; }

</style>
