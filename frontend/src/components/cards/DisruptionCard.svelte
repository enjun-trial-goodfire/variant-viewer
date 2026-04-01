<script lang="ts">
  import DisruptionRow from '../DisruptionRow.svelte';
  import Swatch from '../Swatch.svelte';
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

  const DELTA_THRESHOLD = 0.03;
  const DELTA_STEEPNESS = 150;
  function gatedScore(delta: number, z: number): number {
    const gate = 1 / (1 + Math.exp(-DELTA_STEEPNESS * (Math.abs(delta) - DELTA_THRESHOLD)));
    return gate * Math.abs(z);
  }

  const topItems = $derived.by(() => {
    return Object.entries(v.disruption ?? {})
      .map(([head, d]) => {
        const delta = d.var - d.ref;
        const score = gatedScore(delta, d.z);
        return { head, display: headsConfig[head]?.display ?? head, eval: evalStr(head), score, ...d };
      })
      .filter(d => d.score > 0.3)
      .sort((a, b) => b.score - a.score)
      .slice(0, 15);
  });

  // Use Ryo's curated group order from heads.json _meta
  const curatedGroups = $derived(g.heads?._meta?.curated_disruption_groups ?? []);

  const allGroups = $derived.by(() => {
    const groups: Record<string, typeof topItems> = {};
    for (const [head, d] of Object.entries(v.disruption ?? {})) {
      const info = headsConfig[head];
      if (!info || info.quality === 'removed') continue;
      const group = info.group ?? 'Other';
      // Only include groups in the curated list
      if (curatedGroups.length && !curatedGroups.includes(group)) continue;
      (groups[group] ??= []).push({ head, display: info.display ?? head, eval: evalStr(head), score: 0, ...d });
    }
    for (const items of Object.values(groups)) items.sort((a, b) => b.z - a.z);
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
      <span class="section-title" style="margin:0">Top Disruptions</span>
      <button class="help-btn" onclick={() => showHelp = !showHelp}>?</button>
    </div>
    <!-- title-cell spans cols 1-2, so next child is col 3 (ref/var bars) -->
    <div class="legend-cell">
      <span class="ll">benign</span>
      <Swatch color="#27a" label="Benign" /><Swatch color="#6ac" label="Leaning benign" /><Swatch color="#bbb" label="Neutral" /><Swatch color="#d88" label="Leaning pathogenic" /><Swatch color="#c55" label="Pathogenic" />
      <span class="ll">pathogenic</span>
    </div>
    <div class="legend-cell">
      <span class="ll">decrease</span>
      <Swatch color="var(--decrease)" label="Feature decreased" /><Swatch color="var(--increase)" label="Feature increased" />
      <span class="ll">increase</span>
    </div>
    <div></div>
  </div>

  <div class="help-panel" class:open={showHelp}>
    <div class="help-panel-inner">
      Each row shows a biological feature. Left bars: ref (faded) and var (solid) probe scores. Right bar: &Delta; (var &minus; ref) with z-score.
    </div>
  </div>

  {#each topItems as item}
    <DisruptionRow name={item.display} ref={item.ref} var={item.var} z={item.z} ref_lr={item.ref_lr} var_lr={item.var_lr} head={item.head} evalStr={item.eval} description={headsConfig[item.head]?.description} distributions={g.distributions} />
  {/each}

  {#if showAll}
    <div class="section-title" style="margin-top:16px">All Disruptions</div>
    {#each allGroups as [group, items]}
      <div class="profile-group">
        <div class="profile-group-title">{group}</div>
        {#each items as item}
          <DisruptionRow name={item.display} ref={item.ref} var={item.var} z={item.z} ref_lr={item.ref_lr} var_lr={item.var_lr} head={item.head} evalStr={item.eval} description={headsConfig[item.head]?.description} distributions={g.distributions} />
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
    --disruption-grid: minmax(80px, 140px) 55px 1fr 1fr 90px;
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
    grid-column: 1 / 3;
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
