<script lang="ts">
  import { tierColor } from '../../lib/colors';
  import ScoreRow from '../ScoreRow.svelte';
  import PathogenicityLegend from '../PathogenicityLegend.svelte';
  import HeadHistogram from '../HeadHistogram.svelte';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  let showHelp = $state(false);

  const headsConfig = $derived(g.heads?.heads ?? {});

  const predRows = $derived.by(() => {
    const rows: Array<{key: string; name: string; rawVal: number; displayVal: number; color: string; damaging: boolean; isProbe: boolean; isEvo2: boolean}> = [];

    rows.push({
      key: '_pathogenic',
      name: 'Evo2 Pathogenicity',
      rawVal: v.pathogenicity,
      displayVal: v.pathogenicity,
      color: tierColor(v.pathogenicity, false),
      damaging: v.pathogenicity > 0.5,
      isProbe: false,
      isEvo2: true,
    });

    for (const [key, info] of Object.entries(headsConfig)) {
      if (!info.predictor) continue;
      const p = info.predictor;
      const dbVal = (v.gt ?? {})[key];
      const probeEffect = (v.effect ?? {})[key];
      const probeVal = probeEffect?.value;
      if (dbVal == null && probeVal == null) continue;
      const primary = dbVal ?? probeVal;
      const invert = p.invert ?? false;
      const phredMax = p.phredMax;
      // CADD uses PHRED scoring: convert PHRED/max to percentile
      const displayVal = phredMax
        ? 1 - Math.pow(10, -primary * phredMax / 10)
        : invert ? 1 - primary : primary;
      const damaging = invert ? primary < p.threshold : primary > p.threshold;
      rows.push({
        key,
        name: p.display ?? info.display ?? key,
        rawVal: primary,
        displayVal,
        color: tierColor(phredMax ? displayVal : primary, invert && !phredMax),
        damaging,
        isProbe: dbVal == null,
        isEvo2: false,
      });
    }

    return rows.sort((a, b) => {
      if (a.isEvo2) return -1;
      if (b.isEvo2) return 1;
      return (headsConfig[a.key]?.predictor?.order ?? 99) - (headsConfig[b.key]?.predictor?.order ?? 99);
    });
  });

  const nDam = $derived(predRows.filter(r => r.damaging).length);
  const distributions = $derived(g.distributions ?? {});
  let expandedKey = $state<string | null>(null);

  function getHist(key: string, invert: boolean) {
    const distKey = key === '_pathogenic' ? 'pathogenic' : key;
    const d = distributions[distKey];
    if (!d) return null;
    const hist = d.benign ? d : d.gt_hist ?? null;
    if (!hist) return null;
    if (!invert) return hist;
    // Flip the histogram for inverted predictors (e.g. SIFT → 1-SIFT)
    return { ...hist, benign: [...hist.benign].reverse(), pathogenic: [...hist.pathogenic].reverse() };
  }

  function toggle(key: string, invert: boolean) {
    expandedKey = expandedKey === key ? null : (getHist(key, invert) ? key : null);
  }
</script>

{#if predRows.length > 1}
  <div class="card">
    <div class="header-row">
      <div class="title-cell">
        <span class="section-title" style="margin:0">Computational Predictors</span>
        <button class="help-btn" onclick={() => showHelp = !showHelp}>?</button>
      </div>
      <div class="legend-cell">
        <PathogenicityLegend />
      </div>
      <div></div>
    </div>

    <div class="help-panel" class:open={showHelp}>
      <div class="help-panel-inner">
        Evo2 Pathogenicity is our probe's prediction. Other scores are from established tools. Values with * are Evo2's internal predictions. Click any row to see the distribution.
      </div>
    </div>

    {#each predRows as r}
      <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
      <div onclick={() => toggle(r.key, headsConfig[r.key]?.predictor?.invert ?? false)} style="cursor:pointer">
        <ScoreRow
          label="{r.name}{r.isEvo2 ? ' ✦' : ''}"
          value={r.displayVal}
          color={r.color}
          suffix={r.isProbe ? '<sup style="font-size:8px;color:var(--text-muted)">*</sup>' : ''}
        />
      </div>

      {#if expandedKey === r.key}
        {@const hist = getHist(r.key, headsConfig[r.key]?.predictor?.invert ?? false)}
        {#if hist}
          <div class="hist-row">
            <div></div>
            <div><HeadHistogram histogram={hist} variantValue={headsConfig[r.key]?.predictor?.invert ? r.displayVal : r.rawVal} headName={r.name} /></div>
            <div></div>
          </div>
        {/if}
      {/if}
    {/each}

    <div style="font-size:11px;color:var(--text-muted);margin-top:8px;padding-top:6px;border-top:1px solid var(--border)">
      Consensus: <b>{nDam}/{predRows.length}</b> {nDam > predRows.length / 2 ? 'deleterious (supports PP3)' : 'benign (supports BP4)'}
      {#if predRows.some(r => r.isProbe)}
        <br>* predicted by Evo2 (no database value available)
      {/if}
    </div>
  </div>
{/if}

<style>
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
  .hist-row {
    display: grid;
    grid-template-columns: minmax(100px, 160px) 1fr 45px;
    gap: 8px;
  }
</style>
