<script lang="ts">
  import { tierColor } from '../../lib/colors';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  let showHelp = $state(false);

  const predictors = [
    {name: 'PhyloP 100-way', key: 'phylop_100way', threshold: 0.5, invert: false, desc: 'Evolutionary conservation'},
    {name: 'PhastCons 100-way', key: 'phastcons_100way', threshold: 0.5, invert: false, desc: 'Conserved element probability'},
    {name: 'GERP', key: 'gerp_c', threshold: 0.5, invert: false, desc: 'Evolutionary rate profiling'},
    {name: 'CADD', key: 'cadd_c', threshold: 0.333, invert: false, desc: 'Combined annotation-dependent depletion'},
    {name: 'AlphaMissense', key: 'alphamissense_c', threshold: 0.564, invert: false, desc: 'Deep learning missense predictor'},
    {name: 'REVEL', key: 'revel_c', threshold: 0.5, invert: false, desc: 'Rare exome variant ensemble'},
    {name: '1 \u2212 SIFT', key: 'sift_c', threshold: 0.05, invert: true, desc: 'Sequence homology tolerance (inverted)'},
    {name: 'PolyPhen-2', key: 'polyphen_c', threshold: 0.85, invert: false, desc: 'Structure-based damage prediction'},
    {name: 'EVE', key: 'eve_c', threshold: 0.5, invert: false, desc: 'Evolutionary model of variant effect'},
    {name: 'SpliceAI', key: 'spliceai_max_c', threshold: 0.2, invert: false, desc: 'Splice disruption predictor'},
  ];

  const gt = $derived(v.gt || {});
  const effLookup = $derived(v.effect || {});

  const predRows = $derived(predictors.map(p => {
    const dbVal = gt[p.key];
    const probeVal = effLookup[p.key];
    if (dbVal == null && probeVal == null) return null;
    const primary = dbVal ?? probeVal;
    const damaging = p.invert ? primary < p.threshold : primary > p.threshold;
    return { ...p, db: dbVal, probe: probeVal, primary, damaging };
  }).filter(Boolean) as Array<{name: string; key: string; threshold: number; invert: boolean; desc: string; db: number | undefined; probe: number | undefined; primary: number; damaging: boolean}>);

  const nDam = $derived(predRows.filter(r => r.damaging).length);
  const consensus = $derived(nDam > predRows.length / 2 ? 'deleterious' : 'benign');
</script>

{#if predRows.length > 1}
  <div class="card">
    <button class="card-help-btn" onclick={() => showHelp = !showHelp}>?</button>
    {#if showHelp}
      <div class="card-help open">
        <div class="card-help-inner">
          <b>Computational Predictors.</b> Scores from established clinical variant assessment tools.
          Values with * are Evo2's internal predictions.
        </div>
      </div>
    {/if}

    <div class="section-title">Computational Predictors</div>

    {#each predRows as r}
      {@const displayVal = r.invert ? 1 - r.primary : r.primary}
      {@const color = tierColor(r.primary, r.invert)}
      {@const src = r.db != null ? '' : '*'}
      <div class="profile-row">
        <div class="profile-label">{r.name}</div>
        <div class="profile-bar-container">
          <div class="profile-bar" style="width:{Math.max(2, displayVal * 100)}%;background:{color}"></div>
        </div>
        <div class="profile-value" style="color:{color};font-weight:600;text-align:left">
          {displayVal.toFixed(2)}{#if src}<sup style="font-size:8px;color:var(--text-muted)">{src}</sup>{/if}
        </div>
      </div>
    {/each}

    <div style="font-size:11px;color:var(--text-muted);margin-top:8px;padding-top:6px;border-top:1px solid var(--border)">
      Consensus: <b>{nDam}/{predRows.length}</b> {consensus}
      {nDam > predRows.length / 2 ? '(supports PP3)' : '(supports BP4)'}
      {#if predRows.some(r => r.db == null)}
        <br>* predicted by Evo2 (no database value available)
      {/if}
    </div>
  </div>
{/if}
