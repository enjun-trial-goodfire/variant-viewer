<script lang="ts">
  import { humanConsequence, truncate, LABEL_DISPLAY, labelClass, navigate } from '../../lib/helpers';
  import HelpCard from '../HelpCard.svelte';
  import type { Variant } from '../../lib/types';

  interface Props { variant: Variant; }
  let { variant: v }: Props = $props();

  const neighbors = $derived(v.neighbors ?? []);
  const isVUS = $derived(v.label === 'VUS' && neighbors.length > 0);
  const character = $derived(
    v.nP > v.nB * 2 ? 'predominantly pathogenic'
    : v.nB > v.nP * 2 ? 'predominantly benign'
    : 'mixed'
  );
  const characterColor = $derived(
    character.includes('pathogenic') ? 'var(--negative)'
    : character.includes('benign') ? '#27a'
    : 'var(--text-secondary)'
  );
</script>

<HelpCard helpText="<b>Nearest Neighbors.</b> The 10 most similar variants by cosine similarity in Evo2's probe activation embedding space. Click any row to view that variant.">
  <div class="section-title">Nearest Neighbors</div>

  {#if isVUS}
    <div class="vus-callout">
      This variant is <b>unclassified (VUS)</b>. Its nearest neighbors suggest
      <b style="color:{characterColor}">{character}</b> character.
    </div>
  {/if}

  <div class="neighbor-summary">
    <strong>{v.nP}</strong> pathogenic, <strong>{v.nB}</strong> benign, <strong>{v.nV}</strong> VUS among {neighbors.length} closest
  </div>

  <table>
    <thead>
      <tr><th>Variant</th><th>Gene</th><th>Consequence</th><th>Label</th><th>Score</th><th>Sim</th></tr>
    </thead>
    <tbody>
      {#each neighbors as nb}
        <tr onclick={() => navigate(`variant/${nb.id}`)}>
          <td class="mono" title={nb.id}>{truncate(nb.id)}</td>
          <td>{nb.gene}</td>
          <td>{humanConsequence(nb.consequence)}</td>
          <td><span class="label-badge {labelClass(nb.label)}">{LABEL_DISPLAY[nb.label] || nb.label}</span></td>
          <td class="mono">{(nb.score * 100).toFixed(0)}%</td>
          <td class="mono">{(nb.similarity * 100).toFixed(0)}%</td>
        </tr>
      {/each}
    </tbody>
  </table>
</HelpCard>
