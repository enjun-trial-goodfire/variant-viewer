<script lang="ts">
  import { truncate, labelClass, navigate } from '../../lib/helpers';
  import HelpCard from '../HelpCard.svelte';
  import type { Variant } from '../../lib/types';

  interface Props { variant: Variant; }
  let { variant: v }: Props = $props();

  const neighbors = $derived(v.neighbors ?? []);
  const nP = $derived(neighbors.filter(n => n.label.includes('pathogenic')).length);
  const nB = $derived(neighbors.filter(n => n.label.includes('benign')).length);
  const nV = $derived(neighbors.length - nP - nB);
  const isVUS = $derived(v.label === 'VUS' && neighbors.length > 0);
  const character = $derived(
    nP > nB * 2 ? 'predominantly pathogenic'
    : nB > nP * 2 ? 'predominantly benign'
    : 'mixed'
  );
  const characterColor = $derived(
    character.includes('pathogenic') ? 'var(--pathogenic)'
    : character.includes('benign') ? 'var(--benign)'
    : 'var(--text-secondary)'
  );
</script>

<HelpCard
  title="Nearest Neighbors"
  helpText="<b>Nearest Neighbors.</b> The 10 most similar variants by cosine similarity in Evo2's probe activation embedding space. Click any row to view that variant."
>

  {#if isVUS}
    <div class="vus-callout">
      This variant is <b>unclassified (VUS)</b>. Its nearest neighbors suggest
      <b style="color:{characterColor}">{character}</b> character.
    </div>
  {/if}

  <div class="neighbor-summary">
    <strong>{nP}</strong> pathogenic, <strong>{nB}</strong> benign, <strong>{nV}</strong> VUS among {neighbors.length} closest
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
          <td>{nb.consequence_display}</td>
          <td><span class="label-badge {labelClass(nb.label)}">{nb.label_display}</span></td>
          <td class="mono" style="color:{nb.score > 0.7 ? 'var(--pathogenic)' : nb.score < 0.3 ? 'var(--benign)' : 'var(--text-secondary)'}">{(nb.score * 100).toFixed(0)}%</td>
          <td class="mono" style="color:var(--text-muted)">{(nb.similarity * 100).toFixed(0)}%</td>
        </tr>
      {/each}
    </tbody>
  </table>
</HelpCard>
