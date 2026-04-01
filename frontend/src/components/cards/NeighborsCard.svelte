<script lang="ts">
  import { scoreColor } from '../../lib/colors';
  import { humanConsequence, truncate, LABEL_DISPLAY } from '../../lib/helpers';
  import type { Variant } from '../../lib/types';

  interface Props { variant: Variant; }
  let { variant: v }: Props = $props();

  let showHelp = $state(false);

  function navigate(path: string) { location.hash = `#/${path}`; }

  function labelClass(label: string) {
    if (label.includes('pathogenic')) return 'label-pathogenic';
    if (label.includes('benign')) return 'label-benign';
    return 'label-vus';
  }
</script>

<div class="card">
  <button class="card-help-btn" onclick={() => showHelp = !showHelp}>?</button>
  {#if showHelp}
    <div class="card-help open">
      <div class="card-help-inner">
        <b>Nearest Neighbors.</b> The 10 most similar variants by cosine similarity in Evo2's probe activation embedding space.
      </div>
    </div>
  {/if}

  <div class="section-title">Nearest Neighbors</div>

  {#if v.label === 'VUS' && v.neighbors.length}
    {@const character = v.nP > v.nB * 2 ? 'predominantly pathogenic' : v.nB > v.nP * 2 ? 'predominantly benign' : 'mixed'}
    {@const cc = character.includes('pathogenic') ? 'var(--negative)' : character.includes('benign') ? '#27a' : 'var(--text-secondary)'}
    <div class="vus-callout">
      This variant is <b>unclassified (VUS)</b>. Its nearest neighbors suggest
      <b style="color:{cc}">{character}</b> character.
    </div>
  {/if}

  <div class="neighbor-summary">
    <strong>{v.nP}</strong> pathogenic, <strong>{v.nB}</strong> benign, <strong>{v.nV}</strong> VUS among {v.neighbors.length} closest
  </div>

  <table>
    <thead>
      <tr><th>Variant</th><th>Gene</th><th>Consequence</th><th>Label</th><th>Score</th><th>Sim</th></tr>
    </thead>
    <tbody>
      {#each v.neighbors as nb}
        <tr class={nb.label.includes('pathogenic') ? 'pathogenic' : nb.label.includes('benign') ? 'benign' : ''}
            style="cursor:pointer" onclick={() => navigate(`variant/${nb.id}`)}>
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
</div>
