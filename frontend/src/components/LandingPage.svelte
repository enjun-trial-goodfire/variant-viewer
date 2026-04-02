<script lang="ts">
  import { onMount } from 'svelte';
  import { getUmap } from '../lib/api';
  import { umapData } from '../lib/stores';
  import UmapCanvas from './UmapCanvas.svelte';
  import type { UmapData } from '../lib/types';

  let umap = $state<UmapData | null>(null);
  let umapMode = $state<'predicted' | 'labeled'>('predicted');

  onMount(async () => {
    const data = await getUmap();
    if (data) {
      umap = data;
      umapData.set(data);
    }
  });
</script>

<div style="max-width:900px;margin:0 auto 32px;text-align:center">
  <p style="font-size:15px;line-height:1.7;color:var(--text-secondary)">
    EVEE (Evo Variant Effect Explorer) provides pathogenicity predictions for all 4.2 million ClinVar variants using a covariance probe trained on embeddings from Evo2, a 7 billion parameter DNA foundation model. The classifier achieves an overall AUROC of 0.997 for identifying benign from pathogenic ClinVar variants, outperforming existing approaches across all variant consequence types. Each variant page shows aggregated metadata, predicted pathogenicity, LLM-synthesized interpretations, annotation disruption profiles, and nearest neighbor evidence.
  </p>
</div>

{#if umap}
  <div class="umap-landing">
    <h2 style="font-size:22px">Variant Embedding Space</h2>
    <p style="font-size:14px">30k variants. Click a point to view its card.</p>
    <div style="margin-bottom:10px;display:flex;gap:4px;justify-content:center">
      <button class="umap-toggle" class:active={umapMode === 'predicted'} onclick={() => umapMode = 'predicted'}>Pathogenicity</button>
      <button class="umap-toggle" class:active={umapMode === 'labeled'} onclick={() => umapMode = 'labeled'}>ClinVar label</button>
    </div>
    <UmapCanvas data={umap} mode={umapMode} />
    <div style="margin-top:10px;font-size:11px;color:var(--text-muted);display:flex;gap:12px;justify-content:center;align-items:center">
      {#if umapMode === 'predicted'}
        <span style="display:inline-flex;align-items:center;gap:6px">
          <span style="font-size:10px">0%</span>
          <span style="display:inline-block;width:120px;height:10px;border-radius:3px;background:linear-gradient(to right, rgb(33,120,171), #bbb 50%, rgb(204,85,85))"></span>
          <span style="font-size:10px">100%</span>
        </span>
      {:else}
        {#each [['pathogenic','#c55'],['likely path.','#d88'],['VUS','#a1a1a1'],['likely benign','#6ac'],['benign','#2178ab']] as [name, color]}
          <span style="display:inline-flex;align-items:center;gap:3px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color}{name==='VUS'?';opacity:0.6':''}"></span>
            {name}
          </span>
        {/each}
      {/if}
    </div>
  </div>
{/if}
