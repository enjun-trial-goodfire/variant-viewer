<script lang="ts">
  import { onMount } from 'svelte';
  import { getVariant } from '../lib/api';
  import { globalData, currentVariant } from '../lib/stores';
  import type { Variant, GlobalData } from '../lib/types';
  import VerdictCard from './cards/VerdictCard.svelte';
  import InterpretationCard from './cards/InterpretationCard.svelte';
  import DisruptionCard from './cards/DisruptionCard.svelte';
  import PredictorsCard from './cards/PredictorsCard.svelte';
  import NeighborsCard from './cards/NeighborsCard.svelte';
  import DistributionCard from './cards/DistributionCard.svelte';
  import PopulationCard from './cards/PopulationCard.svelte';

  interface Props { variantId: string; }
  let { variantId }: Props = $props();

  let variant = $state<Variant | null>(null);
  let error = $state('');
  let global = $state<GlobalData | null>(null);

  const unsub = globalData.subscribe(g => { global = g; });

  async function load(id: string) {
    error = '';
    variant = null;
    try {
      const v = await getVariant(id);
      variant = v;
      currentVariant.set(v);
    } catch {
      error = id;
    }
  }

  $effect(() => {
    load(variantId);
  });

  onMount(() => unsub);
</script>

{#if error}
  <div style="text-align:center;padding:60px 24px;color:var(--text-muted)">
    <div style="font-size:16px;font-weight:600;margin-bottom:4px">Variant not found</div>
    <div style="font-size:13px;font-family:monospace">{error}</div>
  </div>
{:else if variant && global}
  <VerdictCard {variant} {global} />
  <InterpretationCard variantId={variant.id} />
  {#if variant.attribution?.length}
    <DisruptionCard {variant} {global} />
  {/if}
  <PredictorsCard {variant} {global} />
  <NeighborsCard {variant} />
  <DistributionCard {variant} distributions={global.distributions} />
  {#if variant.gnomad_pop && Object.values(variant.gnomad_pop).some(f => f != null && f > 0)}
    <PopulationCard {variant} />
  {/if}
{:else}
  <div style="text-align:center;padding:60px 24px;color:var(--text-muted)">Loading...</div>
{/if}
