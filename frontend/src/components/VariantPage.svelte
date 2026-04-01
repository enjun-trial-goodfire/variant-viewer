<script lang="ts">
  import { getVariant } from '../lib/api';
  import { globalData, currentVariant } from '../lib/stores';
  import type { Variant } from '../lib/types';
  import VerdictCard from './cards/VerdictCard.svelte';
  import InterpretationCard from './cards/InterpretationCard.svelte';
  import DisruptionCard from './cards/DisruptionCard.svelte';
  import EffectsCard from './cards/EffectsCard.svelte';
  import PredictorsCard from './cards/PredictorsCard.svelte';
  import NeighborsCard from './cards/NeighborsCard.svelte';
  // DistributionCard removed — pathogenicity histogram is now in PredictorsCard
  import PopulationCard from './cards/PopulationCard.svelte';

  interface Props { variantId: string; }
  let { variantId }: Props = $props();

  let variant = $state<Variant | null>(null);
  let error = $state('');

  async function load(id: string) {
    error = '';
    variant = null;
    try {
      variant = await getVariant(id);
      currentVariant.set(variant);
    } catch {
      error = id;
    }
  }

  $effect(() => { load(variantId); });

  const hasPopulation = $derived(
    variant?.gnomad_pop && Object.values(variant.gnomad_pop).some(f => f > 0)
  );
</script>

{#if error}
  <div style="text-align:center;padding:60px 24px;color:var(--text-muted)">
    <div style="font-size:16px;font-weight:600;margin-bottom:4px">Variant not found</div>
    <div style="font-size:13px;font-family:monospace">{error}</div>
  </div>
{:else if variant && $globalData}
  <VerdictCard {variant} global={$globalData} />
  <InterpretationCard variantId={variant.variant_id} />
  {#if Object.keys(variant.disruption ?? {}).length}
    <DisruptionCard {variant} global={$globalData} />
  {/if}
  {#if Object.keys(variant.effect ?? {}).length}
    <EffectsCard {variant} global={$globalData} />
  {/if}
  <PredictorsCard {variant} global={$globalData} />
  {#if variant.neighbors?.length}
    <NeighborsCard {variant} />
  {/if}
  {#if hasPopulation}
    <PopulationCard {variant} />
  {/if}
{/if}
