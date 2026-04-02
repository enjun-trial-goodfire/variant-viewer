<script lang="ts">
  import HeadHistogram from '../HeadHistogram.svelte';
  import HelpCard from '../HelpCard.svelte';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  const dist = $derived(g.distributions?.pathogenic);
</script>

{#if dist}
  <HelpCard title="Score Distribution" helpText="Histogram of predicted pathogenicity scores across all 232K labeled ClinVar variants. Blue bars = benign-labeled variants, red bars = pathogenic-labeled variants. The vertical black line marks this variant's predicted score. Good separation between the blue and red distributions indicates the probe discriminates well.">
    <HeadHistogram histogram={dist} variantValue={v.score_pathogenic} headName="Pathogenicity" />
  </HelpCard>
{/if}
