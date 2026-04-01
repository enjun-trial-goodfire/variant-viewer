<script lang="ts">
  import ScoreRow from './ScoreRow.svelte';
  import HeadHistogram from './HeadHistogram.svelte';

  interface Props {
    head: string;
    display: string;
    value: number;
    lr: number;
    description?: string;
    distributions?: Record<string, any>;
  }
  let { head, display, value, lr, description, distributions }: Props = $props();

  let expanded = $state(false);
  const histData = $derived(distributions?.[head]?.benign ? distributions[head] : null);
</script>

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div onclick={() => expanded = !expanded} style="cursor:pointer">
  <ScoreRow label={display} {value} {lr} />
</div>

{#if expanded}
  <div class="hist-row">
    <div></div>
    <div class="hist-detail">
      {#if description}
        <div class="head-desc">{description}</div>
      {/if}
      {#if histData}
        <HeadHistogram histogram={histData} variantValue={value} headName={display} />
      {/if}
    </div>
  </div>
{/if}

<style>
  .hist-row {
    display: grid;
    grid-template-columns: minmax(100px, 160px) 1fr 45px;
    gap: 8px;
  }
  .hist-detail {
    padding: 4px 0;
  }
  .head-desc {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.5;
    padding: 6px 10px;
    margin-bottom: 6px;
    background: var(--bg-hover);
    border-left: 2px solid var(--accent);
    border-radius: 0 4px 4px 0;
  }
</style>
