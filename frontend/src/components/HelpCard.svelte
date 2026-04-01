<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    helpText: string;
    class?: string;
    legend?: Snippet;
    children: Snippet;
  }
  let { title, helpText, class: className = '', legend, children }: Props = $props();

  let showHelp = $state(false);
</script>

<div class="card {className}">
  <div class="card-header">
    <div class="title-group">
      <div class="section-title" style="margin:0">{title}</div>
      <button class="help-btn" onclick={() => showHelp = !showHelp}>?</button>
    </div>
    {#if legend}
      <div class="card-legend">{@render legend()}</div>
    {/if}
  </div>
  <div class="help-panel" class:open={showHelp}>
    <div class="help-panel-inner">{@html helpText}</div>
  </div>
  {@render children()}
</div>

<style>
  .card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .title-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .card-legend {
    flex: 1;
    display: flex;
    justify-content: center;
  }
</style>
