<script lang="ts">
  import Swatch from './Swatch.svelte';

  interface Props { inline?: boolean; }
  let { inline = false }: Props = $props();
</script>

{#if inline}
  <div class="legend-inline">
    <span class="legend-label">benign</span>
    <Swatch color="#27a" label="Benign" />
    <Swatch color="#6ac" label="Leaning benign" />
    <Swatch color="#bbb" label="Neutral" />
    <Swatch color="#d88" label="Leaning pathogenic" />
    <Swatch color="#c55" label="Pathogenic" />
    <span class="legend-label">pathogenic</span>
  </div>
{:else}
  <div class="legend-grid">
    <div></div>
    <div class="legend-inline">
      <span class="legend-label">benign</span>
      <Swatch color="#27a" label="Benign" />
      <Swatch color="#6ac" label="Leaning benign" />
      <Swatch color="#bbb" label="Neutral" />
      <Swatch color="#d88" label="Leaning pathogenic" />
      <Swatch color="#c55" label="Pathogenic" />
      <span class="legend-label">pathogenic</span>
    </div>
    <div></div>
  </div>
{/if}

<style>
  .legend-grid {
    display: grid;
    grid-template-columns: minmax(100px, 160px) 1fr 45px;
    gap: 8px;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .legend-inline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
  }
  .legend-label {
    font-size: 9px;
    color: var(--text-muted);
  }
</style>
