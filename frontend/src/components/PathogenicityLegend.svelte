<script lang="ts">
  import Swatch from './Swatch.svelte';
</script>

<span class="legend">
  <span class="endpoint left">benign</span>
  <Swatch color="#2178ab" label="Benign" />
  <Swatch color="#66aacc" label="Leaning benign" />
  <Swatch color="#bbbbbb" label="Uncertain" />
  <Swatch color="#dd8888" label="Leaning pathogenic" />
  <Swatch color="#cc5555" label="Pathogenic" />
  <span class="endpoint right">pathogenic</span>
</span>

<style>
  .legend { display: flex; align-items: center; justify-content: center; gap: 3px; }
  .endpoint { font-size: 9px; color: var(--text-muted); width: 55px; }
  .left { text-align: right; }
  .right { text-align: left; }
</style>
