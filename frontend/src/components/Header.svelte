<script lang="ts">
  import { search } from '../lib/api';
  import { truncate, navigate } from '../lib/helpers';
  import type { SearchResult } from '../lib/types';

  let query = $state('');
  let results = $state<SearchResult[]>([]);
  let showResults = $state(false);
  let searchTimeout: ReturnType<typeof setTimeout>;

  function onInput() {
    clearTimeout(searchTimeout);
    const q = query.trim();
    if (q.length < 2) { showResults = false; return; }
    if (q.includes(':') && q.split(':').length >= 4) {
      showResults = false;
      navigate(`variant/${q}`);
      return;
    }
    searchTimeout = setTimeout(async () => {
      results = await search(q);
      showResults = true;
    }, 300);
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key !== 'Enter') return;
    const q = query.trim();
    showResults = false;
    if (q.includes(':') && q.split(':').length >= 4) {
      navigate(`variant/${q}`);
    } else {
      search(q).then(r => { if (r.length) navigate(`variant/${r[0].v}`); });
    }
  }

  function onClickOutside(e: MouseEvent) {
    if (!(e.target as HTMLElement).closest('.search-container')) {
      showResults = false;
    }
  }
</script>

<svelte:document on:click={onClickOutside} />

<div class="header" style="view-transition-name: header">
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions a11y_click_events_have_key_events -->
  <h1 onclick={() => navigate('')} style="cursor: pointer">
    EVEE <span style="font-size:11px;font-weight:400;color:var(--text-muted);margin-left:8px">Evo Variant Effect Explorer</span>
  </h1>
  <div class="search-container">
    <input
      type="text"
      bind:value={query}
      oninput={onInput}
      onkeydown={onKeydown}
      placeholder="Search gene or variant (BRCA1, chr17:43093110:C:T)"
      autocomplete="off"
    />
    {#if showResults}
      <div class="search-results visible">
        {#if results.length === 0}
          <div class="search-item"><span class="meta">No results</span></div>
        {:else}
          {#each results as r}
            <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
            <div class="search-item" role="button" tabindex="0" onclick={() => { showResults = false; navigate(`variant/${r.v}`); }}>
              <span class="vid" title={r.v}>{truncate(r.v)}</span>
              <span class="meta">{r.c} · {r.l}</span>
            </div>
          {/each}
        {/if}
      </div>
    {/if}
  </div>
  <div class="header-branding">
    <img src="/goodfire-color.svg" alt="Goodfire">
    <img src="/Mayo_Clinic_Logo_2023.png" alt="Mayo Clinic" style="height:48px">
  </div>
</div>

<style>
  .header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: center; gap: 16px;
    position: sticky; top: 0; z-index: 100;
  }
  h1 { font-size: 20px; font-weight: 600; white-space: nowrap;
    transition: color 0.15s; }
  h1:hover { color: var(--accent); }
  input { width: 100%; padding: 8px 14px;
    border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px;
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace;
    transition: border-color 0.15s, box-shadow 0.15s; }
  input:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-light); }
  .search-container { flex: 1; max-width: 560px; margin: 0 auto; position: relative; }
  .header-branding { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  .header-branding img { height: 24px; }
  .search-results { position: absolute; top: 100%; left: 0; right: 0; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: var(--radius); margin-top: 4px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08); z-index: 200; max-height: 400px; overflow-y: auto; }
  .search-results:not(.visible) { display: none; }
  .search-item { padding: 8px 14px; cursor: pointer; display: flex; justify-content: space-between;
    align-items: center; transition: background 0.1s; }
  .search-item:hover { background: var(--bg-hover); }
  .search-item .vid { font-family: monospace; font-size: 12px; }
  .search-item .meta { font-size: 11px; color: var(--text-muted); }
</style>
