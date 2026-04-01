<script lang="ts">
  import { onMount } from 'svelte';
  import { getGlobal } from './lib/api';
  import { globalData, currentHash } from './lib/stores';
  import Header from './components/Header.svelte';
  import LandingPage from './components/LandingPage.svelte';
  import VariantPage from './components/VariantPage.svelte';

  let route = $state<'landing' | 'variant'>('landing');
  let variantId = $state('');

  function handleHash() {
    const hash = location.hash || '#/';
    currentHash.set(hash);
    if (hash.startsWith('#/variant/')) {
      variantId = decodeURIComponent(hash.slice('#/variant/'.length));
      route = 'variant';
    } else {
      route = 'landing';
    }
  }

  onMount(() => {
    handleHash();
    window.addEventListener('hashchange', handleHash);

    // Load global data on startup
    getGlobal().then(data => {
      globalData.set(data);
    });

    return () => window.removeEventListener('hashchange', handleHash);
  });
</script>

<Header />

<div class="container" id="content">
  {#if route === 'landing'}
    <LandingPage />
  {:else if route === 'variant'}
    <VariantPage {variantId} />
  {/if}
</div>

<div id="global-tip"></div>
<div id="umap-tooltip"></div>
