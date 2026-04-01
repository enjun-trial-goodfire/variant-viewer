<script lang="ts">
  import { onMount } from 'svelte';
  import { getGlobal } from './lib/api';
  import { globalData } from './lib/stores';
  import Header from './components/Header.svelte';
  import LandingPage from './components/LandingPage.svelte';
  import VariantPage from './components/VariantPage.svelte';

  let route = $state<'landing' | 'variant'>('landing');
  let variantId = $state('');

  function applyHash() {
    const hash = location.hash || '#/';
    if (hash.startsWith('#/variant/')) {
      variantId = decodeURIComponent(hash.slice('#/variant/'.length));
      route = 'variant';
    } else {
      route = 'landing';
    }
  }

  function handleHash() {
    if (document.startViewTransition) {
      document.startViewTransition(() => applyHash());
    } else {
      applyHash();
    }
  }

  onMount(() => {
    applyHash(); // initial route — no transition
    window.addEventListener('hashchange', handleHash);
    getGlobal().then(data => globalData.set(data));
    return () => window.removeEventListener('hashchange', handleHash);
  });
</script>

<Header />

<div class="container">
  {#if route === 'landing'}
    <LandingPage />
  {:else if route === 'variant'}
    <VariantPage {variantId} />
  {/if}
</div>

<div id="global-tip"></div>
<div id="umap-tooltip"></div>
