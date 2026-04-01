<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';
  import type { Variant, DistributionData } from '../../lib/types';

  interface Props { variant: Variant; distributions: Record<string, any>; }
  let { variant: v, distributions }: Props = $props();

  let canvasEl: HTMLCanvasElement;
  let chart: Chart | null = null;
  let showHelp = $state(false);

  function render() {
    const distData = distributions?.pathogenic as DistributionData;
    if (!distData || !canvasEl) return;
    const bins = distData.bins;
    const mb = Math.min(bins - 1, Math.max(0, Math.floor(v.score * bins)));

    chart?.destroy();
    chart = new Chart(canvasEl.getContext('2d')!, {
      type: 'bar',
      data: {
        labels: Array.from({ length: bins }, (_, i) => `${(i / bins * 100).toFixed(0)}%`),
        datasets: [
          { label: 'Benign', data: distData.benign, backgroundColor: 'rgba(34,119,170,0.4)', borderWidth: 0 },
          { label: 'Pathogenic', data: distData.pathogenic, backgroundColor: 'rgba(204,85,85,0.4)', borderWidth: 0 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: { x: { ticks: { maxTicksLimit: 10 } }, y: {} },
        plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } } },
      },
      plugins: [{
        id: 'variantMarker',
        afterDraw(c) {
          const meta = c.getDatasetMeta(0);
          if (!meta.data[mb]) return;
          const ctx = c.ctx;
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(meta.data[mb].x, c.chartArea.top);
          ctx.lineTo(meta.data[mb].x, c.chartArea.bottom);
          ctx.strokeStyle = '#333';
          ctx.lineWidth = 2.5;
          ctx.stroke();
          ctx.font = '11px sans-serif';
          ctx.fillStyle = '#333';
          const tx = meta.data[mb].x;
          const lbl = 'This variant';
          const tw = ctx.measureText(lbl).width;
          ctx.fillText(lbl, tx + tw + 8 > c.chartArea.right ? tx - tw - 4 : tx + 4, c.chartArea.top + 12);
          ctx.restore();
        },
      }],
    });
  }

  $effect(() => { v; distributions; render(); });
  onMount(() => { render(); });
  onDestroy(() => { chart?.destroy(); });
</script>

<div class="card">
  <button class="card-help-btn" onclick={() => showHelp = !showHelp}>?</button>
  {#if showHelp}
    <div class="card-help open">
      <div class="card-help-inner">
        <b>Score Distribution.</b> Histogram of predicted pathogenicity across all 232K labeled ClinVar variants.
        The vertical line marks this variant.
      </div>
    </div>
  {/if}

  <div class="section-title">Score Distribution</div>
  <div class="distribution-container">
    <canvas bind:this={canvasEl}></canvas>
  </div>
</div>
