<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as echarts from 'echarts/core';
  import { BarChart } from 'echarts/charts';
  import { GridComponent, TooltipComponent, MarkLineComponent } from 'echarts/components';
  import { CanvasRenderer } from 'echarts/renderers';
  echarts.use([BarChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer]);

  interface HistData {
    benign: number[];
    pathogenic: number[];
    bins: number;
    range: [number, number];
    _bTotal: number;
    _pTotal: number;
  }

  interface Marker {
    value: number;
    color: string;
    label?: string;
  }

  interface Props {
    histogram: HistData;
    markers?: Marker[];
    title?: string;
  }
  let { histogram, markers = [], title }: Props = $props();

  let container: HTMLDivElement;
  let chart: echarts.ECharts | null = null;

  function render() {
    if (!container || !histogram?.benign) return;
    chart?.dispose();
    chart = echarts.init(container, undefined, { renderer: 'canvas' });

    const { benign, pathogenic, bins, range: [lo, hi], _bTotal, _pTotal } = histogram;
    const step = (hi - lo) / bins;

    const labels = Array.from({ length: bins }, (_, i) => {
      const v = lo + (i + 0.5) * step;
      return Math.abs(hi - lo) <= 2 ? v.toFixed(2) : v.toFixed(1);
    });

    // Mark lines without labels (legend is below the chart)
    const markLineData = markers.map(m => {
      const idx = Math.max(0, Math.min(bins - 1, Math.floor((m.value - lo) / (hi - lo) * bins)));
      return {
        xAxis: idx,
        lineStyle: { color: m.color, width: 1.5, type: 'solid' },
        label: { show: false },
      };
    });

    chart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      grid: { left: 30, right: 10, top: title ? 22 : 10, bottom: 30 },
      ...(title ? { title: { text: title, textStyle: { fontSize: 9, color: '#999', fontWeight: 'normal' }, left: 'center', top: 0 } } : {}),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (ps: any) => {
          const idx = ps[0]?.dataIndex;
          if (idx == null) return '';
          const loE = (lo + idx * step).toFixed(3);
          const hiE = (lo + (idx + 1) * step).toFixed(3);
          const b = benign[idx] || 0;
          const p = pathogenic[idx] || 0;
          const bCount = Math.round(b * _bTotal);
          const pCount = Math.round(p * _pTotal);
          return `${loE} \u2013 ${hiE}<br>Benign: ${bCount}<br>Pathogenic: ${pCount}`;
        },
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { fontSize: 8, color: '#999', interval: Math.max(0, Math.floor(bins / 5) - 1) },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { fontSize: 8, color: '#999' },
        splitLine: { lineStyle: { color: '#f0ede8' } },
      },
      series: [
        {
          name: 'Benign',
          type: 'bar',
          data: benign,
          itemStyle: { color: 'rgba(34,119,170,0.5)' },
          barGap: 0,
          ...(markLineData.length ? {
            markLine: { silent: true, symbol: 'none', data: markLineData },
          } : {}),
        },
        {
          name: 'Pathogenic',
          type: 'bar',
          data: pathogenic,
          itemStyle: { color: 'rgba(204,85,85,0.5)' },
          barGap: 0,
        },
      ],
    });
  }

  $effect(() => { histogram; markers; render(); });
  onMount(() => render());
  onDestroy(() => chart?.dispose());
</script>

<div class="hist-wrap">
  <div bind:this={container} class="score-histogram"></div>
  {#if markers.length}
    <div class="hist-legend">
      {#each markers as m}
        <span class="legend-item">
          <span class="legend-line" style="border-color:{m.color}"></span>
          <span class="legend-label">{m.label} = {m.value.toFixed(3)}</span>
        </span>
      {/each}
    </div>
  {/if}
</div>

<style>
  .hist-wrap { width: 100%; }
  .score-histogram {
    width: 100%;
    aspect-ratio: 1;
  }
  .hist-legend {
    display: flex;
    gap: 10px;
    justify-content: center;
    padding: 2px 0;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 9px;
    color: var(--text-muted);
  }
  .legend-line {
    display: inline-block;
    width: 14px;
    height: 0;
    border-top: 2px;
  }
  .legend-label { white-space: nowrap; }
</style>
