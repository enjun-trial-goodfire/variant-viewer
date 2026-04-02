<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as echarts from 'echarts/core';
  import { BarChart } from 'echarts/charts';
  import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components';
  import { CanvasRenderer } from 'echarts/renderers';
  echarts.use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

  interface HistData {
    benign: number[];
    pathogenic: number[];
    bins: number;
    range: [number, number];
    _bTotal: number;
    _pTotal: number;
  }

  interface Props {
    histogram: HistData;
    variantValue: number;
    headName: string;
  }
  let { histogram, variantValue, headName }: Props = $props();

  let container: HTMLDivElement;
  let chart: echarts.ECharts | null = null;

  function render() {
    if (!container || !histogram?.benign) return;

    chart?.dispose();
    chart = echarts.init(container, undefined, { renderer: 'canvas' });

    const { benign, pathogenic, bins, range: [lo, hi] } = histogram;
    const step = (hi - lo) / bins;
    // Labels as percentages: "0%", "5%", "10%", ... "95%"
    const labels = Array.from({ length: bins }, (_, i) => `${Math.round((lo + (i + 0.5) * step) * 100)}%`);

    // Marker bin for this variant
    const markerBin = Math.max(0, Math.min(bins - 1, Math.floor((variantValue - lo) / (hi - lo) * bins)));

    chart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      grid: { left: 35, right: 10, top: 8, bottom: 30 },
      legend: { show: false },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (ps: any) => {
          const idx = ps[0]?.dataIndex;
          if (idx == null) return '';
          const loEdge = Math.round((lo + idx * step) * 100);
          const hiEdge = Math.round((lo + (idx + 1) * step) * 100);
          const b = benign[idx] || 0;
          const p = pathogenic[idx] || 0;
          const bCount = Math.round(b * histogram._bTotal);
          const pCount = Math.round(p * histogram._pTotal);
          const total = bCount + pCount;
          const pctPath = total > 0 ? ((pCount / total) * 100).toFixed(0) : '\u2014';
          return `Score: ${loEdge}\u2013${hiEdge}%<br>Benign: ${bCount}<br>Pathogenic: ${pCount}<br>% Pathogenic: ${pctPath}%`;
        },
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          fontSize: 8,
          color: '#999',
          interval: Math.floor(bins / 5) - 1,
        },
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
          itemStyle: { color: '#27a' },
          barGap: 0,
          // Marker line on the benign series
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: '#333', width: 2, type: 'dashed' },
            label: {
              show: true,
              formatter: 'this variant',
              fontSize: 9,
              position: 'insideStartTop',
            },
            data: [{ xAxis: markerBin }],
          },
        },
        {
          name: 'Pathogenic',
          type: 'bar',
          data: pathogenic,
          itemStyle: { color: '#c55' },
          barGap: 0,
        },
      ],
    });
  }

  $effect(() => {
    histogram; variantValue;
    render();
  });

  onMount(() => render());
  onDestroy(() => chart?.dispose());
</script>

<div bind:this={container} class="histogram-container"></div>

<style>
  .histogram-container {
    width: 100%;
    height: 200px;
  }
</style>
