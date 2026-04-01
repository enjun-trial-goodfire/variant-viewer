<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as echarts from 'echarts/core';
  import { ScatterChart } from 'echarts/charts';
  import { GridComponent, VisualMapComponent, TooltipComponent } from 'echarts/components';
  import { CanvasRenderer } from 'echarts/renderers';

  echarts.use([ScatterChart, GridComponent, VisualMapComponent, TooltipComponent, CanvasRenderer]);

  interface HeatmapData {
    data: [number, number, number, number][]; // [bin_i, bin_j, pct_pathogenic, count]
    bins: number;
  }

  interface Props {
    heatmap: HeatmapData;
    variantRef: number;  // raw score [0, 1]
    variantVar: number;  // raw score [0, 1]
    headName: string;
  }
  let { heatmap, variantRef, variantVar, headName }: Props = $props();

  let container: HTMLDivElement;
  let chart: echarts.ECharts | null = null;

  function render() {
    if (!container || !heatmap?.data?.length) return;

    chart?.dispose();
    chart = echarts.init(container, undefined, { renderer: 'canvas' });

    const { data, bins } = heatmap;
    const maxCount = Math.max(...data.map(d => d[3]));
    const step = 1.0 / bins;  // score step per bin (e.g. 0.1 for 10 bins)

    // Map bin indices to score centers: bin 0 → step/2, bin 9 → 1 - step/2
    const toScore = (bi: number) => (bi + 0.5) * step;

    // Scatter data: [ref_score_center, var_score_center, pct, count]
    const scatterData = data.map(([bi, bj, pct, count]) => [toScore(bi), toScore(bj), pct, count]);

    // Variant marker position (snap to bin center)
    const vRefCenter = toScore(Math.min(bins - 1, Math.max(0, Math.floor(variantRef * bins))));
    const vVarCenter = toScore(Math.min(bins - 1, Math.max(0, Math.floor(variantVar * bins))));

    // Tick positions: bin edges (0, 0.1, 0.2, ... 1.0)
    const ticks = Array.from({ length: bins + 1 }, (_, i) => i * step);

    // Max symbol size in pixels: chart is square, so use container width
    const chartSize = Math.min(container.clientWidth - 100, 350); // approx plot area
    const maxSymbolPx = (chartSize / bins) * 0.85;

    chart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      grid: {
        left: 'center',
        top: 40,
        bottom: 45,
        width: chartSize,
        height: chartSize,
      },
      tooltip: {
        formatter: (p: any) => {
          if (p.seriesIndex !== 0) return '';
          const [ref, va, pct, count] = p.data;
          return `<b>${headName}</b><br>` +
            `ref: ${(ref - step/2).toFixed(1)}–${(ref + step/2).toFixed(1)}<br>` +
            `var: ${(va - step/2).toFixed(1)}–${(va + step/2).toFixed(1)}<br>` +
            `<b style="color:${pct > 50 ? '#c55' : '#2178ab'}">${pct}%</b> pathogenic (n=${count})`;
        },
      },
      xAxis: {
        type: 'value',
        name: 'ref score',
        nameLocation: 'center',
        nameGap: 28,
        nameTextStyle: { fontSize: 11, color: '#888' },
        min: 0,
        max: 1,
        interval: step,
        axisLabel: {
          fontSize: 9,
          color: '#999',
          formatter: (v: number) => {
            // Show every other label to avoid clutter
            const idx = Math.round(v / step);
            return idx % 2 === 0 ? v.toFixed(1) : '';
          },
        },
        splitLine: { show: true, lineStyle: { color: '#e8e0d4', width: 1 } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: 'var score',
        nameLocation: 'center',
        nameGap: 35,
        nameTextStyle: { fontSize: 11, color: '#888' },
        min: 0,
        max: 1,
        interval: step,
        axisLabel: {
          fontSize: 9,
          color: '#999',
          formatter: (v: number) => {
            const idx = Math.round(v / step);
            return idx % 2 === 0 ? v.toFixed(1) : '';
          },
        },
        splitLine: { show: true, lineStyle: { color: '#e8e0d4', width: 1 } },
        axisTick: { show: false },
      },
      visualMap: {
        show: true,
        type: 'continuous',
        dimension: 2,
        min: 0,
        max: 100,
        text: ['pathogenic', 'benign'],
        textStyle: { fontSize: 9, color: '#999' },
        orient: 'horizontal',
        itemWidth: 10,     // thickness (vertical) when horizontal
        itemHeight: 140,   // length (horizontal) when horizontal
        left: 'center',
        top: 2,
        inRange: {
          color: ['#2178ab', '#66aacc', '#bbbbbb', '#dd8888', '#cc5555'],
        },
        seriesIndex: [0],
      },
      series: [
        // Sized squares: pct → color (via visualMap), count → size
        {
          type: 'scatter',
          data: scatterData,
          symbol: 'rect',
          symbolSize: (val: number[]) => {
            const frac = Math.sqrt(val[3] / maxCount);
            return Math.max(3, maxSymbolPx * frac);
          },
          itemStyle: { opacity: 0.9, borderColor: '#fff', borderWidth: 0.5 },
          encode: { x: 0, y: 1, tooltip: [0, 1, 2, 3] },
        },
        // Variant position: subtle highlighted cell background
        {
          type: 'scatter',
          data: [[vRefCenter, vVarCenter]],
          symbol: 'rect',
          symbolSize: maxSymbolPx * 1.1,
          itemStyle: { color: 'rgba(0,0,0,0.08)', borderColor: '#333', borderWidth: 1.5 },
          z: 1,  // behind the data squares
          silent: true,
        },
      ],
    });
  }

  $effect(() => {
    heatmap; variantRef; variantVar;
    render();
  });

  onMount(() => render());
  onDestroy(() => chart?.dispose());
</script>

<div bind:this={container} class="heatmap-container"></div>

<style>
  .heatmap-container {
    width: 100%;
    height: 420px;
  }
</style>
