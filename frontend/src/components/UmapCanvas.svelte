<script lang="ts">
  import { onMount } from 'svelte';
  import { scoreColor } from '../lib/colors';
  import { LABEL_NAMES } from '../lib/helpers';
  import type { UmapData } from '../lib/types';

  interface Props {
    data: UmapData;
    mode: 'predicted' | 'labeled';
  }

  let { data, mode }: Props = $props();

  let canvasEl: HTMLCanvasElement;

  function navigate(path: string) {
    location.hash = `#/${path}`;
  }

  function getColors(umapData: UmapData, umapMode: string) {
    const { score, labels } = umapData;
    const lc: Record<string, number[]> = {
      pathogenic: [.80,.33,.33,.8], likely_pathogenic: [.86,.54,.28,.8],
      benign: [.13,.47,.67,.8], likely_benign: [.42,.67,.80,.8], VUS: [.63,.63,.63,.4]
    };
    if (umapMode === 'labeled') return labels.map(l => lc[LABEL_NAMES[l] || 'VUS'] || lc.VUS);
    return score.map(s => {
      if (s < 0.5) { const t = s * 2; return [.13+t*.53, .47+t*.20, .67, .8]; }
      const t = (s - .5) * 2; return [.67+t*.13, .67-t*.33, .67-t*.33, .8];
    });
  }

  function draw() {
    if (!canvasEl || !data) return;
    const ctx = canvasEl.getContext('2d')!;
    const dpr = devicePixelRatio || 1;
    const W = 900, H = 600;
    canvasEl.style.width = W + 'px';
    canvasEl.style.height = H + 'px';
    canvasEl.width = W * dpr;
    canvasEl.height = H * dpr;
    ctx.scale(dpr, dpr);

    const { x, y, score, ids, genes, labels, gene_list } = data;
    const labelStr = labels.map(l => LABEL_NAMES[l] || '?');
    const geneStr = gene_list ? genes.map(g => gene_list[g] || '?') : genes.map(String);
    const n = x.length;
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const yMin = Math.min(...y), yMax = Math.max(...y);
    const pad = 15, w = W - 2 * pad, h = H - 2 * pad;
    const px = new Float32Array(n), py = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      px[i] = pad + (x[i] - xMin) / (xMax - xMin) * w;
      py[i] = pad + (y[i] - yMin) / (yMax - yMin) * h;
    }

    ctx.fillStyle = '#faf8f5';
    ctx.fillRect(0, 0, W, H);
    const colors = getColors(data, mode);
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.random() * (i + 1) | 0;
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const i of order) {
      const c = colors[i];
      const alpha = (mode === 'labeled' && labelStr[i] === 'VUS') ? .35 : .7;
      ctx.fillStyle = `rgba(${(c[0]*255)|0},${(c[1]*255)|0},${(c[2]*255)|0},${alpha})`;
      ctx.beginPath();
      ctx.arc(px[i], py[i], 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    const tooltip = document.getElementById('umap-tooltip')!;

    function findNearest(e: MouseEvent) {
      const cr = canvasEl.getBoundingClientRect();
      const mx = (e.clientX - cr.left) * (W / cr.width);
      const my = (e.clientY - cr.top) * (H / cr.height);
      let best = -1, bestD = 12;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(mx - px[i], my - py[i]);
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }

    let raf = false;
    canvasEl.onmousemove = (e: MouseEvent) => {
      if (raf) return;
      raf = true;
      requestAnimationFrame(() => {
        raf = false;
        const i = findNearest(e);
        if (i >= 0) {
          const s = score[i], lbl = labelStr[i];
          const c = lbl.includes('pathogenic') ? '#c55' : lbl.includes('benign') ? '#27a' : '#666';
          tooltip.style.display = 'block';
          tooltip.style.left = (e.clientX + 14) + 'px';
          tooltip.style.top = (e.clientY - 10) + 'px';
          tooltip.innerHTML = `<b>${geneStr[i] || '?'}</b> <span style="color:${c};font-weight:600">${lbl}</span><br>
            <span style="font-family:monospace;font-size:10px">${ids[i]}</span><br>
            Pathogenicity: <b style="color:${scoreColor(s)}">${(s * 100).toFixed(0)}%</b>`;
          canvasEl.style.cursor = 'pointer';
        } else {
          tooltip.style.display = 'none';
          canvasEl.style.cursor = 'crosshair';
        }
      });
    };

    canvasEl.onclick = (e: MouseEvent) => {
      tooltip.style.display = 'none';
      const i = findNearest(e);
      if (i >= 0) navigate(`variant/${ids[i]}`);
    };

    canvasEl.onmouseleave = () => { tooltip.style.display = 'none'; };
  }

  $effect(() => {
    // Redraw when mode changes
    mode;
    draw();
  });

  onMount(() => { draw(); });
</script>

<div class="umap-wrap">
  <canvas bind:this={canvasEl}></canvas>
</div>
