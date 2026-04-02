// ── Color scales ─────────────────────────────────────────────────────
//
// Two independent axes:
//   Classification: blue (#27a) → light blue (#6ac) → gray (#bbb) → pink (#d88) → red (#c55)
//   Disruption:     orange (#DB8A48) for decrease, green (#4a9) for increase

/** Continuous score → color (blue at 0, gray at 0.5, red at 1). */
export function scoreColor(s: number): string {
  if (s == null || isNaN(s)) return 'var(--text-muted)';
  return `rgb(${Math.round(215 * s)}, 80, ${Math.round(178 * (1 - s))})`;
}

/** 5-tier classification color from a likelihood ratio (0 = benign, 1 = pathogenic). */
export function barColor(lr: number): string {
  if (lr > 0.9) return '#cc5555';
  if (lr > 0.75) return '#dd8888';
  if (lr < 0.1) return '#2178ab';
  if (lr < 0.25) return '#66aacc';
  return '#bbbbbb';
}

/** 5-tier color for predictor bars (same scale, with optional inversion). */
export function tierColor(val: number, invert: boolean): string {
  return barColor(invert ? 1 - val : val);
}

/** Continuous color matching HeadHeatmap colorbar: #2178ab → #66aacc → #bbbbbb → #dd8888 → #cc5555 */
export function lrColor(lr: number): string {
  const stops = [[0x21,0x78,0xab],[0x66,0xaa,0xcc],[0xbb,0xbb,0xbb],[0xdd,0x88,0x88],[0xcc,0x55,0x55]];
  const t = Math.max(0, Math.min(1, lr)) * (stops.length - 1);
  const i = Math.min(Math.floor(t), stops.length - 2);
  const f = t - i;
  const [r,g,b] = stops[i].map((c, j) => Math.round(c + (stops[i+1][j] - c) * f));
  return `rgb(${r},${g},${b})`;
}
