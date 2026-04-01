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
  if (lr > 0.9) return '#c55';
  if (lr > 0.75) return '#d88';
  if (lr < 0.1) return '#27a';
  if (lr < 0.25) return '#6ac';
  return '#bbb';
}

/** 5-tier color for predictor bars (same scale, with optional inversion). */
export function tierColor(val: number, invert: boolean): string {
  return barColor(invert ? 1 - val : val);
}

/** Disruption direction color (orange = decreased, green = increased). */
export function deltaColor(z: number, sign: number): { text: string; bar: string } {
  if (z < 1) return { text: 'var(--text-muted)', bar: '#ccc' };
  const color = sign < 0 ? 'var(--decrease)' : 'var(--increase)';
  return { text: color, bar: color };
}
