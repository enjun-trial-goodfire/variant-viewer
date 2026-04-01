import type { HeadDistribution } from './types';

export function scoreColor(s: number): string {
  return `rgb(${Math.round(215 * s)}, 80, ${Math.round(178 * (1 - s))})`;
}

export function barColor(v: number): string {
  if (v > 0.9) return '#c55';
  if (v > 0.75) return '#DB8A48';
  if (v < 0.1) return '#27a';
  if (v < 0.25) return '#6ac';
  return '#bbb';
}

export function headColor(
  head: string,
  value: number,
  isDelta: boolean,
  distributions: Record<string, HeadDistribution> | null,
): string {
  if (!distributions) return '#bbb';
  let dist = distributions[head] as any;
  if (!dist) return '#bbb';
  if (dist.ref) dist = dist.ref;
  if (!dist.benign) return '#bbb';
  const [lo, hi] = dist.range || [0, 1];
  const bins = dist.bins || 40;
  const mapped = (value - lo) / (hi - lo);
  const bin = Math.max(0, Math.min(bins - 1, Math.floor(mapped * bins)));
  const bRaw = dist.benign[bin] || 0;
  const pRaw = dist.pathogenic[bin] || 0;
  if (bRaw + pRaw < 5) return '#bbb';
  const bTotal = dist.benign.reduce((a: number, x: number) => a + x, 0) || 1;
  const pTotal = dist.pathogenic.reduce((a: number, x: number) => a + x, 0) || 1;
  return barColor(pRaw / pTotal / (pRaw / pTotal + bRaw / bTotal));
}

export function tierColor(val: number, invert: boolean): string {
  const v = invert ? 1 - val : val;
  if (v >= 0.8) return '#c55';
  if (v >= 0.6) return '#DB8A48';
  if (v >= 0.4) return '#bbb';
  if (v >= 0.2) return '#6ac';
  return '#27a';
}

export function deltaColor(z: number, sign: number): { text: string; bar: string } {
  return z < 1
    ? { text: 'var(--text-muted)', bar: '#ccc' }
    : { text: sign < 0 ? '#c55' : '#4a9', bar: sign < 0 ? '#c55' : '#4a9' };
}
