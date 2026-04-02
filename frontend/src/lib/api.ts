import type { GlobalData, HeadConfig, Interpretation, SearchResult, UmapData, Variant } from './types';

declare global {
  interface Window { __APP_CONFIG__?: { API_BASE?: string }; }
}

const API_BASE = window.__APP_CONFIG__?.API_BASE || '';
function api(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : `/api${path}`;
}

// Heads config — loaded once by getGlobal(), drives normalizeVariant().
let _heads: Record<string, HeadConfig> = {};
let _globalReady: Promise<void> | null = null;
let _resolveGlobal: (() => void) | null = null;
_globalReady = new Promise(resolve => { _resolveGlobal = resolve; });

export async function getGlobal(): Promise<GlobalData> {
  let result: GlobalData;
  if (API_BASE) {
    const [heads, dist] = await Promise.all([
      fetch('/heads.json').then(r => r.json()),
      fetch('/statistics.json').then(r => r.json()),
    ]);
    result = { heads, distributions: dist };
  } else {
    result = await (await fetch('/api/global')).json();
  }
  _heads = result.heads?.heads ?? {};
  _resolveGlobal?.();
  return result;
}

export async function getUmap(): Promise<UmapData | null> {
  const resp = await fetch(API_BASE ? '/umap.json' : '/api/umap');
  return resp.ok ? resp.json() : null;
}

export async function getVariant(id: string): Promise<Variant> {
  await _globalReady;  // ensure heads.json is loaded before normalizing
  const resp = await fetch(api(`/variants/${encodeURIComponent(id)}`));
  if (!resp.ok) throw new Error(`Variant not found: ${id}`);
  return normalizeVariant(await resp.json());
}

export async function search(query: string): Promise<SearchResult[]> {
  const q = query.trim();
  if (q.length < 2) return [];
  const resp = await fetch(api(`/variants/search?q=${encodeURIComponent(q)}`));
  return resp.ok ? resp.json() : [];
}

export function fetchInterpretation(
  variantId: string,
  signal: AbortSignal,
  onResult: (interp: Interpretation) => void,
  onLoading: () => void,
  onError: () => void,
): void {
  const url = api(`/variants/${encodeURIComponent(variantId)}/analysis`);
  onLoading();
  let attempt = 0;
  function poll() {
    if (signal.aborted) return;
    attempt++;
    fetch(url, { signal })
      .then(r => {
        if (signal.aborted) return;
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json().then((data: any) => {
          if (data?.status === 'ok' || data?.result) onResult(data.result || data);
          else if ((data?.status === 'processing' || data?.processing_status === 'processing') && attempt < 20) {
            setTimeout(poll, Math.min(2000 * 1.5 ** (attempt - 1), 30000));
          } else onError();
        });
      })
      .catch(e => { if (e.name !== 'AbortError') onError(); });
  }
  poll();
}

// ── Flat row → Variant ──────────────────────────────────────────────
//
// Column convention (the contract):
//   ref_{h}, var_{h}, z_{h}, dist_{h}, spread_{h}  → disruption
//   eff_{h}                                         → effect
//   gt_{h}                                          → ground truth
//   everything else                                 → metadata
//
// _heads (from heads.json) is the schema.

const JSON_FIELDS = ['acmg', 'clinical_features', 'submitters', 'domains', 'neighbors', 'attribution'];

function normalizeVariant(row: Record<string, any>): Variant {
  const disruption: Variant['disruption'] = {};
  const effect: Variant['effect'] = {};
  const gt: Record<string, number> = {};

  for (const [h, info] of Object.entries(_heads)) {
    if (info.category === 'disruption') {
      disruption[h] = {
        ref: row[`ref_${h}`], var: row[`var_${h}`],
        z: row[`z_${h}`],
        dist: row[`dist_${h}`], spread: row[`spread_${h}`],
      };
    } else {
      effect[h] = { value: row[`eff_${h}`] };
    }
    if (info.predictor && row[`gt_${h}`] != null) gt[h] = row[`gt_${h}`];
  }

  for (const f of JSON_FIELDS) {
    if (typeof row[f] === 'string') row[f] = JSON.parse(row[f]);
  }

  return { ...row, disruption, effect, gt } as Variant;
}
