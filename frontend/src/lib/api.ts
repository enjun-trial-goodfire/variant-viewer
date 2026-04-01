import type { GlobalData, Interpretation, SearchResult, UmapData, Variant } from './types';

export async function getGlobal(): Promise<GlobalData> {
  return (await fetch('/api/global')).json();
}

export async function getUmap(): Promise<UmapData | null> {
  const resp = await fetch('/api/umap');
  return resp.ok ? resp.json() : null;
}

export async function getVariant(id: string): Promise<Variant> {
  const resp = await fetch(`/api/variants/${encodeURIComponent(id)}`);
  if (!resp.ok) throw new Error(`Variant not found: ${id}`);
  return normalizeVariant(await resp.json());
}

export async function search(query: string): Promise<SearchResult[]> {
  const q = query.trim();
  if (q.length < 2) return [];
  const resp = await fetch(`/api/variants/search?q=${encodeURIComponent(q)}`);
  return resp.ok ? resp.json() : [];
}

export function fetchInterpretation(
  variantId: string,
  signal: AbortSignal,
  onResult: (interp: Interpretation) => void,
  onLoading: () => void,
  onError: () => void,
): void {
  const url = `/api/interpret/${encodeURIComponent(variantId)}`;
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
          if (data?.status === 'ok') onResult(data);
          else if (data?.status === 'processing' && attempt < 20) {
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
// The DB stores precomputed per-variant-per-head values:
//   ref_score_{h}, var_score_{h}  → disruption ref/var
//   z_{h}                         → precomputed z-score
//   ref_lr_{h}, var_lr_{h}        → precomputed likelihood ratios
//   score_{h}                     → effect value
//   lr_{h}                        → precomputed effect likelihood ratio
//   gt_{h}                        → ground truth

const JSON_FIELDS = new Set(['acmg', 'clinical_features', 'submitters', 'domains', 'neighbors', 'attribution']);
const GNOMAD_POP_COLS = new Set(['gnomad_afr_af', 'gnomad_amr_af', 'gnomad_asj_af', 'gnomad_eas_af', 'gnomad_fin_af', 'gnomad_nfe_af', 'gnomad_sas_af']);

function normalizeVariant(row: Record<string, any>): Variant {
  const disruption: Variant['disruption'] = {};
  const effect: Variant['effect'] = {};
  const gt: Record<string, number> = {};
  const gnomad_pop: Record<string, number> = {};
  const meta: Record<string, any> = {};

  // First pass: collect ref_score values to know which disruption heads exist
  for (const [k, v] of Object.entries(row)) {
    if (v == null) continue;
    if (k.startsWith('ref_score_')) {
      const h = k.slice(10);
      const va = row[`var_score_${h}`];
      if (va != null) {
        disruption[h] = {
          ref: v, var: va,
          z: row[`z_${h}`] ?? 0,
          ref_lr: row[`ref_lr_${h}`] ?? 0.5,
          var_lr: row[`var_lr_${h}`] ?? 0.5,
        };
      }
    } else if (k.startsWith('score_') && k !== 'score_pathogenic') {
      const h = k.slice(6);
      effect[h] = { value: v, lr: row[`lr_${h}`] ?? 0.5 };
    } else if (k.startsWith('gt_')) {
      gt[k.slice(3)] = v;
    } else if (GNOMAD_POP_COLS.has(k)) {
      if (v > 0) gnomad_pop[k.slice(7, -3)] = v;
    } else if (JSON_FIELDS.has(k)) {
      meta[k] = typeof v === 'string' ? JSON.parse(v) : v;
    } else if (!k.startsWith('var_score_') && !k.startsWith('z_') && !k.startsWith('ref_lr_') && !k.startsWith('var_lr_') && !k.startsWith('lr_')) {
      meta[k] = v;
    }
  }

  return { ...meta, disruption, effect, gt, gnomad_pop } as Variant;
}
