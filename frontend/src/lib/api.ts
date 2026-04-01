import type { GlobalData, Interpretation, SearchResult, UmapData, Variant } from './types';

declare global {
  interface Window {
    __APP_CONFIG__?: { API_BASE?: string; STATIC_BASE?: string };
  }
}

const config = {
  API_BASE: window.__APP_CONFIG__?.API_BASE || '',
  STATIC_BASE: window.__APP_CONFIG__?.STATIC_BASE || '',
};

const isApiMode = !!config.API_BASE;

function apiUrl(path: string): string {
  if (isApiMode) return `${config.API_BASE}${path}`;
  return `/api${path}`;
}

export async function getGlobal(): Promise<GlobalData> {
  if (isApiMode) {
    const resp = await fetch(`${config.STATIC_BASE || config.API_BASE}/global.json`);
    return resp.json();
  }
  const resp = await fetch('/api/global');
  return resp.json();
}

export async function getUmap(): Promise<UmapData | null> {
  if (isApiMode) {
    const resp = await fetch(`${config.STATIC_BASE || config.API_BASE}/umap.json`);
    if (!resp.ok) return null;
    return resp.json();
  }
  const resp = await fetch('/api/umap');
  if (!resp.ok) return null;
  const data = await resp.json();
  return data;
}

export async function getVariant(id: string): Promise<Variant> {
  if (isApiMode) {
    const resp = await fetch(`${config.API_BASE}/variants/${encodeURIComponent(id)}`);
    if (!resp.ok) throw new Error(`Variant not found: ${id}`);
    return resp.json();
  }
  const resp = await fetch(`/api/variants/${encodeURIComponent(id)}`);
  if (!resp.ok) throw new Error(`Variant not found: ${id}`);
  return resp.json();
}

export async function search(query: string): Promise<SearchResult[]> {
  const q = query.trim();
  if (q.length < 2) return [];
  if (isApiMode) {
    const resp = await fetch(`${config.API_BASE}/variants/search?q=${encodeURIComponent(q)}`);
    if (!resp.ok) return [];
    return resp.json();
  }
  const resp = await fetch(`/api/variants/search?q=${encodeURIComponent(q)}`);
  if (!resp.ok) return [];
  return resp.json();
}

export function fetchInterpretation(
  variantId: string,
  signal: AbortSignal,
  onResult: (interp: Interpretation) => void,
  onLoading: () => void,
  onError: () => void,
): void {
  const useApi = isApiMode;
  const url = useApi
    ? `${config.API_BASE}/variants/${encodeURIComponent(variantId)}/analysis`
    : `/api/interpret/${encodeURIComponent(variantId)}`;

  onLoading();

  let attempt = 0;
  const maxAttempts = 20;

  function poll() {
    if (signal.aborted) return;
    attempt++;

    fetch(url, { signal })
      .then(r => {
        if (signal.aborted) return null;
        if (!useApi) {
          if (!r.ok) throw new Error(`${r.status}`);
          return r.json().then((interp: Interpretation) => {
            if (interp && interp.status === 'ok') onResult(interp);
            else onError();
          });
        }
        if (r.status === 200) {
          return r.json().then((data: any) => {
            if (data.result) onResult(data.result);
            else if (data.status === 'ok') onResult(data);
            else onError();
          });
        }
        if (r.status === 202) {
          if (attempt >= maxAttempts) { onError(); return; }
          const delay = Math.min(2000 * Math.pow(1.5, attempt - 1), 30000);
          setTimeout(poll, delay);
          return;
        }
        throw new Error(`${r.status}`);
      })
      .catch(e => {
        if (e.name === 'AbortError') return;
        onError();
      });
  }

  poll();
}
