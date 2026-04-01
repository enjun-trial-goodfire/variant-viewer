// UMAP stores integer label indices for compactness — this maps them back.
export const LABEL_NAMES = ['benign', 'pathogenic', 'VUS'];

export function truncate(s: string, n = 30): string {
  return s.length > n ? s.slice(0, n - 2) + '\u2026' : s;
}


export function labelClass(label: string): string {
  if (label.includes('pathogenic')) return 'label-pathogenic';
  if (label.includes('benign')) return 'label-benign';
  return 'label-vus';
}

export function navigate(path: string): void {
  const target = path ? `#/${path}` : '#/';
  if (location.hash === target) {
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  } else {
    location.hash = target;
  }
}
