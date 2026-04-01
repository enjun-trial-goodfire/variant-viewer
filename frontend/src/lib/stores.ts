import { writable } from 'svelte/store';
import type { GlobalData, UmapData, Variant } from './types';

export const globalData = writable<GlobalData | null>(null);
export const umapData = writable<UmapData | null>(null);
export const currentVariant = writable<Variant | null>(null);
