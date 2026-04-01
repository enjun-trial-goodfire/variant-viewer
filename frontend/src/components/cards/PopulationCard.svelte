<script lang="ts">
  import HelpCard from '../HelpCard.svelte';
  import type { Variant } from '../../lib/types';

  interface Props { variant: Variant; }
  let { variant: v }: Props = $props();

  const pops: [string, string][] = [
    ['afr', 'African'], ['amr', 'American'], ['asj', 'Ashkenazi'],
    ['eas', 'East Asian'], ['fin', 'Finnish'], ['nfe', 'European'], ['sas', 'South Asian'],
  ];

  const maxAF = $derived(Math.max(...pops.map(([k]) => v.gnomad_pop[k] || 0), 1e-6));
</script>

<HelpCard
  title="Population Frequency"
  helpText="<b>Population Frequency.</b> gnomAD exome allele frequencies across ancestry groups. Variants absent from gnomAD are more likely to be rare and potentially pathogenic."
>
  <div style="font-size:13px;margin-bottom:10px">
    gnomAD exome: <b>{v.gnomad_display}</b>
    {#if v.gnomad_label}
      · <span style="color:var(--text-muted)">{v.gnomad_label}</span>
    {/if}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px;font-size:12px">
    {#each pops as [k, label]}
      {@const f = v.gnomad_pop[k]}
      {@const w = f && f > 0 ? Math.max(2, (f / maxAF) * 100) : 0}
      {@const fStr = f && f > 0 ? (f >= 0.01 ? `${(f * 100).toFixed(1)}%` : `1 in ${Math.round(1/f).toLocaleString()}`) : '—'}
      <div style="display:flex;align-items:center;gap:6px">
        <span style="width:70px;color:var(--text-secondary);flex-shrink:0">{label}</span>
        <span style="flex:1;height:8px;background:var(--bg-track);border-radius:3px;overflow:hidden">
          <span style="display:block;height:100%;width:{w}%;background:var(--accent);border-radius:3px"></span>
        </span>
        <span style="width:90px;text-align:right;font-family:monospace;font-size:10px;color:var(--text-muted);white-space:nowrap">{fStr}</span>
      </div>
    {/each}
  </div>
</HelpCard>
