<script lang="ts">
  import { barColor } from '../lib/colors';

  interface Props {
    label: string;
    value: number;
    lr?: number;
    color?: string;
    suffix?: string;
  }
  let { label, value, lr, color, suffix }: Props = $props();

  const barBg = $derived(color ?? barColor(lr ?? 0.5));
  const barWidth = $derived(Math.max(2, value * 100));
</script>

<div class="profile-row">
  <div class="profile-label">{label}</div>
  <div class="profile-bar-container">
    <div class="profile-bar" style="width:{barWidth}%;background:{barBg}"></div>
  </div>
  <div class="profile-value" style="color:{barBg};font-weight:600">
    {(value * 100).toFixed(0)}%{#if suffix}{@html suffix}{/if}
  </div>
</div>
