<script lang="ts">
  interface Props {
    color: string;
    label?: string;
  }
  let { color, label }: Props = $props();
</script>

<span class="swatch" style="background:{color}">
  {#if label}
    <span class="tip">{label}</span>
  {/if}
</span>

<style>
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    cursor: default;
    position: relative;
  }
  .swatch:hover {
    transform: scale(1.3);
  }
  .tip {
    display: none;
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(30, 30, 30, 0.92);
    color: #fff;
    font-size: 10px;
    white-space: nowrap;
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
    z-index: 100;
  }
  .swatch:hover .tip {
    display: block;
  }
</style>
