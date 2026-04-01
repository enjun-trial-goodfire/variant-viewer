<script lang="ts">
  import { headColor, deltaColor, barColor } from '../../lib/colors';
  import { extractDelta } from '../../lib/helpers';
  import type { Variant, GlobalData } from '../../lib/types';

  interface Props { variant: Variant; global: GlobalData; }
  let { variant: v, global: g }: Props = $props();

  let showHelp = $state(false);
  let showAllDisruptions = $state(false);

  function displayName(head: string) { return g.display?.[head] || head; }
  function evalBadge(head: string) {
    const e = g.eval?.[head];
    return e ? `${e.metric}=${e.value}` : '';
  }

  function computeZ(head: string, delta: number) {
    const stats = g.head_stats?.[head];
    if (stats?.std && stats.std > 0) return (delta - stats.mean) / stats.std;
    return 0;
  }

  function navigate(path: string) { location.hash = `#/${path}`; }
</script>

<div class="card">
  <button class="card-help-btn" onclick={() => showHelp = !showHelp}>?</button>
  {#if showHelp}
    <div class="card-help open">
      <div class="card-help-inner">
        <b>Disruption Profile.</b> Each row shows a biological feature predicted by an Evo2 probe head.
        Left bars show ref (faded) and var (solid) predictions. Right bar shows delta with z-score.
      </div>
    </div>
  {/if}

  <div class="section-title">Top Disruptions</div>
  <div class="profile-legend" style="display:block">
    <div>Left: ref (faded) / var (solid) &nbsp;|&nbsp; Color = likelihood ratio:
      <span style="color:#c55">&#9632;</span> pathogenic
      <span style="color:#DB8A48">&#9632;</span> leaning path.
      <span style="color:#bbb">&#9632;</span> neutral
      <span style="color:#6ac">&#9632;</span> leaning benign
      <span style="color:#27a">&#9632;</span> benign
    </div>
    <div>Right: &Delta; (var &minus; ref) &nbsp;|&nbsp;
      <span style="color:var(--negative)">&#9632;</span> decreased (&minus;)
      <span style="color:var(--positive)">&#9632;</span> increased (+)
    </div>
  </div>

  {#each v.attribution as item}
    {@const d = extractDelta(v.disruption?.[item.name])}
    {@const z = Math.abs(item.z || 0)}
    {@const sign = d.delta < 0 ? -1 : 1}
    {@const dc = deltaColor(z, sign)}
    {@const refColor = headColor(item.name, d.ref, false, g.distributions)}
    {@const varColor = headColor(item.name, d.var, false, g.distributions)}
    {@const zBarW = Math.min(z / 4 * 50, 50)}
    {@const pctLabel = z > 0 ? `${d.delta >= 0 ? '+' : '\u2212'}${z.toFixed(1)}\u03C3` : ''}
    <div class="profile-row">
      <div class="profile-label">
        {displayName(item.name)}
        {#if evalBadge(item.name)}
          <span style="font-size:9px;color:var(--text-muted)"> {evalBadge(item.name)}</span>
        {/if}
      </div>
      <div style="display:flex;flex-direction:column;gap:1px;flex:1;background:#f0ede8;border-radius:2px;padding:1px">
        <div style="height:7px;border-radius:2px;width:{d.ref > 0 ? Math.max(2, d.ref*100) : 0}%;background:{refColor};opacity:0.5"></div>
        <div style="height:7px;border-radius:2px;width:{d.var > 0 ? Math.max(2, d.var*100) : 0}%;background:{varColor}"></div>
      </div>
      <div style="flex:1;height:16px;position:relative;margin:0 4px;background:#f5f2ee;border-radius:3px;overflow:hidden">
        <div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#ddd"></div>
        <div style="position:absolute;top:3px;height:10px;border-radius:2px;{d.delta >= 0 ? `left:50%;width:${zBarW}%` : `right:50%;width:${zBarW}%`};background:{dc.bar}"></div>
      </div>
      <div class="profile-value" style="color:{dc.text};font-weight:600">{d.delta > 0 ? '+' : ''}{d.delta.toFixed(3)}</div>
      <div style="width:40px;text-align:right;font-family:monospace;font-size:10px;color:{dc.text};flex-shrink:0">{pctLabel}</div>
    </div>
  {/each}

  <!-- Expandable all disruptions -->
  {#if showAllDisruptions}
    <div class="section-title" style="margin-top:16px">All Disruptions</div>
    {#each Object.entries(g.heads?.disruption || {}) as [groupName, keys]}
      {@const items = keys.filter(k => v.disruption?.[k] != null).map(k => {
        const d = v.disruption[k];
        const delta = Array.isArray(d) ? d[1] - d[0] : d;
        const zAbs = Math.abs(computeZ(k, delta));
        return { key: k, delta, ref: Array.isArray(d) ? d[0] : 0, var: Array.isArray(d) ? d[1] : 0, zAbs };
      }).sort((a, b) => b.zAbs - a.zAbs)}
      {#if items.length}
        <div class="profile-group">
          <div class="profile-group-title">{groupName}</div>
          {#each items as item}
            {@const z = item.zAbs}
            {@const sign = item.delta < 0 ? -1 : 1}
            {@const dc = deltaColor(z, sign)}
            {@const refColor = headColor(item.key, item.ref, false, g.distributions)}
            {@const varColor = headColor(item.key, item.var, false, g.distributions)}
            {@const zBarW = Math.min(z / 4 * 50, 50)}
            {@const pctLabel = z > 0 ? `${item.delta >= 0 ? '+' : '\u2212'}${z.toFixed(1)}\u03C3` : ''}
            <div class="profile-row">
              <div class="profile-label">{displayName(item.key)}</div>
              <div style="display:flex;flex-direction:column;gap:1px;flex:1;background:#f0ede8;border-radius:2px;padding:1px">
                <div style="height:7px;border-radius:2px;width:{item.ref > 0 ? Math.max(2,item.ref*100) : 0}%;background:{refColor};opacity:0.5"></div>
                <div style="height:7px;border-radius:2px;width:{item.var > 0 ? Math.max(2,item.var*100) : 0}%;background:{varColor}"></div>
              </div>
              <div style="flex:1;height:16px;position:relative;margin:0 4px;background:#f5f2ee;border-radius:3px;overflow:hidden">
                <div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#ddd"></div>
                <div style="position:absolute;top:3px;height:10px;border-radius:2px;{item.delta >= 0 ? `left:50%;width:${zBarW}%` : `right:50%;width:${zBarW}%`};background:{dc.bar}"></div>
              </div>
              <div class="profile-value" style="color:{dc.text};font-weight:600">{item.delta > 0 ? '+' : ''}{item.delta.toFixed(3)}</div>
              <div style="width:40px;text-align:right;font-family:monospace;font-size:10px;color:{dc.text};flex-shrink:0">{pctLabel}</div>
            </div>
          {/each}
        </div>
      {/if}
    {/each}

    <div class="section-title" style="margin-top:16px">Variant Effects</div>
    <div class="profile-legend">Variant-level effect predictions (from diff view).</div>
    {#each Object.entries(g.heads?.effect || {}) as [groupName, keys]}
      {@const items = keys.filter(k => v.effect?.[k] != null && !k.startsWith('pfam_')).map(k => ({key: k, val: v.effect[k]})).sort((a, b) => Math.abs(b.val) - Math.abs(a.val))}
      {#if items.length}
        <div class="profile-group">
          <div class="profile-group-title">{groupName}</div>
          {#each items as item}
            {@const color = headColor(item.key, item.val, false, g.distributions)}
            <div class="profile-row">
              <div class="profile-label">{displayName(item.key)}</div>
              <div class="profile-bar-container">
                <div class="profile-bar" style="width:{Math.max(2, item.val * 100)}%;background:{color}"></div>
              </div>
              <div class="profile-value">{item.val.toFixed(3)}</div>
            </div>
          {/each}
        </div>
      {/if}
    {/each}
  {/if}

  <div class="show-more" onclick={() => showAllDisruptions = !showAllDisruptions}>
    {showAllDisruptions ? 'Hide all disruptions' : 'Show all disruptions'}
  </div>
</div>
