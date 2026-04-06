/**
 * Build hg38 Genome Browser URLs with a minimal track set for EVEE embeds.
 *
 * UCSC URL reference: https://genome.ucsc.edu/goldenPath/help/hgTracksHelp.html
 *   - hideTracks=1 then trackName=pack|dense turns on only those tracks (see “Optional URL parameters”).
 *   - ignoreCookie=1 avoids merging the user’s saved cart (keeps the view predictable in iframes).
 *
 * Track symbolic names match hg38 Track Controls / Table Browser “primary table” where possible.
 * UCSC renames or retires tracks between releases; verify in the browser if a track disappears.
 */
export function buildUcscHg38ContextUrl(
  chrom: string,
  start: number,
  end: number,
  apiKey?: string,
): string {
  const q = new URLSearchParams();

  q.set('db', 'hg38');
  q.set('position', `${chrom}:${start}-${end}`);

  // Deterministic view: do not blend in the visitor’s last session tracks.
  q.set('ignoreCookie', '1');

  // Global hide, then re-enable only the tracks we want (UCSC docs; not “allTracks=”).
  q.set('hideTracks', '1');

  // Less visual noise in the image (documented optional params).
  q.set('guidelines', 'off');

  // Remove the entire track-control strip under the graphic (collapsed group headers, visibility menus,
  // “Collapse all” / “Expand all”, per-group rows). Cart variable used by UCSC kent hgTracks.c:
  //   showTrackControls = cartUsualBoolean(cart, "trackControlsOnMain", TRUE);
  // Same effect as Track Configuration → Configure Image → uncheck “Show track controls under main graphic”
  // (hgTracksHelp.html). Value "0" is interpreted as false by the cart.
  //
  // Not the same as clicking “Collapse all track groups” (button hgt.collapseGroups): that only folds
  // groups via per-group vars like hgt.group_<name>_close, with no single stable public URL for all groups.
  q.set('trackControlsOnMain', '0');

  // --- RefSeq Genes (NCBI) -----------------------------------------------
  // Composite + subtrack selection (hgTracksHelp example: refSeqComposite=full&refGene_sel=1).
  q.set('refSeqComposite', 'pack');
  q.set('refGene_sel', '1');

  // --- GENCODE / gene catalog --------------------------------------------
  // “knownGene” is the classic UCSC Genes track on hg38, built from GENCODE (Basic set historically).
  // A separately labeled “GENCODE Basic” hub track may use wgEncodeGencodeBasicV* (version changes).
  q.set('knownGene', 'pack');

  // --- Common dbSNP -------------------------------------------------------
  // Requested name: snp151Common. If UCSC upgrades the assembly default (e.g. dbSNP 153+), update here.
  q.set('snp151Common', 'dense');

  // --- 100 vertebrates (multiz alignment) --------------------------------
  // “Vertebrate Multiz Alignments (100 species)” → table/track id multiz100way on hg38.
  q.set('multiz100way', 'dense');

  // Chromosome ideogram (cytoBand): hide to reduce vertical clutter in small iframes.
  q.set('cytoBand', 'hide');

  if (apiKey) q.set('apiKey', apiKey);

  return `https://genome.ucsc.edu/cgi-bin/hgTracks?${q.toString()}`;
}
