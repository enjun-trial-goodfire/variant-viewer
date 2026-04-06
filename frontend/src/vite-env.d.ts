/// <reference types="svelte" />
/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** UCSC Genome Browser — append to hgTracks URLs to bypass CAPTCHA on programmatic/embed loads (private deploys only; exposed in client bundle). */
  readonly VITE_UCSC_API_KEY?: string;
}
