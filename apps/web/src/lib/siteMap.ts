/** Static route manifest: the single source of truth for the router, the
 *  header nav, the footer, and the site-map test. Dynamic routes (/gallery/:id)
 *  and the not-found route are handled separately in App.tsx. */
export interface SiteRoute {
  path: string;
  id: string; // matches the page root's data-page attribute
  label: string;
  inNav: boolean; // show in the header primary nav
  inFooter: boolean;
  webgl: boolean; // mounts the 3D splat viewer (cannot render under jsdom)
}

export const SITE_ROUTES: SiteRoute[] = [
  { path: "/", id: "home", label: "Home", inNav: false, inFooter: true, webgl: true },
  { path: "/features", id: "features", label: "Features", inNav: true, inFooter: true, webgl: false },
  { path: "/features/layer-inspector", id: "feature-layer-inspector", label: "Layer Inspector", inNav: false, inFooter: false, webgl: false },
  { path: "/features/truth-meter", id: "feature-truth-meter", label: "Truth Meter", inNav: false, inFooter: false, webgl: false },
  { path: "/features/physics-sandbox", id: "feature-physics-sandbox", label: "Physics Sandbox", inNav: false, inFooter: false, webgl: true },
  { path: "/features/relight-studio", id: "feature-relight-studio", label: "Relight Studio", inNav: false, inFooter: false, webgl: true },
  { path: "/how-it-works", id: "pipeline", label: "How it works", inNav: true, inFooter: true, webgl: false },
  { path: "/gallery", id: "gallery", label: "Gallery", inNav: true, inFooter: true, webgl: false },
  { path: "/pricing", id: "pricing", label: "Pricing", inNav: true, inFooter: true, webgl: false },
  { path: "/docs", id: "docs", label: "Docs", inNav: true, inFooter: true, webgl: false },
  { path: "/self-host", id: "self-host", label: "Self-host", inNav: false, inFooter: true, webgl: false },
  { path: "/studio", id: "studio", label: "Open Studio", inNav: false, inFooter: false, webgl: true },
];
