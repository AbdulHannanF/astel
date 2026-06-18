import { Link } from "react-router-dom";

import { SITE_ROUTES } from "../../lib/siteMap.ts";

export function SiteFooter(): React.JSX.Element {
  const footerRoutes = SITE_ROUTES.filter((r) => r.inFooter);

  return (
    <footer className="site-footer">
      <div className="site-footer__content">
        <div className="site-footer__brand">
          <Link to="/" className="site-footer__logo">
            <span className="brand__word">Astel</span>
          </Link>
          <p className="site-footer__tagline">
            Geometry-accurate, world-aware Gaussian splat assets.
            <br />
            Splats only. Radical honesty. Self-host ready.
          </p>
          <p className="site-footer__copy mono">
            © {new Date().getFullYear()} Astel
          </p>
        </div>

        <nav className="site-footer__nav" aria-label="Footer navigation">
          {footerRoutes.map((route) => (
            <Link key={route.path} to={route.path}>
              {route.label}
            </Link>
          ))}
        </nav>
      </div>
    </footer>
  );
}
