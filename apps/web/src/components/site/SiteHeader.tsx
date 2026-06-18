import { useState } from "react";
import { Link, NavLink } from "react-router-dom";

import { SITE_ROUTES } from "../../lib/siteMap.ts";

export function SiteHeader(): React.JSX.Element {
  const navRoutes = SITE_ROUTES.filter((r) => r.inNav);
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => setMenuOpen((v) => !v);
  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="site-header">
      <Link to="/" className="brand" onClick={closeMenu}>
        <img src="/favicon.svg" alt="" aria-hidden />
        <span className="brand__word">Astel</span>
      </Link>

      {/* Desktop nav */}
      <nav className="site-header__nav" aria-label="Primary navigation">
        {navRoutes.map((route) => (
          <NavLink key={route.path} to={route.path}>
            {route.label}
          </NavLink>
        ))}
      </nav>

      <div className="site-header__actions">
        <Link className="site-cta" to="/studio">
          Open Studio
        </Link>
        {/* Mobile hamburger */}
        <button
          type="button"
          className="site-header__burger"
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={menuOpen}
          onClick={toggleMenu}
        >
          <span className="site-header__burger-bar" />
          <span className="site-header__burger-bar" />
          <span className="site-header__burger-bar" />
        </button>
      </div>

      {/* Mobile drawer */}
      {menuOpen && (
        <nav
          className="site-header__mobile-nav"
          aria-label="Mobile navigation"
        >
          {navRoutes.map((route) => (
            <NavLink key={route.path} to={route.path} onClick={closeMenu}>
              {route.label}
            </NavLink>
          ))}
          <Link className="site-cta site-cta--mobile" to="/studio" onClick={closeMenu}>
            Open Studio
          </Link>
        </nav>
      )}
    </header>
  );
}
