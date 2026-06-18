import { Outlet } from "react-router-dom";

import { SiteFooter } from "./SiteFooter.tsx";
import { SiteHeader } from "./SiteHeader.tsx";

export function SiteLayout(): React.JSX.Element {
  return (
    <div className="site">
      <SiteHeader />
      <main className="site-main">
        <Outlet />
      </main>
      <SiteFooter />
    </div>
  );
}
