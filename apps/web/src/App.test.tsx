import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { AppRoutes } from "./App.tsx";
import { SITE_ROUTES } from "./lib/siteMap.ts";

describe("site map", () => {
  // WebGL pages mount the splat viewer, which cannot render under jsdom; they
  // are covered by live smoke instead.
  for (const route of SITE_ROUTES.filter((r) => !r.webgl)) {
    it(`renders ${route.path} (${route.id})`, () => {
      render(
        <MemoryRouter initialEntries={[route.path]}>
          <AppRoutes />
        </MemoryRouter>,
      );
      expect(
        document.querySelector(`[data-page="${route.id}"]`),
      ).not.toBeNull();
    });
  }

  it("renders the not-found page for unknown routes", () => {
    render(
      <MemoryRouter initialEntries={["/no-such-page"]}>
        <AppRoutes />
      </MemoryRouter>,
    );
    expect(document.querySelector('[data-page="not-found"]')).not.toBeNull();
  });
});
