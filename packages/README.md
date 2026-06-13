# packages/

Reserved for shared workspace packages (TypeScript libraries consumed by
`apps/*`, and later the published `@astel/sdk`). Empty at M1; the pnpm
workspace already globs `packages/*` so a new package here is picked up
automatically.

Planned tenants:
- `@astel/manifest` — TS types + reader/writer for the `.astel` package format
  (mirrors `docs/specs/schemas/`).
- `@astel/sdk` — the public TypeScript client (REST + SSE), shipped to npm in M5.
