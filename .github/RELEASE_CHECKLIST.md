# Release checklist (alpha)

Pre-release sanity for a tagged RedHop version.

## Verify

- [ ] `cargo build --workspace` and `cargo test --workspace` pass.
- [ ] `cd python && maturin develop --release --features files,semantic && python -m pytest tests/` passes. The `files,semantic` features mirror the published wheel; without them the loader_errors and rerank tests can't exercise the right surface.
- [ ] `cd nodejs && npm run build && npm test` passes.
- [ ] Examples run: `python python/examples/{basic_rag,compare_strategies,economics_demo}.py`.
- [ ] Hermetic benchmark regenerates: `cargo run -p redhop-examples --example bench_context_strategies --release`.
- [ ] CLI smoke: `redhop compare`, `analyze-context`, `benchmark`, `report`.
- [ ] No stale `NeoRAG`/`neorag` branding (except the preserved `neotrace` wire format and lab-repo paths).
- [ ] README and in-repo doc links resolve.
- [ ] `README` / `python/README` examples are accurate against the current API.

## Version bump

- [ ] Workspace `version` in root `Cargo.toml`.
- [ ] `python/pyproject.toml` version (and `python/Cargo.toml`).
- [ ] `nodejs/package.json` version.
- [ ] Note changes in the changelog.

## Publish (when ready)

Publishing is tag-driven — push a `v<version>` tag and three workflows
fire in parallel:

- [ ] `release-crates.yml` — `cargo publish -p redhop` (single
  consolidated crate; no in-order multi-crate publish anymore).
- [ ] `release-python.yml` — `maturin build --release --features
  semantic,files` per platform, then `maturin publish` to PyPI.
- [ ] `release-node.yml` — `napi build --release` per target in
  `napi.triples`, then `npm publish` the meta + per-platform packages.
- [ ] `create-release.yml` — drafts the GitHub Release from CHANGELOG.

Pre-tag manual checks:

- [ ] Tag matches all three version fields above.
- [ ] CI is green on the commit being tagged (`gh run list --workflow CI`).
- [ ] Update the docs website (separate repo).
