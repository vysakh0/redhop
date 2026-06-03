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
- [ ] Note changes in the changelog.

## Publish (when ready)

- [ ] `cargo publish` core crates in dependency order (`redhop-core` → `redhop-context` → …).
- [ ] Build wheels for the target platforms (cibuildwheel / maturin-action) and `maturin publish` to PyPI.
- [ ] Tag the release; update the docs website.
