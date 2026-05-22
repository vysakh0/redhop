# Release checklist (alpha)

Pre-release sanity for a tagged RedHop version.

## Verify

- [ ] `cargo build --workspace` and `cargo test --workspace` pass.
- [ ] `cd python && maturin develop --release && python -m pytest tests/` passes.
- [ ] Examples run: `python python/examples/{basic_rag,compare_strategies,economics_demo}.py`.
- [ ] Hermetic benchmark regenerates: `cargo run -p redhop-examples --example bench_context_strategies --release`.
- [ ] CLI smoke: `redhop compare`, `analyze-context`, `benchmark`, `report`.
- [ ] No stale `NeoRAG`/`neorag` branding (except the preserved `neotrace` wire format and lab-repo paths).
- [ ] Docs build: `mdbook build docs/book`; internal links resolve.
- [ ] `README` / `python/README` examples are accurate against the current API.

## Version bump

- [ ] Workspace `version` in root `Cargo.toml`.
- [ ] `python/pyproject.toml` version (and `python/Cargo.toml`).
- [ ] Note changes in `ROADMAP.md` / changelog.

## Publish (when ready)

- [ ] `cargo publish` core crates in dependency order (`redhop-core` → `redhop-context` → …).
- [ ] Build wheels for the target platforms (cibuildwheel / maturin-action) and `maturin publish` to PyPI.
- [ ] Tag the release; attach the mdBook build or publish the docs site.
