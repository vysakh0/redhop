<!-- Keep PRs focused. See CONTRIBUTING.md. -->

**What & why**
What this changes and the motivation.

**How verified**
- [ ] `cargo test --workspace`
- [ ] Python tests (`cd python && maturin develop --release --features files,semantic && python -m pytest tests/`) — if bindings touched
- [ ] Node tests (`cd nodejs && npm run build && npm test`) — if bindings touched
- [ ] examples / benchmark run — if relevant

**Evidence** (if this changes a default, strategy, or a documented behavior)
Link the finding (`docs/findings/...`) that justifies it. Defaults change only
with measurement.

**Scope check**
- [ ] No new framework surface (agents / workflows / orchestration / graph / decomposition)
- [ ] Rust stays the source of truth (bindings wrap, don't fork logic)
- [ ] No over-claiming; caveats preserved
