# Rust usage

```toml
[dependencies]
redhop-context = "0.1"
redhop-core    = "0.1"
```

```rust
use redhop_context::{build_context, ContextConfig, ContextStrategy};
use redhop_core::Query;

let cfg = ContextConfig {
    token_budget: 12_000,
    strategy: ContextStrategy::ReasoningPreserving,
    ..Default::default()
};
let ctx = build_context(&query, &retrieved, &cfg);

let prompt = ctx.text();      // assembled context
let report = &ctx.report;     // ContextReport telemetry
println!("{}", report.render(None));
```

`retrieved` is a `&[redhop_core::RetrievalResult]` — whatever your retriever
produced. `ContextConfig::default()` uses `ReasoningPreserving` with safe
thresholds.

## Functions

- `build_context(query, retrieved, cfg) -> BuiltContext` — budget-aware.
- `filter_context(query, retrieved, cfg) -> BuiltContext` — no budget cap.
- `analyze_context(query, retrieved, cfg) -> ContextReport` — non-destructive.
- `context_economics(query, retrieved, cfg) -> ContextEconomics`.

`BuiltContext` holds `chunks` and `report`; `report.render(before)` produces the
human-readable Context Optimization Report (pass the `analyze_context` report as
`before` for the token/density deltas).

The crate is `#![forbid(unsafe_code)]`, has no async, and pulls only `serde` +
`unicode-segmentation` — the default build is offline and lightweight.
