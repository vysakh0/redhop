# redhop-context

The core of [RedHop](https://github.com/vysakh0/redhop): reasoning-preserving
context assembly under a finite token budget, with first-class observability.

Given a query and the chunks a retriever returned, it builds the prompt context
a downstream LLM sees — pruning distractors, **preserving reasoning-critical
"second-hop" evidence**, and reporting exactly what it did. Not a retriever, not
a reranker, not a framework.

```rust
use redhop_context::{build_context, ContextConfig};
use redhop_core::Query;

let ctx = build_context(
    &query,
    &retrieved,                       // &[redhop_core::RetrievalResult]
    &ContextConfig { token_budget: 12_000, ..Default::default() },  // default = ReasoningPreserving
);
let prompt = ctx.text();
println!("{}", ctx.report.render(None));   // Context Optimization Report
```

## API

- `build_context` — budget-aware assembly → `BuiltContext`.
- `filter_context` — filter junk, no budget truncation.
- `analyze_context` — non-destructive diagnostics → `ContextReport`.
- `context_economics` — economics of a chunk set as-is.

Strategies: `reasoning_preserving` (default), `distractor_filtered`,
`max_density`, `redundancy_pruned`, `raw_topk`.

## Why the default is `reasoning_preserving`

On multi-hop questions the second hop is low-relevance-to-query by construction,
so relevance-based pruning drops it (the "second-hop tax"). This crate's default
keeps query-relevant seeds *and* rescues low-relevance chunks linked to a seed,
dropping only unlinked junk. Measured, with CIs, in the
[evidence layer](https://github.com/vysakh0/redhop/tree/main/docs/findings).

`#![forbid(unsafe_code)]`, no async; the default build pulls only `serde` and
`unicode-segmentation`. Apache-2.0.
