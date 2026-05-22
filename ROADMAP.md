# Roadmap

RedHop is **alpha**. The architecture, APIs, findings, CLI, Python bindings,
and packaging are stable enough for public visibility and first external users.
This roadmap is deliberately bounded — RedHop stays a reasoning-preserving
context optimization library, not a framework.

## Done

- **Core context API** — `build_context` / `filter_context` / `analyze_context`
  / `context_economics` with the `ContextReport` telemetry.
- **Strategies** — `raw_topk`, `distractor_filtered`, `max_density`,
  `redundancy_pruned`, and the default `reasoning_preserving`.
- **Evidence layer** — `docs/findings/` (with a falsified-hypotheses registry),
  reproducible `benchmarks/`, captured `reports/`.
- **CLI** — `redhop compare` / `analyze-context` / `benchmark` / `report`.
- **Python bindings** — pyo3 + maturin, abi3 wheel (`import redhop`).
- **Rename + packaging hygiene** — clean RedHop naming, publish metadata.

## Next (productization, not research)

- **Publish** — PyPI alpha (`pip install redhop`) and crates.io for the core
  crates; CI multi-platform wheel builds (cibuildwheel / maturin-action).
- **Docs site** — host the mdBook (`docs/book/`).
- **npm bindings** — `napi-rs` wrapper over the same context API.
- **Enterprise PDF demo** — extraction → ingestion diagnostics (OCR noise /
  duplicates / fragmentation) → context optimization → economics report.

## Research frontier (measurement-gated, not speculative)

These extend the *signal*, not the architecture class:

- **Semantic-linkage rescue** — embedding similarity instead of lexical Jaccard
  for the bridge link, to rescue paraphrase-linked second hops.
- **Larger / more datasets** — single-hop vs multi-hop splits; non-HotpotQA
  workloads; cross-generator end-to-end runs.
- **Full-gold-retention labeling** at scale (already prototyped in the n=300 run).

Each must arrive with a finding doc (hypothesis, setup, metrics, caveats,
reproduce command, verdict) before it changes a default.

## Explicitly out of scope

Agents, planners, workflow/orchestration DAGs, graph traversal, query
decomposition, embedded LLMs or vector DBs, RL controllers. RedHop composes
*under* your stack; it does not replace it.
