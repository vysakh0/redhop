# Architecture

RedHop is a small Rust workspace with thin language bindings and an evidence
layer. Rust is the source of truth; everything else wraps it.

```text
            your retriever                         your LLM
                  │                                    ▲
                  ▼                                    │
   ┌─────────────────────────────────────────────────────────┐
   │  redhop-context   build_context / filter_context /        │
   │                   analyze_context / context_economics      │
   │                   → ContextReport (observability)          │
   └─────────────────────────────────────────────────────────┘
        ▲                 ▲                      ▲
        │                 │                      │
   redhop (Python,    redhop CLI            redhop-core
   pyo3/maturin)      (compare/analyze/     (traits, types:
                       benchmark/report)     Chunk, Query, …)
```

## Crates

- **`redhop-core`** — traits and data types (`Chunk`, `Query`, `RetrievalResult`,
  …). No logic, just the shared vocabulary.
- **`redhop-context`** — the heart: context assembly, the strategies, the
  `ContextReport`. `#![forbid(unsafe_code)]`, no async, minimal deps.
- **`redhop-cli`** — the `redhop` binary (compare/analyze/benchmark/report).
- **Supporting crates** — `chunking`, `retrieval`, `reranking`, `diagnostics`,
  `storage`, `embeddings` (feature-gated ONNX), `calibration`, `pipeline`. These
  back the research/benchmark harnesses and are optional to the context path.

## Bindings

- **Python** (`python/`) — pyo3 + maturin, abi3 wheel. A thin wrapper over
  `redhop-context`; no logic duplicated.
- **CLI** — wraps the same public functions; JSON in, human/JSON out.

## Evidence layer

- **`docs/findings/`** — hypothesis → result → mechanism per finding, with a
  falsified-hypotheses registry.
- **`benchmarks/`** — the reproducible harnesses that regenerate every claim.
- **`reports/`** — captured raw run outputs.

## Design rules

- The default build is offline and lightweight; heavyweight backends (ONNX,
  tokenizers) are feature-gated.
- Defaults change only with a measured finding.
- No agents, planners, workflows, graph traversal, or embedded models.
