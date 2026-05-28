# API Stability

RedHop is **0.x (alpha)**. This document states what you can rely on, what may
change, and the boundary the project intentionally keeps.

## What RedHop is (and is not)

RedHop is a **reasoning-preserving context runtime** that sits between your documents
and an LLM. It owns chunking, internal retrieval, context allocation,
reasoning-safe optimization, observability, and token economics.

It is **not** a retriever you wire up, a vector database, an agent/workflow/
orchestration framework, or a PDF parser. You bring text (from PyMuPDF, Marker,
Unstructured, your own OCR, …); RedHop owns everything from chunking onward. That
narrow scope is deliberate and will be preserved.

## Stable surface

These are the supported entry points. Within 0.x we avoid breaking them, and any
breaking change is called out in [CHANGELOG.md](../CHANGELOG.md).

**Python (`redhop`)**
- `Document.from_text(text, chunk_size=…, chunk_overlap=…, strategy=…, …)` and
  `Document.from_chunks(chunks, …)` — `chunk_size`/`chunk_overlap` are index-time
- `Document.context(query, budget=…)` — `budget` is a query-time override (no
  re-indexing); `Document.analyze(query)`, `Document.n_chunks`
- `build_context(query, retrieved_chunks, strategy=..., token_budget=..., ...)`
- `filter_context(...)`, `analyze_context(...)`, `context_economics(...)`
- `grounding_score(query, text)`, `link_strength(a, b)`
- `ContextReport` getters (`strategy`, `requested_strategy`, `auto_decision`,
  `input_tokens`, `total_tokens`, `n_input_chunks`, `n_selected`,
  `input_distractor_ratio`, `evidence_density`, `retained_evidence_ratio`,
  `second_hop_rescue_count`, `estimated_waste_tokens`, …) and `BuiltContext`
  (`.text()`, `.chunks`, `.report`).

**Rust**
- `redhop_document::{Document, DocumentConfig}`
- `redhop_context::{build_context, filter_context, analyze_context,
  context_economics, ContextConfig, ContextStrategy, ContextReport,
  BuiltContext, AutoDecision, grounding_score, link_strength}`

**Stable semantics**
- `ContextStrategy` string/serde names (`raw_topk`, `distractor_filtered`,
  `redundancy_pruned`, `max_density`, `reasoning_preserving`, `auto`).
- `AutoDecision` values (`passthrough`, `prune`, `not_auto`).
- The structured `ContextReport` fields above (parse these for telemetry).

## Experimental / may change without notice

- **Internal retrieval** — `redhop-retrieval` (`Bm25Retriever`, dense, hybrid),
  `redhop-pipeline`, `redhop-diagnostics`, `redhop-orchestration`, and the other
  lower crates are implementation detail, not the public product API. Retrieval
  is intentionally **not** surfaced through `Document`; do not depend on it.
- **Default values** are evidence-tuned and may shift as measurements improve —
  e.g. `auto_passthrough_max_tokens` (the dilution gate), `distractor_min_grounding`,
  `link_min_jaccard`, `candidate_k`, chunk sizes. Pin them explicitly if you need
  stability. Each default traces to a finding in [docs/findings/](findings/).
- **The rendered report text** (`str(report)` / the "RedHop Decision Report"
  layout) is for humans and may be reworded. For programmatic use, read the
  structured fields / `auto_decision`, not the string.
- The `NeoTrace` wire format (`neotrace/1`) is versioned separately and is not
  governed by this document.

## Versioning

- **0.x:** minor releases may contain breaking changes; they will be documented
  in the changelog. Prefer pinning a version.
- **≥1.0:** semantic versioning — breaking changes only in major releases, with
  deprecation notices first.

## A note on claims

RedHop's defaults exist because a specific failure was *measured* (see the
[evidence layer](findings/README.md)). We avoid SOTA/"solves reasoning" framing:
the value is measured behavior, conservative defaults, and an interpretable,
observable runtime. Operational claims are benchmark-backed or they are not made.
