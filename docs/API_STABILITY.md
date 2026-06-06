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
- `Document.from_text(text, source="document", chunk_size=…, chunk_overlap=…,
  strategy=…, language=…, …)`, `Document.from_chunks(chunks, …, language=…)`,
  `Document.from_file(path, …, language=…)`, `Document.from_bytes(data,
  source, …, language=…)`, `Document.from_folder(path, …, language=…)`
  — `chunk_size`/`chunk_overlap` are index-time; `language=` selects
  any of the 18 Snowball Porter2 languages (or errors on unknown names).
- `Document.context(query, budget=…)` — `budget` is a query-time
  override (no re-indexing); `Document.analyze(query)`,
  `Document.n_chunks`, `Document.n_files`, `Document.skipped_files`.
- `build_context(query, retrieved_chunks, strategy=..., token_budget=..., ...)`,
  `filter_context(...)`, `analyze_context(...)`, `context_economics(...)`,
  `grounding_score(query, text)`, `link_strength(a, b)`.
- `ContextReport` getters (`strategy`, `requested_strategy`, `auto_decision`,
  `input_tokens`, `total_tokens`, `n_input_chunks`, `n_selected`,
  `input_distractor_ratio`, `evidence_density`, `retained_evidence_ratio`,
  `second_hop_rescue_count`, `estimated_waste_tokens`, …) and `BuiltContext`
  (`.text()`, `.chunks`, `.report`).

**Node (`redhop`)**
- `Document.fromText(text, opts)`, `Document.fromChunks(chunks, opts)`,
  `Document.fromFile(path, opts)`, `Document.fromBytes(data, source, opts)`,
  `Document.fromFolder(path, folderOpts)` — `opts.language` selects any
  of the 18 Snowball Porter2 languages (or throws on unknown names).
- `Document.context(query, budget?, neighbors?, includeHeading?)` →
  `BuiltContext { text, chunks, citations, report }`.
- `Document.analyze(query)` → `Report` (same shape as
  `context().report`) without paying assembly cost.
- `Document.chunkCount`, `Document.nFiles`, `Document.skippedFiles`
  getters.
- Top-level `groundingScore(query, text)` and `linkStrength(a, b)`.

**Rust (`redhop`)**
- `redhop::Document` + `redhop::DocumentConfig` (re-exported from
  `redhop::document`) — the high-level surface.
- `redhop::{build_context, filter_context, analyze_context,
  context_economics, ContextConfig, ContextStrategy, ContextReport,
  BuiltContext, AutoDecision, grounding_score, link_strength}` —
  re-exported from `redhop::context`.
- `redhop::analyzer::{Analyzer, SnowballAnalyzer, default_english}` —
  the pluggable analyzer surface attached via `Document::with_analyzer`.
- `redhop::{citations, Citation, FolderOptions, LoadOptions}` plus, behind
  the `files` feature, `redhop::{read_file, read_bytes, read_folder,
  read_folder_with}`.

**Stable semantics**
- `ContextStrategy` string/serde names (`raw_topk`, `distractor_filtered`,
  `redundancy_pruned`, `max_density`, `reasoning_preserving`, `auto`).
- `AutoDecision` values (`passthrough`, `prune`, `not_auto`).
- The structured `ContextReport` fields above (parse these for telemetry).

## Known call-shape asymmetries (Python vs Node)

The Python and Node bindings expose the same *fields* and the same *string
values* (parity is actively tested at the data level), but two call shapes
differ for idiomatic reasons. They are stable within 0.x — call them as
documented below; both are correct, neither will be silently flipped.

1. **`from_text` arguments.**
   - Python: `Document.from_text(text, source="document", …)` — `source`
     is the second positional argument (with a default).
   - Node: `Document.fromText(text, options?)` — `source` lives inside the
     options bag: `Document.fromText(text, { source: "policy.md" })`.
2. **`BuiltContext.text`.**
   - Python: `ctx.text()` — callable method (idiomatic for the pyo3 binding
     since the underlying Rust value is borrowed).
   - Node: `ctx.text` — string property on the returned object (idiomatic
     napi-rs `#[napi(object)]` shape).

Everything else — `report.strategy`, `report.requested_strategy` (Python) /
`report.requestedStrategy` (Node), `auto_decision` / `autoDecision`,
strategy string values, the chunks and citations arrays, etc. — has the
same shape across both bindings.

## Experimental / may change without notice

- **Internal retrieval** — `redhop::retrieval` (`Bm25Retriever`, dense,
  hybrid) inside the consolidated crate, plus the sibling crates
  `redhop-pipeline`, `redhop-diagnostics`, `redhop-orchestration`,
  `redhop-observability`, `redhop-calibration`, and `redhop-benchmarks`
  — implementation detail, not the public product API. Retrieval is
  intentionally **not** surfaced through `Document`; do not depend on
  these surfaces directly.
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
[evidence layer](findings/README.md)). SOTA/"solves reasoning" framing is
avoided: the value is measured behavior, conservative defaults, and an
interpretable, observable runtime. Operational claims are benchmark-backed
or they are not made.
