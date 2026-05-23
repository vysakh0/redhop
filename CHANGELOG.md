# Changelog

All notable changes to RedHop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to the
versioning policy in [docs/API_STABILITY.md](docs/API_STABILITY.md) (0.x alpha:
minor releases may break; breaking changes are noted here).

## [Unreleased]

### Added
- **`Document` runtime** (`redhop-document`, exposed in Python as
  `redhop.Document`): the high-level "reason over a document" surface —
  `from_text` / `from_chunks` → lazy chunk + in-memory BM25 index →
  `context(query)` / `analyze(query)`. Retrieval is internal; no retriever,
  vector DB, or query-engine wiring is surfaced.
- **`ContextStrategy::Auto`** — size-gated policy: pass small contexts through,
  prune large/diluted ones. Gate (`auto_passthrough_max_tokens`, default 1500)
  calibrated by a size sweep. Exposed in Python as `strategy="auto"`.
- **RedHop Decision Report** — `ContextReport` rendering reworked into an
  interpretable three-layer report (decision / economics / diagnostics).
  `auto_decision()` (`Passthrough` / `Prune` / `NotAuto`) plus
  `requested_strategy` / `input_tokens` exposed in Rust and Python.
- **Embedding-free lexical dedup** in `RedundancyPruned` (term-set Jaccard) so it
  works in the local BM25 path without a vector model.
- Public observability primitives `grounding_score` / `link_strength`.
- Real-document evaluation on CUAD contracts + context-dilution findings; new
  findings docs and the `docs/retrievaltips.md` guide. CI workflow, changelog,
  and API-stability doc.

### Changed
- **Default chunk size 256 → 128 tokens.** A chunk_size × budget × dataset sweep
  (vs LangChain/LlamaIndex) showed granularity — not the assembly strategy — is
  the lever: finer chunks lift multi-hop ≥0.8 evidence retention 54%→77% (ahead
  of both frameworks) and tie at large budgets. See
  `docs/findings/CHUNK_GRANULARITY.md`.
- **`Document` API split by cost:** `chunk_size` / `chunk_overlap` on `from_text`
  (index-time), `budget` on `context(query, budget=...)` (query-time, no
  re-indexing). Rust: `Document::context_with`.
- BM25 query handling reduces arbitrary natural-language queries to a clean word
  bag, so `doc.context(question)` never crashes on punctuation/quotes (ranking
  unchanged).
- Context defaults are evidence-tuned; see the evidence layer for the
  justification behind each.

### Notes
- Findings are preserved including falsified hypotheses; operational claims are
  benchmark-backed. The architecture is intentionally bounded — no agents,
  workflows, planners, or orchestration.
