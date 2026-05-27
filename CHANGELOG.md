# Changelog

All notable changes to RedHop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to the
versioning policy in [docs/API_STABILITY.md](docs/API_STABILITY.md) (0.x alpha:
minor releases may break; breaking changes are noted here).

## [Unreleased]

## [0.1.0] - 2026-05-27

First public release — `pip install redhop` (PyPI); npm and crates.io to follow.

### Added
- **Retrieval tiers** on `Document` (`retrieval=`): `"lexical"` (BM25, default,
  zero-dependency), `"hybrid"` (BM25 prunes to a candidate pool → dense rerank of
  that pool only — no ANN, no vector DB), and `"semantic"` (global exact-cosine
  dense over every chunk, for small/bounded synonym-heavy corpora). Rust:
  `RetrievalMode::{Lexical, Hybrid { candidate_pool }, Dense}`,
  `Document::with_embedder` / `with_query_embedder`, and a reusable
  `LocalRerankRetriever` in `redhop-retrieval`. Evidence:
  `docs/findings/LOCAL_RERANK.md`, `SEMANTIC_MISMATCH.md`, `DENSE_RERANK_CEILING.md`.
- **Bring-your-own embedder** for the semantic tiers: `embedder_model` /
  `embedder_tokenizer` / `embedder_dim` / `embedder_pooling` (`"cls"` | `"mean"`),
  plus `embedder_query_prefix` / `embedder_passage_prefix` for asymmetric models
  (E5). Any ONNX bi-encoder works; RedHop ships none and never bundles a default.
- **File & folder loaders** — `Document.from_file` and `from_folder`
  (`redhop-files`): read + chunk text-based files (txt/markdown/source/…) and
  whole directories, each chunk tagged with its source path.
- **Self-contained wheel** — the ONNX engine and file parsers compile into the
  single `pip install redhop` wheel (no Python deps, no `[onnx]` extra).
- **Findings layer extended** — `SEMANTIC_ZERO_DEP` (the non-contextual ceiling;
  MaxSim/RM3/static-embeddings falsified), `DENSE_RERANK_CEILING` (0.80 plateau is
  the second-hop tax; no cheap escalation trigger), `SPEED_VS_FRAMEWORKS` (the
  "speed moat" claim largely falsified — honest correction).
- **Open-source tooling** — `ruff` (Python lint + format), `cargo-deny`
  (license/advisory gate, `deny.toml`), coverage (`cargo-llvm-cov` + `pytest-cov`)
  wired into CI; `SECURITY.md`.
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
