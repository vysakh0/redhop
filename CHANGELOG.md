# Changelog

All notable changes to RedHop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to the
versioning policy in [docs/API_STABILITY.md](docs/API_STABILITY.md) (0.x alpha:
minor releases may break; breaking changes are noted here).

## [Unreleased]

## [0.1.4] - 2026-06-01

Code-search ergonomics + a regression guard around the embedding cache.
Follow-on to 0.1.3's BM25-quality theme: the retrieval is now precise, but
the way the assembled context is presented for **code** files leaves the
user staring at a `def` line without the implementation. Default neighbor
expansion for code chunks closes that gap.

### Changed
- **`Document::context(query)` on a code chunk now attaches ±1 neighbor
  chunks by default.** Code is chunked as fixed-token windows so a 50-line
  function often spans 2-3 chunks; a hit on the chunk containing the `def`
  line would previously cite only the signature, omitting the body in the
  next chunk. With `DocumentConfig::code_neighbors_default = 1`
  (the new default), citations on code hits include the surrounding
  implementation. **Behavior change** for code-shaped corpora — set
  `code_neighbors_default: 0` to restore the old chunk-only behavior. No
  effect on prose corpora (the auto-expansion fires only on chunks tagged
  `metadata["kind"] == "code"`).

### Added
- **`DocumentConfig::code_neighbors_default: usize`** (default `1`) — the
  knob that drives the change above. Inherited via the Python / Node
  bindings' default config; no public binding-surface change.

### Fixed
- (No code fixes — 0.1.3 already closed the BM25 quality gaps. See the
  Notes section for the verified-not-broken story.)

### Notes
- **Embedding persistence verified.** The 0.1.3 audit suspected that
  `read_folder_with(persist=true)` re-embedded every chunk on reload
  (paying ~30-60 sec of bge-small cost per cold start on a 1000-chunk
  codebase). The machinery is already correct: `embedded_chunks()`
  populates the `Chunk::embedding` field from the retriever cache before
  writing `index.json`, `Embedding` is `Serialize`/`Deserialize`, and
  `LocalRerankRetriever::index` short-circuits any chunk that comes back
  with an embedding already set. Round-trip test
  (`crates/redhop/tests/embedding_persistence.rs`) now pins this — a
  reload triggers exactly 1 embed call (the query), not N+1 (the query +
  every chunk). Locked in as a regression guard.
- Three new integration tests for the code-neighbor default
  (`crates/redhop/tests/code_neighbor_expansion.rs`): code chunks get a
  neighbor, prose corpora are unaffected, opt-out via
  `code_neighbors_default = 0` works as expected. 103/103 tests pass.

## [0.1.3] - 2026-05-31

Retrieval-quality release. Fixes the structural hybrid-fusion bug reported in
[issue #1](https://github.com/vysakh0/redhop/issues/1) and a family of
silent-search-miss bugs the audit surfaced: BM25 and the grounding scorer were
disagreeing on what a "token" is. Behavior on existing corpora will shift —
rankings get sharper for queries with morphological variants, code identifiers,
filenames, headings, version suffixes, and stopwords.

### Fixed
- **Hybrid retrieval now RRF-fuses BM25 with the dense rerank** instead of
  returning `dense.truncate(top_k)` on the pure-prose pool path. A chunk
  BM25 ranked #1 that the dense model demoted past `top_k` is no longer
  silently dropped. Restores the documented "hybrid ≥ either tier" contract.
  Result `score.method` becomes `RetrievalMethod::Hybrid` for every hybrid
  result, and `breakdown.fused` is populated. (Issue #1.)
- **BM25 now stems** (Snowball English/Porter2) — same algorithm the
  grounding scorer already used. Queries like `compression` now match a
  chunk containing `compress_video`; `running` matches `runs`, etc.
- **BM25 now drops stopwords**, matching the grounding scorer's
  `STOPWORDS` list. A stopword-padded query like `"what is the refund
  window"` ranks the same chunk first as the bare `"refund window"`.
- **BM25 now splits camelCase / PascalCase identifiers** at index time. A
  query for `compress` now reaches `compressVideo`; `http` reaches
  `HTTPResponse`. Original identifiers still self-match.
- **BM25 now splits letter ↔ digit boundaries.** `parseV2` indexes as
  `parseV2 + parse + V + 2`; `Phi3` as `Phi3 + Phi + 3` — so the base
  name reaches versioned identifiers and model names.
- **BM25 now searches `source` and `heading` in addition to `text`.** A
  query for the filename `auth.rs` reaches a chunk from that file even
  when the chunk text itself doesn't mention the filename; a query for a
  Markdown heading reaches the chunk under that heading. The `source`
  field type changed from `STRING` (exact-match only) to an analyzed
  TextField using the same stemming pipeline.
- **`sanitize_query` no longer over-strips.** It used to replace every
  non-alphanumeric character with a space to defend against Tantivy
  QueryParser meta-syntax, degrading `v1.2.3` to three single-char
  tokens, `.NET` to `NET`, `e-mail` to a single character. It now
  replaces only the chars QueryParser actually parses as syntax
  (`+ - : * ? ^ ~ \ ( ) [ ] { } " < >`); everything else passes through
  to the analyzer, which tokenizes it consistently with indexed text.
  Uppercase `AND`/`OR`/`NOT` are neutralized by a lowercase pass.

### Added
- **`ContextReport.low_confidence_retrieval: bool`** +
  **`low_confidence_threshold: f32`** — a programmatic signal that fires
  when every selected chunk has grounding ≤ the threshold (default `0.10`,
  same as `distractor_min_grounding`). Lets callers detect "no good
  match found" without inferring from `n_selected == 0`. Exposed on the
  Python report (`report.low_confidence_retrieval`), the Node report
  (`report.lowConfidenceRetrieval`), and rendered as a one-line warning
  in `report.render()`. Both new fields use `#[serde(default)]` so older
  deserialized reports stay compatible.
- **`ContextConfig.low_confidence_max_grounding: f32`** — the threshold
  the signal applies. Defaults to `distractor_min_grounding`.
- **`DocumentConfig.min_candidates: usize`** (default `0` = off) — an
  opt-in floor on the number of candidates delivered to the assembler.
  Under `hybrid` / `semantic`, if the primary retriever returns fewer
  than this, a BM25 fallback over the same chunks tops the result up.
  Exposed in `LoadOptions.min_candidates: Option<usize>` (Rust loaders)
  and `Options.minCandidates: Option<u32>` (napi). Pair with
  `low_confidence_retrieval` to detect when the fallback fires with
  weak chunks. No-op under `lexical` (the primary already is BM25).

### Notes
- Five separate test regressions cover the BM25 fixes — see
  `crates/redhop/src/retrieval/bm25.rs` and
  `crates/redhop/src/retrieval/local_rerank.rs`. Two more cover the new
  knob and the low-confidence signal. 99/99 tests pass under
  `cargo test -p redhop --features files`.
- Two existing tests had top-1 assertions pinned to specific BM25
  micro-stats; both are loosened to assert the structural invariant
  (RRF was applied, dense breakdown is present) rather than the brittle
  ordering.

## [0.1.2] - 2026-05-28

npm meta-tarball slim-down. PyPI and crates.io bumped to keep version parity;
no source changes on those sides.

### Fixed
- **npm `redhop` meta package was 158 MB unpacked** because
  `napi artifacts --dir artifacts` deposits ALL platform `.node` files in
  `nodejs/` root before publish, and `"files": ["*.node", ...]` in
  `nodejs/package.json` slurped them all into the meta tarball. Removing
  the `*.node` glob from the `files` array drops the meta to **~28 KB**;
  a Mac M1 user's full `npm install redhop` goes from ~185 MB to ~27 MB.
  The local-file-first branch in `index.js` was only meaningful during
  dev (after `napi build` leaves the binary at `nodejs/redhop.<target>.node`);
  installed-from-npm consumers now follow the `optionalDependencies`
  path the optional-deps mechanism is designed for.

## [0.1.1] - 2026-05-28

**npm-only release** to ship the Windows platform package under a non-blocked
name. PyPI and crates.io stayed at 0.1.0 — no source changes affecting them.

### Changed
- **Windows platform package renamed** from the napi-default
  `redhop-win32-x64-msvc` to **`redhop-win-x64`** because npm's spam
  detector blocked the default name. Confirmed by publishing an
  identical tarball under the new name (same metadata, same binary).
  The meta `redhop`'s `index.js` and `optionalDependencies` reference
  the new name; the binary file inside the platform package is still
  `redhop.win32-x64-msvc.node` (napi-rs derives that from the Rust
  target triple). A small `nodejs/scripts/publish-npm.mjs` replaces
  `napi prepublish` so the rename is wired through all the publish
  steps.

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
