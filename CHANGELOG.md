# Changelog

All notable changes to RedHop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to the
versioning policy in [docs/API_STABILITY.md](docs/API_STABILITY.md) (0.x alpha:
minor releases may break; breaking changes are noted here).

## [Unreleased]

Post-v0.1.4 work on `main` — not yet tagged. Queued for the next
release. Targets **0.2.0** because `ContextConfig` and `DocumentConfig`
grew new required fields for the pluggable analyzer — callers using
struct field literals from outside the crate need to add
`analyzer: ...`. Callers using `..Default::default()` are unaffected.

### Added

- **Pluggable lexical analyzer.** The new `crate::analyzer::Analyzer`
  trait + `SnowballAnalyzer` (18 Snowball Porter2 languages) is a
  first-class extension point: one analyzer drives BOTH the BM25
  retriever AND the grounding scorer, so the two layers can't drift on
  what "the same term" means (the bug class fixed by hand four times
  through 0.1.3-0.1.4). Design rationale in
  `docs/design/ANALYZER_PLUGIN.md`; usage in `docs/LANGUAGE.md`.
- **`Document::with_analyzer(Arc<dyn Analyzer>)`** — mirrors
  `with_embedder`. Swaps the analyzer for both layers in lockstep.
- **`LoadOptions::language: Option<String>`** — string-routed access to
  the 18 builtins (`"english"`, `"german"`, `"french"`, …). Unknown
  language names return an error (no silent fallback to English).
- **Python `language` kwarg** on every `Document.from_*` constructor.
- **Node `language` field** on `Options`.
- **`ContextConfig::analyzer`** + **`DocumentConfig::analyzer`** — the
  analyzer flows end-to-end (loaders → `DocumentConfig` → `Document` →
  `Bm25Retriever` / `LocalRerankRetriever` / fallback BM25 / grounding
  scorer).
- **Quality suite T41-T45** pinning the analyzer plugin end-to-end:
  German `Bücher`↔`Buch` morphology, French `manger`↔`mange`
  inflections, proof that `with_analyzer` swaps both layers in lockstep,
  unknown-language error, and per-Document analyzer isolation (no leak
  through the OnceLock default or Tantivy's tokenizer manager).
- **Document binding parity** (Node):
  - `Document.analyze(query)` — pure diagnostics, returns the same
    `Report` shape as `context().report` without paying assembly cost.
    Was missing on Node (Python and Rust had it).
  - `Document.nFiles` getter (u32) — number of source files actually
    indexed. `1` for single-source ctors, the readable count for
    `fromFolder`.
  - `Document.skippedFiles` getter (`SkippedFile[]`) — `{source, reason}`
    pairs for files `fromFolder` couldn't parse. Was previously a
    silent skip with no introspection. Mirrors Python's
    `doc.skipped_files`.
- **Core**: `Document` carries `n_files()` and `skipped_files()`
  accessors. Single-source constructors default to `1` and empty.
  `read_folder_with` (both simple + persisted paths) now records
  `(source, reason)` for each file it skips instead of silently
  dropping them.

### Fixed
- **All-stopword query no longer crashes BM25.** A query that the analyzer
  pipeline reduces to zero positive terms (`""`, `"   "`,
  `"the and is of in or"`) used to surface as a hard Tantivy error:
  `Invalid query: Only excluding terms given`. The retriever now traps that
  error class (and the `empty query` class) and returns an empty result —
  the only sensible behavior for a no-signal query.
  Caught by the new `quality_suite::t25` on its first run.

### Added
- **ASCII folding for accented characters** (`café` ↔ `cafe`,
  `Süßigkeit` ↔ `Sussigkeit`, `naïve` ↔ `naive`). Both layers (BM25
  analyzer + grounding scorer's `normalize`) fold combining diacritics via
  NFKD so European Latin content is reachable from both accented and
  unaccented forms. Verified empirically before the change (`cafe` query
  used to miss a `café` chunk). New tests T27, T28, T39 pin this.
- **`crates/redhop/tests/quality_suite.rs`** — a 40-test behavior-level
  suite organized by what a user perceives, not by code structure. Covers
  tokenization (T01-T07), multi-field reach (T08-T09), document structure
  (T10-T13), context assembly (T14-T20), hybrid contract (T21-T22),
  edge cases (T23-T26), Unicode/multilingual (T27-T30), adversarial
  queries (T31-T34), nested markdown (T35), cross-format mixed corpus
  (T36), and non-English pinning (T37-T40). Found two real bugs on its
  first runs (the empty-query crash and the accent-folding gap).
- **`docs/LANGUAGE.md`** — the honest scope of non-English support, by
  family, plus the names of crates and the code locations to plug in
  for German morphology / Chinese word-segmentation / etc.
- **`README.md`** "Language support" subsection under "Retrieval tiers"
  — a 3-row matrix calibrating expectations for non-English content
  without bloating the README.

### Changed
- Two pre-existing tests (`reorders_pool_by_query_embedding`,
  `separate_query_embedder_drives_the_query_side` from earlier work)
  remained loosened from previous releases — top-1 assertions were
  pinned to specific BM25 micro-stats that flip under tokenizer changes.
  They now assert the structural invariant (RRF applied, dense breakdown
  present) rather than the brittle ordering.

### Notes
- `unicode-normalization` promoted from transitive (via tantivy) to a
  direct dep of redhop. Used for the grounding scorer's NFKD fold.
- Workspace test count: 314/314 pass under `cargo test --workspace`
  (was 260 at the v0.1.4 tag).
- CI gates remain clean: `cargo fmt --all -- --check`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## [0.1.4] - 2026-06-01

Citation ergonomics — for both code and prose. Follow-on to 0.1.3's
BM25-quality theme: retrieval is now precise, but the **assembled context**
returned to the LLM left the user staring at a `def` line without the
implementation, or at a deep section paragraph without its parent heading.
Both gaps closed by default; both have explicit opt-out knobs. Plus three
prose-side fixes that surfaced during the audit (setext headings, PDF
heading heuristic, plumbing).

### Changed
- **`Document::context(query)` on a code chunk now attaches ±1 neighbor
  chunks by default.** Code is chunked as fixed-token windows so a 50-line
  function often spans 2-3 chunks; a hit on the chunk containing the `def`
  line would previously cite only the signature, omitting the body in the
  next chunk. With `DocumentConfig::code_neighbors_default = 1`
  (the new default), citations on code hits include the surrounding
  implementation. **Behavior change** for code-shaped corpora — set
  `code_neighbors_default: 0` to restore the old chunk-only behavior. No
  effect on prose corpora (fires only on chunks tagged
  `metadata["kind"] == "code"`).
- **`Document::context(query)` on a prose chunk with a section heading
  now attaches the section's opener chunk by default.** A query that
  lands deep inside `## Refunds → ### Eligibility` previously cited only
  the matched chunk — the LLM lost the section title. With
  `DocumentConfig::prose_heading_default = true` (the new default), the
  section's first chunk is attached. **Behavior change** for hierarchical
  prose — set `prose_heading_default: false` to disable. Only fires on
  chunks that carry non-empty `metadata["heading"]` (markdown, DOCX,
  PPTX, XLSX, and — new in this release — PDF).
- **Markdown sections now recognize setext headings** (`Title\n=====`
  for H1, `Title\n-----` for H2) in addition to ATX (`#`/`##`/…). YAML
  frontmatter (`---` ... `---` at file start) is detected and excluded
  from setext scanning so its closing fence doesn't get treated as an
  H2 underline. Pandoc output / older docs / man pages now produce the
  same section structure as their ATX equivalents.
- **PDF chunks now carry best-effort heading metadata.** A per-page
  heuristic lifts the first short, non-paragraph-shaped line into
  `Section::heading` (rejecting page-number footers, body lines ending
  in sentence punctuation, and lines ending in a digit). Lets the BM25
  heading-field search added in 0.1.3 actually reach PDF chunks by
  topic; previously `metadata["heading"]` was always `None` on PDFs.

### Added
- **`DocumentConfig::code_neighbors_default: usize`** (default `1`).
- **`DocumentConfig::prose_heading_default: bool`** (default `true`).
  Both inherited via the Python / Node bindings' default config; no
  binding-surface change for callers who don't override.

### Fixed
- (No code-bug fixes — 0.1.3 already closed the BM25 quality gaps. See
  the Notes section for the verified-not-broken embedding-persistence
  story.)

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
- Eleven new tests across the citation-ergonomics theme: 3 for the code
  neighbor default, 3 for the prose heading default, 3 for setext
  headings + frontmatter handling, 2 for the PDF heading heuristic.
  **111/111 tests pass** under
  `cargo test -p redhop --features files`.

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
