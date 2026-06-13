# Changelog

All notable changes to RedHop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to the
versioning policy in [docs/API_STABILITY.md](docs/API_STABILITY.md) (0.x alpha:
minor releases may break; breaking changes are noted here).

## [0.4.0] — 2026-06-13

**Architecture cleanup + Python API breaking change.** Two changes worth
your attention before upgrading.

1. **Python `Document.from_*` takes `options=` instead of 18 keyword
   arguments.** The new shape mirrors what the Node binding has had since
   day one. Every existing call that passes any chunking / retrieval /
   embedder / language knob must move it into a
   `redhop.DocumentOptions(...)` (or `redhop.FolderOptions(...)` for
   `from_folder`). Calls with no kwargs are unchanged.

2. **The unused `AdaptiveOrchestrator` subsystem is removed.** Five
   workspace crates (`crates/{orchestration,pipeline,diagnostics,calibration,observability}`,
   all `publish = false`), their public trait surface in `redhop::core`
   (`DiagnosticsEngine`, `RegimeClassifier`, `DiagnosticsReport`, the
   whole `core::state` module), 39 dependent measurement probes, and 5
   findings that measured the deleted controller. Net: −23,744 LOC of
   Rust. None of it was reachable from the published `redhop` crate, the
   FFI bindings, or the CLI. The project's stated discipline (no graphs /
   agents / planners) is now reflected in the workspace it ships.

### Changed (breaking)

- **Python `Document.from_*`** — every constructor now accepts only the
  positional payload plus a keyword-only `options=` struct:

  ```python
  # before
  doc = redhop.Document.from_text(text, retrieval="hybrid", language="german")
  doc = redhop.Document.from_folder("./docs", persist=True, ignore=["*.lock"])
  doc = redhop.Document.from_bytes(data, source="x.pdf", retrieval="hybrid")

  # after
  doc = redhop.Document.from_text(
      text,
      options=redhop.DocumentOptions(retrieval="hybrid", language="german"),
  )
  doc = redhop.Document.from_folder(
      "./docs",
      options=redhop.FolderOptions(persist=True, ignore=["*.lock"]),
  )
  doc = redhop.Document.from_bytes(  # source is now positional, not a kwarg
      data, "x.pdf",
      options=redhop.DocumentOptions(retrieval="hybrid"),
  )
  ```

  Two new pyclasses (`redhop.DocumentOptions`, `redhop.FolderOptions`)
  carry every knob. `FolderOptions` has the folder-specific fields
  (`recursive`, `gitignore`, `ignore`, `persist`, `index_dir`) at the top
  level and nests `options=DocumentOptions(...)` for chunking / retrieval
  knobs — same shape Node's `FolderOptions` has.

- **Rust `redhop::core`** — no longer re-exports any of the orchestration
  surface: `DiagnosticsEngine`, `RegimeClassifier`, `DiagnosticsReport`,
  `DiagnosticsWarning`, `RetrievalState`, `RetrievalRegime`,
  `RegimeDistribution`, `ConfidenceProfile`, `ClassificationTrace`,
  `RuleFire`, `TakenAction`, `AbstainReason`, `ActionCost`, `Budget`,
  `RerankerLevel`, `RetrievalAction`, `StopReason`. The
  `redhop::core::state` module is gone. The corresponding traits in
  `redhop::core::traits` are gone. None of these had any implementation
  in the published surface — they were the trait/state surface of the
  deleted orchestration layer.

### Removed

- **Workspace crates** — `crates/orchestration`, `crates/pipeline`,
  `crates/diagnostics`, `crates/calibration`, `crates/observability`. All
  were `publish = false`, never reached crates.io.
- **Internal measurement probes** — 39 examples in `crates/examples` that
  depended on the orchestration layer (`adaptive_loop`,
  `adaptive_eval_*`, `bge_*`, `calibration_sweep`, `ce_escalation_*`,
  `context_economics`, `diagnostics`, `emit_*`, `hybrid_old_vs_new`,
  `ingestion_diagnostics`, `layered_diagnostics`, `method_pair_regret`,
  `musique_chunk_sweep`, `musique_embedder_swap`,
  `musique_hybrid_recall`, `musique_recall_diagnostic`,
  `neotrace_import`, `observability_report`, `quickstart`,
  `rag_with_claude`, `real_corpus_calibration`, `real_embedding_bakeoff`,
  `real_pdf_validation`, `regime_classification`, `second_hop_retention`,
  `semantic_local_rerank`, `semantic_natural`,
  `semantic_reasoning_rerank`, `signal_ablation`,
  `distractor_answer_correlation`, `export_semantic_pool`,
  `export_rerank_pool`). The 20 still-useful probes (`cuad_*`, `eval_*`,
  `query_set_analyzer_probe`, `multilingual_*`, `semantic_mismatch`,
  `chat_rag`, `document_dense`, `enrich_code_search`,
  `spider_enrich_probe`, `sub_idf_reweighting_probe`, `ce_smoke`) are
  preserved.
- **Architecture docs** — `docs/INTEROPERABILITY.md` and
  `docs/NEOTRACE_SCHEMA.md` described the deleted Python ↔ Rust
  orchestration seam. `docs/ARCHITECTURE.md` rewritten to match the
  actual published surface.
- **Findings** — `ADAPTIVE_CONTROLLER.md`, `ADAPTIVE_REAL_SUBSTRATE.md`,
  `SUBSTRATE_COUPLING.md`, `REAL_WORKLOAD.md`, `EMBEDDING_BAKEOFF.md` —
  all five measured the deleted controller.

### Performance

- **Python GIL** — every `Document.from_*` entry point plus
  `Document.context`, `context_with_rewrites`, and `analyze` wraps its
  pure-Rust work in `py.allow_threads(...)`. A Python server can now
  answer requests on sibling threads while one folder index or one
  retrieval is running. The previous behavior held the GIL across the
  entire walk → parse → chunk → index pipeline.

### Internal

- **`embeddings::CachedEmbedder` mutex** — switched from
  `std::sync::Mutex` to `parking_lot::Mutex` (already a workspace dep).
  7 `.unwrap()`s on `.lock()` removed; the lock is now infallible. Holds
  the same "never across `.await`" discipline as before.
- **Module headers** — the legacy `//! # redhop-context`, `//! # redhop-core`,
  `//! # redhop-chunking`, `//! # redhop-retrieval`, `//! # redhop-document`,
  `//! # redhop-embeddings`, `//! # redhop-storage`, `//! # redhop-reranking`
  headers (from before the pre-0.2 crate consolidation) are gone.
- **Rustdoc broken-link guard** — `crates/redhop/src/lib.rs` now warns on
  `rustdoc::broken_intra_doc_links`. Five preexisting broken links fixed.
- **Python binding cleanup** — `python/src/lib.rs` shed 510 LOC of
  duplicated logic now that all five entry points route through
  `redhop::text` / `redhop::read_file_with` / `redhop::read_bytes_with` /
  `redhop::chunks_typed` / `redhop::read_folder_with`. The Python and
  Node bindings now use the same loader functions.

### Migration

Most callers only need to lift their kwargs into a `DocumentOptions`:

| before | after |
| --- | --- |
| `Document.from_text(text)` | `Document.from_text(text)` (unchanged) |
| `Document.from_text(text, language="german")` | `Document.from_text(text, options=DocumentOptions(language="german"))` |
| `Document.from_file(p, token_budget=400, candidate_k=3)` | `Document.from_file(p, options=DocumentOptions(token_budget=400, candidate_k=3))` |
| `Document.from_bytes(data, source="x.pdf")` | `Document.from_bytes(data, "x.pdf")` (`source` is now positional) |
| `Document.from_folder(p, persist=True, ignore=["*.lock"])` | `Document.from_folder(p, options=FolderOptions(persist=True, ignore=["*.lock"]))` |
| `Document.from_folder(p, persist=True, retrieval="hybrid")` | `Document.from_folder(p, options=FolderOptions(persist=True, options=DocumentOptions(retrieval="hybrid")))` |

Node, Rust, and CLI users see no API change. Rust callers who depended
on `redhop::core::DiagnosticsReport` or any of the deleted state types
from the published crate — that was the orchestration layer that was
never reachable end-to-end. File an issue if you had a real use case.

## [0.3.4] — 2026-06-10

**Retrieval diagnostics + bring-your-own-pipeline observability.**
Two threads in this release:
(1) `ctx.report.diagnosis` — every Decision Report now carries
query-level facts about how the query interacted with the corpus and
the retrieved candidates, plus a closed registry of bounded hints
that fire on documented failure shapes with a measured-finding
citation on each one;
(2) workload audit + observability export — `summarize_diagnoses`
aggregates across N reports into one focus recommendation, and
`redhop.otel.report_to_attributes` flattens any report into OpenTelemetry
or Langfuse-compatible span attributes with zero new dependencies.
The combination lets users point RedHop's diagnostics at their
existing LangChain / LlamaIndex / pgvector pipeline in ~10 lines, no
migration. See `docs/DIAGNOSE_YOUR_PIPELINE.md`.

### Added — self-service retrieval diagnostic

- **`ctx.report.diagnosis`** (Rust + Python + Node). The Decision
  Report now carries query-level facts about how the query interacted
  with the corpus and the retrieved candidates, plus a small closed
  registry of bounded hints that fire on documented failure shapes.
  Every hint cites a findings or docs file. The three failure shapes
  catalogued in `docs/CHOOSING_A_CONFIG.md` (vocabulary mismatch,
  polysemy, templated boilerplate) each produce a recognizable hint
  with the evidence path attached. Healthy queries produce zero hints.
  Always computed, never alters retrieval, no new configuration knobs.
  Design: `docs/design/REPORT_DIAGNOSIS.md`. Examples:
  `examples/{python,nodejs,rust}/12_diagnosis.*`.

### Added — workload audit + observability export

- **`summarize_diagnoses(reports)`** (Rust + Python + Node). Aggregates
  per-query diagnoses across N `ContextReport`s into a single
  workload summary: hint histogram, failure rates, top vocabulary
  gaps, and at most one focus recommendation citing the measured
  finding behind it. Six focus codes (`vocab_mismatch`,
  `templated_queries`, `underdetermined_queries`, `weak_retrieval`,
  `healthy`, `sample_too_small`) resolved by a fixed priority order.
  Mirrors the existing `eval::summarize` shape.
- **`redhop.otel.report_to_attributes(report)`** (Python). Flattens a
  Decision Report into OpenTelemetry-legal span attributes (or
  Langfuse metadata) under a `redhop.` namespace. Zero new
  dependencies. Node and Rust ship the same conventions as a copy-
  paste snippet in the docs page.
- **New docs page**: `docs/DIAGNOSE_YOUR_PIPELINE.md` + website mirror.
  Four-step walk-through for pointing RedHop's diagnostics at an
  existing LangChain / LlamaIndex / pgvector / hand-rolled pipeline
  without migrating, including the honesty section ("`analyze_context`
  reports waste, only `build_context` removes it"). Design:
  `docs/design/WORKLOAD_AUDIT.md`. Example #13:
  `examples/{python,nodejs,rust}/13_workload_audit.*`.

### Fixed

- **`docs/CHOOSING_A_CONFIG.md`**: removed the stale "hybrid sometimes
  returns fewer candidates than lexical" warning that pointed at issue
  #1. The strict-superset contract was restored in 0.3.1 by the
  pure-rerank + BM25-tail fill, with a regression test
  (`local_rerank.rs::pure_rerank_lets_dense_win_when_it_disagrees_with_bm25`).

### Discipline notes

- `ContextReport` gained a new `diagnosis: Diagnosis` field. In
  0.x-alpha this is documented as additive-but-technically-breaking
  for Rust callers that construct `ContextReport` via struct literal
  from outside the crate. Users who build through the public API
  (`Document::context`, `build_context`, `analyze_context`) are
  unaffected. Python and Node bindings are dynamic and additive.
- 13 new 🟡-convention thresholds across the per-query hint and
  workload-focus registries are logged in `DEFAULT_PROVENANCE.md`
  with a re-validation entry. Folded into the existing 0.3 sweep
  backlog.

## [0.3.3] — 2026-06-09

**Answer-quality eval surface (Rust + Python + Node) + audit of
defaulted-on heuristics.** Two threads in this release:
(1) a new in-process `evaluate(...)` / `critique(...)` surface for
closed-set answer-quality metrics — lexical + LLM-judged, with
claim-decomposed faithfulness and TP/FP/FN correctness; calibrated
against Ragas at n=200 HotpotQA (r=+0.664, MAE=0.151);
(2) the defaulted-on heuristics audit (five measured, one
already-flipped in 0.3.2, two API smells fixed).

### Added — eval surface

- **`evaluate(query, ctx, answer=, gold_answer=, judge=, decompose_faithfulness=, decompose_correctness=)`** —
  in-process answer-quality eval that returns one `EvalReport` blending
  lexical (CI-deterministic) and LLM-judged metrics. Available in Rust
  (`redhop::evaluate(...)`), Python (`redhop.evaluate(...)`), and Node
  (`evaluateWithJudge(...)` for the async judge path). Faithfulness,
  relevancy, correctness in `_lexical` (no LLM) and `_judged` (opt-in)
  flavors; gold-relative metrics (`context_recall`,
  `context_precision`, `answer_token_recall`) when `gold` is provided.
- **`critique(answer, aspects, judge=, context=, query=)`** — open-ended
  user-defined dimensions (harmfulness, conciseness, brand voice,
  etc.). One LLM call per aspect; polarity-corrected scores so high =
  good across the report regardless of `highIsGood`. Returns a
  `CritiqueReport` with per-aspect scores in input order.
- **`summarize(reports)`** — aggregates a sequence of per-case
  `EvalReport`s into a means + N + share-flagged summary, the same
  shape RedHop's runtime uses for its Decision Report.
- **Judge surface** — `Judge.from_callable(fn).cached()` (Python),
  `Judge.fromCallable(fn, name).cached()` (Node), and the Rust
  `Judge` trait with `CachedJudge` and `CallableJudge` wrappers. One
  caching layer for any user-supplied LLM caller; an LRU sized by
  the caller. Single primitive supports faithfulness, relevancy,
  correctness, critique, and decomposed paths.
- **Claim-decomposed faithfulness** (`decompose_faithfulness=True`):
  extracts atomic claims via a few-shot LLM call, then batch-verifies
  all of them in a single second LLM call. Two LLM calls regardless
  of how many claims were extracted. `gpt-4o-mini` correlates with
  Ragas's faithfulness at r=+0.664 on n=200 HotpotQA (see
  [docs/findings/EVAL_JUDGED_CALIBRATION.md](docs/findings/EVAL_JUDGED_CALIBRATION.md)).
  Verifier prompt includes paraphrase-positive examples + negative
  entity-substitution examples to balance strictness and recall.
- **TP/FP/FN correctness** (`decompose_correctness=True`): mirrors
  decomposed-faithfulness on the answer-vs-gold axis. Extracts
  claims from both the answer and the gold answer, classifies each as
  TP / FP / FN, returns F₁. Diagnostic counters
  (`n_correctness_tp/fp/fn`) surface the intermediate categorisation.
- **Refusal-aware decomposition** — "I don't know" answers correctly
  produce `mean_faithfulness_judged = None` (0 claims extracted)
  instead of being scored as a vacuous 1.0. Surfaces refusals as a
  distinct category, not as faithfulness=1.
- **Diagnostic counters** on `EvalReport`:
  `n_faithfulness_claims_extracted`, `n_faithfulness_claims_supported`,
  `n_correctness_tp`, `n_correctness_fp`, `n_correctness_fn`. Surface
  intermediate classifier counts so callers can debug WHY a metric
  landed where it did.

### Added — eval evidence

- [docs/COMPARISON_RAGAS.md](docs/COMPARISON_RAGAS.md) — public-facing
  head-to-head with Ragas on claim-decomposed faithfulness. n=200
  HotpotQA, gpt-4o-mini, with Claude haiku as third-judge tie-breaker.
- [docs/findings/EVAL_JUDGED_CALIBRATION.md](docs/findings/EVAL_JUDGED_CALIBRATION.md)
  rewritten end-to-end: three-layer evidence (5-case wiring probe →
  5-case Ragas side-by-side → n=200 HotpotQA correlation + third-judge
  subset). Documents the v0→v4 prompt iteration that fixed four
  traceable failure modes (paraphrase rejection, comparative
  hallucination, compound-attribution dilution, wrong-entity
  substitution). Calls out single-shot LLM noise as a measured
  property of the workload (gpt-4o-mini at temp=0 is not
  deterministic through OpenRouter — ~20–30% per-case variance).
- [docs/findings/ANSWER_QUALITY_EVAL.md](docs/findings/ANSWER_QUALITY_EVAL.md) —
  full API tour for the new `evaluate(...)` + `critique(...)` surface.
- [docs/findings/EVAL_VS_RAGAS_SOURCE.md](docs/findings/EVAL_VS_RAGAS_SOURCE.md) —
  source-read comparison of the two libraries' implementations.
- `bench/eval_correlation_hotpot.py` — runs the n=200 Pearson r / MAE
  measurement on HotpotQA against Ragas with configurable context mode
  (`supporting` / `distractor_only` / `all`).
- `bench/eval_third_judge.py` — Claude haiku tie-breaker via the local
  `claude -p` CLI; no API key needed.
- `bench/eval_faith_trace.py` — diagnostic harness for tracing claim
  extraction + per-claim verifier votes on specific qids. Has a
  `--variant v0/v1/v2/v3/v4` flag for prompt iteration without
  rebuilding the Rust crate.
- `bench/eval_judged_calibration.py` — the 5-case wiring probe with
  optional Ragas side-by-side when installed.
- `bench/select_third_judge_subset.py` — filters contested cases from
  a correlation-bench JSON so the third-judge run stays cheap.

### Breaking — Rust only

- `redhop::evaluate(...)` signature now takes six parameters:
  `(query, ctx, answer, gold, judge, config)` instead of the prior
  three-parameter shape. The Python and Node bindings absorb this via
  kwargs / options and are NOT breaking. Pass `None` for `answer` and
  `judge` and `EvalConfig::default()` for `config` to match the old
  behavior.

### Added — defaulted-on heuristics audit

- **`code_neighbors_default` / `codeNeighborsDefault`** — surfaces the
  ±N adjacent-chunk auto-pull on code chunks as a constructor kwarg on
  Python `from_text`/`from_file`/`from_bytes`/`from_folder`/`from_chunks`
  and as a field on the Node `Options` struct. Default `1` (unchanged
  behavior). Pass `0` to disable, or `2`/`3` for more aggressive
  expansion under loose token budgets. See
  [docs/findings/CODE_NEIGHBORS_DEFAULT.md](docs/findings/CODE_NEIGHBORS_DEFAULT.md)
  for the measured budget tradeoff.
- **`prose_heading_default` / `proseHeadingDefault`** — surfaces the
  auto-attach of section-heading chunks to prose hits as the same
  constructor-level kwarg / option. Default `true` (unchanged). Pass
  `false` for memory-tight workloads where the heading isn't
  load-bearing. See
  [docs/findings/PROSE_HEADING_DEFAULT.md](docs/findings/PROSE_HEADING_DEFAULT.md)
  for the measured +7pt ≥0.8 lift at typical budgets.
- **`crates/redhop/src/load.rs`** — `LoadOptions` now exposes
  `code_neighbors_default` and `prose_heading_default` as
  `Option<usize>` / `Option<bool>`. Threads through `read_folder_with`
  for parity with the in-memory loaders.
- **Audit finding docs.** Five new findings on the defaulted-on
  heuristics audit:
  [`RAW_ANALYZER`](docs/findings/RAW_ANALYZER.md) (flipped in 0.3.2),
  [`HYBRID_CANDIDATE_POOL`](docs/findings/HYBRID_CANDIDATE_POOL.md)
  (inert knob — don't tune),
  [`PROSE_HEADING_DEFAULT`](docs/findings/PROSE_HEADING_DEFAULT.md)
  (+7pt at typical budgets),
  [`BM25_SOURCE_FIELD`](docs/findings/BM25_SOURCE_FIELD.md) (+4pt with
  signal, 0pt with noise),
  [`CODE_NEIGHBORS_DEFAULT`](docs/findings/CODE_NEIGHBORS_DEFAULT.md)
  (budget-dependent compromise).
- **Cross-binding parity tests** for the two new kwargs (Python
  `test_loaders.py`, Node `smoke.cjs`).

### Changed

- **No default values changed in this release.** All defaults remain
  what they were after 0.3.2 — the new kwargs default to the existing
  Rust values (`code_neighbors_default=1`, `prose_heading_default=true`).
  Existing callers see zero behavior change; the new kwargs are an
  opt-out / tune surface only.

## [0.3.2] — 2026-06-08

**Breaking: the default text analyzer flipped to `RawAnalyzer`.** New
`Document` objects no longer apply English Snowball stemming,
CamelCaseSplitter, or stopword filtering by default. Measurement on
three workloads (CUAD, HotpotQA, MuSiQue) showed the simpler pipeline
matches or beats English Snowball on retention AND latency:

| Workload | english ≥0.8 | raw ≥0.8 | Δ | english p50 | raw p50 |
|---|---:|---:|---:|---:|---:|
| CUAD | 86% | **91%** | **+5pts** | 6.4ms | 3.8ms |
| HotpotQA | 100% | 100% | 0 | 2.9ms | 2.3ms |
| MuSiQue | 90% | **97%** | **+7pts** | 3.4ms | 2.3ms |

The mechanism: Snowball stem collisions (`settles`/`settled`/`settling`
→ `settl`) inflate BM25 scores on chunks that share *any* form, drowning
out the discriminating proper nouns. See
[docs/findings/RAW_ANALYZER.md](docs/findings/RAW_ANALYZER.md) for the
full probe.

### Migration

- **Most users:** rebuild the index against 0.3.2; expect rank shifts
  but usually higher retention. Re-run your eval if you have one.
- **Code search / inflection-heavy workloads:** pass
  `language="english"` to restore the previous behavior
  (CamelCaseSplitter, stopwords, Snowball stem).
- **Multilingual paths unchanged:** `language="german"`, `"french"`,
  etc. still route to the corresponding Snowball analyzer.

```python
# Before 0.3.2 default (still available via opt-in):
doc = redhop.Document.from_text(text, language="english")

# 0.3.2 default (no argument):
doc = redhop.Document.from_text(text)
```

## [0.3.1] — 2026-06-08

The **post-release honesty audit**. After 0.3.0 shipped, a critical
review of the public claims surfaced ten methodological issues, then
a second reviewer pass surfaced five follow-ups. This release addresses
all of them. The most consequential change is a measurement, not code:
running `bench/compare.py` at n=300 with the same Stripper applied to
*every* system showed that LlamaIndex actually benefits more from the
same preprocessing than RedHop does — so the previously-published
"+4.7 over LlamaIndex" framing was apples-to-oranges and has been
retired across all user-facing docs.

### Added

- **`Stripper::is_effective_on(query)`** (Rust + Python + Node): returns
  a `StripperEffect` reporting the analyzer's view of the query
  (`original_tokens` / `stripped_tokens`), which configured boilerplate
  terms fired (`removed_terms`) vs sat silent (`unused_boilerplate`),
  AND a `probable_silent_no_op` list — the subset of unused boilerplate
  whose raw lowercased substring DOES appear in the query (= the actual
  bug). Empty `probable_silent_no_op` means every configured term that
  should have fired, did. 2 Rust + 2 Python + 2 Node tests, plus parity
  test asserting the field matches across bindings.
- **`bench/compare.py` fair-preprocessing arm.** New
  `cuad_stripped_items(n)` iterator applies `redhop.Stripper(CUAD_BOILERPLATE)`
  to every system's CUAD query before retrieval. The fair n=300 result:
  LlamaIndex 86% → 94%, RedHop topk 82% → 88%, LangChain 73% → 79%.
  Raw run persisted at `reports/framework_comparison_fair_preprocessing_2026-06-08.txt`.
- **`bench/query_set_analyzer_calibration.py`.** 7-workload calibration:
  5 obviously-distinct (P=R=1.00 on these) + 2 boundary-adjacent. Both
  boundary cases at template_word_share 0.291 and 0.334 stay quiet — the
  shipped 0.50 threshold is conservative; the actual crossover lives in
  `0.334 < threshold ≤ 0.50` (unprobed).
- **`scripts/check_feature_matrix.sh`** + matching `feature-matrix` job
  in `.github/workflows/ci.yml`. Sweeps 4 feature combinations × 2 crates
  = 8 `cargo check --all-targets` runs (redhop core; Python binding;
  the Node binding ships single-config by design). Found and fixed two
  latent test-compile breaks under `--no-default-features` (see Fixed).
- **6 new cross-binding parity tests** covering the 0.3.0 surface:
  Stripper.apply, Stripper.is_effective_on, Vocabulary.apply,
  Vocabulary.enrich, analyze_query_set, context_with_rewrites. The
  pre-0.3.1 parity coverage only touched build_context and friends.

### Changed (claim corrections; no runtime behavior change)

- **CUAD framing rewritten** in README.md, python/README.md,
  nodejs/README.md, comparison.mdx, benchmarks.md, overview.mdx,
  alternatives/llamaindex.mdx, llms.txt §"Why this matters", SVG alt
  texts + `<desc>`. The "+4.7 over LlamaIndex" headline is retired
  because the fair-preprocessing measurement disproved its like-for-like
  reading. The recipe's value is now framed as a **reproducible
  in-process workflow** (audit trail + Decision Report + `evaluate`),
  not an architectural retrieval lead. RedHop's clearer architectural
  win is called out separately: multi-hop (80% on HotpotQA vs LlamaIndex
  72%, n=300, no preprocessing).
- **SPIDER_ENRICH downgraded from "Confirmed" to "Suggestive"** and
  restructured to separate the cleanly-measured algorithmic lift
  (arm B, +0.128 mean recall, deterministic enrichment from schema
  metadata) from the curator-conflicted marginal lift (arm C, +0.067
  on top, hand-curated synonyms by the author of the questions). New
  Methodology Limitations section names the conflict, sample-schema
  selection bias, and what the probe does vs does not support.
  `VOCABULARY_ENRICH` header flipped from "bidirectional measured
  evidence" → "asymmetric measured evidence (negative clean; positive
  suggestive)".
- **"Four-corner rule" → "four-corner observation"** in findings/README,
  SPIDER_ENRICH, CUAD_HYBRID_RERANK, CUAD_ENRICH_DEFINITIONS_NULL,
  SUB_IDF_AUTO_DROP_NULL, examples README, the cuad_hybrid_rerank probe
  comment, plus the memory file (renamed). The cleanly-falsified
  corners are the load-bearing evidence; the curated corners are
  CUAD-only with author-curator overlap on the positive arms — a
  universal-rule framing wasn't warranted.
- **`analyze_query_set` calibration headline rewritten** from "P=R=1.00
  on 5 workloads" (true but uninformative — all 5 sat at obvious
  extremes) to "**Confirmed on the extremes; boundary behavior bounded
  but unmeasured**." QUERY_SET_ANALYZER.md now leads with the honest
  scope and includes the boundary table.
- **Stripper-alone retention aligned to 87.7%** (from `CUAD_CLAUSE_EXPANSION`'s
  controlled three-arm run) across all forward-looking surfaces.
- **`evaluate` docstrings** (Rust, Python, Node) now distinguish
  self-eval (measures *focus*) from gold-conditional (measures
  *correctness*). Without gold, evaluate's metrics tell you whether the
  context is query-focused, not whether the right answer is in there.
- **`Stripper`'s analyzer-token semantics** documented in the new
  helper's rustdoc + Python/Node docstrings + `llms.txt`. The
  silent-no-op failure mode is now explicitly named alongside the fix.
- **`feedback_discipline` memory refined.** The "bounded architecture"
  rule used to forbid "chains" wholesale; refined to distinguish
  composition of compiled deterministic rewrites (in scope — that's
  what `context_with_rewrites` is) from runtime branching among options
  (still out of scope).

### Changed (runtime behavior)

- **`retrieval="hybrid"` is now pure rerank, not RRF fusion.** The
  previous behavior fused BM25 and dense rankings with Reciprocal Rank
  Fusion (RRF) to guarantee BM25-strong hits never got demoted by the
  dense step (the original "issue #1" safety). The
  [MULTIHOP_CONSTANT_CHUNKING](docs/findings/MULTIHOP_CONSTANT_CHUNKING.md)
  probe revealed RRF was burying compositional-multi-hop bridge
  passages (low BM25 rank + high dense rank get averaged-down). Now
  `LocalRerankRetriever::retrieve` returns the dense-sorted top_K
  directly; unembedded code chunks from the BM25 pool are appended at
  the tail to preserve the issue-#1 safety for them. Measured impact
  (n=100 each, same `bench/multihop_hybrid_competitors_probe.py`
  harness): MuSiQue ≥0.8 retention 26% → **34% (+8)**, HotpotQA ≥0.8
  83% → **81% (−2)**. Net positive: 4× the MuSiQue benefit vs the
  HotpotQA cost. Two unit tests updated to assert dense-winner ordering
  rather than RRF-method markers
  ([crates/redhop/src/retrieval/local_rerank.rs](crates/redhop/src/retrieval/local_rerank.rs)).
  Users who want explicit RRF fan-out can still construct
  `HybridRetriever::rrf(...)` from `crate::retrieval::hybrid`.

### Fixed

- **Two latent test-compile breaks under `--no-default-features`**, both
  caught by the new feature-matrix sweep:
  - `crates/redhop/tests/files_extract.rs` imported `redhop::files::*`
    unconditionally — gated with `#![cfg(feature = "files")]`.
  - `crates/redhop/tests/quality_suite.rs` imported `redhop::read_bytes`
    and used `serde_json::Value` metadata-access patterns whose type
    inference relies on `files`-gated traits — gated with the same
    attribute. CI now runs `cargo test -p redhop --features files,semantic`
    in addition to the default-features `cargo test --workspace` so these
    integration tests don't silently never execute.
- **3 stale numbers in `docs/findings/CUAD_CLAUSE_EXPANSION.md`** (lines
  23, 26, 150) that contradicted the same file's header — aligned to
  81.3% → 87.7% / +4.7 with the fair-preprocessing caveat next to the
  +4.7 number.

## [0.3.0] — 2026-06-07

The **workflow + measurement** release. Ships a new public-API surface
that closes the templated-workload retention gap end-to-end, in all three
bindings (Rust, Python, Node): `analyze_query_set`, the `QueryRewrite`
trait with two built-in implementations (`Stripper` and `Vocabulary`),
`Document::context_with_rewrites(...)` to compose them with an audit
trail, `Vocabulary::enrich(...)` as the chunk-side mirror, and `evaluate`
for deterministic A/B with no LLM judge. On the CUAD framework comparison
the full **detect → compile → context_with_rewrites → A/B** workflow takes
≥0.8 retention from **81.3% → 90.7%** — a 9.4-point lift over raw BM25,
beating LlamaIndex's 86% by 4 points, at native BM25 latency (~2.5ms/query)
on default lexical retrieval. Worked example, hand-curated CUAD clause-name
dictionary, and a 6-arm probe contrasting the workflow vs hybrid+cross-encoder
live in `docs/findings/CUAD_CLAUSE_EXPANSION.md` and
`docs/findings/CUAD_HYBRID_RERANK.md`.

`Vocabulary.enrich(...)` ships with **bidirectional measured evidence on the
regime rule it follows.** Positive side: `docs/findings/SPIDER_ENRICH.md`
measured **+0.19 mean column recall** on Spider-shape schema retrieval (curated
workload synonyms; n=30, candidate_k=10). Negative side:
`docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md` measured **−2.0 pts** on
CUAD prose chunks. The two findings together complete the **four-corner
rule** with measured evidence on all four corners: workload-pervasive
signal manipulation fails on either side of the pipeline; only
workload-curated semantics work. See `docs/findings/VOCABULARY_ENRICH.md`
for the regime rule, use-case ranking, and failure modes.

**Breaking on the manual-chunks path (Python + Node):** the typed
`redhop.Chunk(text, *, source=None, id=None, metadata=None, ...)` constructor
becomes the *only* accepted input shape for `Document.from_chunks` and the
low-level `build_context` / `filter_context` / `analyze_context` /
`context_economics` entry points. Bare strings and dicts both raise
`ValueError` with a migration hint pointing at the new constructor. The
trade-off is intentional: the dict path didn't expose chunk metadata at all,
so manually-constructed chunks couldn't carry `page` / `heading` / `line`
into citations — a real functional gap, not just ergonomics. The typed
`Chunk` closes that gap and surfaces `source` (provenance) and `id`
(identity) as the two distinct concepts they already are in the Rust core
(see Breaking below for the migration).

### Added

#### Templated-workload helpers (Rust + Python + Node)

- **`analyze_query_set(queries) → QuerySetReport`** — diagnostic that takes
  a representative sample of your queries and reports whether they share
  enough boilerplate to be templated, which terms are doing the dilution,
  and a coarse `estimated_dilution_cost` band. Cross-workload probe
  (`docs/findings/QUERY_SET_ANALYZER.md`): CUAD fires (share 0.66, cost
  high); HotpotQA + MuSiQue both stay quiet (0.00 and 0.12, both
  `is_templated=False`). Conservative by design — false positives push
  users toward a workaround that won't help, which is worse than staying
  quiet.
- **`QueryRewrite` trait + `Stripper` + `Vocabulary`** — compiled,
  observable, token-level-correct replacement for the function-form
  rewrites originally drafted for this release. Each `QueryRewrite`
  implementation returns a `RewriteResult { query, record }` so every
  stage's `{stage, from, to, matched, added, removed}` lands on
  `ContextReport::query_rewrites` automatically when called through
  the chain.
  - **`Stripper::new(boilerplate)`** — compiled boilerplate-removal
    rewrite. Matches at token granularity through the analyzer (with a
    surface-form fallback for tokens like "of"/"the" that stem to
    empty), so a single-token strip cannot accidentally erase a
    substring inside a longer word (an `"of"` strip does **not** erase
    the `"of"` inside `"office"`). Replaces the substring-based
    `drop_template_terms` function originally drafted for 0.3.0.
  - **`Vocabulary::new(entries)` / `Vocabulary::bidirectional(entries)`**
    — compiled workload-curated equivalence classes. Tokenizes keys,
    synonyms, and the query through the same analyzer the BM25 index
    uses, so a vocabulary key `"ip"` cannot fire on the `"ip"` inside
    `"recipient"`. Bidirectional mode treats every class member as a
    trigger (PTO ↔ "paid time off" ↔ "vacation"). The CUAD probe
    (`docs/findings/CUAD_CLAUSE_EXPANSION.md`) shows +3.0 points on top
    of the template-stripped baseline (the new token-level matching
    re-validates at 90.7% vs the substring-based predecessor's 90.3%
    — same workload, +0.4 from analyzer alignment).
  - **`Document::context_with_rewrites(query, &[&stripper, &vocab])`**
    — runs the chain left-to-right through retrieval. Each stage sees
    the previous stage's output; the per-stage `RewriteRecord`s land on
    `ctx.report.query_rewrites` automatically.
  - **Future-extensible.** Both `Stripper` and `Vocabulary` are
    `QueryRewrite` implementations; user code can ship its own (e.g. a
    workload-specific normalizer) and chain it alongside the built-ins.
    The trait is exported on the public API surface.
  - **`Vocabulary::enrich(chunk) → RewriteResult`** — chunk-side
    mirror of `apply` shipped as a primitive on **mechanism reasoning
    with asymmetric measured evidence**. The mechanism (a chunk-side
    doc2query variant) and the regime hypothesis
    (`expected value ∝ shortness × opacity × dictionary-exists`) are
    well-grounded; the *positive* prediction (short opaque coded
    units — schema columns, API symbols, error codes) is **not yet
    measured by RedHop**. Spider/BIRD as the schema-regime probe is
    queued, not run. The *negative* prediction (long prose chunks
    + workload-pervasive vocabulary will dilute, not help) has been
    measured directly:
    [`CUAD_ENRICH_DEFINITIONS_NULL`](docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md)
    regressed retention −2.0pt vs the 90.7% workflow baseline
    (~24-point loss on the 17/50 affected contracts). This completes
    the four-corner rule from CUAD_PRF_NULL + SUB_IDF_AUTO_DROP_NULL
    onto the chunk side: workload-pervasive signal manipulation fails
    on either side of the pipeline. Users adopting `enrich` should
    A/B on their own corpus with `redhop::evaluate(...)` —
    the regime rule is a hypothesis, not a guarantee. Audit trail
    (per-chunk `RewriteRecord` with `stage: "enrich"`) returned to
    the caller so the A/B is auditable. Synthetic demo (not a
    benchmark): `crates/examples/examples/enrich_code_search.rs`.
    Full asymmetric-evidence framing + use case predictions + failure
    modes in `docs/findings/VOCABULARY_ENRICH.md`.
- **`evaluate(query, ctx, gold) → EvalReport`** — in-process retrieval-eval
  scorer, no LLM judge. Self-eval (`mean_grounding`, `evidence_density`,
  `retained_evidence_ratio`, `second_hop_rescues`, `low_confidence`,
  `estimated_waste_tokens`) is always populated; gold-relative metrics
  (`context_recall`, `context_precision`, `answer_token_recall`) are
  optionally unlocked by passing `gold_chunks` and/or `gold_answer`.
  Composite `overall` blends whichever fields are available. Designed as
  a *refraction* of the same primitives the runtime uses to make its
  Decision Report — a low `overall` and `report.low_confidence_retrieval`
  are the same signal viewed twice, not independent measurements, so eval
  and runtime can never disagree. Rationale, contract details, and the
  10 / 11 / 9 Rust / Python / Node tests pin in
  `docs/findings/EVALUATE_API.md`.

#### Findings (the evidence layer)

New findings document what was tried, what worked, and what was
falsified across this release:

- **Confirmed** — `QUERY_SET_ANALYZER`, `CUAD_RECALL_GAP`,
  `CUAD_CLAUSE_EXPANSION`, `MULTILINGUAL_ANALYZER`, `EVALUATE_API`,
  `CUAD_HYBRID_RERANK` (substitute-not-stack rule), `VOCABULARY_ENRICH`
  (confirmed on both sides of the regime rule), `SPIDER_ENRICH`
  (the positive-side validation for `Vocabulary.enrich(...)`: curated
  chunk-side enrichment on a Spider-shape sample lifted mean column
  recall +0.19 from 0.77 → 0.97, ≥0.8 retention 63% → 93%).
- **Null result / falsified** — `CUAD_PRF_NULL` (unweighted PRF on
  boilerplate-heavy corpora), `CUAD_CHUNK_FRAGMENTATION_NULL` (chunker
  isn't the CUAD lever), `SUB_IDF_AUTO_DROP_NULL` (corpus-only IDF
  manipulation fails in both directions),
  `CUAD_ENRICH_DEFINITIONS_NULL` (chunk-side enrich on per-contract
  Definitions regressed −2.0 pts vs the 90.7% workflow baseline;
  ~24-point loss on the 17/50 contracts where Definitions were
  extractable — chunk-side parallel to CUAD_PRF_NULL's failure mode,
  measured directly).
- **The four-corner rule is now measured on all four corners.**
  Workload-pervasive signal manipulation fails on either side of the
  pipeline; only workload-curated semantics work:
  query-side curated wins (`CUAD_CLAUSE_EXPANSION` +3.0pt) ·
  query-side auto fails (`CUAD_PRF_NULL` −3.7pt) ·
  chunk-side curated wins (`SPIDER_ENRICH` +0.19 mean recall) ·
  chunk-side auto fails (`CUAD_ENRICH_DEFINITIONS_NULL` −2.0pt).

#### Examples

Eleven new harnesses under `crates/examples/examples/`:
`cuad_query_preprocessing`, `cuad_chunk_strategy_sweep`,
`cuad_chunk_fragmentation`, `cuad_clause_expansion`, `cuad_hybrid_rerank`,
`cuad_perf`, `cuad_prf`, `cuad_rust_vs_python_path`,
`multilingual_query_set_probe`, `query_set_analyzer_probe`,
`sub_idf_reweighting_probe`.

#### Documentation

- New workflow-lift chart `.github/workflow_lift.svg` embedded in the
  root README + binding READMEs — surfaces the 81 → 88 → 90.7% story
  visually.
- Root README, `python/README.md`, `nodejs/README.md` "Templated
  workloads" section rewritten to detect → strip → (optional) vocabulary →
  A/B with `Stripper` / `Vocabulary` / `context_with_rewrites` tabled.
- `docs/CHOOSING_A_CONFIG.md` step 3 leads with the new "two paths up
  the same hill" decision table contrasting `retrieval="hybrid"` (the
  one-knob alternative) vs BM25 + the helpers (best-quality).

#### Chat-RAG and chronology preservation

- **`ContextConfig::preserve_order: bool`** — new field (default `false`,
  no behavior change for existing callers). When set, the assembled
  context emits selected chunks in **source-document order** instead of
  the strategy's relevance-emitted order. The selection step is
  untouched; only the final ordering changes. Designed for chat
  histories, narrative transcripts, and sequential logs where
  chronology / causality matters and a relevance-ranked emission would
  destroy the meaning ("after the refund came in" reads strangely if
  presented before "ordered the laptop").
- The sort key is `(source, chunk_position)` where `chunk_position`
  prefers a `chunk_index` metadata field (stamped automatically by
  `Document::from_chunks_with` based on input order, so caller-supplied
  chunks via `from_chunks` get a stable chronology key for free) and
  falls back to the chunker's existing `sentence_range.start` for
  text-loaded paths.
- Exposed across all three bindings:
  - **Rust** — `ContextConfig { preserve_order: true, .. }`; flows
    through `LoadOptions::preserve_order` for the `text()` /
    `chunks()` paths.
  - **Python** — `redhop.Document.from_text(text, preserve_order=True)`
    and `from_chunks` / `from_file` / `from_bytes`; also exposed on the
    low-level `redhop.build_context(query, chunks, preserve_order=True)`
    and `redhop.filter_context(...)`.
  - **Node** — `Document.fromText(text, { preserveOrder: true })` and
    siblings; also a `preserveOrder?: boolean` field on the
    `ContextOptions` shape consumed by `buildContext` and `filterContext`.
- Worked example:
  [`crates/examples/examples/chat_rag.rs`](../../crates/examples/examples/chat_rag.rs)
  shows a 12-turn chat where, on the query `"shipping refund label
  return"`, the strategy picks four turns by relevance — preserve_order
  off emits them in `[turn-08, turn-03, turn-05, turn-06]` (relevance);
  preserve_order on emits them in `[turn-03, turn-05, turn-06, turn-08]`
  (chronological), so the LLM reads what was said in the order it was
  said. 3 new Rust unit tests pin the contract
  (`preserve_order_off_emits_relevance_order`,
  `preserve_order_on_emits_document_order`,
  `preserve_order_groups_by_source`).

### Changed

- **Package registry URLs now point at `https://www.redhopai.com`** as
  the canonical `Homepage`, with the GitHub repo kept as `Repository`
  (PyPI) / `repository` (npm) / `repository` (crates.io). Before this,
  PyPI displayed two identical "Homepage" and "Repository" links both
  pointing at GitHub; npm displayed neither. PyPI also gains
  `Documentation`, `Changelog`, `Issues`, and `Evidence layer` link
  entries; npm gains `homepage`, `repository`, `bugs`, and an
  expanded `keywords` array (`reasoning`, `embeddings` added).
- **Findings master table refreshed** with new rows on
  `/docs/benchmarks/` (website) and `docs/findings/README.md` (repo).
  Framework comparison row updated: the CUAD headline is now
  `90.7%` via `Stripper` + `Vocabulary` (was `88%` via strip alone),
  beating LlamaIndex by 4 points. `VOCABULARY_ENRICH` row promoted from
  *asymmetric measured evidence* to *confirmed on both sides of the
  regime rule* after the `SPIDER_ENRICH` probe landed.
- **`RewriteResult.query` field renamed to `RewriteResult.text`**
  (Rust). The same struct is the output of both query-side
  `QueryRewrite::apply` and chunk-side `Vocabulary::enrich`. The old
  `query` field name read awkwardly on the enrich path
  (`vocab.enrich(chunk_text).query` describes a chunk, not a query);
  `text` is neutral and accurate for both directions. The audit-record
  `stage` field is the signal of which side of the pipeline emitted
  the result (`"strip"` / `"vocabulary"` / `"enrich"`). Pre-publish
  rename — no callers exist outside the repo yet, but flagging for
  anyone building from source on a pre-release commit.
- **User-facing docs (`README.md`, `python/README.md`, `nodejs/README.md`,
  website) elevate the rewrite chain + audit trail + `evaluate` to a
  dedicated "Show your work" section.** The 0.3.0 differentiator
  versus other RAG frameworks is *every transform is observable on
  the same Decision Report and every change is A/B-scoreable without
  an LLM judge*; the previous docs surfaced the 3-call surface plus
  citations but understated the rewrite/audit/evaluate combo. The new
  section appears on every binding's README and as both a homepage
  card and a section on the website.

### Fixed

- **`Document.from_folder` was constructing `LoadOptions` without
  `preserve_order` under `--features files,semantic`.** Caught
  locally while writing `examples/python/07_retrieval_tiers.py` (a
  full-feature build). The bug was hidden in the lean (no-features)
  default build because the missing-field code path was behind
  `#[cfg(feature = "files")]`. The default published wheel ships with
  `features = ["files", "semantic"]`, so end users would have hit it.
  Fixed; all 4 feature configurations (`--no-default-features`,
  `--features files`, `--features semantic`, `--features files,semantic`)
  now compile cleanly.

### Breaking — `redhop.Chunk` is now the only accepted manual-chunks shape

- **`Document.from_chunks` + `build_context` + `filter_context` +
  `analyze_context` + `context_economics` now require typed
  `redhop.Chunk(...)` instances.** Bare strings and plain dicts both
  raise `ValueError` with a migration hint:
  ```
  chunk 0: expected redhop.Chunk(text, source=..., ...); got str. As of
  0.3.0, strings and dicts are no longer accepted — wrap your input as
  `redhop.Chunk(text, source='myfile.txt')`.
  ```
- **What the new constructor looks like:**
  ```python
  redhop.Chunk(
      text,
      source=None,       # provenance: file path / URL / logical handle
      id=None,            # identity: stable id, defaults to c0, c1, …
      metadata=None,      # open dict; citations read page/heading/line
      token_count=None,   # auto from whitespace if omitted
      embedding=None,     # for pre-computed dense vectors
  )
  ```
  Node mirrors with `new redhop.Chunk(text, { source, id, metadata, tokenCount, embedding })`.
- **Why this is now a breaking change instead of a backward-compat additive:**
  the dict path didn't accept `metadata` at all, so manually-supplied
  chunks couldn't carry page/heading/line into citations. The two-ways-
  to-do-it cleanup is incidental; closing the metadata gap is the real
  reason. Strict typing also surfaces `source` (provenance) and `id`
  (identity) as distinct concepts the way the Rust core has always
  treated them — the dict path conflated them in practice.
- **Migration:**
  | Before | After |
  | --- | --- |
  | `from_chunks(["a", "b"])` | `from_chunks([redhop.Chunk("a"), redhop.Chunk("b")])` |
  | `from_chunks([{"text": "a", "source": "x.md"}])` | `from_chunks([redhop.Chunk("a", source="x.md")])` |
  | `from_chunks([{"text": "a", "id": "x", "source": "y.md"}])` | `from_chunks([redhop.Chunk("a", id="x", source="y.md")])` |
  | `buildContext(q, [{ id, text }, ...])` (Node) | `buildContext(q, [new Chunk(text, { id }), ...])` |
- **What's new on the typed-chunks path:** citations now pick up `page`,
  `heading`, and `line` from `metadata={...}` on chunks the user built
  themselves. Before 0.3.0 those fields were always `None` on the
  manual-chunks path — only the file loaders populated them.
- **Rust callers unaffected.** The `redhop::core::Chunk` struct hasn't
  changed shape. `Document::from_chunks(Vec<Chunk>)` still takes
  `Vec<redhop::core::Chunk>` exactly as it did. A new public facade
  `redhop::chunks_typed(Vec<Chunk>, &LoadOptions)` was added so the
  bindings can route pre-formed chunks through the indexing pipeline
  without going through the chunker (preserving 1-to-1 chunk identity).

### Breaking (Node only — Python and Rust callers unaffected)

- **Node `BuiltContext` is now a `#[napi]` class** (was a plain
  `#[napi(object)]`). The four exposed properties (`text`, `chunks`,
  `citations`, `report`) remain readable as JS properties via getters, so
  existing user code that does `ctx.text`, `ctx.chunks`, etc., continues
  to work unchanged. The TypeScript type changes from
  `interface BuiltContext { … }` to `class BuiltContext { … }`. The
  reason for the change is that `redhop.evaluate(query, ctx, …)` needs
  access to the underlying Rust struct (chunk IDs, the full report shape)
  which a plain object can't carry.
  - **What breaks:** if you were `JSON.stringify(ctx)`, class getters
    aren't enumerable by default and the output will be `{}` instead of
    the four-field object. Project to a plain object explicitly:
    `JSON.stringify({ text: ctx.text, chunks: ctx.chunks, citations: ctx.citations, report: ctx.report })`.
    No other behavior changes.

## [0.2.2] - 2026-06-06

The **binding parity + evidence layer** release. No breaking changes for any
binding's callers. The Node binding gains 14 missing `Report` fields, the
documentation gets its first visual presentation (badges, charts,
architecture diagram), and the evidence layer grows by five new findings
that document what was tried, what worked, and what was falsified honestly.

### Added

#### Node binding — full `Report` field-surface parity with Python

- **`Report` gains 14 fields** + a permanent alias: `strategy`,
  `requestedStrategy`, `inputTokens`, `tokenBudget`, `tokenUtilization`,
  `nInputChunks`, `nSelected`, `inputDistractorRatio`,
  `reasoningPreservationDelta`, `distractorsPruned`, `removedTotal`,
  `evidenceDensity`, `distractorRatio`, `estimatedWasteTokens`, plus
  `secondHopRescueCount` (== `secondHopRescues`, the existing short
  name; both names will always be present and equal). Before this
  release Node's `Report` exposed roughly half of Python's surface —
  programmatic callers using `report.totalTokens` could not read
  `report.nSelected` or any of the economics fields. All additions are
  non-breaking; no existing field changed name or shape.
- **`docs/API_STABILITY.md`** gains a "Known call-shape asymmetries"
  section documenting the two pre-existing idiomatic differences
  between the Python and Node bindings (`from_text` positional vs
  options-bag `source`; `ctx.text()` callable in Python vs `ctx.text`
  property in Node). These are stable within 0.x — neither will be
  silently flipped.

#### README + binding-page presentation

- **Multi-registry badges at the top of every README** (root, Python,
  Node). PyPI / crates.io / npm version numbers, license, and a link
  to the evidence layer. Brand color (`#e11d48`) on the registry
  badges.
- **A retention-vs-frameworks bar chart** (`.github/retention_vs_frameworks.svg`)
  showing the measured head-to-head numbers from
  [`FRAMEWORK_COMPARISON.md`](docs/findings/FRAMEWORK_COMPARISON.md):
  HotpotQA multi-hop (RedHop 77%, LangChain 71%, LlamaIndex 72%) and
  CUAD contracts (RedHop 82%, LangChain 73%, LlamaIndex 86%). Hand-rolled
  SVG, no fake screenshots, every number traces to
  [`reports/framework_comparison.txt`](reports/framework_comparison.txt).
- **A pipeline architecture diagram** (`.github/architecture.svg`)
  showing the five stages — Document → Chunking → Retrieval →
  Allocation → BuiltContext — with the calibrating finding named under
  each internal stage and a "YOU BRING / REDHOP OWNS / YOU GET" scope
  label.
- **A Decision Report visual** (`.github/decision_report.svg`) —
  terminal-styled SVG rendering of `ctx.report` output. Same content
  as the ASCII block (which is preserved under a collapsed `<details>`
  for copy-paste).
- **A References section** at the bottom of the root README, citing
  the named work each piece of the runtime leans on: BM25 (Robertson &
  Zaragoza 2009), Porter2 (Porter 2001), RRF (Cormack et al. 2009),
  Lost-in-the-Middle (Liu et al. 2023), NQC (Shtok et al. 2012), MDR
  (Xiong et al. 2021), and the HotpotQA/MuSiQue/CUAD evaluation
  datasets. Each citation links to the finding doc that uses that
  work.
- The binding READMEs (`python/README.md`, `nodejs/README.md`) get
  the architecture + Decision Report visuals via absolute
  `raw.githubusercontent.com` URLs so they render on PyPI and npm
  package pages (not just on GitHub).

#### Structural test suite (`crates/redhop/tests/`)

- **`proptest_invariants.rs`** — 9 property-based invariants
  pinning `build_context`'s behavior under random valid inputs:
  `build_context_never_panics`, `resolved_strategy_is_never_auto`,
  `auto_decision_triangle_holds`, `selection_is_subset_of_input`,
  `no_duplicate_chunk_ids_in_output`, `token_budget_respected`,
  `report_counts_match_reality`, `report_ratios_are_finite_and_in_range`,
  `build_context_is_deterministic`. Adds `proptest = "1"` as a
  dev-dependency. Catches edge cases hand-written tests miss by
  construction (empty input, NaN scores, all-stopword query, single-token
  budget, …) — the bug class structural tests exist to close off.
- **`default_calibration.rs`** — 9 pins binding each tuned default
  on `ContextConfig::default()` / `DocumentConfig::default()` to the
  finding that calibrated it (`token_budget = 8192`,
  `auto_passthrough_max_tokens = 1500`, `distractor_min_grounding = 0.10`,
  `redundancy_max_cosine = 0.92`, `link_min_jaccard = 0.12`,
  `low_confidence_max_grounding ≡ distractor_min_grounding`,
  `target_tokens = 128`, `candidate_k = 20`, `Document.strategy = Auto`).
  Silent default drift becomes impossible; intentional drift is
  documented in the same commit that makes it.
- **`public_api_snapshot.rs`** — 5 compile-time guards against silent
  rename/removal of public symbols, plus type-bound signature pins for
  load-bearing functions and exhaustive-match pins on
  `ContextStrategy` / `AutoDecision` / `RetrievalMethod` string
  variants. Caught 4 real signature mistakes during authoring.
- **`golden_quality.rs`** — 6 end-to-end retrieval-quality canaries
  on small inline corpora: G01 lexical-keyword match, G02 stemming
  finds morphological variants, G03 distractor doesn't dominate
  relevant chunk, G04 second-hop chunk survives default assembly, G05
  low-confidence signal fires on off-corpus queries, G06 citations
  carry correct source. The canary that fires between benchmark runs
  when RedHop "got dumber" on a real query shape.
- **Three field-set parity tests** (`test_report_field_surface_parity`,
  `test_built_context_field_surface_parity`,
  `test_context_economics_field_surface_parity` in
  `python/tests/test_parity_node.py`). Each compares the SET of fields
  each binding exposes for the named return type. Auto-catches the gap
  class where a new `#[getter]` (Python) or `pub` field (Node) appears
  on one side without the other keeping up — the failure mode that hid
  the 14-field Node `Report` gap until a smoke test stumbled on
  `strategy`.

#### Evidence layer — five new finding documents

- **[`MUSIQUE_RECALL_GAP.md`](docs/findings/MUSIQUE_RECALL_GAP.md)** —
  decomposes the dense recall gap between HotpotQA (0.76) and MuSiQue
  (0.28) into five distinct contributors (gold density, retrieval signal
  type, wide-net coverage, embedder capacity, chunking) and documents an
  attempted full-pool RRF refactor of `RetrievalMode::Hybrid` that an
  honest A/B benchmark falsified. Branch `feature/hybrid-full-pool-rrf`
  on origin holds the working refactor as a research record; main keeps
  the existing Hybrid behavior. Includes 5 reproducible example
  harnesses under `crates/examples/examples/musique_*.rs` and
  `hybrid_old_vs_new.rs` (the A/B that closed the question).
- **[`RERANKING_LIMITS.md`](docs/findings/RERANKING_LIMITS.md) Update —
  2026-06-06 (kind-label gate)** — falsifies both directions of the
  HotpotQA-type-label gate proposed in the original finding's "open
  problem" section. Closes that probe.
- **[`RERANKING_LIMITS.md`](docs/findings/RERANKING_LIMITS.md) Update —
  2026-06-06 (later, grounding gate)** — documents the discovery and
  cross-corpus falsification of a `grounding_top1 ≤ 0.35` gate that
  worked on HotpotQA (+0.031 lift, robust to 5-fold CV) but failed to
  generalize to MuSiQue. Also covers an NQC + WIG cross-corpus probe
  that didn't port. Closes the CE-gate research direction with a
  measured negative result. Includes the Phase A feature-logging
  harness (`crates/examples/examples/ce_gate_feature_log{,_musique}.rs`)
  for any future probe.
- **[`DENSE_RERANK_CEILING.md`](docs/findings/DENSE_RERANK_CEILING.md)
  Update — 2026-06-06** — falsifies MDR single-pass as a uniform policy
  (−0.05 vs dense baseline) while documenting a real +0.027 lift on the
  subset of queries where dense had a gold in the pool but missed it.
  Closes the single-shot MDR probe.
- **[`LOCAL_RERANK.md`](docs/findings/LOCAL_RERANK.md) Update —
  2026-06-06** — notes the status of `LocalRerankRetriever` after the
  MuSiQue investigation: it is now a building block rather than the
  default `Hybrid`, but the "semantic recall without ANN" contract it
  established is intact and the working refactor is preserved on
  `feature/hybrid-full-pool-rrf` for future re-evaluation.

### Changed

- **`python/tests/test_parity_node.py`** now pins `strategy` +
  `requested_strategy` data-value parity (in addition to the field-set
  parity above). The harness previously *normalized away* these fields
  rather than testing them — direct dict-key access means a future
  regression that drops either field on either side fails with a clear
  `KeyError` instead of silently passing.
- Documentation polish: `python/README.md`'s `from_text` row shows the
  optional `source=` parameter; `nodejs/README.md` lists the full
  `report` shape; the top-level README's `"hybrid"` row in the
  Retrieval tiers table accurately reflects the shipped semantics.

### Notes on the runtime

- **`RetrievalMode::Hybrid` is unchanged** for this release. A full-pool
  RRF refactor was built end-to-end on `feature/hybrid-full-pool-rrf`
  (commit `c81ffbe`, all tests passing, fmt + clippy clean) but a direct
  A/B benchmark falsified the ship decision: at the user-facing
  `candidate_k = 20` the new composition gave only +0.0074 on MuSiQue
  and +0.0017 on HotpotQA (both below the +0.02 pre-registered ship bar)
  and regressed HotpotQA at K=4 by −0.011. The wide-K wins are real
  (+0.07 MuSiQue@50, +0.034 HotpotQA@50) but not user-facing.
  `LocalRerankRetriever`'s BM25-prune-then-RRF composition is still
  what `RetrievalMode::Hybrid` resolves to. Full A/B numbers and the
  ship-decision audit are in
  [`MUSIQUE_RECALL_GAP.md`](docs/findings/MUSIQUE_RECALL_GAP.md).

## [0.2.1] - 2026-06-06

The **robustness + bugfix** patch release. Two real bugs fixed (one BM25
edge case, one cross-binding serde-compat break), one new Python helper,
and ~30 new tests pinning load-bearing contracts across the codebase.

### Fixed

- **BM25: silent wildcard fallback on no-signal queries.** Queries whose
  every term was filtered out (stopwords only, or all-out-of-vocab)
  silently fell back to a match-all wildcard, returning the corpus's
  top-BM25 chunks as if the query had matched something. Now returns
  an empty result set with a clear signal.
- **`ContextReport.removed` and `.economics` missing `#[serde(default)]`.**
  A binding payload from an older RedHop binary missing these fields
  would error on deserialize — a silent cross-version compatibility break
  for Python/Node callers shuttling `ContextReport` across the FFI as
  JSON. Both target types already derive `Default`; the fix is a no-op
  for fresh payloads and gracefully fills in zeros for old ones.

### Added

- **`redhop.context_with_timeout` (Python).** Thin `ThreadPoolExecutor`
  watchdog around `Document.context()` for agent integrations that need
  to bail on slow queries:

  ```python
  try:
      ctx = redhop.context_with_timeout(doc, q, timeout_ms=5000)
  except TimeoutError:
      ...
  ```

  Forwards `budget` / `neighbors` / `include_heading`. Scope is
  deliberately Python-only — true Rust-side cancellation needs hooks in
  Tantivy/ONNX that don't exist yet, and the docstring + `TimeoutError`
  message document the limitation.

- **`docs/DEFAULT_PROVENANCE.md`** — every tuned default in
  `ContextConfig` / `DocumentConfig` linked back to the finding that
  justifies it (so callers can audit which numbers are calibrated vs
  arbitrary).

### Internal — robustness tests

Seven new test passes (~30 tests) pinning load-bearing contracts that
were previously informal:

- **Determinism** — same input → same output, Rust + cross-binding parity.
- **Internal invariants** — 7+ consistency invariants across the strategy
  matrix (selected ⊆ input, `removed.total` matches drop count, etc.).
- **Concurrency** — `Send + Sync` audit + 1024-call parallel stress.
- **Adversarial loaders** — 9 tests covering corrupt PDFs, symlink loops,
  deep recursion, malformed DOCX/PPTX/XLSX.
- **Auto-gate boundary** — pins the inclusive `<=` semantics at
  1499/1500/1501 input tokens + the custom-gate path.
- **Serde round-trip** — every cross-FFI type (`Chunk`, `Score`,
  `ContextReport`, ...) survives JSON round-trip; forward-compat
  exercised via a minimal pre-0.1.3 payload.
- **Strategy semantics** — 7 differential tests pinning the contrasts
  between all 5 `ContextStrategy` variants on a shared corpus
  (catches accidental strategy convergence).
- **Persisted cache** — incremental cache hit/miss contract for
  `read_folder_with(persist=true)`: per-file `(mtime, size)` skip,
  no-op reload doesn't rewrite, fingerprint invalidation on config
  change, deleted-file cleanup.

No public API changes. Python and Node callers are unaffected aside
from the new `context_with_timeout` helper.

## [0.2.0] - 2026-06-03

The **binding-parity + non-English** release. Three months of incremental
quality work plus a focused arc on cross-binding consistency: Python, Node,
and Rust all expose the same surface, return the same values for the same
inputs, and drift is now actively prevented in CI. The Rust crate also gains
a pluggable lexical analyzer, closing the structural bug class (BM25 ↔
grounding-scorer disagreement) that 0.1.3–0.1.4 fixed by hand four times.

### Breaking changes

Two source-level breaks for Rust callers; `..Default::default()` and the
`pip`/`npm` consumers are unaffected:

1. **`ContextConfig` + `DocumentConfig` grew new required fields**
   (`analyzer: Arc<dyn Analyzer>`) for the pluggable lexical analyzer.
   Callers constructing those structs via field literals from outside
   the crate need to add `analyzer: redhop::analyzer::default_english()`.
2. **`ContextConfig::default().token_budget`** changed from **2048 → 8192**
   to align with the Python binding's long-standing default (which was
   shipping to PyPI users that whole time). Rust callers relying on the
   old 2048 default will now get a 4× larger assembled context. Set
   `token_budget: 2048` explicitly to restore the old behavior. Python +
   Node users see no change.

### Added

#### Pluggable lexical analyzer

- **`crate::analyzer::Analyzer` trait** + **`SnowballAnalyzer`** (18
  Snowball Porter2 languages). First-class extension point: one analyzer
  drives BOTH the BM25 retriever AND the grounding scorer, so the two
  layers structurally cannot disagree on what "the same term" means.
  Design rationale in `docs/design/ANALYZER_PLUGIN.md`; usage in
  `docs/LANGUAGE.md`.
- **`Document::with_analyzer(Arc<dyn Analyzer>)`** — mirrors
  `with_embedder`. Swaps the analyzer for both layers in lockstep.
- **`LoadOptions::language: Option<String>`** — string-routed access to
  the 18 builtins (`english`, `german`, `french`, `spanish`, `italian`,
  `portuguese`, `dutch`, `russian`, `swedish`, `norwegian`, `danish`,
  `finnish`, `romanian`, `hungarian`, `turkish`, `arabic`, `greek`,
  `tamil`). Unknown language names return an error (no silent fallback
  to English).
- **Python `language` kwarg** on every `Document.from_*` constructor.
- **Node `language` field** on `Options`.

#### Binding parity (Node catches up to Python)

- **`Document.analyze(query)`** — pure diagnostics, returns the same
  `Report` shape as `context().report` without paying assembly cost.
- **`Document.nFiles`** getter — number of source files indexed (`1`
  for single-source ctors, the readable count for `fromFolder`).
- **`Document.skippedFiles`** getter — `SkippedFile[]` (`{source,
  reason}` pairs) for files `fromFolder` couldn't parse. Was a silent
  skip with no introspection before.
- **`buildContext` / `filterContext` / `analyzeContext` /
  `contextEconomics`** top-level functions — the low-level "I do my own
  retrieval, just want RedHop for assembly" surface. Mirrors Python's
  same-named functions; takes `ChunkInput[]` + `ContextOptions`.
- **`groundingScore(query, text)`** + **`linkStrength(a, b)`** — the
  observability primitives the strategies use internally, exposed so
  external code reuses RedHop's exact relevance notion instead of
  reimplementing.

#### Tests + infrastructure

- **`crates/redhop/tests/quality_suite.rs`** — 45-test behavior-level
  suite organized by what a user perceives, not by code structure.
  Covers tokenization (T01-T07), multi-field reach (T08-T09), document
  structure (T10-T13), context assembly (T14-T20), hybrid contract
  (T21-T22), edge cases (T23-T26), Unicode/multilingual (T27-T30),
  adversarial queries (T31-T34), nested markdown (T35), cross-format
  mixed corpus (T36), non-English pinning (T37-T40), and the analyzer
  plugin (T41-T45). Found two real bugs on its first runs (an
  empty-query BM25 crash and an accent-folding gap), and a binding bug
  via T41-T44 (`from_chunks` silently dropping `language=` in Python).
- **`python/tests/test_parity_node.py`** + **`nodejs/test/parity_runner.cjs`**
  — cross-binding parity harness. 6 tests hand identical inputs to
  Python and Node and diff structured outputs (caught the
  `analyzeContext` / `contextEconomics` `token_budget` divergence on
  its first run).
- **`crates/cli/tests/cli_smoke.rs`** — first-ever CLI integration
  tests. Asserts `--help` works on each subcommand + a real
  `analyze-context -` stdin pipe.
- **Node CI job** — `.github/workflows/ci.yml` now builds the napi
  addon and runs `npm test` on PRs. Previously PRs only exercised
  Rust + Python.
- **ASCII folding** (`café` ↔ `cafe`, `Süßigkeit` ↔ `Sussigkeit`,
  `naïve` ↔ `naive`) in both BM25 and the grounding scorer (via NFKD).
  New tests T27, T28, T39 pin this.

#### Documentation

- **`docs/LANGUAGE.md`** — honest scope of non-English support, by
  family + the `Analyzer` plugin's public API (Rust / Python / Node).
- **`docs/design/ANALYZER_PLUGIN.md`** — rewritten to describe the
  shipped surface (was originally a proposal with several deviations).
- **README "Language support" section** + per-package READMEs
  (`python/README.md`, `nodejs/README.md`) — `language=` examples.
- **`docs/ARCHITECTURE.md`** — refreshed against the post-consolidation
  workspace (the pre-0.2 split of `redhop-{core,context,…}` into
  separate crates was rolled into one published `redhop` crate; diagram
  and crate-name references updated).
- **`docs/API_STABILITY.md`** — full Node section added; Python section
  updated with `language=`, `n_files`, `skipped_files`; Rust section
  updated with the consolidated module paths.

### Changed

- **Python folder walker unified with Rust's `read_folder_with`** —
  −429 LOC in `python/src/lib.rs` (≈25% of the file). Removed the
  parallel `build_folder_persisted`, `collect_files`, `PersistedIndex`,
  `CachedFile`, `fingerprint`, etc. Both bindings now share Rust's
  single implementation; on-disk index format is byte-compatible with
  the previous Python writer, so existing caches reload cleanly.
- **`strategy_from_str` + `retrieval_from_str`** consolidated to a
  single source of truth in `redhop::load`. Python's wrappers now
  forward to the Rust functions with `map_err` instead of duplicating
  the match arms.
- **`Document` carries `n_files()` and `skipped_files()` accessors** on
  the Rust struct. Single-source constructors default to `1` / empty;
  `read_folder_with` (both simple and persisted paths) now records
  `(source, reason)` for each skipped file instead of silently dropping
  them.
- **MSRV bumped 1.75 → 1.77** across all three workspace declarations
  (workspace, `python/Cargo.toml`, `nodejs/Cargo.toml`) — the napi-rs
  2.x in the Node binding sets the actual floor; the inconsistency
  meant a 1.75 user hit a mysterious napi error instead of a clear MSRV
  one.

### Fixed

- **All-stopword query no longer crashes BM25.** A query the analyzer
  pipeline reduces to zero positive terms (`""`, `"   "`, `"the and is
  of in or"`) used to surface as a hard Tantivy error (`Invalid query:
  Only excluding terms given`). The retriever now traps that error
  class (and the `empty query` class) and returns an empty result.
  Caught by `quality_suite::t25` on its first run.
- **Python `Document.from_chunks` silently dropped `language=`** — the
  pyo3 signature accepted the kwarg but the call into `doc_config`
  passed `None` instead of the user's value. Caught by the new Python
  analyzer test suite on its first run.
- **Node `analyzeContext` / `contextEconomics`** were honoring the
  user's `token_budget` option; Python's equivalents hardcode
  `usize::MAX` because these are no-budget pure-analysis surfaces.
  Caught by the cross-binding parity tests on their first run.
- **Node `index.d.ts`** was stale — the `language` field and
  `minCandidates` field were present on the Rust `Options` struct but
  hadn't been regenerated. TypeScript users got "Object literal may
  only specify known properties" on perfectly valid options.

### Notes

- `unicode-normalization` promoted from transitive (via tantivy) to a
  direct dep of redhop. Used for the grounding scorer's NFKD fold.
- **Workspace test count**: 320/320 (Rust) + 81/81 (Python, +1 BGE
  fixture skip) + Node smoke + analyzer suites. Was 260 at the v0.1.4
  tag.
- CI gates: `cargo fmt --all -- --check`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test --workspace`, `cargo doc
  --workspace --no-deps --features files,semantic` (warning-free), the
  cross-binding parity suite, and the Node smoke + analyzer suites.
  All six CI jobs green.
- **`[package.metadata.docs.rs] all-features = true`** added to the
  redhop crate so the published doc page on docs.rs shows the
  `files` + `semantic` items instead of just the lean lexical surface.
- 21 example files swept clean of hardcoded `/Users/vysakh/...` paths;
  they resolve datasets/models/exports through
  `redhop_examples::{data_path, exports_path, model_path,
  bge_small_paths, ms_marco_paths}` helpers that honor
  `REDHOP_{DATA,EXPORTS,MODELS}_DIR` env vars.


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
