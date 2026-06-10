# RedHop · Rust examples

Runnable Rust examples covering the 0.3.0 API surface. Each file is
self-contained: data inline, no external services, no model downloads
(except where noted), no LLM calls.

## Setup

```bash
# From the repo root — this is a workspace member.
cargo run -p redhop-rust-examples --example 01_quickstart --release
```

The crate is `redhop-rust-examples` (in `Cargo.toml`). The `--release`
flag is recommended because the first build links the full `redhop`
dependency graph; subsequent runs are incremental.

For an external project, add to your own `Cargo.toml`:

```toml
[dependencies]
redhop = "0.3"   # add features = ["files", "semantic"] for examples 07 & 11
anyhow = "1"
```

## What's here

**Core API surface (no model download, no FS access):**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 01 | [`01_quickstart.rs`](examples/01_quickstart.rs) | `Document::from_text(source, text)` → `doc.context(query)` → `redhop::citations(&ctx)` + `ctx.report`. The 3-call surface. |
| 02 | [`02_structured_corpus.rs`](examples/02_structured_corpus.rs) | `redhop::core::Chunk::new(...)` with `.with_metadata(HashMap)` for typed chunks. source-vs-id distinction. Open metadata (page/heading/line) flowing to citations. |
| 03 | [`03_templated_workload.rs`](examples/03_templated_workload.rs) | `analyze_query_set` → `Stripper::new` → `Vocabulary::new` → `doc.context_with_rewrites(query, &[&s, &v])` with the per-stage `ctx.report.query_rewrites` audit trail. |
| 04 | [`04_chunk_enrich.rs`](examples/04_chunk_enrich.rs) | `vocab.enrich(chunk_text)` returning `RewriteResult` at ingest. **Read the honest-framing notice at the top** — enrich is shipped with asymmetric measured evidence; A/B on your own corpus before adopting. |
| 05 | [`05_evaluate_ab.rs`](examples/05_evaluate_ab.rs) | `redhop::evaluate(&Query, &ctx, EvalGold::Chunks(&[ids]))` — deterministic A/B with no LLM judge. Reproduces the same +0.12 lift seen in the Python and Node mirrors. |
| 06 | [`06_chat_rag.rs`](examples/06_chat_rag.rs) | `ContextConfig { preserve_order: true, .. }` via `Document::from_chunks_with(chunks, cfg)` — same retrieval selection, chronological emission. |

**Retrieval tiers and assembly options:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 07 | [`07_retrieval_tiers.rs`](examples/07_retrieval_tiers.rs) | `LoadOptions { retrieval: Some("lexical"/"hybrid"/"semantic"), model: Some("bge-small"), .. }`. First run of hybrid/semantic downloads bge-small (~80MB). |
| 08 | [`08_structural_expansion.rs`](examples/08_structural_expansion.rs) | `doc.context_expanded(query, budget, candidate_k, neighbors, include_heading)` — same selection, padded with adjacent context and section headings. |
| 09 | [`09_multilingual.rs`](examples/09_multilingual.rs) | `LoadOptions { language: Some("german"), .. }` — routes the whole pipeline through the right Snowball stemmer. 18 languages supported; unknown strings *error* rather than silently fall back to English. |
| 10 | [`10_strategy_choice.rs`](examples/10_strategy_choice.rs) | The four strategies via `build_context(&query, &chunks, &cfg)` — Auto, RawTopK, DistractorFiltered, ReasoningPreserving. Demonstrates the second-hop rescue with `ctx.report.second_hop_rescue_count`. |

**Loaders:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 11 | [`11_folder_indexing.rs`](examples/11_folder_indexing.rs) | `read_folder_with(path, &FolderOptions { ignore, persist, .. })` with `.gitignore` support, custom ignore globs, incremental on-disk cache. Plus `read_bytes_with(data, "source.pdf", ...)` for S3/GCS/DB blobs. |

**Observability:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 12 | [`12_diagnosis.rs`](examples/12_diagnosis.rs) | `ctx.report.diagnosis` carries per-query facts about how the query met the corpus (`query_terms`, `zero_match_terms`, `term_stats`, `score_spread`) plus bounded hints (`VocabMismatch`, `LowDiscriminationQuery`, `UnderdeterminedQuery`, …) each citing the measured finding behind it. Healthy queries fire no hints. |
| 13 | [`13_workload_audit.rs`](examples/13_workload_audit.rs) | The bring-your-own-pipeline (BYO) loop: `analyze_context(&query, &results, &cfg)` observes what an external retriever returned, `summarize_diagnoses(&reports)` aggregates a workload into one focus recommendation. Walk-through: [`docs/DIAGNOSE_YOUR_PIPELINE.md`](../../docs/DIAGNOSE_YOUR_PIPELINE.md). |

## How these relate to `crates/examples/`

The `crates/examples/examples/` directory has 59 other Rust example
files — those are **measurement probes** (the evidence layer behind
`docs/findings/`), not API showcases. The split:

- [`examples/rust/`](.) — *how* to use the API. Real-world scenarios,
  inline data, demo-shaped output. **This is what you want if you're
  learning the surface or building something.**
- [`../../crates/examples/examples/`](../../crates/examples/examples/) — *what is true*
  on a measured workload. Reproducible benchmarks behind every claim in
  [`docs/findings/`](../../docs/findings/).

## How to read these in order

If you're new to RedHop, run them top-to-bottom:

1. **01–06 (core API surface)** — Quickstart through chat RAG.
2. **07–10 (retrieval and assembly options)** — Tier selection,
   structural expansion, multilingual, strategy choice.
3. **11 (loaders)** — Filesystem / blob / persistent index.

Each file's `//!` doc-comment block spells out the real-world
scenario it's modeling and links the relevant finding in
`docs/findings/` where applicable.

## Equivalent Python and Node.js examples

Each `.rs` file here mirrors the same-numbered `.py` file under
[`../python/`](../python/) and `.cjs` file under
[`../nodejs/`](../nodejs/). The scenarios, data, and output structure
are identical — pick the language that matches your project.
