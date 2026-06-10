# RedHop · Node.js examples

Runnable examples covering the 0.3.0 Node.js API surface. Each file is
self-contained: data inline, no external services, no model downloads
(except where noted), no LLM calls.

## Setup

```bash
npm install redhop
```

Then from the repo root:

```bash
node examples/nodejs/01_quickstart.cjs
```

## What's here

**Core API surface (no model download, no FS access):**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 01 | [`01_quickstart.cjs`](01_quickstart.cjs) | Load a document, ask a question, read the Decision Report. The 3-call surface. |
| 02 | [`02_structured_corpus.cjs`](02_structured_corpus.cjs) | `new Chunk(text, { source, id, metadata })` for content you already chunked elsewhere (FAQs, schema rows, code symbols). Citations carry `source` / `heading` / `page` / `line` through automatically. |
| 03 | [`03_templated_workload.cjs`](03_templated_workload.cjs) | The templated-workload workflow: `analyzeQuerySet` → `Stripper` → `Vocabulary` → `doc.contextWithRewrites(...)` with a per-stage audit trail on `ctx.report.queryRewrites`. |
| 04 | [`04_chunk_enrich.cjs`](04_chunk_enrich.cjs) | Chunk-side `vocab.enrich(...)` at ingest time for short, opaque coded retrieval units. **Read the honest-framing notice at the top** — enrich is shipped with asymmetric measured evidence; A/B on your own corpus before adopting. |
| 05 | [`05_evaluate_ab.cjs`](05_evaluate_ab.cjs) | Deterministic A/B with `evaluate(query, ctx, { goldChunks: [...] })` — no LLM judge, no API key, no money spent. Same primitives the runtime uses for its Decision Report. |
| 06 | [`06_chat_rag.cjs`](06_chat_rag.cjs) | `preserveOrder: true` for chat histories — relevance-driven selection, chronological emission. The trick is one config flag; the contrast on the same chat is the demo. |

**Retrieval tiers and assembly options:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 07 | [`07_retrieval_tiers.cjs`](07_retrieval_tiers.cjs) | The three `retrieval=` tiers — `lexical` (BM25, default), `hybrid` (BM25 + dense rerank), `semantic` (global dense). First run of hybrid/semantic downloads `bge-small` (~80MB). |
| 08 | [`08_structural_expansion.cjs`](08_structural_expansion.cjs) | `doc.context(query, budget, neighbors, includeHeading)` — same retrieval selection, padded with adjacent context and section headings within the token budget. |
| 09 | [`09_multilingual.cjs`](09_multilingual.cjs) | `language: "german"`/`"french"`/… — routes the whole pipeline through the right Snowball stemmer. 18 languages supported; unknown strings error rather than silently fall back to English. |
| 10 | [`10_strategy_choice.cjs`](10_strategy_choice.cjs) | Assembly strategies — `auto` (default size-gated), `raw_topk` (pass-through), `reasoning_preserving` (multi-hop rescue, `secondHopRescueCount` on the report), `distractor_filtered` (naive baseline). Shows the second-hop tax live. |

**Loaders:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 11 | [`11_folder_indexing.cjs`](11_folder_indexing.cjs) | `Document.fromFolder(path, { ... })` with `.gitignore`, custom `ignore` globs, `persist: true` for incremental on-disk caching, and `fromBytes(...)` for S3 / GCS / DB blobs. |

**Observability:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 12 | [`12_diagnosis.cjs`](12_diagnosis.cjs) | `ctx.report.diagnosis` carries per-query facts about how the query met the corpus (`queryTerms`, `zeroMatchTerms`, `termStats`, `scoreSpread`) plus bounded hints (vocab mismatch, polysemy, templated boilerplate) each citing the measured finding behind it. Healthy queries fire no hints. |
| 13 | [`13_workload_audit.cjs`](13_workload_audit.cjs) | The bring-your-own-pipeline (BYO) loop: `analyzeContext(query, yourChunks)` observes what an external retriever returned, `summarizeDiagnoses(reports)` aggregates a workload into one focus recommendation, plus the OTel / Langfuse attribute snippet for shipping reports to telemetry. Walk-through: [`docs/DIAGNOSE_YOUR_PIPELINE.md`](../../docs/DIAGNOSE_YOUR_PIPELINE.md). |

## How to read these in order

If you're new to RedHop, run them top-to-bottom:

1. **01–06 (core API surface)** — Quickstart through chat RAG.
2. **07–10 (retrieval and assembly options)** — Tier selection,
   structural expansion, multilingual support, strategy choice.
3. **11 (loaders)** — FS / blob / persistent on-disk index.

Each file's comment block spells out the real-world scenario it's
modeling and links to the relevant finding in `docs/findings/` where
applicable.

## Notes on the Node.js binding

- All public surfaces use camelCase: `doc.contextWithRewrites`,
  `ctx.report.queryRewrites`, `ctx.report.autoDecision`,
  `report.secondHopRescueCount`, etc.
- `ctx.text` is a property (not a method). `doc.chunkCount` /
  `doc.nFiles` / `doc.skippedFiles` are properties on the Document.
- Options always come in an options-bag object: `Document.fromText(text,
  { source, retrieval, model, ... })`, `new Chunk(text, { source, id,
  metadata })`, `doc.context(query, budget, neighbors, includeHeading)`
  for the few positional args. See [`../../nodejs/index.d.ts`](../../nodejs/index.d.ts)
  for the complete TypeScript surface.

## Equivalent Python examples

Each `.cjs` file here mirrors the same-numbered `.py` file under
[`../python/`](../python/). The scenarios, data, and output structure
are identical — pick the language that matches your project.
