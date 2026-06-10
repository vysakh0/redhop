# RedHop · Python examples

Runnable examples covering the 0.3.0 Python API surface. Each file is
self-contained: data inline, no external services, no model downloads,
no LLM calls.

## Setup

```bash
pip install redhop          # core (lexical retrieval, in-process)
pip install "redhop[files]" # add PDF / DOCX / PPTX / XLSX loaders
```

Then from the repo root:

```bash
python examples/python/01_quickstart.py
```

## What's here

**Core API surface (no model download, no FS access):**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 01 | [`01_quickstart.py`](01_quickstart.py) | Load a document, ask a question, read the Decision Report. The 3-call surface. |
| 02 | [`02_structured_corpus.py`](02_structured_corpus.py) | `redhop.Chunk(text, source=..., id=..., metadata={...})` for content you already chunked elsewhere (FAQs, schema rows, code symbols). Citations carry `source`/`heading`/`page`/`line` through automatically. |
| 03 | [`03_templated_workload.py`](03_templated_workload.py) | The templated-workload workflow: `analyze_query_set` → `Stripper` → `Vocabulary` → `Document.context_with_rewrites(...)` with a per-stage audit trail on `ctx.report.query_rewrites`. |
| 04 | [`04_chunk_enrich.py`](04_chunk_enrich.py) | Chunk-side `Vocabulary.enrich(...)` at ingest time for short, opaque coded retrieval units. **Read the honest-framing notice at the top** — enrich is shipped with asymmetric measured evidence; A/B on your own corpus before adopting. |
| 05 | [`05_evaluate_ab.py`](05_evaluate_ab.py) | Deterministic A/B with `redhop.evaluate(query, ctx, gold_chunks=[...])` — no LLM judge, no API key, no money spent. Same primitives the runtime uses for its Decision Report. |
| 06 | [`06_chat_rag.py`](06_chat_rag.py) | `preserve_order=True` for chat histories — relevance-driven *selection*, chronological *emission*. The trick is one config flag; the contrast on the same chat is the demo. |

**Retrieval tiers and assembly options:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 07 | [`07_retrieval_tiers.py`](07_retrieval_tiers.py) | The three `retrieval=` tiers — `lexical` (BM25, default), `hybrid` (BM25 + dense rerank), `semantic` (global dense). First run of hybrid/semantic downloads `bge-small` (~80MB). |
| 08 | [`08_structural_expansion.py`](08_structural_expansion.py) | `neighbors=1` and `include_heading=True` for structured-document QA — same retrieval selection, padded with adjacent context and section headings within the token budget. |
| 09 | [`09_multilingual.py`](09_multilingual.py) | `language="german"`/`"french"`/… — routes the whole pipeline (chunking, BM25 stemming, grounding) through the right Snowball stemmer. 18 languages supported; unknown strings *error* rather than silently fall back to English. |
| 10 | [`10_strategy_choice.py`](10_strategy_choice.py) | Assembly strategies — `auto` (default size-gated), `raw_topk` (pass-through), `reasoning_preserving` (multi-hop rescue, with `second_hop_rescue_count` on the report), `distractor_filtered` (naive baseline). Shows the second-hop tax mechanism live. |

**Loaders (requires `redhop[files]`):**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 11 | [`11_folder_indexing.py`](11_folder_indexing.py) | `Document.from_folder(path, ...)` with `.gitignore`, custom `ignore` globs, `persist=True` for incremental on-disk caching, and `from_bytes(...)` for S3/GCS/DB blobs. |

**Observability:**

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 12 | [`12_diagnosis.py`](12_diagnosis.py) | `ctx.report.diagnosis` carries per-query facts about how the query met the corpus (`query_terms`, `zero_match_terms`, `term_stats`, `score_spread`) plus bounded hints (vocab mismatch, polysemy, templated boilerplate) each citing the measured finding behind it. Healthy queries fire no hints. |
| 13 | [`13_workload_audit.py`](13_workload_audit.py) | The bring-your-own-pipeline (BYO) loop: `redhop.analyze_context(query, your_chunks)` observes what an external retriever returned, `redhop.summarize_diagnoses(reports)` aggregates a workload into one focus recommendation, `redhop.otel.report_to_attributes(report)` flattens any report into OpenTelemetry / Langfuse-compatible attributes. Walk-through: [`docs/DIAGNOSE_YOUR_PIPELINE.md`](../../docs/DIAGNOSE_YOUR_PIPELINE.md). |

## What's not here (yet)

- **Spider / BIRD schema retrieval** — the natural positive probe for
  `Vocabulary.enrich(...)`. Queued post-0.3.0; in the meantime,
  `04_chunk_enrich.py` shows the API and the honest framing.
- **Cross-encoder reranking** (`rerank="cross-encoder"`) — adds
  another ~300MB model and 5-10× latency. Documented in
  [docs/CHOOSING_A_CONFIG.md](../../docs/CHOOSING_A_CONFIG.md);
  consider adding once we measure when it actually helps on top of
  the workflows in 03 and 07.

## How to read these in order

If you're new to RedHop, run them top-to-bottom — they're sequenced to
introduce one concept at a time:

1. **01–06 (core API surface)** — Quickstart through chat RAG. Each
   one introduces a new piece of the surface: the 3-call shape, typed
   chunks, the rewrite chain + audit trail, chunk-side enrich (with
   honesty), A/B eval, chronology preservation.
2. **07–10 (retrieval and assembly options)** — Once you know the
   surface, these show the knobs: which retrieval tier (lexical /
   hybrid / semantic), how to pad hits with structural context,
   multilingual support, which assembly strategy fits which workload.
3. **11 (loaders)** — Where your bytes live: filesystem, in-memory,
   blob storage; how RedHop handles the boring parts (.gitignore,
   persistence, ignore globs).

Each file's docstring spells out the real-world scenario it's modeling
and links to the relevant finding in `docs/findings/` where applicable.
