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

| # | File | What it demonstrates |
| -- | ---- | -------------------- |
| 01 | [`01_quickstart.py`](01_quickstart.py) | Load a document, ask a question, read the Decision Report. The 3-call surface. |
| 02 | [`02_structured_corpus.py`](02_structured_corpus.py) | `redhop.Chunk(text, source=..., id=..., metadata={...})` for content you already chunked elsewhere (FAQs, schema rows, code symbols). Citations carry `source`/`heading`/`page`/`line` through automatically. |
| 03 | [`03_templated_workload.py`](03_templated_workload.py) | The templated-workload workflow: `analyze_query_set` → `Stripper` → `Vocabulary` → `Document.context_with_rewrites(...)` with a per-stage audit trail on `ctx.report.query_rewrites`. |
| 04 | [`04_chunk_enrich.py`](04_chunk_enrich.py) | Chunk-side `Vocabulary.enrich(...)` at ingest time for short, opaque coded retrieval units. **Read the honest-framing notice at the top** — enrich is shipped with asymmetric measured evidence; A/B on your own corpus before adopting. |
| 05 | [`05_evaluate_ab.py`](05_evaluate_ab.py) | Deterministic A/B with `redhop.evaluate(query, ctx, gold_chunks=[...])` — no LLM judge, no API key, no money spent. Same primitives the runtime uses for its Decision Report. |
| 06 | [`06_chat_rag.py`](06_chat_rag.py) | `preserve_order=True` for chat histories — relevance-driven *selection*, chronological *emission*. The trick is one config flag; the contrast on the same chat is the demo. |

## What's not here (yet)

- **Spider / BIRD schema retrieval** — the natural positive probe for
  `Vocabulary.enrich(...)`. Queued post-0.3.0; in the meantime,
  `04_chunk_enrich.py` shows the API and the honest framing.
- **`Document.from_file(...)` walk-through** — once you `pip install
  "redhop[files]"`, the API is the same as `from_text(...)` — see
  the root README for a code snippet.
- **Dense / hybrid retrieval** — `retrieval="hybrid"` adds an
  embedding model download; we kept these examples model-free so they
  run anywhere. The pattern is documented in
  [docs/CHOOSING_A_CONFIG.md](../../docs/CHOOSING_A_CONFIG.md).

## How to read these in order

If you're new to RedHop, run them top-to-bottom — they're sequenced to
introduce one concept at a time:

1. **Quickstart** (01) gives you the 3-call surface and the Decision Report.
2. **Structured corpus** (02) introduces the typed `Chunk` constructor
   and the source-vs-id distinction.
3. **Templated workload** (03) layers in the rewrite chain and the
   audit trail.
4. **Chunk enrich** (04) is the chunk-side mirror of (03)'s query-side
   rewrites, with explicit caveats about its asymmetric evidence.
5. **Evaluate A/B** (05) closes the loop — how to *measure* whether a
   rewrite you adopted in (03) or (04) actually helps on your data.
6. **Chat RAG** (06) is a tangent: same retrieval surface, one config
   flag that matters for chronology-sensitive applications.

Each file's docstring spells out the real-world scenario it's modeling
and links to the relevant finding in `docs/findings/` where applicable.
