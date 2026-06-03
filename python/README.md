# RedHop

**A reasoning-preserving context runtime for RAG.**

Hand it a document and a question. RedHop chunks, retrieves, and allocates the
context your model should actually see — then tells you what it kept, what it dropped,
and why, with citations back to the source. No vector database, no LLM, all in-process.

```python
import redhop

doc = redhop.Document.from_file("contract.pdf")
ctx = doc.context("What is the governing law?")

answer = llm.generate(ctx.text())   # any LLM provider — no lock-in
```

```bash
pip install redhop
```

One self-contained wheel — no Python dependencies. The default lexical tier needs no
model at all; the semantic/rerank tiers download a small model on first use (cached).

## The idea

**Retrieval quality is not the same as reasoning quality.** Transformers tolerate
irrelevant context far better than they tolerate *missing reasoning links* — so the
chunk a multi-hop answer depends on is often low-relevance to the query and gets
silently pruned. RedHop's default keeps it, and makes the trade-off visible. It is
**not** a retriever, vector database, agent framework, or workflow engine — it does one
thing: turn a document and a query into the right prompt context, and explain the
decision.

## It explains every decision

Every call returns a **Decision Report** — what it kept, what it dropped, and *why*,
including when it deliberately leaves a small context untouched.

```python
print(ctx.report)
```

```text
RedHop Decision Report
══════════════════════

Decision: Auto → pruning (intervened on a diluted context)

  Why:
    - large/diluted contexts dilute attention; pruning recovers signal density
  Result:
    - removed distractor chunks, kept all query-relevant evidence
    - preserved a second-hop link a plain relevance filter would drop

Diagnostics
───────────
  Chunks:             24 → 3
  Second-hop rescues: 1
```

Read the fields directly via `ctx.report.auto_decision`, `total_tokens`,
`retained_evidence_ratio`, or call `doc.analyze(query)` for the report **without**
assembling a context.

## Cite the evidence

Every selected chunk remembers where it came from:

```python
for c in ctx.citations:
    print(c["source"], c["page"], c["heading"])
    # contract.pdf  3     None      ->  "contract.pdf, p.3"
    # notes.md      None  "Refunds" ->  "notes.md -> Refunds"
```

## Loading documents

| On-ramp | For |
| --- | --- |
| `Document.from_text(text)` | text you already have |
| `Document.from_chunks([...])` | content you already chunked |
| `Document.from_file("x.pdf")` | a file — PDF, DOCX, PPTX, XLSX, Markdown, or text/code |
| `Document.from_bytes(data, source="x.pdf")` | bytes you fetched (S3 / GCS / HTTP / DB) |
| `Document.from_folder("./docs", persist=True)` | a whole directory, with an optional incremental on-disk index |

## Retrieval tiers — no vector database

Start at the lexical default — it handles most document QA because the words
in the question are usually the words in the answer — and climb only when the
failure shape calls for it. All in-process, no ANN, no index server.

```python
# Default — most docs (code, API refs, runbooks, financial reports, handbooks)
doc = redhop.Document.from_file("contract.pdf")
ctx = doc.context("What is the governing law?")

# Structured docs with parallel clauses (regional overrides, per-region sub-sections):
doc = redhop.Document.from_file("msa.pdf", retrieval="hybrid", model="bge-small")
ctx = doc.context("What law applies in the UK?", include_heading=True, neighbors=1)

# Synonym-mismatch corpora (HR FAQs, support tickets where users phrase
# things very differently from the docs). Cross-encoder adds 5–10× latency
# — verify it helps on your corpus before enabling.
doc = redhop.Document.from_file("support.md",
    retrieval="hybrid", model="bge-small", rerank="cross-encoder")
```

The 60-second decision guide with trade-offs and query-writing tips:
[CHOOSING_A_CONFIG](https://github.com/vysakh0/redhop/blob/main/docs/CHOOSING_A_CONFIG.md).

## Non-English content

Default is English Snowball. Swap with the `language=` kwarg — any of
the 18 Snowball Porter2 languages (`arabic, danish, dutch, english,
finnish, french, german, greek, hungarian, italian, norwegian,
portuguese, romanian, russian, spanish, swedish, tamil, turkish`):

```python
doc = redhop.Document.from_text(german_text, language="german")
# Now `Buch` finds chunks containing `Bücher` (and vice versa)
```

One analyzer drives both BM25 retrieval AND the grounding scorer, so
they can't drift on what "the same term" means. Unknown names raise
(we don't silently fall back to English). See the
[language guide](https://github.com/vysakh0/redhop/blob/main/docs/LANGUAGE.md)
for the full breakdown and the calibration disclaimer (we ship the
stemmers; eval-corpus ranking quality on a real domain corpus is the
user's call).

## Assembly strategies

| `strategy=` | What it does |
| --- | --- |
| `reasoning_preserving` *(default)* | keep query-relevant seeds **and** rescue low-relevance chunks linked to one; drop only unlinked junk |
| `distractor_filtered` | drop everything below a query-grounding bar |
| `max_density` | greedily pack the densest chunks into the budget |
| `raw_topk` | keep retrieval order until the budget fills |
| `auto` | size-gated: pass small contexts through, prune large/diluted ones |

Already have chunks from your own retriever? Use `redhop.build_context(query,
retrieved_chunks=chunks, ...)` for the low-level surface.

## Documentation

Full docs, the comparison vs LangChain / LlamaIndex, and the evidence behind every
default: **https://redhopai.com**

Apache-2.0. Also available for **Node.js** (`npm install redhop`) and **Rust**
(`cargo add redhop`).
