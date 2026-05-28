<p align="center">
  <img src=".github/icon.svg" width="96" height="96" alt="RedHop">
</p>

<h1 align="center">RedHop</h1>

<p align="center"><b>A reasoning-preserving context runtime for RAG.</b></p>

<p align="center">
Hand it a document and a question. It chunks, retrieves, and allocates the
context your model should actually see — then tells you what it kept, what it
dropped, and why, with citations back to the source. No vector database, no LLM,
all in-process.
</p>

---

RedHop is the layer between your documents and the LLM. It is **not** a vector
database, an agent framework, or a workflow engine — it does one thing: turn a
document and a query into the right prompt context, and explain the decision.

The core idea it's built on: **retrieval quality is not the same as reasoning
quality.** Transformers tolerate irrelevant context far better than they tolerate
*missing reasoning links* — so the chunk a multi-hop answer depends on is often
low-relevance to the query and gets silently pruned. RedHop's default keeps it and
makes the trade-off visible. The reasoning behind every default — including the
hypotheses that failed — lives in the [evidence layer](docs/findings/README.md).

## Install

> **Alpha — 0.1.x.** Published on PyPI, crates.io, and npm.

```bash
pip install redhop                            # Python  — on PyPI
cargo add redhop --features files,semantic    # Rust    — on crates.io
npm install redhop                            # Node.js — on npm
```

The same surface is available in all three. The embedding/reranking models
auto-download on first use; the default lexical tier needs no model at all.

## The basic approach

Point it at a file and ask. Parsing, chunking, retrieval, and token-budgeting all
happen inside — you think in documents and queries, not retrieval infrastructure.

```python
import redhop

doc = redhop.Document.from_file("contract.pdf")
ctx = doc.context("What is the governing law?")

answer = llm.generate(ctx.text())   # any LLM provider — no lock-in
```

`ctx` carries everything you need to prompt the model *and* show your work:
`ctx.text()` (the assembled prompt), `ctx.report` (the decision), and
`ctx.citations` (where it came from).

Same call in Node and Rust:

```js
const doc = Document.fromFile("contract.pdf");
const ctx = doc.context("What is the governing law?");
```

```rust
let mut doc = redhop::read_file("contract.pdf")?;
let ctx = doc.context("What is the governing law?")?;
```

Already have chunks from your own retriever? Hand them straight in with
`Document.from_chunks([...])` (or the lower-level `redhop.build_context(...)`),
and everything below still applies.

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

You can also read the fields directly — `ctx.report.auto_decision`,
`total_tokens`, `retained_evidence_ratio` — or call `doc.analyze(query)` to get the
report **without** assembling a context (pure diagnostics).

## Cite the evidence

Every selected chunk remembers where it came from, so you can show the evidence
trail instead of just pasting text:

```python
for c in ctx.citations:
    print(c["source"], c["page"], c["heading"])
    # contract.pdf  3     None      →  "contract.pdf, p.3"
    # notes.md      None  "Refunds" →  "notes.md → Refunds"
```

`source` plus whichever of `page` / `heading` / `line` the format provides — no
separate store, no second lookup.

## Loaders

Several on-ramps, all returning a `Document` with the same options:

| On-ramp | For |
| --- | --- |
| `from_text` | text you already have (your own parser/OCR, a DB field) |
| `from_chunks` | content you already chunked |
| `from_file` | a file on disk — PDF, DOCX, PPTX, XLSX, Markdown, or text/code |
| `from_bytes` | bytes you fetched yourself — S3 / Azure Blob / GCS / HTTP / DB blobs |
| `from_folder` | a whole directory in one index, with an optional incremental on-disk cache |

Code files are chunked verbatim and labeled with their nearest definition; prose is
sentence-packed; each format carries the structural location it has (page, heading,
line) for citations.

## Retrieval tiers — no vector database

Retrieval is a ladder. Start at the lexical default — it handles most document
QA because the words in the question are usually the words in the answer —
and climb only when the failure shape calls for it. All tiers run in-process,
with no ANN and no index server.

| `retrieval=` | What it does | Reach for it when |
| --- | --- | --- |
| `"lexical"` *(default)* | BM25 — zero dependencies, fully offline, ~50ms warm | most document QA: code, API refs, runbooks, financial reports, handbooks, mixed folders |
| `"hybrid"` | BM25 prunes to a pool, a dense model reorders it | the doc has parallel near-duplicate clauses (regional overrides, per-region sub-sections) — pair with `context(include_heading=True, neighbors=1)` |
| `"semantic"` | dense over every chunk, exact cosine | queries and answers share no vocabulary at all (rare in practice for document QA) |

Set `rerank="cross-encoder"` to add a second-stage scorer that reads each
`(query, passage)` pair jointly — useful for true synonym-mismatch corpora
(HR/support KBs where users phrase things very differently from the docs).
Adds 5–10× query latency, so **verify it helps on your own corpus before
enabling.**

→ Not sure which to pick? See [docs/CHOOSING_A_CONFIG.md](docs/CHOOSING_A_CONFIG.md)
  — the 60-second decision guide with code recipes.

## Assembly strategies

How the context is built from the retrieved candidates. The default is reasoning-preserving;
the others are there when you want a different trade-off.

| `strategy=` | What it does | When |
| --- | --- | --- |
| `reasoning_preserving` *(default)* | keep query-relevant seeds **and** rescue low-relevance chunks linked to a seed; drop only unlinked junk | multi-hop / general |
| `distractor_filtered` | drop everything below a query-grounding bar | single-hop |
| `max_density` | greedily pack the densest chunks into the budget | tight budgets |
| `raw_topk` | keep retrieval order until the budget fills | baseline / no optimization |
| `auto` | size-gated: pass small contexts through untouched, prune large/diluted ones | when you don't want to choose |

## Documentation

- **Choosing a configuration**: [docs/CHOOSING_A_CONFIG.md](docs/CHOOSING_A_CONFIG.md) — 60-second decision guide
- **Retrieval & context tips**: [docs/retrievaltips.md](docs/retrievaltips.md)
- **Comparison** (vs LangChain / LlamaIndex): [docs/COMPARISON.md](docs/COMPARISON.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Evidence layer** (every finding, including the falsified ones): [docs/findings/](docs/findings/README.md)
- **Python**: [python/README.md](python/README.md) · **Node**: [nodejs/README.md](nodejs/README.md)
- **API stability**: [docs/API_STABILITY.md](docs/API_STABILITY.md) ·
  **FAQ**: [FAQ.md](FAQ.md) · **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
