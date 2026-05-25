# Direction: RedHop as the in-process retrieval runtime (no vector DB)

> **Status:** Plan (approved direction). Captures the repositioning + the
> `from_file`/`from_folder` ingestion layer + the restored three-tier ladder.
> Sequencing and open decisions at the end.

## Positioning

**RedHop is the in-process retrieval runtime for people who won't run a vector DB** —
coding agents, ad-hoc document QA, desktop/file apps, CI scripts. Point it at files,
ask a question, get the right context. No service, no index server, no DB to operate.

The gap: for *ephemeral / local / per-session* retrieval, standing up Chroma / FAISS /
pgvector is absurd overhead, and grep is too dumb. Nobody owns "point at a folder, ask,
no infra." That's the wedge — sharper than "context layer between docs and the LLM."

## The retrieval ladder (three tiers, by corpus size)

| corpus | tier (`retrieval=`) | mechanism | cost |
| ------ | ------------------- | --------- | ---- |
| any size, keyword-dense | **`lexical`** (default) | BM25 inverted index | no model; sub-linear query; scales |
| **thousands of local files, no DB** | **`hybrid`** (restore) | BM25 prune → dense rerank the ~50-pool | embeds only the pool/query — bounded |
| one doc / small bounded set | **`semantic`** | global dense, exact cosine over all chunks | embed-all up front; O(N)/query |
| huge + persistent + multi-user | *external vector store* | ANN | **out of scope — handoff** |

**Restore `hybrid`** (the BM25-prune→rerank tier we collapsed): it's the semantic
option that *scales without a vector DB*, because it embeds only the BM25-selected
pool per query — exactly the agent/folder case. `semantic` (global dense) stays the
best-recall choice for small/bounded corpora. See [GLOBAL_DENSE](findings/GLOBAL_DENSE.md)
and [LOCAL_RERANK](findings/LOCAL_RERANK.md) (un-supersede the latter as a tier).

> **Open naming decision.** Either keep three explicit tiers
> (`lexical`/`hybrid`/`semantic`), or make **one adaptive `semantic`** that picks
> global-dense for small corpora and BM25-prune+rerank for large (by chunk count).
> Adaptive is the friendliest UX; three explicit gives control. Recommendation:
> ship `hybrid` explicitly now, layer an adaptive default later.

## Ingestion: `from_file` / `from_folder`

The DX that sells the wedge — collapse the parse→text→chunk dance into one line:

```python
redhop.Document.from_file("contract.pdf").context("governing law?")
redhop.Document.from_folder("./docs", retrieval="hybrid").context("refund policy?")
```

### Parsing stack (reuse from `../diwadi-mono`)
- **PDF:** `pdfium-render` 0.8 (high quality, **needs the pdfium native binary** — a
  weight/portability tension) **or** the custom pure-Rust `lopdf`-based parser in
  `diwadi-mono/crates/diwadi-desktop/src/pdf_parser` (lighter, no binary; lift into a
  shared crate).
- **DOCX:** `docx-rs` 0.4. **XLSX/sheets:** `calamine` 0.32.
- **PPTX / OOXML:** `zip` 6 + `quick-xml` 0.37 (parse the XML ourselves).
- **Type detection:** `mime_guess`. Markdown/txt/code: read directly.

### Design rules
- **Feature-gated** (`redhop[files]`): the core stays "text in, bring your own
  parser" — `from_text` remains the contract and the escape hatch. `from_file`/
  `from_folder` are batteries-included convenience on top, so the bounded-core
  identity survives and the default install stays lean.
- **pdfium binary dep is the open call:** prefer the pure-Rust lopdf parser for the
  default `[files]` build; offer `[files-pdfium]` for higher-fidelity PDF. (Decide.)
- **Citations / metadata (high value, nearly free here):** stamp each chunk with
  `path` + `page`/`heading` during ingestion, so the Decision Report and assembled
  context can cite *"contract.pdf p.3"*. Huge for agents and QA.

## Persistence (the hard part `from_folder` forces)

A folder of thousands of files **cannot** re-parse + re-index every run. So
`from_folder` requires an on-disk index:
- **BM25 index → disk** (Tantivy supports it): parse + chunk + index once, reload.
- **Embeddings: nothing to persist for `hybrid`** — they're query-time on the pool.
  (For `semantic`/global, you'd persist the vectors, or just don't use it on folders.)
- **Incremental re-index** by mtime — re-parse only changed files. Essential for
  apps/agents where files change. `.gitignore`/glob/size filters for `from_folder`.

RedHop is currently in-memory/ephemeral (lazy index, not persisted). Persistence is
**required, not polish**, for the folder use case.

## Distribution play
An **MCP server / tool wrapper** so coding agents call RedHop directly
(`search_files(query)` over a folder) with zero glue — straight into the agent
audience.

## Scope flags (honest)
- **Parsing is a support burden** (why we punctured it originally). Own supported
  formats + limits; keep `from_text` as the fallback when a file parses badly.
- **Docs ≠ code.** Codebase retrieval (AST-aware chunking, symbols, `.gitignore`) is a
  bigger, separate play. Start with *documents* (pdf/docx/pptx/md/txt); don't promise
  great *code* retrieval yet.
- **Identity shift:** we were proudly "not a parser." This is a deliberate
  repositioning — name it, feature-gate it, keep the core narrow.

## Sequencing
1. **Restore `hybrid` tier** — code (engine exists), docs, benchmarks. *(do first)*
2. **`from_file`** — single file, ephemeral, `redhop[files]`, citations. Easy, high DX.
3. **`from_folder` + persistence + incremental** — the real engineering; folder default
   tier = `lexical`/`hybrid`, never global `semantic`.
4. **Adaptive `semantic`** (optional) and **MCP wrapper** (distribution).
5. Evolve the site hero to the runtime/no-vector-DB positioning.
