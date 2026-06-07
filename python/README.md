# RedHop

**A reasoning-preserving context runtime for RAG.**

[![PyPI](https://img.shields.io/pypi/v/redhop?label=pypi&color=e11d48)](https://pypi.org/project/redhop/)
[![Python](https://img.shields.io/pypi/pyversions/redhop?color=e11d48)](https://pypi.org/project/redhop/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/vysakh0/redhop/blob/main/LICENSE)
[![Evidence layer](https://img.shields.io/badge/evidence-layer-blue)](https://github.com/vysakh0/redhop/tree/main/docs/findings)

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

## How it compares

Measured on identical documents + budgets + BM25 retrieval, RedHop **beats both
frameworks on multi-hop evidence retention** (80% vs LangChain 71%, LlamaIndex 72%)
and **beats LangChain on contracts** (82% vs 73%). It trails LlamaIndex by 4 points
on CUAD's raw-template query — that gap is mechanism-known and closeable with a
6-line query preprocessor (RedHop reaches 88%, +2 over LlamaIndex); see
[CUAD_RECALL_GAP.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_RECALL_GAP.md).
All without a vector database, an agent framework, or model finetuning.

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/retention_vs_frameworks.svg" alt="Evidence retention vs LangChain vs LlamaIndex" width="100%">
</p>

Methodology + raw runs: [FRAMEWORK_COMPARISON.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/FRAMEWORK_COMPARISON.md)
· [framework_comparison_2026-06-06.txt](https://github.com/vysakh0/redhop/blob/main/reports/framework_comparison_2026-06-06.txt).

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/architecture.svg" alt="RedHop pipeline" width="100%">
</p>

Five stages: you bring documents and a query, RedHop owns parsing, chunking,
retrieval, and context allocation, and you get a `BuiltContext` with the
assembled prompt, citations, and a Decision Report. Each stage has an
evidence-backed default that traces to a finding in
[`docs/findings/`](https://github.com/vysakh0/redhop/tree/main/docs/findings).

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

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/decision_report.svg" alt="Sample Decision Report" width="100%">
</p>

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
| `Document.from_text(text, source="document")` | text you already have |
| `Document.from_chunks([redhop.Chunk(...), ...])` | content you already chunked — pass typed `redhop.Chunk(text, source=..., id=..., metadata={...})` instances |
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

Already have chunks from your own retriever? Wrap each as `redhop.Chunk(text,
source=..., id=..., metadata={...})` and pass into
`redhop.build_context(query, retrieved_chunks=chunks, ...)` (low-level) or
`redhop.Document.from_chunks(chunks)` (full indexing).

## Templated workloads — the +9 retention lift (BM25, no model needed)

If every query in your workload follows a fixed template — legal QA
("*Highlight the parts (if any) of this contract related to X. Details: …*"),
support-ticket triage ("*Help me with X, my account is Y, the error is Z*"),
form-filled queries from a structured UI — **BM25 weights every query term
by corpus IDF, not by how often the term repeats across your query set**.
The boilerplate words dilute the real signal words, and retention suffers.
This is the mechanism behind the 4-point CUAD gap on the head-to-head;
closing it doesn't need a vector DB or a different retriever — it needs two
small preprocessing helpers on the query side.

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/workflow_lift.svg" alt="CUAD retention rises 81% → 88% → 90.3% across the detect → strip → expand workflow; LlamaIndex is at 86%" width="100%">
</p>

**Measured** on the CUAD framework comparison (n=300, BM25, budget 2,000 tok):

| step | helper | retention | Δ |
| ---- | ------ | ---------:| -:|
| raw 24-word template | — | 81.3% | — |
| + strip the wrapper | `Stripper` | 87.7% | **+6.4** |
| + add workload synonyms | `Vocabulary` | **90.7%** | **+3.0** |

**RedHop with the full workflow is at 90.7% — beating LlamaIndex by 4 points
on the same setup, at native BM25 latency (~2.5ms/query).** Mechanism +
worked clause dict:
[CUAD_CLAUSE_EXPANSION.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_CLAUSE_EXPANSION.md).

Recommended workflow: **detect → strip → (optional) expand → A/B**. The
rewrite chain runs inside `Document.context_with_rewrites(...)` so each
stage's audit trail lands on `report.query_rewrites` automatically.

```python
import redhop

# 1 — Detect. Hand a representative sample of your queries to the analyzer.
report = redhop.analyze_query_set(my_queries[:300])
# report.is_templated            → True / False
# report.template_word_share     → e.g. 0.66 on CUAD
# report.boilerplate_terms       → ["highlight", "contract", "lawyer", …]
# report.estimated_dilution_cost → "high" | "medium" | "low" | "none"

if report.is_templated:
    # 2 — Compile the rewrite chain.
    stripper = redhop.Stripper(report.boilerplate_terms)

    # 3 — (optional) Vocabulary. If your workload has known topic synonyms
    #     (clause types, error codes), compile them once.
    vocab = redhop.Vocabulary({
        # YOUR keys → synonyms; CUAD worked example in CUAD_CLAUSE_EXPANSION.md
        "change of control": ["merger", "successor", "acquisition"],
    })

    # 4 — Run the chain through retrieval; audit lands on report.query_rewrites.
    doc = redhop.Document.from_file("contract.pdf")
    ctx_a = doc.context(user_query)                              # baseline
    ctx_b = doc.context_with_rewrites(user_query, [stripper, vocab])
    eval_a = redhop.evaluate(user_query, ctx_a, gold_chunks=gold_ids)
    eval_b = redhop.evaluate(user_query, ctx_b, gold_chunks=gold_ids)
    print(eval_b.overall - eval_a.overall)   # the lift, deterministically
```

- **Only matters if your queries are templated.** `analyze_query_set` is
  conservative by design — HotpotQA and MuSiQue both register quiet
  (`is_templated=False`) in the cross-workload probe; CUAD fires. If
  yours doesn't fire, skip this section.
- **The analyzer measures the *shape* of your query set, not your
  retention.** It says "this *looks* like a templated workload" with
  the boilerplate terms it found; it does **not** promise a specific
  lift. Always A/B on your gold-evidence sample before committing.
- **For single-doc extraction workloads also set `strategy="raw_topk"`.**
  `auto` routes large contexts to `reasoning_preserving`, which solves a
  multi-hop problem contract extraction doesn't have. RawTopK beats it
  by ~4 points at every chunk size on CUAD.
- **We deliberately don't ship a CUAD-specific `strip_template()`
  helper.** Templates are workload-specific; baking one in would make
  the wrong call for the next workload. `Stripper(...)` and
  `Vocabulary({...})` take *your* boilerplate / synonym dict so the
  call stays on your side.
- **Or take the one-knob alternative — `retrieval="hybrid"`.**
  Dense reads chunks as semantic content rather than counting tokens,
  so the boilerplate ratio stops mattering. Substitutes for stripping
  by a different mechanism (+5.3 on raw CUAD at ~10ms/query). On CUAD
  specifically, BM25 + strip + vocabulary still wins — 90.7% / 2.5ms
  vs hybrid+CE 89.0% / 683ms. The two paths are *substitutes*, not
  complements; pick one. See
  [CUAD_HYBRID_RERANK.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_HYBRID_RERANK.md).

| helper | what it does | finding |
| ------ | ------------ | ------- |
| `analyze_query_set(queries)` | Inspects your queries; flags whether they're templated and which terms are doing the dilution | [QUERY_SET_ANALYZER](https://github.com/vysakh0/redhop/blob/main/docs/findings/QUERY_SET_ANALYZER.md) |
| `Stripper(boilerplate)` | Compiled token-level boilerplate strip; word-boundary safe (an `"of"` strip does not erase `"of"` inside `"office"`). Plugs into the rewrite chain so the audit trail is captured | [CUAD_RECALL_GAP](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_RECALL_GAP.md) · [MULTILINGUAL_ANALYZER](https://github.com/vysakh0/redhop/blob/main/docs/findings/MULTILINGUAL_ANALYZER.md) |
| `Vocabulary({key: [synonyms]})` | Compiled workload-curated equivalence classes — appends high-IDF synonyms when the token-level key matches. `Vocabulary.bidirectional({...})` for symmetric maps (PTO ↔ paid time off). Opposite mechanism to PRF (falsified) | [CUAD_CLAUSE_EXPANSION](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_CLAUSE_EXPANSION.md) |
| `vocab.enrich(chunk_text)` | Chunk-side mirror — applied at *ingest* to raise the semantic floor of short, opaque, coded retrieval units (schema columns, API symbols, error codes, defined contract terms). Same compiled vocab, different job: `apply` patches anticipated query reformulations; `enrich` makes opaque content matchable for queries you can't anticipate | [VOCABULARY_ENRICH](https://github.com/vysakh0/redhop/blob/main/docs/findings/VOCABULARY_ENRICH.md) |
| `Document.context_with_rewrites(query, [stripper, vocab])` | Runs the chain through retrieval; per-stage audit lands on `report.query_rewrites` | (same finding as above) |
| `evaluate(query, ctx, gold_chunks=, gold_answer=)` | Deterministic A/B scoring against gold; no LLM judge. Same primitives the Decision Report uses | [EVALUATE_API](https://github.com/vysakh0/redhop/blob/main/docs/findings/EVALUATE_API.md) |

Decision rule + the recipe on the docs site:
[Choosing a configuration → "Templated queries with heavy boilerplate"](https://www.redhopai.com/docs/choosing-a-config/#3-templated-queries-with-heavy-boilerplate).

## Documentation

Full docs, the comparison vs LangChain / LlamaIndex, and the evidence behind every
default: **https://www.redhopai.com**

Apache-2.0. Also available for **Node.js** (`npm install redhop`) and **Rust**
(`cargo add redhop`).
