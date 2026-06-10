# RedHop

**A reasoning-preserving context runtime for RAG.**

[![PyPI](https://img.shields.io/pypi/v/redhop?label=pypi&color=e11d48)](https://pypi.org/project/redhop/)
[![Python](https://img.shields.io/pypi/pyversions/redhop?color=e11d48)](https://pypi.org/project/redhop/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/vysakh0/redhop/blob/main/LICENSE)
[![Evidence layer](https://img.shields.io/badge/evidence-layer-blue)](https://github.com/vysakh0/redhop/tree/main/docs/findings)

Hand it a document and a question. RedHop chunks, retrieves, and allocates the
context your model should actually see, then tells you what it kept, what it dropped,
and why, with citations back to the source. No vector database, no LLM, all in-process.

## Get started in 60 seconds

```bash
pip install redhop
```

```python
import redhop

doc = redhop.Document.from_file("contract.pdf")    # parses + chunks + indexes
ctx = doc.context("What is the governing law?")    # retrieves + assembles
answer = llm.generate(ctx.text())                  # any LLM — no lock-in
```

That's it. `ctx.citations` tells you where the answer came from.
`ctx.report` explains what was kept, dropped, and why.

## How it compares

Measured on identical documents + budgets + BM25 retrieval, RedHop **beats both
frameworks on multi-hop evidence retention** (80% vs LangChain 71%, LlamaIndex 72%)
and **beats LangChain on contracts** (82% vs 73%). On CUAD's raw-template query
LlamaIndex leads by 4 (LlamaIndex 86% vs RedHop 82% ≥0.8 retention).

**Honest fair-preprocessing result** (`bench/compare.py`, n=300, 2026-06-08):
applying `Stripper(boilerplate)` to every system's query lifts everyone:
LlamaIndex 86% → 94%, RedHop 82% → 88%, LangChain 73% → 79%. LlamaIndex
actually benefits more from the same Stripper than RedHop does. RedHop
reaches **90.7%** by additionally layering a hand-authored 34-key
clause-name `Vocabulary` on top, but that recipe was not applied to
LlamaIndex, and the +4.7 framing previously reported here is RedHop-with-
recipe vs LlamaIndex-default, not a like-for-like comparison.

RedHop's clearer architectural lead is **multi-hop retention**, replicated on
two datasets at n=300: **HotpotQA ≥0.8 retention 80% vs LlamaIndex 72%, LangChain
71% (+8)**. **MuSiQue ≥0.8 retention 22% vs LlamaIndex 17%, LangChain 19% (+3 to
+5)**. Compositional multi-hop is harder, the magnitude shrinks but the lead
holds at the ≥0.8 threshold. `raw_topk` matches `reasoning_preserving` on both,
so the edge is RedHop's chunking + BM25 defaults rather than the assembly
strategy.

**Push multi-hop further with `retrieval="hybrid"`**: measured +12 ≥0.8 on
HotpotQA (71% → 83%) and +8 ≥0.5 on MuSiQue (66% → 74%) at n=100, at
~90-120× per-query latency (3ms → 250-400ms). Stripper and candidate_k tuning
don't help on multi-hop: only dense rerank pierces the lexical-vs-semantic
gap on bridge passages.

**Apples-to-apples hybrid vs LangChain/LlamaIndex (same bge-small, n=100,
post pure-rerank fix):** HotpotQA: RedHop hybrid wins (81% ≥0.8 vs
LangChain 77%, LlamaIndex 67%). MuSiQue: LangChain leads narrowly
(39% vs RedHop 34%, LlamaIndex 31%). The 0.3.1 audit traced the
MuSiQue gap to RedHop's RRF fusion burying bridge passages with low
BM25 + high dense rank. This release switches the default to pure
rerank. Net: HotpotQA −2, MuSiQue +8 (close to predicted +10). Latency
profile (2-5× slower than competitors' hybrid) is a separate open item.
See [MULTIHOP_HYBRID_COMPETITORS.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/MULTIHOP_HYBRID_COMPETITORS.md)
+ [MULTIHOP_CONSTANT_CHUNKING.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/MULTIHOP_CONSTANT_CHUNKING.md).

What RedHop's CUAD recipe offers is a reproducible, in-process, audited path
from 82% → 87.7% → 90.7% using `Stripper` + `Vocabulary` with a Decision
Report. The primitives are reusable on any templated workload. See
[CUAD_CLAUSE_EXPANSION.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_CLAUSE_EXPANSION.md),
[MUSIQUE_MULTIHOP.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/MUSIQUE_MULTIHOP.md),
and [MULTIHOP_HYBRID.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/MULTIHOP_HYBRID.md).

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
irrelevant context far better than they tolerate *missing reasoning links*, so the
chunk a multi-hop answer depends on is often low-relevance to the query and gets
silently pruned. RedHop's default keeps it, and makes the trade-off visible. It is
**not** a retriever, vector database, agent framework, or workflow engine. It does one
thing: turn a document and a query into the right prompt context, and explain the
decision.

## It explains every decision

Every call returns a **Decision Report**: what it kept, what it dropped, and *why*,
including when it deliberately leaves a small context untouched.

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/decision_report.svg" alt="Sample Decision Report" width="100%">
</p>

Read the fields directly via `ctx.report.auto_decision`, `total_tokens`,
`retained_evidence_ratio`, or call `doc.analyze(query)` for the report **without**
assembling a context. When retrieval looks weak, `ctx.report.diagnosis` lists
the query terms that appear nowhere in the corpus and fires bounded hints
(vocab mismatch, templated boilerplate, polysemy) with a link to the
measured finding behind each one. Example:
[`12_diagnosis.py`](https://github.com/vysakh0/redhop/blob/main/examples/python/12_diagnosis.py).

## Already running retrieval somewhere else?

Point the same diagnostics at your existing LangChain / LlamaIndex /
pgvector pipeline without migrating. Three calls, no behavior change:

```python
# Hand the candidates your retriever returned to RedHop for diagnosis.
texts = your_retriever.invoke(query)        # whatever you have today
chunks = [redhop.Chunk(t, id=str(i), source="external") for i, t in enumerate(texts)]
report = redhop.analyze_context(query, chunks)

# Aggregate across a workload: one focus recommendation, citing the finding.
reports = [redhop.analyze_context(q, retrieve(q)) for q in production_queries]
summary = redhop.summarize_diagnoses(reports)
print(summary)

# Ship the report to your telemetry (OpenTelemetry, Langfuse, anything).
from redhop.otel import report_to_attributes
span.set_attributes(report_to_attributes(report))
```

Walk-through: [`docs/DIAGNOSE_YOUR_PIPELINE.md`](https://github.com/vysakh0/redhop/blob/main/docs/DIAGNOSE_YOUR_PIPELINE.md).
Example: [`13_workload_audit.py`](https://github.com/vysakh0/redhop/blob/main/examples/python/13_workload_audit.py).

## Cite the evidence

Every selected chunk remembers where it came from:

```python
for c in ctx.citations:
    print(c["source"], c["page"], c["heading"])
    # contract.pdf  3     None      ->  "contract.pdf, p.3"
    # notes.md      None  "Refunds" ->  "notes.md -> Refunds"
```

## Show your work: query rewrites with an audit trail

Every transformation between the raw query and what BM25 actually saw is
recorded on the same Decision Report. Compile a `Stripper` (boilerplate
removal), a `Vocabulary` (workload-curated synonyms), or both, run them as
a chain via `doc.context_with_rewrites(...)`, and the per-stage records
land on `ctx.report.query_rewrites`:

```python
stripper = redhop.Stripper(["highlight", "the", "parts", "of", "this", "contract"])
vocab    = redhop.Vocabulary({"change of control": ["merger", "successor", "acquisition"]})

ctx = doc.context_with_rewrites(query, [stripper, vocab])

for rec in ctx.report.query_rewrites:
    print(rec.stage, "matched=", rec.matched, "added=", rec.added)
```

The same `Vocabulary` works **chunk-side** at ingest via `vocab.enrich(chunk_text)`.
It lifts retrieval **+0.19 mean recall** on schema-style corpora
([SPIDER_ENRICH](https://github.com/vysakh0/redhop/blob/main/docs/findings/SPIDER_ENRICH.md)),
and is measured to *hurt* (−2.0pt) on long prose chunks
([CUAD_ENRICH_DEFINITIONS_NULL](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md)).
A/B with `redhop.evaluate(...)` to confirm before adopting.

## Score the change: deterministic, or LLM-judged when you need it

`redhop.evaluate(...)` runs in two modes. Use deterministic in CI on
every PR. Opt into a judge when you want faithfulness / relevancy /
correctness against generated answers.

**Deterministic**: no API calls, ~ms per query. Returns
`context_recall` / `context_precision` / `answer_token_recall` /
`faithfulness_lexical` / `relevancy_lexical` / `correctness_lexical`
+ a composite `overall`. Same primitives the Decision Report uses.

```python
ctx_a = doc.context(user_query)
ctx_b = doc.context_with_rewrites(user_query, [stripper, vocab])
eval_a = redhop.evaluate(user_query, ctx_a, gold_chunks=gold_ids)
eval_b = redhop.evaluate(user_query, ctx_b, gold_chunks=gold_ids)
print("lift on overall:", eval_b.overall - eval_a.overall)
```

**LLM-judged**: pass `judge=` plus your own LLM caller (OpenAI,
Anthropic, OpenRouter, local). Adds `faithfulness_judged` /
`relevancy_judged` / `correctness_judged` to the same report.
Claim-decomposed faithfulness (`decompose_faithfulness=True`) is
substantively equivalent to Ragas: r=+0.664, MAE=0.151 on n=200
HotpotQA, see [COMPARISON_RAGAS](https://github.com/vysakh0/redhop/blob/main/docs/COMPARISON_RAGAS.md).
TP/FP/FN F₁ via `decompose_correctness=True`.

```python
def my_llm(prompt, system):
    # Your LLM SDK call — return a float or {"score": float}.
    return float(openai_client.chat.completions.create(...).choices[0].message.content)

judge = redhop.Judge.from_callable(my_llm).cached()
report = redhop.evaluate(
    user_query, ctx,
    answer="The refund window is thirty days.",
    gold_answer="thirty days",
    judge=judge,
    decompose_faithfulness=True,
    decompose_correctness=True,
)
```

For user-defined aspects (harmfulness, conciseness, brand voice…),
`redhop.critique(answer, aspects=[...], judge=...)` runs one judge
call per aspect with polarity-corrected scores. Aggregate test sets
with `redhop.summarize(reports)`.

Full API + field list:
[ANSWER_QUALITY_EVAL](https://github.com/vysakh0/redhop/blob/main/docs/findings/ANSWER_QUALITY_EVAL.md).

## Loading documents

| On-ramp | For |
| --- | --- |
| `Document.from_text(text, source="document")` | text you already have |
| `Document.from_chunks([redhop.Chunk(...), ...])` | content you already chunked: pass typed `redhop.Chunk(text, source=..., id=..., metadata={...})` instances |
| `Document.from_file("x.pdf")` | a file: PDF, DOCX, PPTX, XLSX, Markdown, or text/code |
| `Document.from_bytes(data, source="x.pdf")` | bytes you fetched (S3 / GCS / HTTP / DB) |
| `Document.from_folder("./docs", persist=True)` | a whole directory, with an optional incremental on-disk index |

## Retrieval tiers: no vector database

Start at the lexical default (it handles most document QA because the words
in the question are usually the words in the answer) and climb only when the
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

Default is a minimal analyzer (tokenize + lowercase + ASCII fold, no
stemmer), measured to beat English Snowball on every English workload
we tested ([RAW_ANALYZER](https://github.com/vysakh0/redhop/blob/main/docs/findings/RAW_ANALYZER.md)).
Swap with the `language=` kwarg: `"english"` for code search /
inflection-heavy English content, or any of the 18 Snowball Porter2
languages (`arabic, danish, dutch, english, finnish, french, german,
greek, hungarian, italian, norwegian, portuguese, romanian, russian,
spanish, swedish, tamil, turkish`):

```python
doc = redhop.Document.from_text(german_text, language="german")
# Now `Buch` finds chunks containing `Bücher` (and vice versa)
```

One analyzer drives both BM25 retrieval AND the grounding scorer, so
they can't drift on what "the same term" means. Unknown names raise
(we don't silently fall back to English). See the
[language guide](https://github.com/vysakh0/redhop/blob/main/docs/LANGUAGE.md)
for the full breakdown and the calibration disclaimer (we ship the
stemmers, but eval-corpus ranking quality on a real domain corpus is the
user's call).

## Assembly strategies

| `strategy=` | What it does |
| --- | --- |
| `reasoning_preserving` *(default)* | keep query-relevant seeds **and** rescue low-relevance chunks linked to one, and drop only unlinked junk |
| `distractor_filtered` | drop everything below a query-grounding bar |
| `max_density` | greedily pack the densest chunks into the budget |
| `raw_topk` | keep retrieval order until the budget fills |
| `auto` | size-gated: pass small contexts through, prune large/diluted ones |

Already have chunks from your own retriever? Wrap each as `redhop.Chunk(text,
source=..., id=..., metadata={...})` and pass into
`redhop.build_context(query, retrieved_chunks=chunks, ...)` (low-level) or
`redhop.Document.from_chunks(chunks)` (full indexing).

## Templated workloads: the +9 retention lift (BM25, no model needed)

If every query in your workload follows a fixed template, such as legal QA
("*Highlight the parts (if any) of this contract related to X. Details: …*"),
support-ticket triage ("*Help me with X, my account is Y, the error is Z*"),
or form-filled queries from a structured UI, **BM25 weights every query term
by corpus IDF, not by how often the term repeats across your query set**.
The boilerplate words dilute the real signal words, and retention suffers.
This is the mechanism behind the 4-point CUAD gap on the head-to-head.
Closing it doesn't need a vector DB or a different retriever: it needs two
small preprocessing helpers on the query side.

<p align="center">
  <img src="https://raw.githubusercontent.com/vysakh0/redhop/main/.github/workflow_lift.svg" alt="RedHop CUAD retention rises 81.3% → 87.7% → 90.7% via Stripper then Vocabulary. LlamaIndex is at 86% (raw template). Fair-preprocessing footnote: the same Stripper applied to LlamaIndex's query lifts it to 94%, and the Vocabulary recipe was not applied to LlamaIndex." width="100%">
</p>

**Measured** on the CUAD framework comparison (n=300, BM25, budget 2,000 tok):

| step | helper | retention | Δ |
| ---- | ------ | ---------:| -:|
| raw 24-word template | — | 81.3% | — |
| + strip the wrapper | `Stripper` | 87.7% | **+6.4** |
| + add workload synonyms | `Vocabulary` | **90.7%** | **+3.0** |

**RedHop with the full workflow is at 90.7%, beating LlamaIndex by 4 points
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
  conservative by design: HotpotQA and MuSiQue both register quiet
  (`is_templated=False`) in the cross-workload probe, while CUAD fires. If
  yours doesn't fire, skip this section.
- **The analyzer measures the *shape* of your query set, not your
  retention.** It says "this *looks* like a templated workload" with
  the boilerplate terms it found. It does **not** promise a specific
  lift. Always A/B on your gold-evidence sample before committing.
- **For single-doc extraction workloads also set `strategy="raw_topk"`.**
  `auto` routes large contexts to `reasoning_preserving`, which solves a
  multi-hop problem contract extraction doesn't have. RawTopK beats it
  by ~4 points at every chunk size on CUAD.
- **We deliberately don't ship a CUAD-specific `strip_template()`
  helper.** Templates are workload-specific. Baking one in would make
  the wrong call for the next workload. `Stripper(...)` and
  `Vocabulary({...})` take *your* boilerplate / synonym dict so the
  call stays on your side.
- **Or take the one-knob alternative: `retrieval="hybrid"`.**
  Dense reads chunks as semantic content rather than counting tokens,
  so the boilerplate ratio stops mattering. Substitutes for stripping
  by a different mechanism (+5.3 on raw CUAD at ~10ms/query). On CUAD
  specifically, BM25 + strip + vocabulary still wins: 90.7% / 2.5ms
  vs hybrid+CE 89.0% / 683ms. The two paths are *substitutes*, not
  complements, so pick one. See
  [CUAD_HYBRID_RERANK.md](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_HYBRID_RERANK.md).

| helper | what it does | finding |
| ------ | ------------ | ------- |
| `analyze_query_set(queries)` | Inspects your queries and flags whether they're templated and which terms are doing the dilution | [QUERY_SET_ANALYZER](https://github.com/vysakh0/redhop/blob/main/docs/findings/QUERY_SET_ANALYZER.md) |
| `Stripper(boilerplate)` | Compiled token-level boilerplate strip, word-boundary safe (an `"of"` strip does not erase `"of"` inside `"office"`). Plugs into the rewrite chain so the audit trail is captured | [CUAD_RECALL_GAP](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_RECALL_GAP.md) · [MULTILINGUAL_ANALYZER](https://github.com/vysakh0/redhop/blob/main/docs/findings/MULTILINGUAL_ANALYZER.md) |
| `Vocabulary({key: [synonyms]})` | Compiled workload-curated equivalence classes: appends high-IDF synonyms when the token-level key matches. `Vocabulary.bidirectional({...})` for symmetric maps (PTO ↔ paid time off). Opposite mechanism to PRF (falsified) | [CUAD_CLAUSE_EXPANSION](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_CLAUSE_EXPANSION.md) |
| `vocab.enrich(chunk_text)` | Chunk-side mirror. **Measured to lift retrieval +0.19 mean recall on Spider-shape schemas**. Use it when your retrieval units are short and opaque (schema columns, error codes, API symbols, defined contract terms). Measured to *hurt* (−2.0pt) on long prose chunks, so don't use it there. A/B with `redhop.evaluate(...)` against your gold before adopting | [SPIDER_ENRICH](https://github.com/vysakh0/redhop/blob/main/docs/findings/SPIDER_ENRICH.md) + [VOCABULARY_ENRICH](https://github.com/vysakh0/redhop/blob/main/docs/findings/VOCABULARY_ENRICH.md) + [CUAD_ENRICH_DEFINITIONS_NULL](https://github.com/vysakh0/redhop/blob/main/docs/findings/CUAD_ENRICH_DEFINITIONS_NULL.md) |
| `Document.context_with_rewrites(query, [stripper, vocab])` | Runs the chain through retrieval. Per-stage audit lands on `report.query_rewrites` | (same finding as above) |
| `evaluate(query, ctx, gold_chunks=, gold_answer=, judge=, decompose_faithfulness=, decompose_correctness=)` | A/B scoring against gold. Deterministic-by-default (lexical, no LLM). Opt-in `judge=` adds LLM-judged faithfulness/relevancy/correctness, with claim-decomposition and TP/FP/FN modes. Same primitives the Decision Report uses | [ANSWER_QUALITY_EVAL](https://github.com/vysakh0/redhop/blob/main/docs/findings/ANSWER_QUALITY_EVAL.md) · [COMPARISON_RAGAS](https://github.com/vysakh0/redhop/blob/main/docs/COMPARISON_RAGAS.md) |
| `critique(answer, aspects, judge=)` | LLM-judged scoring for user-defined dimensions (harmfulness, conciseness, brand voice…). One judge call per aspect, polarity-corrected so high = good | [ANSWER_QUALITY_EVAL](https://github.com/vysakh0/redhop/blob/main/docs/findings/ANSWER_QUALITY_EVAL.md) |
| `ctx.report.diagnosis` | Query-level facts (terms missing from the corpus, score spread, per-term DF) plus a closed registry of bounded hints (vocab mismatch, polysemy, templated boilerplate). Every hint cites the measured finding behind it. Always computed, observation only | [REPORT_DIAGNOSIS](https://github.com/vysakh0/redhop/blob/main/docs/design/REPORT_DIAGNOSIS.md) · [CHOOSING_A_CONFIG](https://github.com/vysakh0/redhop/blob/main/docs/CHOOSING_A_CONFIG.md) |
| `summarize_diagnoses(reports)` | Workload-level aggregation: hint histogram, failure rates, top vocabulary gaps, and at most one focus recommendation citing the finding behind it. Six focus codes (`vocab_mismatch`, `templated_queries`, `underdetermined_queries`, `weak_retrieval`, `healthy`, `sample_too_small`) | [WORKLOAD_AUDIT](https://github.com/vysakh0/redhop/blob/main/docs/design/WORKLOAD_AUDIT.md) |
| `redhop.otel.report_to_attributes(report)` | Flattens any Decision Report into OpenTelemetry / Langfuse-compatible span attributes under a `redhop.` namespace. Zero new dependencies. Pair with `analyze_context` to instrument an existing LangChain / LlamaIndex / pgvector pipeline | [DIAGNOSE_YOUR_PIPELINE](https://github.com/vysakh0/redhop/blob/main/docs/DIAGNOSE_YOUR_PIPELINE.md) |

Decision rule + the recipe on the docs site:
[Choosing a configuration → "Templated queries with heavy boilerplate"](https://www.redhopai.com/docs/choosing-a-config/#3-templated-queries-with-heavy-boilerplate).

## Documentation

Full docs, the comparison vs LangChain / LlamaIndex, and the evidence behind every
default: **https://www.redhopai.com**

Apache-2.0. Also available for **Node.js** (`npm install redhop`) and **Rust**
(`cargo add redhop`).
