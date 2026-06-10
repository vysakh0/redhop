# Diagnose your existing RAG pipeline

You already run retrieval. Maybe it is LangChain BM25 over your
contracts, LlamaIndex over a notebook of code, pgvector behind an
internal API, or something you wrote yourself. You do not want to
migrate. You want to know **why retrieval sometimes returns the wrong
thing**, and which single knob the data says to reach for first.

This page walks you through pointing RedHop's Decision Report at
your existing pipeline. Three calls, no behavior change, ~10 lines,
no new SDKs.

> Without gold labels this measures *failure shapes*, not retention or
> answer quality. `analyze_context` reports waste; only
> `build_context` (the native RedHop path) removes it. If the summary
> says `healthy`, RedHop has nothing to sell you, and that is the
> correct outcome.

## Who this is for

You retrieve with one of:

- **LangChain** (BM25Retriever, vector stores, ensemble retrievers)
- **LlamaIndex** (BM25, dense indexes, retriever modules)
- **pgvector** / **Weaviate** / **Qdrant** / **Milvus** (you query directly)
- A hand-rolled retriever inside your application code

You want a structured answer to "what is going wrong, and what does
the evidence say to do about it?" without changing the pipeline.

## Step 1: one query, zero behavior change

The shortest path. Hand RedHop the texts your retriever already
returned. `analyze_context` observes the candidate pool without
modifying it.

```python
import redhop

# Your existing pipeline; here is a LangChain sketch.
# from langchain_community.retrievers import BM25Retriever
# retriever = BM25Retriever.from_texts(my_corpus_texts)
# texts = [d.page_content for d in retriever.invoke(query)]

texts = retriever_invoke(query)              # whatever you have today
chunks = [redhop.Chunk(t, id=str(i), source="external")
          for i, t in enumerate(texts)]
report = redhop.analyze_context(query, chunks)

print(report)                                # the Decision Report
print(report.diagnosis["hints"])             # any bounded hints that fired
```

What you get back: per-candidate facts about how the query interacted
with what your retriever produced. The score spread, the empty-context
flag, the low-confidence flag, the query terms that appear in none of
the retrieved candidates. Layer-1 facts. Corpus-wide stats are not
computable here because RedHop does not see your corpus, only what
came out of your retriever.

## Step 2: corpus-level diagnosis

Two lines to upgrade. Load the same chunks into a Document so RedHop
can build a vocabulary map and tell you which query terms appear
**nowhere in your corpus** (the canonical paraphrase failure: the user
says "cancel" and "money back", the doc says "refund" and
"termination for convenience").

```python
doc = redhop.Document.from_chunks(
    [redhop.Chunk(t, id=str(i), source="corpus")
     for i, t in enumerate(my_corpus_texts)]
)

ctx = doc.context(query)
print(ctx.report.diagnosis["zero_match_terms"])  # query terms missing from the corpus
print(ctx.report.diagnosis["term_stats"])        # per-term corpus frequency
```

RedHop indexes a copy in memory; your retrieval is untouched. The
report on `ctx` carries the full diagnosis with `corpus_stats_available
= True`. The five hint codes from
[`docs/CHOOSING_A_CONFIG.md`](CHOOSING_A_CONFIG.md) (vocabulary
mismatch, polysemy, templated boilerplate, plus
`empty_context` / `low_confidence`) can now fire with full evidence.

## Step 3: audit the workload

One query tells you about one query. The real value is across hundreds
of production queries. Collect the reports, call `summarize_diagnoses`,
read the focus.

```python
reports = [doc.context(q).report for q in production_queries]
summary = redhop.summarize_diagnoses(reports)
print(summary)
```

Rendered output (excerpt):

```
RedHop Workload Audit
═════════════════════

  Reports aggregated: 247

Hint histogram
──────────────
  - empty_context               42  (17%)
  - vocab_mismatch              81  (33%)
  - low_confidence              57  (23%)

Rates
─────
  Empty-context rate:    17%
  Low-confidence rate:   23%
  Corpus-stats coverage: 100%

Top zero-match terms
────────────────────
  "cancel", "refund", "subscription", "trial", "billing", "charge", ...

Focus
─────
  Code: vocab_mismatch
  33% of queries had most terms missing from the corpus. Top gap terms:
  "cancel", "refund", "subscription", "trial", "billing", "charge".
  Rephrasing toward the documents' vocabulary is the measured first
  fix, and dense retrieval (retrieval="hybrid") was measured to lift
  retention on exactly this shape.
      evidence: docs/findings/MULTIHOP_HYBRID.md
```

Exactly one focus per summary. Resolution is by a fixed priority
order (vocab mismatch outranks templated outranks underdetermined
outranks weak-retrieval), with thresholds documented in
[`DEFAULT_PROVENANCE.md`](DEFAULT_PROVENANCE.md). The cited finding
is what justifies the recommendation. RedHop never auto-applies the
fix.

### What each focus code points at

| Focus | What it means | Where the fix lives |
| --- | --- | --- |
| `vocab_mismatch` | Queries use different words than the docs do | Rephrase, or `retrieval="hybrid"`, or build a `Vocabulary` |
| `templated_queries` | Fixed wrappers dominate the queries | `analyze_query_set` + `Stripper` |
| `underdetermined_queries` | Short queries can't pick between candidates | UI prompt for one more keyword |
| `weak_retrieval` | High empty/low-confidence but no specific shape | Inspect `top_zero_match_terms`; your corpus may not cover the questions |
| `healthy` | No failure shape above the threshold | No intervention indicated |
| `sample_too_small` | Fewer than 20 queries aggregated | Run more |

## Step 4: ship it to your telemetry

Decision Reports are intrinsic, structured, and serializable. Every
field flattens cleanly into OpenTelemetry attributes or Langfuse
metadata. RedHop imports no SDK; you attach the dict.

### OpenTelemetry (Python)

```python
from opentelemetry import trace
from redhop.otel import report_to_attributes

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("rag.query") as span:
    ctx = doc.context(query)
    span.set_attributes(report_to_attributes(ctx.report))
    # Send the full report body as a span event for the rich details.
    span.add_event("rag.report", attributes={"redhop.report": ctx.report.json()})
```

### Langfuse (Python)

```python
from langfuse import Langfuse
from redhop.otel import report_to_attributes

langfuse = Langfuse()
ctx = doc.context(query)
langfuse.trace(
    name="rag.query",
    input=query,
    output=ctx.text(),
    metadata=report_to_attributes(ctx.report),
)
```

### Node.js

There is no helper module on Node; the conventions table is small
enough to inline. Drop this next to your retrieval call:

```javascript
function reportToAttributes(report, prefix = "redhop.") {
  const d = report.diagnosis;
  const attrs = {
    [prefix + "strategy"]: report.strategy,
    [prefix + "auto_decision"]: report.autoDecision,
    [prefix + "input_tokens"]: Number(report.inputTokens),
    [prefix + "total_tokens"]: Number(report.totalTokens),
    [prefix + "token_budget"]: Number(report.tokenBudget),
    [prefix + "n_selected"]: Number(report.nSelected),
    [prefix + "retained_evidence_ratio"]: Number(report.retainedEvidenceRatio),
    [prefix + "evidence_density"]: Number(report.evidenceDensity),
    [prefix + "estimated_waste_tokens"]: Number(report.estimatedWasteTokens),
    [prefix + "second_hop_rescues"]: Number(report.secondHopRescueCount),
    [prefix + "low_confidence"]: Boolean(report.lowConfidenceRetrieval),
    [prefix + "diagnosis.empty_context"]: Boolean(d.emptyContext),
    [prefix + "diagnosis.n_candidates"]: Number(d.nCandidates),
    [prefix + "diagnosis.hints"]: d.hints.map((h) => h.code),
    [prefix + "diagnosis.zero_match_terms"]: d.zeroMatchTerms.slice(0, 16),
  };
  if (d.scoreSpread != null) attrs[prefix + "diagnosis.score_spread"] = Number(d.scoreSpread);
  return attrs;
}
```

### Rust

Either build the attribute map directly from the public struct, or
serialize with `serde_json::to_value(&ctx.report)` and walk the
result. The conventions table below pins the attribute names and types
that downstream dashboards can rely on.

### Attribute conventions (all bindings)

All keys under the `redhop.` namespace. Every value is one of:
`bool`, `int`, `float`, `string`, `list[string]`. Optional fields are
omitted rather than emitted as `null`.

| Attribute | Type | Source |
| --- | --- | --- |
| `redhop.strategy` | string | `report.strategy` |
| `redhop.requested_strategy` | string | `report.requested_strategy` |
| `redhop.auto_decision` | string | `report.auto_decision` |
| `redhop.input_tokens` | int | `report.input_tokens` |
| `redhop.total_tokens` | int | `report.total_tokens` |
| `redhop.token_budget` | int | `report.token_budget` |
| `redhop.n_input_chunks` | int | `report.n_input_chunks` |
| `redhop.n_selected` | int | `report.n_selected` |
| `redhop.retained_evidence_ratio` | float | `report.retained_evidence_ratio` |
| `redhop.evidence_density` | float | `report.economics.evidence_density` |
| `redhop.estimated_waste_tokens` | int | `report.economics.estimated_waste_tokens` |
| `redhop.second_hop_rescues` | int | `report.second_hop_rescue_count` |
| `redhop.low_confidence` | bool | `report.low_confidence_retrieval` |
| `redhop.diagnosis.empty_context` | bool | `diagnosis.empty_context` |
| `redhop.diagnosis.n_candidates` | int | `diagnosis.n_candidates` |
| `redhop.diagnosis.hints` | string[] | hint codes, in fire order |
| `redhop.diagnosis.zero_match_terms` | string[] | display-ordered, capped at 16 |
| `redhop.diagnosis.score_spread` | float | omitted when `None` |

Excluded on purpose: hint `message` strings (size, cardinality; the
`evidence` path is recoverable from the code), per-term `term_stats`
rows (belong in the JSON body of an event, not the attribute set),
the rendered report (use `report.json()` for the full body).

## Honest limits

- This measures **failure shapes**, not retention or answer quality.
  You will see "your query terms appear nowhere in the corpus", not
  "your retrieval recall is X%". For graded answer quality on a gold
  set, see [`redhop.evaluate`](../examples/python/05_evaluate_ab.py).
- `analyze_context` (Step 1) **observes** waste; it does not remove it.
  Only the native `build_context` / `Document.context` path runs the
  reasoning-preserving prune. The wins in
  [`COMPARISON.md`](COMPARISON.md) are measured on the native path.
- If `summary.focus["code"] == "healthy"`, RedHop has nothing to
  recommend, and that is the correct outcome.

## Reference

- Runnable example: [`examples/python/13_workload_audit.py`](../examples/python/13_workload_audit.py)
  (and the [Node](../examples/nodejs/13_workload_audit.cjs) and
  [Rust](../examples/rust/examples/13_workload_audit.rs) mirrors).
- Per-query diagnosis design: [`design/REPORT_DIAGNOSIS.md`](design/REPORT_DIAGNOSIS.md).
- Workload-audit design: [`design/WORKLOAD_AUDIT.md`](design/WORKLOAD_AUDIT.md).
- The five hint codes and the failure shapes they map to:
  [`CHOOSING_A_CONFIG.md`](CHOOSING_A_CONFIG.md).
- Threshold provenance: [`DEFAULT_PROVENANCE.md`](DEFAULT_PROVENANCE.md).
