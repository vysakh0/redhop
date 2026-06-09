# `evaluate` — in-process scoring of an assembled context, no LLM judge

> **Status:** **Shipped** (Rust + Python + Node, 10 Rust unit tests + 11
> Python + 9 Node assertion blocks, all green).
>
> **TL;DR:** [QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md) gave users a way
> to *detect* templated-workload dilution; [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md)
> showed the workaround (subtraction at the query boundary). The
> remaining gap was the **A/B step** — once you have a candidate fix,
> how do you measure whether it actually helped on *your* gold data
> without paying an LLM judge per call? `evaluate` is that gap closed.
> Zero LLM calls, deterministic, returns a single composite `overall`
> plus the components, with optional ground-truth signals unlocking
> additional gold-relative metrics.

## What this is

```rust
pub fn evaluate(
 query: &Query,
 ctx: &BuiltContext,
 gold: EvalGold<'_>,
) -> EvalReport;
```

Python:

```python
report = redhop.evaluate(query, ctx, gold_chunks=[...], gold_answer="...")
```

Node:

```js
const report = redhop.evaluate(query, ctx, { goldChunks: [...], goldAnswer: "..." });
```

`EvalReport` fields, identical across all three surfaces (with the usual
snake_case / camelCase shift on Node):

| field | populated when | what it measures |
| ----- | -------------- | ---------------- |
| `mean_grounding` | always | mean grounding score over selected chunks |
| `evidence_density` | always | fraction of context tokens that are query-relevant |
| `retained_evidence_ratio` | always | fraction of input evidence that made it through assembly |
| `second_hop_rescues` | always | bridge passages saved by the reasoning-preserving rescue |
| `low_confidence` | always | every selected chunk is at-or-below the grounding ceiling |
| `estimated_waste_tokens` | always | tokens spent on below-bar chunks |
| `context_recall` | `gold_chunks` provided | `|selected ∩ gold| / |gold|` |
| `context_precision` | `gold_chunks` provided | `|selected ∩ gold| / |selected|` |
| `answer_token_recall` | `gold_answer` provided | fraction of stemmed gold-answer terms in the assembled context |
| `faithfulness_lexical` | `answer` provided | **lexical**: fraction of answer sentences with ≥half their content terms in the context (lexical proxy — not real hallucination detection; use an LLM judge for that) |
| `relevancy_lexical` | `answer` provided | **lexical**: token-overlap between query and answer (Snowball-stemmed). Proxy for "did the answer address the question" |
| `correctness_lexical` | `answer` + `gold_answer` | **lexical**: token-overlap between LLM answer and gold answer. Proxy for "did the LLM produce the right tokens" |
| `overall` | always | composite in `[0, 1]` blending whichever fields above are populated |

## Lexical vs judged answer-quality metrics

The `_lexical` and `_judged` suffixes on six of the metrics above are a
deliberate naming choice. The lexical fields are deterministic
token-overlap proxies — cheap, no LLM, runs in CI on every PR. The
judged fields are LLM-scored and unlocked by passing a `Judge`. The
full surface is documented in
[ANSWER_QUALITY_EVAL.md](ANSWER_QUALITY_EVAL.md); this doc focuses on
the underlying design choice.

## Design choice: refraction, not independent measurement

The deliberate distinction from external LLM-judge eval libraries is
**`evaluate` is computed
from the same primitives the runtime already uses to make its Decision
Report.** `mean_grounding` calls the same `grounding_score` that
`ContextStrategy::DistractorFiltered` uses to decide what to drop.
`evidence_density`, `retained_evidence_ratio`, `second_hop_rescues`,
`low_confidence`, `estimated_waste_tokens` are all surfaced unchanged
from `ContextReport`. The composite `overall` blends them with a fixed
formula (no learned weights).

That's a feature, not a bug. **A low `evaluate.overall` and a
`report.low_confidence_retrieval=true` are the same signal refracted
twice — not two independent measurements.** If the runtime says
"this was low-confidence" and the eval says "this scored 0.18", you
are not looking at a discrepancy to debug; you are looking at one
signal viewed from a different vantage. The eval reframes the runtime's
own metadata as a 0–1 score the caller can compare across queries.

This is the right design when:

- You want a **fast, cheap, deterministic** eval that runs at indexing
 speed and never costs you a token.
- You're comparing **arm A vs arm B of the same pipeline** (e.g.,
 templated query vs stripped query, see CUAD_RECALL_GAP), where
 consistency of the metric across the comparison matters more than
 absolute calibration to human judgment.
- You want an eval that **never disagrees** with the runtime's
 self-assessment, so production alerting on `low_confidence` and
 offline A/B reporting on `overall` always tell the same story.

This is **not** the right design when:

- You want **calibrated absolute scores** comparable across pipelines
  with completely different retrieval engines. The composite is
  consistent within RedHop; it's not a benchmark number.

## Where it fits in the user workflow

The detect → strip → A/B workflow shipped with
[QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md) ended with "measure recall
against your gold spans on a sample." That sentence used to mean *write
your own scorer*. With `evaluate`, the workflow now ends with a call:

```python
# 1. Detect
report = redhop.analyze_query_set(my_queries[:300])

if report.is_templated:
 # 2. Strip
 def strip(q): return redhop.drop_template_terms(q, report.boilerplate_terms)

 # 3. A/B — the new bit. No LLM judge, no extra dependencies.
 doc = redhop.Document.from_text(your_document)
 eval_a = redhop.evaluate(
 user_query, doc.context(user_query, strategy="raw_topk"),
 gold_chunks=your_gold_chunk_ids,
 )
 eval_b = redhop.evaluate(
 strip(user_query), doc.context(strip(user_query), strategy="raw_topk"),
 gold_chunks=your_gold_chunk_ids,
 )
 print(eval_b.overall - eval_a.overall) # the lift, deterministically
```

## API contract details worth knowing

- **`gold_chunks=[]` vs `gold_chunks=None`** are different. `[]` means
 "no chunks need to be retrieved" — vacuously perfect recall, undefined
 precision (`None`). `None` means "no gold available; skip the metric."
 Tests pin this.
- **Empty selection** is handled: zero selected chunks plus gold chunks
 given → `context_recall=Some(0.0)`, `context_precision=Some(0.0)`,
 not `NaN`. No `mean_grounding` panic.
- **Stemming is on for `answer_token_recall`.** The metric goes
 through `grounding_score`, which uses English Snowball Porter2
 internally (fixed, independent of the `Document` default) — so
 `"refunds"` in the gold matches `"refund"` in the context regardless
 of whether the document was indexed with the 0.3.2 raw default or an
 explicit `language="english"`. A test pins this.
- **`low_confidence` caps `overall`.** If the runtime flagged the
 context as low-confidence, the composite is capped at 0.25 regardless
 of the other components — a deliberate floor to prevent a
 weak-retrieval-but-high-density situation from scoring well.

## Rust tests (10 total)

`crates/redhop/src/context/eval.rs::tests`. They guard the API contract,
not just the math:

| test | what it pins |
| ---- | ------------ |
| `self_eval_works_without_any_gold` | self-eval populated, gold-relative all `None` |
| `perfect_recall_when_all_gold_in_selected` | recall = precision = 1 in the easy case |
| `partial_recall_when_some_gold_missing_from_selection` | recall = 0.5 when one of two gold missing |
| `answer_token_recall_uses_stemming` | `"refunds"` gold matches `"refund"` context |
| `low_confidence_caps_overall_score` | off-topic query → `overall ≤ 0.25` |
| `empty_gold_chunks_returns_perfect_recall` | `Chunks(&[])` → vacuously perfect recall, no precision |
| `answer_only_gold_leaves_chunk_metrics_none` | `Answer(...)` alone → chunk metrics stay `None` |
| `both_gold_signals_populate_all_three_metrics` | `Both { ... }` → all three populated |
| `precision_distinct_from_recall_with_asymmetric_sets` | 3 selected, 2 gold, 1 hit → recall 0.5, precision ≈ 0.33 |
| `empty_built_context_is_handled_gracefully` | zero-budget selection → recall 0/precision 0, no NaN, finite overall |

## Binding-surface tests

- **Python**: `python/tests/test_evaluate.py`, 11 pytest functions
 including the full detect → strip → evaluate workflow as an
 end-to-end smoke test.
- **Node**: `nodejs/test/evaluate.cjs`, 9 assertion blocks wired into
 `npm test`.

Both mirror the Rust contract tests through the FFI boundary, so a
dropped field on `EvalReport`, a wrong `gold_chunks` kwarg shape, or a
misrouted `EvalGold` variant surfaces in the binding tests, not user
code.

## Honest limits

- **`overall` is not a benchmark number.** It's an internally-consistent
 score for comparing arms of the same pipeline. Comparing
 `evaluate.overall` across pipelines with different retrievers or
 rerankers tells you something about *RedHop's* preferences, not about
 retrieval quality in an absolute sense.
- **No CIs.** Single-call evaluation; no bootstrap variance. Pair with
 a sufficient n on your A/B sample if you want statistical
 significance.
- **English-default analyzer assumed for `answer_token_recall`.**
 Non-English workloads will produce a less informative answer-recall
 number until we surface the analyzer language through this API too.
- **No generated-answer evaluation.** This is retrieval + assembly
 only. Answer-quality scoring needs an LLM judge or a downstream
 benchmark; `evaluate` is silent about it.
- **The composite formula is fixed.** No way for the caller to
 reweight the components today. If your workload cares disproportionately
 about (say) `evidence_density`, read the field directly instead of
 the `overall`.

## What this changes

- New public API in all three bindings:
 - Rust: `redhop::evaluate`, `redhop::EvalGold`, `redhop::EvalReport`
 - Python: `redhop.evaluate(query, ctx, gold_chunks=None, gold_answer=None) -> EvalReport`
 - Node: `redhop.evaluate(query, ctx, options?) -> EvalReport`
- Node `BuiltContext` converted from `#[napi(object)]` to `#[napi]`
 class to carry the underlying Rust struct. Field reads
 (`ctx.text`, `ctx.chunks`, `ctx.citations`, `ctx.report`) are
 preserved as getters; existing JS code continues to work.
- `docs/CHOOSING_A_CONFIG.md` step 3 ("Templated queries with heavy
 boilerplate") now ends in a concrete `redhop.evaluate(...)` call
 instead of the phrase "measure recall against your gold spans."
- The detect → strip → A/B workflow documented in the READMEs
 (root + python + nodejs) is end-to-end runnable from the public API
 for the first time.
