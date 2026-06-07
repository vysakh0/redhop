# Choosing a configuration

If you're not sure which `Document` settings to use, this page tells you in
60 seconds, based on what the docs you're loading actually look like.

> **The one rule.** **Start with the lexical default. Add knobs only when you
> have a reason.** The plain default — `Document.from_file(path).context(q)` —
> handles most document QA workloads (code, API refs, runbooks, financial
> reports, handbooks, mixed folders) because the words in the question are
> usually the words in the answer. The other configurations exist for
> specific failure shapes — described below — not as a default progression.

---

## The decision tree

```
                  Your corpus is…
                        │
   ┌────────────────────┼─────────────────────────┐
   │                    │                         │
"normal" — code,    "structured" — a              "synonym-heavy" — HR FAQs,
API refs, runbooks, contract / policy with        support tickets where
handbooks, reports, near-duplicate clauses        users ask in different
mixed folders       (e.g. governing-law           words than the doc
                    overrides per region)         uses
                        │                         │
        ▼               ▼                         ▼
  Document.from_file(p)   Document.from_file(p,    Document.from_file(p,
  doc.context(q)          retrieval="hybrid",     retrieval="hybrid",
                          model="bge-small")      model="bge-small",
                                                  rerank="cross-encoder")
                          doc.context(q,
                              include_heading=True,
                              neighbors=1)
```

That's it. Three recipes cover the measured space.

---

## The recipes, in code

### Default — works for ~5 of 6 doc shapes

No model download. ~50ms warm queries. Zero ONNX runtime.

```python
import redhop

doc = redhop.Document.from_file("contract.pdf")
ctx = doc.context("What is the governing law?")
prompt = ctx.text()         # feed to any LLM
print(ctx.report)           # see what was retrieved and why
```

**When this is right:** code, API references, runbooks, financial reports,
internal docs, well-titled handbooks, mixed folders (`from_folder`). The
queries share vocabulary with the answers — which is the case far more often
than people expect, especially for technical and policy content.

### Structured docs with parallel clauses

Bumps to **hybrid + heading-aware retrieval**. Adds an ~80MB embedding model
download on first run; warm queries climb to ~150ms. Worth it *only* if your
doc has clauses like "main clause X" *and* "EU override of clause X" *and*
"Japan override of clause X" — heading awareness disambiguates them.

```python
doc = redhop.Document.from_file(
    "msa.pdf",
    retrieval="hybrid",
    model="bge-small",
)
ctx = doc.context(
    "What law applies in the UK?",
    include_heading=True,
    neighbors=1,
)
```

**When this is right:** legal contracts with regional variations,
multi-jurisdiction policies, vendor security questionnaires with repeated
sub-sections. **When it's wrong:** clean single-chapter docs — measured to
*hurt* a 101-page handbook (97% → 94%) because `neighbors=1` dilutes well-
structured chapter content.

### Synonym-heavy domains

Adds a cross-encoder reranker — closes the synonym gap (the canonical
"employee left" vs "staff terminated" case). Adds ~300MB of model download
and 5–10× query latency.

```python
doc = redhop.Document.from_file(
    "support_kb.md",
    retrieval="hybrid",
    model="bge-small",
    rerank="cross-encoder",
)
ctx = doc.context("why did the worker leave?")
```

**When this is right:** corpora where queries and answers regularly share
*no surface words* (HR, support FAQs translated from internal phrasing,
multilingual content). **When it's wrong:** anywhere the default already
works — measured to add 0 accuracy on 6 corpora while paying full latency
cost. **Verify on your corpus before adopting.**

---

## Query writing — the part the user controls

The library can only retrieve what your query gives it. Two patterns we
saw fail in this eval that no config fixed:

### 1. One-word polysemy queries

`'vendor'` retrieves §C.10 Vendor Risk Management — not §7.2 Limitation
of Liability (even though §7.2 also mentions vendors). `'settle'`
retrieves §8.5 indemnification ("settle a claim"), not §9.2 arbitration
("settle a dispute"). **All five tiers including cross-encoder rerank
agreed.** This is a structural ambiguity in the doc, not a tier failure.

**Fix it in the query, not the config:** add one disambiguating word.
`'liability cap for vendor'` correctly finds §7.2. `'arbitration forum
to settle disputes'` finds §9.2.

### 2. Natural-language paraphrase with no shared vocabulary

`'How long do I have to cancel and get my money back?'` against a
contract that uses *"refund"* and *"termination for convenience"* —
not "cancel" or "money back" — returns an empty context across every
tier we tested.

**Fix in the query:** use the doc's vocabulary. *"What's the refund
window?"* finds §3.4 immediately. **Fix at the config level (sometimes):**
`retrieval="semantic"` (full dense, BM25 bypassed) returns *something*
where hybrid returns empty — but the result may still not be the right
clause. There's a [known bug](https://github.com/vysakh0/redhop/issues/1)
where hybrid sometimes returns fewer candidates than lexical alone.

### 3. Templated queries with heavy boilerplate

When every query in your workload follows a fixed template — *"Highlight
the parts (if any) of this contract related to X that should be reviewed
by a lawyer. Details: …"*, *"Help me with X, my account is Y, the error
is Z"*, form-filled queries from structured UIs — BM25 weights each term
in the query by corpus IDF, not query-set frequency. So the 19 boilerplate
words **dilute** the 5 real signal words.

**Two paths up the same hill — pick one, don't combine.** Measured on
CUAD ([findings/CUAD_HYBRID_RERANK.md](findings/CUAD_HYBRID_RERANK.md)):

| path | what you do | retention | latency |
| ---- | ----------- | ---------:| -------:|
| **One-knob** | `retrieval="hybrid"` (BGE-small embedder) | ~86–88% | ~10 ms/q |
| **Best-quality** | BM25 default + `analyze_query_set` → `drop_template_terms` + `expand_query_terms` (workload dict) | **90.3%** | ~2.5 ms/q |

Hybrid retrieval reads chunks as semantic content rather than counting
tokens, so the boilerplate ratio stops mattering. It substitutes for
template stripping by a different mechanism. **Running both gives
diminishing returns** — once one mechanism has fixed the boilerplate
dilution, the other adds only +0.3 points. Strip + expand is
Pareto-optimal on CUAD (higher retention AND lower latency) but takes
the upfront work of writing a stripper and building a synonym dict.

**Recommended workflow if you go the best-quality path:** detect → strip
→ (optional) expand → A/B. The first three steps ship as helpers in the
public API (`analyze_query_set`, `drop_template_terms`,
`expand_query_terms`); the fourth is up to you with your own
gold-evidence sample. Decision rule:

```python
import redhop

# 1. Detect — hand a representative sample of your queries to the analyzer.
report = redhop.analyze_query_set(my_queries[:300])
# Cross-workload probe (findings/QUERY_SET_ANALYZER.md):
#   CUAD     → is_templated=True,  share=0.66, cost="high"
#   HotpotQA → is_templated=False, share=0.00, cost="none"
#   MuSiQue  → is_templated=False, share=0.12, cost="none"

if not report.is_templated:
    # Diverse natural-language queries — no template to strip. Skip the rest.
    pass
else:
    # 2. Strip — use the boilerplate the analyzer found, at your boundary.
    def strip(q):
        return redhop.drop_template_terms(q, report.boilerplate_terms)

    # 3. (optional) Expand — when you have a known taxonomy of "topics"
    #    each with predictable synonyms (clause types, error codes,
    #    issue categories), add them with redhop.expand_query_terms.
    #    Adds high-IDF discriminators to the (already-stripped) query
    #    so BM25 ranks the relevant chunk higher. The opposite mechanism
    #    direction from PRF (which fails on boilerplate-heavy corpora,
    #    findings/CUAD_PRF_NULL.md) — this works because the synonyms
    #    are workload-curated, high-IDF, not corpus-frequency-derived.
    expansions = {
        # YOUR workload's keys → synonyms (CUAD example shown in
        # findings/CUAD_CLAUSE_EXPANSION.md).
        "change of control": ["merger", "successor", "acquisition"],
        "non-compete":       ["restraint", "non-competition"],
    }
    def preprocess(q):
        return redhop.expand_query_terms(strip(q), expansions)

    # 4. A/B — redhop.evaluate scores both arms deterministically; no
    #          LLM judge, no extra dependencies. The composite `overall`
    #          plus the components let you compare arms across a sample
    #          of queries. See findings/EVALUATE_API.md for design.
    doc = redhop.Document.from_text(your_document)
    eval_a = redhop.evaluate(
        user_query,
        doc.context(user_query, strategy="raw_topk"),
        gold_chunks=your_gold_chunk_ids,
    )
    eval_b = redhop.evaluate(
        preprocess(user_query),
        doc.context(preprocess(user_query), strategy="raw_topk"),
        gold_chunks=your_gold_chunk_ids,
    )
    # eval_b.overall - eval_a.overall is the per-query lift.
```

The analyzer measures the *shape* of your queries; it does **not**
promise a specific retention lift. On CUAD the lift was measured
directly at +6 points ≥0.8 retention (82% → 88%, overtaking LlamaIndex
at 86%; see [findings/CUAD_RECALL_GAP.md](findings/CUAD_RECALL_GAP.md)).
On a different templated workload the magnitude depends on how much of
your real query signal was being drowned, which is why step 3 matters.

**For contract-style single-doc extraction workloads also override the
Auto strategy and use `strategy="raw_topk"`.** The Auto policy routes
large contexts to `reasoning_preserving`, which solves a multi-hop
problem CUAD doesn't have. RawTopK beats ReasoningPreserving by ~4 points
on CUAD at every chunk size.

> **Why we don't ship a built-in `strip_template()` helper.** Templates
> are workload-specific — CUAD's boilerplate isn't your boilerplate.
> Baking one in would make the wrong call for the next workload.
> `drop_template_terms` takes *your* boilerplate so the call stays on
> your side. See the design rationale in
> [findings/CUAD_RECALL_GAP.md](findings/CUAD_RECALL_GAP.md).
>
> **What about PRF / query expansion?** Tested twice on RedHop with two
> different failure mechanisms; null on both. See
> [findings/CUAD_PRF_NULL.md](findings/CUAD_PRF_NULL.md) — the dilution
> win here is *subtraction* at the query boundary, not *addition*.

---

## Trade-offs at a glance

| | Lexical default | Hybrid + bge | + cross-encoder rerank |
|---|---|---|---|
| First-run model download | none | ~80MB (bge-small) | + ~300MB (cross-encoder) |
| Warm query latency | **~50ms** | ~150ms | ~1000ms |
| Compile-time deps | none | ONNX runtime | ONNX runtime |
| Where it helps | most document QA | regional overrides / parallel sub-sections | synonym-mismatch retrieval |
| Where it hurts | — | adds latency on docs lexical already handles | adds latency without recovering anything *unless* the failure mode is synonym mismatch |

---

## See also

- **Context optimization strategy** (different question — when to prune what
  was retrieved): [docs/retrievaltips.md](retrievaltips.md).
- **Real-dataset evaluations** (CUAD legal, HotpotQA multi-hop): [docs/findings/](findings/).
- **API reference:** the `Document.from_file()` and `context()` kwargs.
