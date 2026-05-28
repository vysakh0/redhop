# Choosing a configuration

If you're not sure which `Document` settings to use, this page tells you in
60 seconds, based on what the docs you're loading actually look like.

> **The one rule.** **Start with the lexical default. Add knobs only when you
> have a reason.** We measured this across 121 labeled queries on 6 real
> document shapes (legal MSA, API ref, financial report, incident runbook,
> 101-page handbook, and a 5-file folder). The plain default
> (`Document.from_file(path)`) won or tied on 5 of 6.
> Full data: [docs/findings/CORPUS_CONFIG_MATRIX.md](findings/CORPUS_CONFIG_MATRIX.md).

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

## What we measured (the matrix)

Pass rates per config × corpus. Best per row in **bold**. Full setup:
[CORPUS_CONFIG_MATRIX.md](findings/CORPUS_CONFIG_MATRIX.md).

| Corpus | lexical (default) | hybrid+bge | +heading+neighbors | +rerank |
|---|---|---|---|---|
| Legal MSA (50p, 8 regional override clauses)  | 26/29 (90%) | 26/29 (90%) | **27/29 (93%)** | 27/29 (93%) |
| API reference (10p)                            | **17/18 (94%)** | 17/18 (94%) | 17/18 (94%) | 17/18 (94%) |
| Financial report (41p)                         | **18/18 (100%)** | 18/18 (100%) | 18/18 (100%) | 18/18 (100%) |
| Incident runbook (44p)                         | **18/18 (100%)** | 18/18 (100%) | 18/18 (100%) | 18/18 (100%) |
| Engineering handbook (101p)                    | **30/31 (97%)** | 30/31 (97%) | 29/31 (94%) ⬇ | 29/31 (94%) ⬇ |
| Cross-PDF routing (5-file folder)              | **8/8 (100%)** | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |

Reading the matrix:
- **Lexical default is the right pick 5/6 times.**
- **Hybrid + heading + neighbors wins +1 on the MSA only** (the one shape
  where regional overrides make heading awareness pay off).
- **Cross-encoder rerank added 0 pass-rate** anywhere in this battery.
  It's not free — verify on your own corpus.
- **Structural expansion can hurt** when the doc is already well-structured
  (handbook went 97% → 94% with neighbors=1).

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

---

## Trade-offs at a glance

| | Lexical default | Hybrid + bge | + rerank |
|---|---|---|---|
| First-run model download | none | ~80MB (bge-small) | +~300MB (cross-encoder) |
| Warm query latency | **~50ms** | ~150ms | ~1000ms |
| Compile-time deps | none | ONNX runtime | ONNX runtime |
| Accuracy lift over default (measured) | — | **0 of 6 corpora** | **0 of 6 corpora** |
| Where it helps | everything in our battery | synonym-only failure mode | synonym-only failure mode |

---

## See also

- **The data:** [docs/findings/CORPUS_CONFIG_MATRIX.md](findings/CORPUS_CONFIG_MATRIX.md) — full eval, including the failure analysis.
- **Context optimization strategy** (different question, when to prune what
  was retrieved): [docs/retrievaltips.md](retrievaltips.md).
- **API reference:** the `Document.from_file()` and `context()` kwargs.
