# CUAD hybrid + cross-encoder rerank — confirmed substitute, **not** a separate lift

> **Status:** **Confirmed with caveat** (n=300, 6 arms, BGE-small embedder
> + ms-marco-MiniLM cross-encoder). `retrieval="hybrid"` is a viable
> *alternative* path to closing the CUAD gap — it gives +5.3 points on
> the raw template query (81.3% → 86.7%) without any user-side
> preprocessing. But it does **not** stack with the strip + expand
> workflow: on the template-stripped query it adds only +0.3, and even
> hybrid + cross-encoder maxes out at 89.0% — **1.3 points below** the
> 90.3% CUAD plateau from [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md),
> at ~270× the latency.
>
> **TL;DR:** Dense semantic retrieval and template stripping are
> *substitute* mechanisms for boilerplate-induced lexical mismatch, not
> complements. Pick one. The cheaper one (strip + expand on default
> BM25) wins on both numbers and latency.

## Question

The four-corner rule from
[SUB_IDF_AUTO_DROP_NULL](SUB_IDF_AUTO_DROP_NULL.md) showed that
query-side IDF manipulation needs semantic awareness in the signal
source. The natural follow-up is whether **a different mechanism
entirely — dense semantic retrieval, with or without a cross-encoder
on top — can close more of the CUAD gap than the strip + expand
plateau we already shipped at 90.3%.**

Hybrid retrieval works by reading the *content* of each chunk, not
counting tokens, so it doesn't suffer from the boilerplate-dilution
failure mode that hurts raw-template BM25. The prediction was that
hybrid would help **the raw query** (where dilution is biggest) and
plateau on the stripped query (where there's nothing left to fix).
The cross-encoder layer on top is the most expensive available rerank
in the runtime — if it adds anything beyond what hybrid does alone,
that's evidence the residual gap is paraphrase-driven (gold span uses
terms the query doesn't).

## Setup

Harness:
[`crates/examples/examples/cuad_hybrid_rerank.rs`](../../crates/examples/examples/cuad_hybrid_rerank.rs).

Two axes, 3 × 2 = **6 arms** so each retrieval tier can be read at
both query preparations:

| axis | values |
| ---- | ------ |
| retrieval | `lexical` (BM25, default) · `hybrid` (BM25-prune + BGE-small dense rerank) · `hybrid + CE` (+ ms-marco-MiniLM-L-6-v2 cross-encoder) |
| query | raw 24-word template · template-stripped |

Same configuration as the other CUAD findings — n=300 from
`cuad_sample.json`, BM25 candidate pool depth = 40, rerank pool = 40,
budget = 2000 tokens, RawTopK strategy, set-based `span_recall` against
the gold answer span. Median + p95 per-query latency measured per arm.

## Results

| arm | n | ≥0.8 retention | mean recall | p50 ms | p95 ms |
| --- | --:| --------------:| -----------:| ------:| ------:|
| A1: lexical / raw           | 300 | 81.3% | 0.905 | **2.9**  | 6.2 |
| A2: lexical / stripped      | 300 | 87.7% | 0.933 | **2.5**  | 3.9 |
| **B1: hybrid / raw**        | 300 | **86.7%** | 0.926 | 10.7   | 1543.4 |
| B2: hybrid / stripped       | 300 | 88.0% | 0.936 | 8.3    | 1559.8 |
| C1: hybrid+CE / raw         | 300 | 84.0% | 0.916 | 749.4  | 2710.0 |
| **C2: hybrid+CE / stripped** | 300 | **89.0%** | 0.945 | 682.7  | 1740.4 |

(p95 spikes in the hybrid arms are first-query warmup on the BGE
embedder. Steady-state latency is closer to p50.)

Deltas relative to the lexical baseline at the same query
preparation:

| arm vs same-prep lexical | Δ on raw | Δ on stripped |
| ------------------------ | --------:| -------------:|
| hybrid (BGE)             | **+5.3** | **+0.3**      |
| hybrid + CE              | +2.7     | +1.3          |

Best cell across the probe: **89.0%** (hybrid+CE / stripped).
Prior CUAD plateau from [CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md):
**90.3%** (BM25 + strip + clause expand).

## What the results tell us

### Dense retrieval substitutes for template stripping; it doesn't stack

The most informative numbers in the table are the deltas. Hybrid lifts
the **raw** query by +5.3 — almost identical to the +6.4 that template
stripping does on its own. But hybrid on the **stripped** query adds
only +0.3.

The mechanism is the same. Both fix the same problem (boilerplate
crowds out the discriminator) by different means:

- **Template stripping** removes the boilerplate, so BM25 scores the
  discriminator cleanly.
- **Dense retrieval** doesn't count tokens at all — it scores the
  *content match* between the query (as an embedding) and each chunk
  (as an embedding), so the boilerplate ratio doesn't matter.

Once one mechanism has neutralized the problem, the other has nothing
left to do. They're **substitutes**, not complements.

### Cross-encoder is dominated by plain hybrid on the raw query

C1 (hybrid + CE on raw) is **2.7 points worse** than B1 (hybrid on
raw) — 84.0% vs 86.7% — at **70× the latency** (749ms vs 10.7ms p50).

The reason: CE reranks the **candidate pool** that BM25 + dense produced.
If BM25's top-40 missed the gold chunk because of template dilution,
the dense rerank inside hybrid can recover it by re-scoring the full
pool with semantic similarity. But once the result is handed to CE,
CE only sees the top results from BM25's pool — and on this workload
that's a narrower window than what the dense stage looks at.

When the raw query already has the dilution problem fixed (B2 → C2,
stripped path), CE does add a small amount: +1.0 (88.0 → 89.0). It
buys you a single point of retention for **65× more latency** than
plain hybrid and **270× more latency** than the lexical baseline.

### Strip + expand on default BM25 still wins on both axes

The best cell on this probe (89.0%, hybrid+CE / stripped) is **1.3
points below** the previously-measured 90.3% from
[CUAD_CLAUSE_EXPANSION](CUAD_CLAUSE_EXPANSION.md) — and that result
ran at the same ~2.5ms lexical latency.

So the practical ranking on CUAD is:

| approach | retention | p50 latency |
| -------- | ---------:| -----------:|
| BM25 + strip + expand (clause dict) | **90.3%** | ~2.5 ms |
| hybrid+CE / stripped                | 89.0%     | 683 ms  |
| hybrid / stripped                   | 88.0%     | 8.3 ms  |
| BM25 + strip (no expand)            | 87.7%     | 2.5 ms  |
| hybrid+CE / raw                     | 84.0%     | 749 ms  |
| BM25 / raw (baseline)               | 81.3%     | 2.9 ms  |

Strip + expand is **Pareto-optimal** on this workload: highest
retention AND lowest latency.

### When hybrid is still the right call

The "+5.3 on the raw template" result is genuinely useful in a narrow
case: **a user who doesn't want to write a template stripper or
maintain a workload dictionary**. Hybrid recovers most of what those
preprocessors would give you, automatically, at the cost of an embedder
model + ~10ms per query. If hybrid is already on for paraphrase
recovery
([GLOBAL_DENSE](GLOBAL_DENSE.md), [LOCAL_RERANK](LOCAL_RERANK.md)),
turning it on for templated workloads costs nothing additional and
makes the workflow simpler.

But on **answer-span retention specifically**, the dict-based
preprocessor is both faster and higher-quality.

## Honest limits

- **Single embedder (BGE-small-en-v1.5).** A larger embedder, a
  contract-domain-fine-tuned embedder, or a hybrid with re-ranking
  weights tuned for legal text could perform differently. None of those
  are claimed; the BGE-small result is what's measured.
- **Single cross-encoder (ms-marco-MiniLM-L-6-v2).** This is a
  general-purpose CE; a legal-domain CE could close the residual gap
  to the strip+expand plateau. Not tested.
- **n=300 from `cuad_sample.json`.** Same sample as the other CUAD
  findings, no bootstrap CIs. The ranking (lexical + strip + expand
  Pareto-dominates) is robust; the precise margins (1.3, 2.7) could
  drift on a larger sample.
- **Latency measured on the host machine** (Apple M5 in the perf
  numbers used elsewhere). Cross-encoder latency on different hardware
  will vary; the *order of magnitude* (sub-10ms lexical vs hundreds of
  ms with CE) is the durable observation.
- **No downstream Tier 3 answer-quality eval.** Whether the 1.3-point
  retention difference between hybrid+CE and strip+expand translates
  to a measurable F1/EM lift on gpt-4o-mini is unmeasured.
- **The raw-query hybrid lift is workload-specific.** On a non-
  templated workload there's no dilution to compensate for, so the
  +5.3 result would not generalize. This finding is specifically about
  *templated* workloads, where the hybrid lift on the raw query
  substitutes for what stripping would have done.

## Reproduce

```bash
REDHOP_MODELS_DIR=/path/to/models \
  cargo run -p redhop-examples --example cuad_hybrid_rerank --features onnx --release
```

Requires the BGE-small ONNX model and the ms-marco-MiniLM-L-6-v2 ONNX
cross-encoder under `$REDHOP_MODELS_DIR/bge-small-en-v1.5/onnx/` and
`$REDHOP_MODELS_DIR/ms-marco-MiniLM-L-6-v2/onnx/` respectively. Runs
end-to-end in a few minutes (CE arms dominate).

## What this changes

No runtime change, no API change. This is workload guidance:

- **For templated CUAD-shape workloads with a known taxonomy
  (legal QA, support triage):** `BM25 + analyze_query_set →
  drop_template_terms + expand_query_terms + evaluate` remains the
  recommended path. Lowest latency, highest retention.
- **For templated workloads where the user can't / won't maintain a
  dict and a stripper:** `retrieval="hybrid"` is a one-knob fallback
  that recovers most of the lift automatically, at ~3-5× latency. Not
  as good as the full workflow but better than raw BM25.
- **For workloads with genuine semantic mismatch (paraphrase, synonym):**
  `retrieval="hybrid"` is the documented recommendation
  ([GLOBAL_DENSE](GLOBAL_DENSE.md), [LOCAL_RERANK](LOCAL_RERANK.md));
  CUAD is *not* such a workload, and this finding confirms it.
- **For pushing absolute retention past 90.3% on CUAD:** none of the
  mechanisms here suffice. A contract-domain fine-tuned embedder or
  cross-encoder remains untested.

The deeper takeaway, worth carrying into future probes: **mechanisms
that solve the same underlying problem don't stack.** Template
stripping and dense retrieval both fix boilerplate-induced lexical
mismatch; combining them gives diminishing returns. Mechanisms that
solve **different** problems (strip + expand) **do** stack
productively, because the discriminator coverage is orthogonal to the
boilerplate suppression. Stacking is mechanism-orthogonality
dependent.
