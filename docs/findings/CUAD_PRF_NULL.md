# PRF on top of template-stripping — NULL RESULT (and the failure mode is worth remembering)

> **Status:** **Null result, falsified.** Pseudo-relevance feedback (PRF)
> applied on top of the [CUAD template-stripping](CUAD_RECALL_GAP.md) fix
> does not lift ≥0.8 retention past the stripped baseline. Across every
> parameter cell we tested, PRF was at-or-below baseline; at default
> parameters it lost 3.7 points.
>
> **The mechanism is sharp** and predicts where unweighted PRF will fail
> on other workloads. That's why this null result is documented rather
> than discarded — it's a workload-shape rule for future investigations.
>
> **This is the second independent PRF falsification on RedHop.** The
> first was an RM3-style PRF on the semantic-mismatch HotpotQA workload
> (see [SEMANTIC_ZERO_DEP](SEMANTIC_ZERO_DEP.md)), where it failed
> because first-pass precision was low and the feedback was built on
> distractors. This one fails for a *different* reason — first-pass
> precision is high, but the most frequent terms in the top chunks are
> corpus-pervasive boilerplate. **Two different failure modes, same
> conclusion: don't reach for unweighted PRF without checking which
> failure mode you're walking into.**
>
> **TL;DR:** On corpora dominated by domain boilerplate (legal,
> medical, regulatory, support tickets), unweighted PRF picks
> high-frequency *corpus-boilerplate* terms as expansion candidates and
> re-injects them into the query. That re-introduces the same dilution
> [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) just removed — from the document
> side instead of the query side. The template-strip win came from
> *subtraction*; PRF is *addition* of the wrong shape.

## Question

[CUAD_RECALL_GAP.md](CUAD_RECALL_GAP.md) showed that stripping the fixed
24-word template from each CUAD question lifts ≥0.8 evidence retention
from **82% → 88%** (n=300, BM25, budget 2000 tok), overtaking
LlamaIndex's 86%. The natural follow-up question: **is there a generic
query-side technique that pushes higher than 88%?**

PRF — pseudo-relevance feedback — is the classic candidate. The
mechanism is general (not CUAD-specific), well-studied, and the
expected lift on diluted queries is typically +2 to +4 points in the IR
literature. Does it work here?

## Setup

The probe harness is at
[`crates/examples/examples/cuad_prf.rs`](../../crates/examples/examples/cuad_prf.rs).
Same configuration as the other CUAD harnesses and `bench/compare.py`,
so the numbers compose cleanly with [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md):

- n = 300 (the first 300 questions from `cuad_sample.json`)
- BM25 lexical retrieval, no embeddings
- budget = 2000 tokens, candidate_k = 40
- strategy = `RawTopK`, default chunker
- set-based `span_recall` (matches `bench/compare.py`)

PRF mechanism (classic Rocchio-style, BM25-friendly):

1. **First pass:** retrieve a small context (default 500 tok) using the
   stripped query → the "expansion pool" of top chunks.
2. **Term mining:** tokenize the pool, drop stop-words (small English
   list plus residual CUAD-template safety net) and original-query
   terms, count term frequency, take the top-N most frequent content
   terms.
3. **Second pass:** augment the stripped query with the chosen terms,
   retrieve the full 2000-tok context, measure recall against gold.

Three arms so the comparison is apples-to-apples with the other CUAD
findings:

- **arm A:** raw 24-word template          → reproduces the 81.3% baseline
- **arm B:** template stripped             → reproduces the 87.7% baseline (the CUAD_RECALL_GAP win)
- **arm C:** template stripped + PRF       → the experiment

Two parameters were swept: **N_terms** (how many expansion terms get
appended) and **pool_budget** (the first-pass token budget).

## Results

### Sweep over N_terms (pool_budget fixed at 500)

| N_terms | arm C ≥0.8 retention | ΔC − B |
| ------- | --------------------:| ------:|
| 2       | 87.3%                | −0.4   |
| 4       | 85.7%                | −2.0   |
| 8 (default) | 84.0%            | **−3.7** |
| 16      | 81.0%                | −6.7   |

**Monotonic.** Every additional expansion term costs retention. PRF
never crossed B (87.7%) at any N.

### Sweep over pool_budget (N_terms fixed at 4)

| pool_budget | arm C ≥0.8 retention | ΔC − B |
| ----------- | --------------------:| ------:|
| 200         | 87.3%                | −0.4   |
| 300         | 86.0%                | −1.7   |
| 500         | 85.7%                | −2.0   |
| 800         | 86.7%                | −1.0   |

**Pool size barely matters** within the range we tested; pool=200 ties
the smallest-N result, suggesting the failure mechanism is dominated by
*what* gets picked from the pool, not *how much* pool is shown.

**Best PRF cell across the grid: 87.3%** (N=2 with either pool=500 or
pool=200), which still lost 0.4 points to the stripped-only baseline.
The headline default-parameter result was −3.7 points.

## What PRF was actually picking

The expansion terms for the first sample query in the dataset
(`"Document Name" + "The name of the contract"`):

```
["distributor", "company", "other", "agreement",
 "party", "names", "right", "shall"]
```

That's the answer. The most frequent content terms in any first-pass
top-chunk set on a CUAD contract are **legal-contract boilerplate**:
`agreement`, `party`, `shall`, `company`, `distributor`, `right`. These
are everywhere in the corpus because they're everywhere in legal
contracts in general — high term frequency, low corpus-level
discrimination. Adding them back to the query is precisely
re-introducing the kind of dilution that
[CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) just removed, just sourced from
the document side instead of the query template.

## The failure mechanism, sharp

PRF's theoretical justification is: "the top chunks from the first
pass contain query-discriminative terms that aren't in the query;
adding them will improve recall on the second pass." This relies on a
hidden assumption — that the **most frequent terms in the top chunks**
are query-discriminative. On a generic web corpus or open-domain QA
that assumption mostly holds, because high-frequency terms in
topically-clustered chunks are topical.

On a domain-boilerplate-heavy corpus the assumption breaks:

1. The corpus has a large vocabulary of high-TF, low-IDF terms shared
   across all documents (legal boilerplate, in CUAD's case).
2. Those terms appear in every top-chunk set, regardless of which
   specific clause the query asked about.
3. Unweighted "top-N by frequency" surfaces them as expansion
   candidates.
4. Re-injecting them dilutes the signal we just sharpened by
   template-stripping.

The win from template-stripping was **subtraction** of high-TF
low-IDF noise; PRF is **addition** of the same flavor of noise from a
different source. We don't get to add it back and still win.

## Predicted scope (this is the rule worth remembering)

**Unweighted PRF will fail on any workload where the document corpus is
dominated by domain boilerplate.** Concrete examples:

- **Legal:** contracts, court filings, regulatory documents (every
  document repeats `party`, `shall`, `pursuant`, `herein`, `whereas`).
- **Medical:** clinical notes, drug labels (every document repeats
  `patient`, `dose`, `mg`, `daily`, `administered`).
- **Regulatory / compliance:** SOX, GDPR, HIPAA docs (every document
  repeats `subject`, `data`, `controller`, `processor`, `consent`).
- **Customer support / tickets:** every ticket repeats `account`,
  `error`, `please`, `reset`, `support`, `thanks`.
- **Internal runbooks:** every runbook repeats `service`, `restart`,
  `oncall`, `escalate`, `verify`.

For those workloads, **don't reach for unweighted PRF as a recall
optimization.** It will land at-or-below baseline. Either fix the
mechanism first (see "what would actually work" below) or pick a
different lever (template stripping at the query boundary, hybrid
retrieval, sentence-boundary chunking).

**Where unweighted PRF probably still helps:** diverse natural-language
corpora where high-TF terms ARE topical (open-domain QA on Wikipedia,
generic search). HotpotQA-shaped workloads. Not measured here, but the
mechanism prediction is consistent with the IR literature.

## What would actually work (and what we deliberately did not try)

The fix to PRF's failure on boilerplate-heavy corpora is conceptually
simple: **weight expansion candidates by corpus IDF, not just by pool
TF.** Terms that are frequent in the pool *and* rare in the corpus are
the discriminative ones. Concretely: `score(t) = tf_in_pool(t) /
log(N / df_corpus(t))` or any standard tf-idf flavor.

We didn't implement that here for two reasons:

1. It requires direct access to RedHop's BM25 internals (the term
   frequency table and corpus document count), which is not on the
   public surface. Wiring that out is a bigger surgical job than this
   probe was worth at this stage.
2. **The warning sign is severe.** Look at the unweighted top terms
   from the sample query (`distributor, company, agreement, party,
   shall, names, right, other`) — every one is corpus-pervasive
   boilerplate. An IDF re-ranking would push them down, but would it
   push them down *enough* to find genuinely query-discriminative
   terms? On CUAD specifically, the discriminative terms are already
   in the stripped query (the clause name itself), so the headroom for
   PRF to add useful terms is small. The mechanism prediction is that
   even IDF-weighted PRF would be flat-to-marginal on this workload.

If a future investigator wants to revisit this, the next step is:
**implement IDF-weighted PRF and measure it on a workload where the
mechanism prediction is favorable (diverse natural-language queries,
non-boilerplate-heavy corpus).** HotpotQA is the obvious candidate.
But the IDF-weighted variant should NOT be claimed as a CUAD recall
optimization without measurement.

## Other candidates this null result rules out, by mechanism

If unweighted PRF fails by re-injecting corpus boilerplate, then a
broader class of "expand the query using document terms" techniques
will fail the same way on the same workloads. By mechanism:

- **Static synonym dictionaries from corpus statistics** (e.g.,
  word2vec-style synonyms trained on the corpus): same boilerplate
  surfaces.
- **Co-occurrence-based query expansion** without IDF weighting: same.
- **Pseudo-document expansion** (replace the query with the first-pass
  top chunk and re-retrieve): worst case of the same failure — replaces
  the discriminative query terms entirely with boilerplate-heavy chunk
  text.

Techniques that **don't share the mechanism** and so are not ruled out:

- **Static workload-specific keyword dictionaries** (hand-curated from
  domain knowledge, not learned from the corpus): CUAD's 40-clause
  taxonomy is a candidate. Workload-specific.
- **Dense retrieval / cross-encoder rerank:** semantic, not lexical;
  doesn't add terms to the query at all.
- **Sentence-boundary chunking:** document-side, not query-side.
- **IDF-weighted PRF:** mechanism in principle works; not measured;
  prediction is marginal-to-flat on CUAD specifically.

## Honest limits

- **One workload (CUAD), one corpus.** The mechanism prediction
  generalizes to other boilerplate-heavy corpora; the *exact magnitudes*
  do not. CUAD's specific numbers wouldn't be the same on, say, a
  medical-notes corpus.
- **One ranking criterion for expansion terms (raw TF).** IDF-weighted,
  RM3-style, or KL-divergence-based PRF was not tested. The mechanism
  argument above predicts IDF-weighted PRF would also be marginal on
  CUAD specifically, but that is a *prediction*, not a measurement.
- **One first-pass strategy (`RawTopK`).** We didn't test PRF on top of
  a `ReasoningPreserving` first pass; the first-pass strategy choice
  could change which chunks dominate the pool. Not measured.
- **n=300, no bootstrap CIs.** Same caveat as the other CUAD findings.
  Magnitudes within ±1 point should be read as within sample noise; the
  trend (monotonic degradation with N, no parameter combo positive) is
  what's solid.
- **No downstream answer eval.** Whether the small 0.4-point loss at
  N=2 translates to a measurable F1/EM regression at Tier 3 is a
  separate question we didn't ask.

## Reproduce

```bash
# Default parameters (pool=500, N=8) — the headline -3.7 result:
cargo run -p redhop-examples --example cuad_prf --release

# Sweep N at fixed pool:
for N in 2 4 8 16; do
    REDHOP_PRF_N=$N cargo run -p redhop-examples --example cuad_prf --release \
      | grep "≥0.8 retention"
done

# Sweep pool at fixed N:
for P in 200 300 500 800; do
    REDHOP_PRF_POOL=$P REDHOP_PRF_N=4 cargo run -p redhop-examples --example cuad_prf --release \
      | grep "≥0.8 retention"
done
```

## What this changes

Concretely — nothing in the runtime, nothing in the public API. RedHop
ships no PRF helper before or after this finding.

Operationally — for anyone (human or AI agent) investigating "what
else can push past 88% on CUAD" or facing the broader question of
"what query-side techniques will lift recall on a boilerplate-heavy
corpus":

1. **Cross unweighted PRF off the list** for boilerplate-heavy corpora.
   It will not help; the failure mechanism is well-understood.
2. **The next plausible candidates** (in expected-payoff order): static
   workload-specific keyword expansion (CUAD clause-name taxonomy),
   sentence-boundary chunking (the LlamaIndex hypothesis), hybrid
   retrieval with cross-encoder rerank. Each has a different
   measurement cost and a different probability of mattering.
3. **The general principle** behind the win that did work: when
   high-frequency low-IDF terms dilute BM25 on a templated workload,
   the lever is **subtraction at the query boundary** (template-strip),
   not **addition at the query boundary** (PRF). This generalizes.
