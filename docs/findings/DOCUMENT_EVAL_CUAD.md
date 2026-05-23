# Document Runtime on Real Contracts — CUAD Eval (Tier 1 + 2, no LLM)

> **Hypothesis:** on a long real document, the `Document` runtime
> (`from_text → context(query)`) cuts tokens hard while keeping the evidence
> needed to answer, fast, and `Auto` makes sensible decisions.
> **Status:** Confirmed (Tier 1) + robustness-characterized (Tier 2), n=50
> contracts / 644 answerable clause queries, free/local/deterministic.
> **Setup:** CUAD v1 (real commercial contracts, ~9.3k tokens each) with gold
> answer spans. Default `Document` (candidate_k=20, budget 2048, `Auto`).
> Metric: gold-span **word-recall** in the assembled context (robust to a long
> clause being split across chunks). No LLM — this is evidence retention, not
> downstream answer quality (that is Tier 3, pending).
> **Headline:** **−80% tokens** (9.3k → 1.9k) with the gold evidence retained at
> **≥0.8 word-recall on 88%** of queries (96% at ≥0.5), at **~1.7ms/query**.
> `Auto` pruned 94% of queries — contracts are large, so the dilution regime is
> the common case on real docs.
> **Reproduce:** `cargo run -p redhop-examples --example eval_cuad_documents --release`
> (`REDHOP_CUAD_PERTURB=dup|ocr` for Tier 2; `REDHOP_CUAD_PATH` for full CUADv1).
> Raw output in [reports/](../../reports/) `cuad_document_eval*.txt`.
> **Caveats:** lexical word-recall proxy (not answer quality); single
> domain/dataset; word-split spans counted by recall not exact match; memory not
> profiled (latency only). The win is conditional on the doc being large/diluted.

---

## Why this eval

Prior dilution evidence ([CONTEXT_DILUTION.md](CONTEXT_DILUTION.md)) was on
synthetic distractor injection over HotpotQA. This runs the **actual product
path** — `Document::from_text → context()` — on **real long documents**
(commercial contracts), and asks the operational question a user has: *if I hand
RedHop a 9k-token contract and a clause question, does it give the LLM a small
context that still contains the answer, and how fast?*

To separate *retrieval* loss from *pruning* loss, every query runs twice on the
same contract: a retrieval-only ceiling (top-20 candidates, no pruning) and the
default end-to-end path (top-20 + `Auto` prune to budget). The gap is what
pruning costs.

## Tier 1 — the product path (baseline, n=50 contracts / 644 queries)

| metric | value |
| ------ | ----- |
| avg full-contract tokens | 9,322 |
| avg assembled tokens | 1,909 |
| **end-to-end token reduction** | **−80%** |
| `Auto` decisions | 608 prune / 36 passthrough |
| retrieval ceiling — mean word-recall | 0.99 (≥0.5: 99%, ≥0.8: 98%) |
| **end-to-end — mean word-recall** | **0.93 (≥0.5: 96%, ≥0.8: 88%)** |
| latency — doc build (chunk+index) | p50 1.0ms / p95 4.5ms |
| latency — per-query `context()` | p50 1.7ms / p95 3.3ms |

Reading:
- **BM25 on legalese held up far better than feared** — the retrieval ceiling is
  0.99 mean recall; the verbose clause questions surface the right contract
  passage in the top-20 on ~98% of queries.
- **Pruning costs ~6 points of mean recall** (0.99 → 0.93; ≥0.8 retention
  98% → 88%). On ~1 in 8 queries, `Auto` pruning drops enough of a long clause to
  fall below 0.8 recall — the honest cost of an 80% token cut.
- **`Auto` intervened on 94% of queries**: real contracts are large, so the
  dilution regime — where pruning is measured to help — is the common case, not
  the exception. The 36 passthroughs are short contracts under the gate.
- **Sub-2ms per query**, ~1ms to chunk+index a whole contract — local-first,
  no vector infra.

## Tier 2 — robustness on messy corpora

Same contracts, perturbed before ingestion (deterministic):

| perturbation | reduction | `Auto` prune | ceiling ≥0.8 | end-to-end ≥0.8 | query p95 | crashes |
| ------------ | --------- | ------------ | ------------ | --------------- | --------- | ------- |
| none (baseline) | −80% | 608/644 | 98% | 88% | 3.3ms | 0 |
| **dup** (3× duplicated) | −93% | 637/644 | 90% | **70%** | 3.7ms | 0 |
| **ocr** (15% words split mid-word) | −81% | 619/644 | 98% | 85% | 3.4ms | 0 |

Reading:
- **No crashes under any perturbation** (after the query-sanitization fix below) —
  the runtime degrades gracefully, it does not fall over.
- **Duplication is the real stressor.** Triplicating the document drops
  end-to-end ≥0.8 retention from 88% → **70%**: the default path does *not*
  deduplicate, so duplicate chunks fill the candidate set and the token budget,
  crowding out unique evidence. (The retrieval ceiling also dips, 98%→90%,
  because duplicates displace the gold chunk from the top-20.) This is an honest
  limitation, not a bug — and it points at `RedundancyPruned`/dedup-awareness for
  known-duplicated corpora, a measured future option, **not** a default change
  here.
- **Moderate OCR fragmentation is tolerated.** Splitting 15% of long words
  mid-word barely moves the retrieval ceiling (98%) and only nudges end-to-end
  ≥0.8 retention 88%→85% — BM25 still matches on the unsplit majority of query
  terms. Heavier fragmentation would degrade more; this motivates semantic
  retrieval as a future option, not a present claim.

## Bug found and fixed by this eval

Natural-language clause questions (parentheses, smart quotes, punctuation)
**crashed the internal BM25 query parser** (`Syntax Error` from Tantivy's
`QueryParser`) — `doc.context("Highlight the parts (if any)… “requirements”…")`
errored instead of retrieving. Fixed by reducing the query to a clean bag of
word tokens before parsing (alphanumerics kept, everything else collapsed to
whitespace); ranking is unchanged. A real natural-language query must never
crash internal retrieval. (`crates/retrieval/src/bm25.rs`.)

## What is established vs not

**Established (free, local, deterministic):**
- The `Document` path delivers a large, real token reduction (−80%) on real
  long documents while retaining the gold evidence on the large majority of
  queries (≥0.8 on 88%), at ~1.7ms/query.
- `Auto` resolves to *prune* on ~94% of real contract queries (large/diluted
  inputs) — the dilution finding holds on genuine documents, not just synthetic.
- The runtime is crash-robust to duplicated and OCR-fragmented input.

**Not established / honest limits:**
- **Not downstream answer quality.** Word-recall is a retention proxy; whether
  the −80% context answers the clause *as well as* the full contract is **Tier 3**
  (needs an LLM; pending).
- Single dataset/domain (contracts, English, lexical). A different domain or a
  semantic-match-heavy query set could shift the retrieval ceiling.
- Memory not profiled (latency only); duplicated-corpus dedup is unaddressed by
  default.

## Next

1. **Tier 3 (LLM):** does the assembled context answer CUAD clauses as well as
   the full contract? The claim that closes the loop — spends credits.
2. Dedup-aware handling for known-duplicated corpora (measure `RedundancyPruned`
   on the `dup` regime before any default change).
