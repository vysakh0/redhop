# Semantic Mismatch — where lexical retrieval fails and dense helps

> **Question (not "are embeddings better"):** *where* does lexical (BM25)
> retrieval fail, and *when* does semantic (dense) retrieval materially help?
> **Status:** Confirmed across controlled + natural data, Tier-1 *and* Tier-3.
> On natural HotpotQA, dense beats BM25 by +0.16 F1 on semantic-heavy questions
> and ~0 on lexical-friendly. The conditional-escalation *value* is proven; a
> good query-time *trigger* is the open problem (the obvious one is a null).
> **Setup:** 25 controlled items across 5 regimes. Each has a GOLD passage
> (semantically right, usually low lexical overlap with the query), a TRAP
> passage (high lexical overlap, wrong meaning — a BM25 attractor), and
> distractors. All 100 passages pooled into one corpus; per query we measure
> whether each retriever ranks gold at the top. BM25 (Tantivy) vs dense
> (BGE-small, **exact cosine** via `FlatVectorIndex` — no ANN) vs hybrid (RRF).
> **Headline:** lexical wins where lexis aligns and **fails completely** where it
> doesn't (R@1 0%, trap beats gold ~100%); dense recovers it (R@1 ~50–80%); and
> **naive RRF hybrid is *worse* than dense alone** in mismatch regimes.
> **Reproduce:** `cargo run -p redhop-examples --example semantic_mismatch --features onnx --release`
> (BGE model via `REDHOP_BGE_MODEL`). Raw: `reports/semantic_mismatch.txt`.

---

## Results (n=25, top-5; R@1 / R@3 / MRR / trap-beats-gold)

| regime | BM25 | dense (BGE) | hybrid (RRF) |
| ------ | ---- | ----------- | ------------ |
| **control** (high lexical overlap) | **100% / 100% / 1.00 / 0%** | 100% / 100% / 1.00 / 0% | 100% / 100% / 1.00 / 0% |
| paraphrase | 0% / 0% / 0.00 / 100% | **50% / 100% / 0.75 / 50%** | 0% / 83% / 0.38 / 100% |
| legal synonymy | 0% / 0% / 0.00 / 100% | **50% / 50% / 0.53 / 33%** | 0% / 33% / 0.21 / 100% |
| reformulation | 0% / 20% / 0.10 / 100% | **80% / 100% / 0.90 / 20%** | 20% / 100% / 0.53 / 80% |
| low overlap | 0% / 0% / 0.00 / 100% | **75% / 100% / 0.88 / 25%** | 0% / 50% / 0.33 / 100% |
| **ALL** | 16% / 20% / 0.18 / 84% | **68% / 88% / 0.79 / 28%** | 20% / 72% / 0.46 / 80% |

Latency (per query, mean): query embedding (BGE) **2.05 ms** — the "semantic
tax"; BM25 retrieve 0.085 ms; dense retrieve (exact cosine) 0.031 ms.

## What this maps

- **Where lexical works, semantics is unnecessary.** On the control set (the
  query terms appear in the gold passage) BM25 is perfect and dense/hybrid add
  nothing. Lexical-friendly, code-like, exact-keyword regimes belong to BM25.
- **Where lexis and meaning diverge, lexical fails completely — not partially.**
  In every semantic-mismatch regime BM25 scored **R@1 = 0%** and the lexical TRAP
  out-ranked the gold passage ~100% of the time. BM25 isn't "a bit worse" here;
  it confidently retrieves the wrong thing.
- **Dense recovers most of it.** BGE exact-cosine lifts R@1 to ~50–80% and R@3 to
  50–100% across paraphrase, synonymy, reformulation, and low-overlap regimes —
  precisely the cases lexical can't see.

## The surprise: naive hybrid is *not* a safe default

RRF hybrid was **worse than dense alone** in every mismatch regime (R@1 0–20%,
trap-beats-gold 80–100%). Reciprocal-rank fusion rewards passages that appear
high in *both* retrievers' lists. The TRAP is lexically *and* somewhat
semantically present, so it scores in both → boosted; the GOLD is only
semantically present → it appears in one list and gets out-voted. So when one
retriever is *systematically wrong* (BM25 in a mismatch regime), blind fusion
**inherits its wrongness** rather than averaging it out.

**Implication:** "just add hybrid" is not the answer. The right move is
**conditional escalation to dense** when a mismatch is suspected — not blind RRF.

## The conditional reading (fits the runtime philosophy)

This is the same shape as every other RedHop finding: **sophistication is
conditional, not universal.**

- **Default lexical (BM25).** It's free of an embedding tax (~2 ms/query),
  exact, and *dominant* where query and document share vocabulary.
- **Escalate to dense only under suspected semantic mismatch** — e.g. when BM25's
  top score is weak / query↔candidate lexical overlap is low. That's the regime
  where dense pays for its 2 ms tax with a 0% → ~70% R@1 swing.
- **Do not blindly hybridize.** Naive RRF can be worse than either pure mode in
  mismatch regimes.

A "lexical-first runtime with conditional semantic escalation" is the
evidence-backed direction — consistent with conditional intervention only when
justified.

## What is established vs not

**Established (controlled, free, deterministic):**
- BM25 fails hard and dense recovers in paraphrase / synonymy / reformulation /
  low-overlap regimes; both are perfect where lexis aligns.
- Naive RRF hybrid is harmful in mismatch regimes (a real, reproducible result).
- The semantic tax is ~2 ms/query (BGE-small, CPU) on top of sub-0.1 ms retrieval.

**Not established / honest caveats / nulls:**
- **This is the mechanism, not field rates.** The traps are *adversarially*
  designed to defeat BM25, so its 0% R@1 is a worst case, not a typical-corpus
  rate. The point is the *boundary*, not the magnitude.
- **No downstream answer quality yet.** Retention/ranking is a proxy; whether
  dense's retrieval win changes *answers* on natural data is **Tier-3**, pending.
- **Single small embedder (BGE-small), synthetic English set, n=25.** A natural
  paraphrase-heavy QA split (e.g. HotpotQA items binned by query↔gold lexical
  overlap) is the right next instrument.
- We did **not** tune a fancier fusion; a *gated* fusion (use dense's order when
  BM25 confidence is low) is an obvious follow-up the data motivates.

## Phase 2 — natural data + downstream answers (HotpotQA distractor)

Does the boundary hold off the synthetic set, and does it change *answers*? We
split 397 HotpotQA items into **lexical-friendly** (query↔gold overlap above the
median 0.857) vs **semantic-heavy** (below), retrieve the 2 gold paragraphs from
each item's 10-paragraph pool with each mode, and answer with gpt-4o-mini.
(`semantic_natural` example + `score_semantic_natural.py`.)

**Tier-1 — gold-paragraph recall@3:**

| subset | BM25 | dense | hybrid |
| ------ | ---- | ----- | ------ |
| lexical-friendly (n=192) | 0.79 | 0.82 | **0.85** |
| **semantic-heavy (n=205)** | **0.61** | **0.84** | 0.75 |
| ALL | 0.70 | 0.83 | 0.80 |

**Tier-3 — downstream answer quality (F1 / EM):**

| subset | BM25 | dense | hybrid |
| ------ | ---- | ----- | ------ |
| lexical-friendly | 0.57 / 0.46 | 0.60 / 0.48 | **0.63 / 0.51** |
| **semantic-heavy** | 0.38 / 0.29 | **0.54 / 0.44** | 0.45 / 0.35 |
| ALL | 0.47 / 0.37 | **0.57 / 0.46** | 0.54 / 0.43 |

Reading:
- **The boundary holds on real data.** Dense beats BM25 by **+0.23 recall** and
  **+0.16 F1 / +0.15 EM** on the semantic-heavy subset, and barely moves the
  lexical-friendly one (+0.03 F1). Less extreme than the synthetic probe (natural
  BM25 is 0.61, not 0%) — but the *direction and conditionality* replicate.
- **This time retrieval *does* translate downstream.** Unlike the framework
  comparison (where a retention lead washed out at answer time), here BM25
  genuinely *misses* the gold evidence on semantic-heavy items, so the model
  can't answer — and dense recovering the evidence recovers the answer.
- **Naive hybrid is fine on natural data.** Best on lexical (0.63 F1), middling on
  semantic (0.45) — its catastrophic failure was specific to the *adversarial*
  synthetic traps, not real questions. (Honest correction to the Phase-1 read:
  "hybrid is harmful" is true only in adversarial-mismatch regimes.)

## Honest negative — the simple escalation gate does not fire

The natural test of "lexical-first + conditional dense escalation" needs a
*query-time, gold-free* trigger. We tried the obvious one — escalate to dense when
BM25's top hit has low lexical overlap with the query (τ ∈ {0.10, 0.20, 0.30}):

| τ | recall | escalated |
| - | ------ | --------- |
| 0.10 / 0.20 / 0.30 | 0.70 / 0.70 / 0.70 | 0% / 1% / 3% |

It **escalates almost nothing and captures none** of dense's +0.13 ALL gain
(always-BM25 0.70 vs always-dense 0.83). On HotpotQA the query shares entity terms
with the top hit *even when retrieval is wrong*, so query↔top-hit overlap stays
high. **The conditional-escalation value is proven (+0.16 F1 where it matters);
the detection signal is the unsolved part.** A BM25 score-margin / spread signal,
or query-side cues, is the next thing to try — this overlap trigger is a null.

## Updated bottom line

Lexical-first is the right default (free, exact, dominant where lexis aligns), and
**escalating to dense on semantic-heavy queries is worth a real +0.16 F1** — but a
*good escalation trigger remains open*, and blind hybrid is only safe off the
adversarial regime. Conditional, evidence-bounded — not "embeddings everywhere."
