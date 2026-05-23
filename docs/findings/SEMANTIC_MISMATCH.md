# Semantic Mismatch — where lexical retrieval fails and dense helps

> **Question (not "are embeddings better"):** *where* does lexical (BM25)
> retrieval fail, and *when* does semantic (dense) retrieval materially help?
> **Status:** Tier-1 (controlled) confirmed — a clean, conditional boundary.
> Tier-3 (downstream answers on natural data) is the pending materiality layer.
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

## Next

1. **Natural-data Tier-1 + Tier-3:** split a real QA set into lexical-friendly vs
   semantic-heavy subsets by query↔gold lexical overlap; compare BM25 / dense /
   hybrid retrieval recall *and* downstream answer F1/EM. Measures real-world
   materiality + whether the boundary holds off the synthetic set.
2. **A confidence-gated escalation probe:** escalate to dense only when BM25's top
   score is weak; measure whether it captures dense's mismatch wins at near-BM25
   average cost.
