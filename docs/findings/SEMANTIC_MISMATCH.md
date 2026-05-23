# Semantic Mismatch — where lexical retrieval fails and dense helps

> **Question (not "are embeddings better"):** *where* does lexical (BM25)
> retrieval fail, and *when* does semantic (dense) retrieval materially help?
> **Status:** Confirmed across controlled + natural data, Tier-1 *and* Tier-3.
> Dense beats BM25 by +0.16 F1 on semantic-heavy questions and ~0 on
> lexical-friendly — the *value* of escalation is real and observable. But a
> cheap, deterministic escalation *trigger* is a **null**: neither query↔hit
> lexical overlap nor BM25 score margin/entropy separates the regimes, and
> conditional escalation gives no selective gain over linear interpolation.
> **The resolution: don't trigger — bound it.** BM25-prune + *local* dense rerank
> over the candidate pool recovers ~96% of global dense's gains at bounded cost
> and no vector infra (Phase 4), conditional on BM25's candidate recall@K.
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

It escalates almost nothing on HotpotQA (the query shares entity terms with the
top hit even when retrieval is wrong, so overlap stays high) — a null.

## Phase 3 — can the escalation be triggered at all? (score-distribution signals)

The hypothesis: maybe semantic-heavy failures show up as *BM25 uncertainty*
(ambiguous / flat score distribution) rather than low query↔hit overlap. We
tested two deterministic, gold-free signals from the BM25 result: **margin**
`(top1−top2)/top1` (1 = confident, 0 = ambiguous) and **normalized entropy** of
the top-k scores (high = flat).

**The crux — do the signals separate the subsets? Barely / no:**

| signal | lexical-friendly | semantic-heavy |
| ------ | ---------------- | -------------- |
| margin (1 = confident) | 0.282 | 0.248 |
| entropy (high = flat)  | 0.955 | 0.968 |

Margin is *slightly* lower on semantic-heavy; entropy is essentially identical
(uniformly flat on both). Neither cleanly distinguishes the regime.

**Trigger sweep (escalate to dense when BM25 looks uncertain):**

| margin < τ | escalate | sem. capture | lex. false | recall |
| ---------- | -------- | ------------ | ---------- | ------ |
| 0.10 | 22% | 25% | 20% | 0.74 |
| 0.20 | 44% | 46% | 42% | 0.77 |
| 0.30 | 61% | 65% | 56% | 0.80 |
| 0.50 | 88% | 92% | 84% | 0.81 |

At every threshold **semantic-capture ≈ lexical-false** — the trigger escalates
both subsets at nearly the same rate, i.e. barely better than random. (Entropy is
worse: it fires ~100% always, collapsing to always-dense.)

**Downstream confirms it — the conditional policy just interpolates linearly:**

| policy | F1 | EM | escalated |
| ------ | -- | -- | --------- |
| always BM25 | 0.47 | 0.37 | 0% |
| always dense | 0.57 | 0.46 | 100% |
| margin < 0.20 | 0.52 | 0.41 | 44% |
| margin < 0.30 | 0.55 | 0.44 | 61% |
| margin < 0.50 | 0.56 | 0.46 | 88% |

F1 rises in lock-step with escalation rate — **no selective gain**. To get
near-dense F1 you escalate ~88% (≈ always-dense). There is no "capture most of
dense's wins at near-BM25 cost."

## Phase 4 — local semantic refinement over lexical topology (the resolution)

> Written up as a standalone finding: **[LOCAL_RERANK.md](LOCAL_RERANK.md)**. The
> summary below is the same result in the context of this study.

Phase 3 left an impasse: dense helps, but no cheap *trigger* tells us when to
escalate. Phase 4 dissolves it with a different shape — **don't decide whether to
escalate; bound the semantic work instead.** BM25 prunes a large corpus to a
candidate pool; dense reranks *only within that pool* (local), never embedding the
whole corpus (global). Tested on a **global** HotpotQA corpus (3,957 deduped
paragraphs, so BM25 top-50 is a real prune), arms = bm25 / global dense / local
rerank / hybrid.

**Recall@3 by subset and arm:**

| subset | bm25 | global dense | local rerank | hybrid |
| ------ | ---- | ------------ | ------------ | ------ |
| lexical-friendly | 0.68 | 0.80 | **0.81** | 0.77 |
| semantic-heavy | 0.50 | 0.80 | 0.79 | 0.67 |
| ALL | 0.59 | 0.80 | **0.80** | 0.72 |

**The crux — BM25 recall@50 (the ceiling local rerank can reach):** lexical
0.987, **semantic 0.937**, ALL 0.961. Gold is almost always *in* BM25's top-50,
even on semantic-heavy queries — BM25 just ranks it low. So dense only needs to
*reorder the pool*, not search the corpus.

**Semantic recovery:** of the 165 queries where global dense beat BM25@3, **local
rerank recovered 158 (96%)**; only 7 (4%) genuinely needed global dense (gold fell
outside the top-50).

**Tier-3 — it holds at the answer level too (gpt-4o-mini, F1 / EM):**

| subset | bm25 | global dense | local rerank | hybrid |
| ------ | ---- | ------------ | ------------ | ------ |
| lexical | 0.47 / 0.39 | 0.57 / 0.47 | **0.59 / 0.47** | 0.56 / 0.47 |
| semantic | 0.27 / 0.21 | 0.52 / 0.42 | 0.50 / 0.40 | 0.38 / 0.30 |
| ALL | 0.37 / 0.29 | **0.54 / 0.44** | **0.54 / 0.43** | 0.47 / 0.38 |

Local rerank's recall win **translates fully to answers**: it equals global dense
at ALL (0.54 = 0.54 F1), is within 0.02 on semantic-heavy (recovering the +0.23 F1
over BM25), marginally better on lexical, and beats hybrid throughout. So the
bounded-dense architecture is sound end-to-end, not just at retrieval.

**Compute:** dense work is bounded to 50 candidates (cosine **0.019 ms**) vs
global search 0.505 ms over 3,957 — and **no global ANN / vector index** is
required; BM25 (which scales trivially) does the corpus-wide pruning.

Reading:
- **Local rerank matches global dense** (0.80 = 0.80) and **beats naive hybrid**
  (0.72) — no fusion poisoning, because dense *reorders* rather than *votes
  against* BM25.
- **It resolves the no-trigger problem.** You don't detect when to escalate;
  local rerank is bounded and cheap enough to run **always** over the BM25
  candidates. "Lexical topology first, semantic refinement second" —
  unconditionally.
- **The honest cap is BM25 *candidate* recall.** It works here because BM25
  recall@50 is 0.94 even on semantic-heavy (there's *partial* lexical overlap to
  surface gold at depth 50). On **pure-synonym** mismatch (the controlled probe,
  where gold shared ~0 query terms), BM25 candidate recall would collapse and
  local rerank would degrade toward BM25 — only global dense could help. The 4%
  "global-dense-only" residual is exactly that tail.
- **Compute is bounded, not free.** The embedder cost is real — it's K candidate
  embeddings (at query time) or the corpus (precomputed once). Local rerank's win
  is *bounding dense to K and dropping the global ANN*, not eliminating the model.

This is a **retrieval-runtime hypothesis, now supported** (not a new algorithm):
on workloads where BM25 has decent first-stage *candidate* recall, BM25-prune +
local dense rerank recovers ~all of global dense's quality at bounded cost and no
vector infra.

## Bottom line — answering the systems question honestly

> *Can retrieval sophistication itself become conditional and observable?*

- **The value is real and observable:** dense materially helps semantic-heavy
  queries (+0.23 recall, +0.16 F1) and is ~neutral elsewhere.
- **But a cheap, deterministic *trigger* does not exist** among the obvious
  candidates. Two independent signal families — query↔hit lexical overlap *and*
  BM25 score margin/entropy — both **fail to separate** the regimes. Conditional
  escalation gives no selective advantage over linear interpolation: cost ≈
  benefit. So *conditional escalation is not operationally viable with a simple
  signal* on this data. Plainly: a null.
- **And it may not need to be.** Dense (weakly) *dominates* BM25 in **both**
  subsets here (lexical 0.60 ≥ 0.57 F1; semantic 0.54 ≥ 0.38) at only a ~2 ms
  absolute embedding tax — so the honest operational choice is **binary**: run
  dense (or hybrid — best on lexical, fine on semantic) when you can afford a
  local embedder; stay on BM25 when you can't, accepting the semantic-heavy loss.
  A trigger would only matter if dense were *wasteful* on lexical-friendly
  queries, and the data says it isn't.

**The resolution (Phase 4):** the trigger was the wrong frame. Instead of
deciding *when* to pay for global dense, **bound the dense work to BM25's
candidate pool and always rerank it locally** — recovering ~96% of global dense's
gains at bounded cost and no vector infra. So:

> **Lexical topology first → local semantic refinement second → runtime
> economics always visible.** No global ANN, no escalation trigger.

This is RedHop's retrieval-runtime direction, now evidence-backed — *conditional
on BM25 having decent first-stage candidate recall* (the honest cap; pure-synonym
workloads with ~zero lexical overlap still need global dense for the residual).

## Next (open, honest)

- **Wire BM25-prune + local rerank into the `Document` path** (behind the onnx
  feature) as the semantic-capable retrieval mode — the product payoff of Phase 4.
- ~~**Tier-3 on the local-rerank arm**~~ **(done — local rerank = global dense on
  answer F1, 0.54 = 0.54 ALL; within 0.02 on semantic-heavy).**
- A second dataset with a stronger semantic tail (MS MARCO / reformulated queries)
  to test where BM25 candidate recall@K finally breaks (the local-rerank cap).
