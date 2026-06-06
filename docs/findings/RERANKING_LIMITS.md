# Cross-Encoder Escalation — the Reranking-Limits Finding

**Hypothesis (from the action-path experiment).** Dense retrieval fails
by returning a semantically tight cluster that misses the orthogonal
second hop. A cross-encoder re-scoring a *wider* net should be the
action whose geometry *matches* that failure — it can pull a
dissimilar-but-relevant chunk up into the final top-k.

**Result: the hypothesis is FALSIFIED.** And the falsification is the
deepest systems result in the project.

```bash
cargo run -p redhop-examples --example ce_escalation_economics --features onnx --release
```

## Numbers (60 HotpotQA items, dense BGE, wide net = top-20, k_final = 4)

| strategy | recall@4 | CE calls |
| -------- | -------- | -------- |
| static dense (no CE) | **0.732** | 0 |
| uniform CE (rerank 20 → 4) | 0.704 | 60 |
| selective CE (controller gate) | 0.704 | 60 |
| oracle (CE only when it helps) | **0.783** | **7** |

- **Uniform cross-encoder reranking made recall WORSE: −0.029.**
- CE was useful on 7/60 queries (12%), **harmful on 10/60 (17%)** — it
  hurt more queries than it helped.
- CE latency: 151 ms/query over 20 candidates (real, expensive).

## Why — the reranking-limits insight

A cross-encoder scores **query↔passage relevance**. On multi-hop
HotpotQA the missing evidence is the **second hop** — a chunk that is
relevant to the *bridge entity*, not to the *original query*. By
construction it has **low query-passage relevance**. So the
cross-encoder does exactly the wrong thing: it **demotes** the
second-hop chunk that dense retrieval did manage to surface in the
wide net, pushing it *below* the top-k cutoff.

> **Multi-hop retrieval failure is not a reranking problem. The missing
> chunk is low-relevance-to-query *by definition*, so no
> query-passage reranker — lexical, semantic, or cross-encoder — can
> recover it. They are all relevance-to-query operations, and the
> second hop is precisely the thing that lacks relevance to the query.**

This generalizes the action-path finding to its limit: not only does
`ExpandTopK` (more similar neighbors) fail to reach the second hop —
*reranking of every kind fails*, and the strongest reranker (the
cross-encoder) fails *hardest*, because it's the most confident about
demoting low-query-relevance chunks. The correct action for multi-hop
is something else entirely (query decomposition / iterative retrieval),
which RedHop deliberately does **not** have and should not grow
speculatively.

## The selective-escalation premise is REAFFIRMED (by the oracle)

The oracle row is the important counterpoint: **+0.051 recall at only 7
CE calls** (vs uniform's −0.029 at 60). The structure is exactly what
selective escalation is for:

- CE helps a **12% minority** and harms a **17% minority**.
- Applying CE uniformly is therefore **strictly worse** than not using
  it (−0.029) — you eat the harm on 17% to get the help on 12%.
- The *only* way CE adds value is **selectively** — fire it on the ~12%
  it helps, skip it on the rest. The oracle proves the headroom exists
  (+0.051) at a fraction of the compute (7 vs 60 calls, ~8.5× cheaper).

So the experiment does not refute selective escalation — it *demands*
it. Uniform reranking is the strategy this data most clearly condemns.

## But the controller's current gate can't discriminate

The selective arm fired CE on **all 60 queries** (`p_easy < 0.5` held
for every query under dense retrieval), so selective collapsed to
uniform. The controller's diagnostic signals **do not separate
"CE will help" from "CE will hurt."** That is the real open problem this
experiment exposes:

> The gating signal for cross-encoder escalation must predict
> *will reranking help or hurt THIS query* — and the current
> regime diagnostics (built around lexical/semantic grounding,
> distractor ratio, dispersion) don't carry that signal under dense
> multi-hop retrieval.

## A new risk-geometry insight: wide-net reranking is not recall-safe

In every prior adaptive experiment, `harmful lift` was 0.000. Here CE
hurt recall on 17% of queries. The difference is the **action class**:

- **Reorder-only rerank** (the lexical reranker used earlier) permutes
  the existing top-k. The *set* is unchanged, so recall@k is unchanged —
  **recall-safe by construction.**
- **Wide-net rerank** (top-N → top-k, what a cross-encoder needs to add
  value) *changes which chunks are in the final k*. It can drop a gold
  chunk below the cutoff — **not recall-safe.**

So "EscalateReranker" is two different actions with different risk
geometry. The conservative controller's no-improvement gate catches a
bad escalation *post hoc* (actual_gain < 0 → stop escalating further),
but the recall damage to that query is already done. **Wide-net
reranking is a genuinely risky action and must be gated more carefully
than reorder-only reranking.**

## What this validates / refutes / opens

**Refuted:** "stronger reranker = aligned geometry = recovers dense's
missed recall." Cross-encoders fail multi-hop second-hop recovery and
make net recall worse when applied uniformly.

**Reaffirmed:** selective escalation is not optional — it's the *only*
way a cross-encoder adds value here (12% help / 17% harm ⇒ uniform is
strictly bad; oracle-selective is +0.051 at 8.5× lower cost).

**Opened:** the gating signal. Predicting CE-helps-vs-hurts is unsolved
by the current diagnostics. The cheapest next probe: do the HotpotQA
`type` (comparison vs bridge) / `level` labels predict CE benefit?
Comparison/single-hop questions are where the gold *is* query-relevant
and CE should help; bridge/multi-hop is where CE hurts. If the label
predicts it, the gate has a cheap, principled signal — and that's a
*measurement*, not an architecture addition.

## Honest limits

- **60-item sample, single run, no CI.** The −0.029 / +0.051 deltas are
  directional; the 12%-help / 17%-harm split is the robust qualitative
  finding.
- **ms-marco MiniLM-L-6 cross-encoder, single-logit.** Smoke-tested
  correct (Paris #1 for "capital of France", score +7.6 vs −11.1 for an
  irrelevant passage). The multi-hop demotion is a property of
  query-passage rerankers generally, not a model bug — consistent with
  known IR results that cross-encoders underperform on multi-hop.
- **HotpotQA is adversarially multi-hop.** On a single-hop or
  comparison-heavy workload, CE would likely help uniformly. The finding
  is workload-shaped: *the action must match the failure geometry, and
  the failure geometry is workload-specific.*

## The throughline

Three experiments, one law, sharpened each time:

1. Substrate in the *sensing* path → no economic effect (actions
   embedding-blind).
2. Substrate in the *retrieval action* path → raw recall up, but the
   controller's actions mis-match dense's failure mode → more waste.
3. Substrate in the *reranking action* path → the strongest reranker
   *hurts* multi-hop recall, because query-passage relevance is the
   wrong signal for second-hop recovery.

**Retriever, failure geometry, and corrective-action geometry must all
align. RedHop's value is recognizing when they don't — and the
conservative controller's job is to refuse to spend compute on
misaligned actions.** That refusal is exactly what a correctly-gated
selective escalator would do here (fire on the 12%, skip the 88%); the
open work is building the gate that can tell them apart.

## Update — 2026-06-06 — the kind-label gate is falsified (in both directions)

The original finding flagged the cheapest probe for the open gating
problem:

> The cheapest next probe: do the HotpotQA `type` (comparison vs bridge)
> / `level` labels predict CE benefit? Comparison/single-hop questions
> are where the gold *is* query-relevant and CE should help;
> bridge/multi-hop is where CE hurts.

We measured it on a **stratified 100 bridge + 100 comparison sample**
(vs the original 60 mixed), same dense BGE retrieval, same wide-net
top-20, same k_final=4, same ms-marco MiniLM-L-6 cross-encoder. Each
strategy's recall (averaged across queries in scope) and CE-call count:

| sample           | strategy         | recall@4 | CE calls | helped | hurt |
| ---------------- | ---------------- | -------: | -------: | -----: | ---: |
| all (n=200)      | static (no CE)   |   0.7643 |        0 |      — |    — |
| all (n=200)      | uniform CE       |   0.7549 |      200 |     20 |   24 |
| all (n=200)      | kind-gate (CE on comparison) | 0.7460 | 100 | — | — |
| all (n=200)      | oracle           |   0.8120 |       20 |      — |    — |
| bridge (n=100)   | static           |   0.6303 |        0 |      — |    — |
| bridge (n=100)   | uniform CE       |   0.6482 |      100 |     17 |   14 |
| bridge (n=100)   | oracle           |   0.7123 |       17 |      — |    — |
| comparison (n=100) | static         |   0.8983 |        0 |      — |    — |
| comparison (n=100) | uniform CE     |   0.8617 |      100 |      3 |   10 |
| comparison (n=100) | oracle         |   0.9117 |        3 |      — |    — |

### Three findings

**1. The proposed kind-gate (fire CE on `kind == comparison`) FAILS.**
It loses to uniform CE by Δrecall = −0.0089 and loses to no-CE static
by Δrecall = −0.0183. Not shippable in this direction.

**2. The original mechanistic story is DIRECTIONALLY INVERTED on this
sample.** Per-type helped-vs-hurt:

- **Bridge** (predicted CE-hurts): helped 17, hurt 14 — CE is net
  *positive* (+0.018), not negative.
- **Comparison** (predicted CE-helps): helped 3, hurt 10 — CE is net
  *negative* (−0.037), not positive.

The cleanest mechanistic re-read: comparison questions in HotpotQA
already have such strong surface-term overlap with their gold (both
named entities appear in the query) that BM25/dense crushes them
without help — there's no headroom for CE to recover, only ceiling to
lose. Bridge questions have actual headroom because dense alone
struggles, leaving room CE can sometimes fill. The "comparison gold IS
query-relevant" intuition was correct; the implication that "CE should
therefore help" missed that CE only helps when **there is headroom to
recover** — and zero headroom is exactly what easy comparison cases
have.

**3. The INVERSE kind-gate (fire CE on `kind == bridge`) WOULD recover
a small positive signal** — combining bridge CE (0.6482) with
comparison static (0.8983) gives 0.7733, beating uniform CE (+0.018)
and no-CE static (+0.009). But: even this inverse gate captures only
**~18% of the oracle headroom** (the oracle's +0.048 vs the inverse
gate's +0.009). The kind label, in either direction, leaves most of
the available CE win on the table.

### What this closes

- **The kind label is NOT a sufficient gate signal — full stop.**
  Neither direction is shippable. Stop testing kind-based variants.
- **The intuition "comparison ⇒ CE helps" is wrong.** The correct
  mechanism rule is "CE only helps where dense leaves headroom" —
  which doesn't map cleanly onto question-type labels.
- **The open problem is RICHER signal.** Margin, dense-score spread,
  query-pool entropy, and other per-query diagnostics RedHop already
  computes are the next probe — those track dense headroom directly,
  not as a proxy through question type. That's a measurement
  follow-up if/when CE-gate becomes a priority again; for now it is
  closed.

### Honest limits of this update

- Single 200-item run, no bootstrap CI. The directional inversion is
  large enough (CE helps bridge net +0.018, hurts comparison net
  −0.037) to be unambiguous against the original 60-sample finding's
  noise, but the per-strategy deltas at this n are ±0.03-level
  significance at best. A two-side stratified rerun at n≥400 would
  settle exact magnitudes — we did not invest that compute because the
  qualitative finding ("kind label is not the signal") is already
  decisive.
- Same models (BGE-small, ms-marco MiniLM-L-6) and same eval (recall@4)
  as the original finding. The inversion is on the same hardware and
  same code path, so it isn't a model swap artifact.
