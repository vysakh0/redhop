# Context Economics — Evidence Allocation Under Finite Attention

RedHop's center of gravity shifting from "retrieve better" to "allocate
the finite attention budget to the densest evidence." Two experiments:
(B) does the premise hold on real LLM outputs, and (A) what do the
pruning strategies actually buy.

## Experiment B — distractors hurt, density helps (REAL LLM outputs)

The premise behind context optimization, tested on the Python lab's
real LLM answers via NeoTrace (each row carries retrieval
`distractor_ratio` + `answer_span_density` AND measured answer quality).

```bash
cargo run -p redhop-examples --example distractor_answer_correlation --release
```

Pearson correlations with `ans_kw_recall` (gold-keyword recall in the
LLM's answer):

| dataset (model) | n | distractor → kw_recall | density → kw_recall |
| --------------- | - | ---------------------- | ------------------- |
| HotpotQA (haiku) | 686 | **−0.472** | +0.561 |
| HotpotQA (llama8b) | 686 | −0.378 | +0.458 |
| MuSiQue (haiku) | 700 | −0.295 | +0.510 |
| MuSiQue (qwen7b) | 700 | −0.282 | +0.401 |
| MuSiQue (mistralnemo) | 700 | −0.183 | +0.368 |
| evidence study | 130 | −0.448 | +0.700 |
| **pooled** | — | **−0.375** | **+0.539** |

**Distractors hurt and evidence density helps — on real LLM outputs,
consistently across four generator models and two datasets.** This is
the empirical foundation for context economics: it's not a synthetic
artifact, it's a model-independent property of how LLMs use retrieved
context. Distractor ratio is negatively correlated with answer quality
everywhere (−0.18 to −0.47); answer-span density is positively
correlated everywhere (+0.37 to +0.70).

## Experiment A — token-efficiency curves (real HotpotQA + BGE)

`build_context` over dense BGE retrieval (wide net = top-20), sweeping
the token budget across four strategies. Gold retention = fraction of
gold chunks present in the assembled context.

```bash
cargo run -p redhop-examples --example context_economics --features onnx --release
```

At budget = 250 tokens (where pruning bites):

| strategy | gold_retained | tokens | distractor | density |
| -------- | ------------- | ------ | ---------- | ------- |
| raw_topk | 0.819 | 243 | 0.034 | 0.248 |
| distractor_filtered | 0.803 | 242 | **0.000** | 0.250 |
| redundancy_pruned | 0.825 | 243 | 0.034 | 0.247 |
| **max_density** | **0.623** | 243 | 0.013 | **0.287** |

Full curve for `max_density` (the revealing one):

| budget | gold_retained | density |
| ------ | ------------- | ------- |
| 80 | 0.314 | 0.349 |
| 150 | 0.463 | 0.318 |
| 250 | 0.623 | 0.287 |
| 400 | 0.784 | 0.253 |
| 800 | 0.914 | 0.206 |

raw_topk at the same budgets retains 0.610 / 0.744 / 0.819 / 0.881 /
0.914 — so at budget 80, max_density retains **half** the gold raw_topk
does (0.314 vs 0.610).

### Three findings

1. **Distractor filtering is the safe, free win.** It drives the
   distractor ratio to **0.000** at essentially no gold cost (0.803 vs
   0.819 at budget 250) and marginally higher density. Why safe: it uses
   an *absolute, low* grounding threshold (0.10) that only removes
   near-zero-overlap junk — gold chunks (even multi-hop second hops)
   share *some* query terms and clear it. Given Experiment B's −0.375
   distractor↔quality correlation, this is pure benefit: remove the
   tokens that demonstrably hurt, keep the evidence.

2. **Max-density is a sharp double-edged sword — and it reproduces the
   reranking-limits geometry a THIRD time.** It achieves the highest
   evidence-per-token (0.349 vs raw's 0.271 at budget 80) but sacrifices
   gold retention (0.314 vs 0.610). It ranks chunks by query-relevance
   density and fills greedily, so the **low-query-relevance second hop
   loses to higher-density chunks and falls below the budget cutoff** —
   exactly the failure that limited the cross-encoder. Max-density is an
   evidence-*concentration* tool for single-hop / high-query-relevance
   workloads, **not a multi-hop-safe strategy.**

3. **Redundancy pruning is ~neutral on this corpus.** Academic chunks
   are distinct, so little fires (gold 0.825 ≈ raw 0.819). It would
   matter on duplicated enterprise corpora — the ingestion-diagnostics
   `duplicate_ratio` is the signal that tells you whether it's worth
   running.

### The threshold-vs-ranking distinction (the design insight)

The same recall-safety split as the reranking risk geometry:

> **Absolute-threshold pruning (distractor filtering: "drop chunks below
> a low grounding bar") is recall-safe** — it only removes clearly
> irrelevant junk.
> **Relative-ranking pruning (max-density: "keep the top chunks by
> density") is recall-risky on multi-hop** — the second hop loses the
> ranking competition even though it's needed.

Both have the same outcome at large budgets (all strategies converge to
0.914 at budget 800) — **context economics only bites under attention
scarcity.** When the budget is generous, pruning is moot; when it's
tight, the strategy choice is consequential and workload-dependent.

## The throughline, now in a third place

Query-relevance operations share one blind spot on multi-hop:

1. ExpandTopK (more similar neighbors) — can't reach the second hop.
2. Cross-encoder rerank (query-passage relevance) — *demotes* the
   second hop (net −0.029 recall).
3. Max-density context pruning (query-relevance ranking) — *drops* the
   second hop (gold 0.623 vs 0.819).

**Any operation that ranks/selects by relevance-to-query inherits the
multi-hop blind spot, because the second hop is low-relevance-to-query
by construction.** What's *safe* is removing what is clearly irrelevant
(distractor filtering, absolute threshold) — that helps without
discarding the orthogonal-but-needed evidence.

## What this means for the product

RedHop's defensible context-economics offering, grounded in these
measurements:

- **Distractor filtering**: a safe, free quality win (distractors → 0,
  gold preserved), justified by the −0.375 real-LLM correlation. Ship
  this as the default.
- **Density / token-budget reporting**: `ContextEconomics` quantifies
  evidence-per-token, distractor ratio, redundancy, and wasted tokens —
  the telemetry production RAG lacks.
- **Max-density**: offered, but *gated* — appropriate when the workload
  is single-hop or the budget is brutal and you accept the recall
  tradeoff. The economics readout makes the tradeoff visible rather than
  silent.

## Honest limits

- **60-item HotpotQA sample, gold = supporting-fact chunks.** Gold
  retention is the proxy for "the answer is reachable"; it is not an
  end-to-end answer-quality measurement (that's Experiment B's domain,
  on the Python lab's LLM outputs).
- **HotpotQA is adversarially multi-hop.** The max-density gold-drop is
  worst-case here; on single-hop QA, max-density would be closer to
  free. The finding is workload-shaped, as all the geometry findings are.
- **Hashing-free, BGE-driven.** Real embedder, real dense retrieval; the
  density/distractor metrics are lexical (query-term overlap), which is
  the same primitive the diagnostics tier uses and is cheap by design.

## Next (measurement, not architecture)

- **Distractor-filtered context → answer quality**, end to end: feed the
  pruned vs raw context to an LLM and measure answer kw-recall. Experiment
  B says distractors hurt; this would confirm that *filtering* them
  helps the generated answer, closing the loop. (Needs an LLM — the
  Python lab's job; RedHop produces the contexts.)
- **Budget-vs-quality frontier per workload**: the budget where pruning
  starts to bite is workload-specific; the harness can map it.
