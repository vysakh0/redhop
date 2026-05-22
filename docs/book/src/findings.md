# Findings

The evidence layer is part of RedHop's identity. Every default exists because a
specific failure was **measured** — and **falsified hypotheses are kept, not
deleted**. Several of the strongest defaults came directly from a hypothesis
that failed.

Each finding records a hypothesis, the experiment, the result (with confidence
intervals where applicable), the caveats, and the implications. The full,
canonical docs live in [`docs/findings/`](https://github.com/redhop/redhop/tree/main/docs/findings)
in the repo; this section summarizes the load-bearing ones.

| Finding | Status | Headline |
| ------- | ------ | -------- |
| [Second-hop tax](./findings-second-hop-tax.md) | Confirmed (n=1327, CIs) | every relevance-based selection taxes the multi-hop second hop; a 0.30 filter keeps only 44% |
| Reasoning preservation | Confirmed (n=300, CIs) | reasoning-preserving beats aggressive filtering end-to-end; gain causally localized to gold reachability |
| [Reranking limits](./findings-reranking-limits.md) | **Falsified** | "a stronger reranker recovers missed recall" — uniform cross-encoder made recall *worse* |
| [Filtering failures](./findings-filtering-failures.md) | Partially falsified | "distractor filtering is a free win" — net benefit is sign-unstable on multi-hop |
| [Context economics](./findings-context-economics.md) | Confirmed | distractors hurt & density helps on real LLM outputs (pooled −0.375 / +0.539) |

## Falsified hypotheses (kept on purpose)

Each was a reasonable prior; the measurement overturned it, and the overturning
produced the real design.

| Hypothesis | Verdict | What it produced |
| ---------- | ------- | ---------------- |
| A stronger reranker recovers multi-hop recall a bi-encoder missed | Falsified | the reranking-limits law; selective (not uniform) escalation |
| Aggressive distractor filtering is a free quality win | Falsified (multi-hop) | `reasoning_preserving`; "don't over-filter" default |
| Distractors strongly degrade strong-generator answers | Falsified (this regime) | reframed the threat to *missing reasoning evidence* |
| Stronger first-stage retrieval reduces controller intervention | Falsified | the retriever↔action coupling law |
| A better embedder helps via sharper diagnostics alone | Partially falsified | the sensing-vs-action-path distinction |
| "more similar neighbors" can reach the missing evidence | Falsified | first sighting of the second-hop tax |

The convergence is the point: reranking failures, aggressive-filtering failures,
max-density failures, and "more neighbors" failures **all reduce to one
geometry** — transformers tolerate irrelevant context but are fragile to missing
reasoning links.
