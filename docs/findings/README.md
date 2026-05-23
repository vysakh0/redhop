# RedHop Evidence Layer

This directory is RedHop's permanent record of *why the library behaves
the way it does*. Every default, every strategy, and every API exists
because a specific retrieval/reasoning failure was **measured** — not
because of a generic retrieval assumption.

The discipline that produced these findings is itself part of the design:
measure aggressively, let hypotheses fail, and extract the real mechanism
afterward. **Falsified hypotheses are preserved here, not deleted** — they
are some of the most valuable knowledge in the repo, and several of
RedHop's strongest defaults come directly from a hypothesis that failed.

## How the evidence is organized

```
docs/findings/   the findings docs (this directory) — hypothesis → result → mechanism
benchmarks/      reproducible measurement harnesses (the `cargo run` examples)
reports/         captured raw outputs of specific runs (e.g. reasoning_preserving_n300/)
```

Each findings doc follows the same shape:

- **Hypothesis** — what we believed going in
- **Status** — Confirmed / Falsified / Partially falsified / Open
- **Setup** — workload, models, sample size, reproduce command
- **Metrics** — exact tables, CIs where we have them
- **Failure cases** — where it breaks / what it cannot do
- **Interpretation** — the mechanism, stated as such
- **Caveats** — honest limits
- **What changed afterward** — the API/default/next-experiment it drove

`SECOND_HOP_TAX.md` and `REASONING_PRESERVATION.md` carry the header block
in full as the template model.

## Master table

| Finding | Status | Headline | Reproduce |
| ------- | ------ | -------- | --------- |
| [SECOND_HOP_TAX](SECOND_HOP_TAX.md) | **Confirmed** (n=1327, CIs) | Every relevance-based selection taxes the multi-hop second hop; a 0.30 filter keeps only 44% of second hops | `cargo run -p redhop-examples --example second_hop_retention --release` |
| [REASONING_PRESERVATION](REASONING_PRESERVATION.md) | **Confirmed** (4 models, n=300, CIs) | Aggressive filtering is net-harmful on all 4 models (−0.06 to −0.15); rescued-subset gain +0.15 to +0.23. Distractor-sensitivity splits by tier not age (non-frontier hurt, frontier inert) | `python python/eval/score_reasoning_qa.py --n 300 --model <id>` |
| [CONTEXT_DILUTION](CONTEXT_DILUTION.md) | **Confirmed (conditional)** (3 models, n=200, CIs) | At ~30k-token contexts, stuffing-it-all-in collapses accuracy; pruning recovers it where dilution bites (gpt-4o-mini +0.211) but is null on dilution-robust models. Win is generic pruning, not ReasoningPreserving | `cargo run -p redhop-examples --example emit_dilution --release` + `python python/eval/score_dilution.py --n 200 --model <id>` |
| [DOCUMENT_EVAL_CUAD](DOCUMENT_EVAL_CUAD.md) | **Confirmed** (50 contracts, 644 q, no LLM) | Document runtime on real contracts: −80% tokens with gold evidence retained (≥0.8 recall on 88%) at ~1.7ms/query; Auto prunes 94% of (large) contracts; crash-robust to dup/OCR; duplication is the main retention stressor | `cargo run -p redhop-examples --example eval_cuad_documents --release` |
| [CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) | **Confirmed** (sweep, vs LangChain/LlamaIndex, no LLM) | Chunk granularity (not strategy) is the lever: default 256→128 lifts multi-hop ≥0.8 retention 54%→77%, ahead of LangChain/LlamaIndex; competitive on contracts (LlamaIndex still leads there) | `bench/.venv/bin/python bench/chunk_sweep.py` |
| [FRAMEWORK_COMPARISON](FRAMEWORK_COMPARISON.md) | **Confirmed: competitive, not dominant** (Tier 1+3, gpt-4o-mini) | Head-to-head vs LangChain/LlamaIndex: RedHop ties LlamaIndex and beats LangChain on answer quality; multi-hop retention edge doesn't translate to a big answer lead; strategy ≠ moat downstream either | `bench/.venv/bin/python bench/tier3.py --n 150` |
| [RERANKING_LIMITS](RERANKING_LIMITS.md) | **Falsified hypothesis** | "A stronger reranker recovers dense's missed recall" — uniform cross-encoder made recall *worse* (−0.029); helps 12% / hurts 17% | `cargo run -p redhop-examples --example ce_escalation_economics --features onnx --release` |
| [DISTRACTOR_ROBUSTNESS](DISTRACTOR_ROBUSTNESS.md) | **Partially falsified** | "Distractor filtering is a free win" — distractors hurt (causal, +0.033), but filtering's net benefit is sign-unstable on multi-hop (the n=20→30 flip) | `cargo run -p redhop-examples --example emit_qa_contexts --release` |
| [CONTEXT_ECONOMICS](CONTEXT_ECONOMICS.md) | **Confirmed** | Distractors hurt & density helps on real LLM outputs (pooled −0.375 / +0.539); max-density pruning drops the second hop | `cargo run -p redhop-examples --example context_economics --features onnx --release` |
| [ADAPTIVE_CONTROLLER](ADAPTIVE_CONTROLLER.md) | **Falsified hypothesis** | "Stronger first-stage retrieval → fewer interventions" — dense BGE *increased* intervention rate (28%→38%) and halved usefulness; controller actions are retriever-coupled | `cargo run -p redhop-examples --example bge_dense_retrieval --features onnx --release` |
| [SUBSTRATE_COUPLING](SUBSTRATE_COUPLING.md) | **Confirmed** | A better embedder in the *sensing* path alone doesn't move economics; it must be in the *action* path. Calibration is substrate-specific | `cargo run -p redhop-examples --example bge_recalibration --features onnx --release` |

Supporting evidence: [ADAPTIVE_REAL_SUBSTRATE](ADAPTIVE_REAL_SUBSTRATE.md),
[EMBEDDING_BAKEOFF](EMBEDDING_BAKEOFF.md) (BGE +99% recall vs hashing),
[REAL_WORKLOAD](REAL_WORKLOAD.md), [INGESTION_PDF](INGESTION_PDF.md).

## Falsified-hypotheses registry

Preserved deliberately. Each was a reasonable prior; the measurement
overturned it, and the overturning is what produced the real design.

| Hypothesis (what we believed) | Verdict | What the data showed | What it produced |
| ----------------------------- | ------- | -------------------- | ---------------- |
| A stronger reranker (cross-encoder) recovers multi-hop recall a bi-encoder missed | **Falsified** | Uniform CE made recall *worse* (−0.029); it *demotes* the low-query-relevance second hop most confidently | The reranking-limits law; selective (not uniform) escalation; reinforced the second-hop tax |
| Aggressive distractor filtering is a free quality win | **Falsified (multi-hop)** | Net effect sign-flipped n=20→30; end-to-end the aggressive *filter* hurt more than the distractors (0.829→0.705) | `ReasoningPreserving`; "don't over-filter" default; the n=300 causal experiment |
| Distractors strongly degrade strong-generator answers | **Falsified (this regime)** | On gap-qualified multi-hop, haiku was distractor-robust (polluted 0.829 ≈ gold 0.830) | Reframed the threat from "distractors" to "premature removal of reasoning evidence" |
| Stronger first-stage retrieval reduces the controller's need to intervene | **Falsified** | Dense BGE *increased* intervention rate and *halved* usefulness — actions matched BM25's failure modes, not dense's | The retriever↔action coupling law; conservative controller's zero-harm guarantee held throughout |
| A better embedder improves retrieval economics by sharpening diagnostics | **Partially falsified** | As a *sensing*-only upgrade it was a near-no-op; recall lift identical (0.062). It must drive the *action* path | The sensing-vs-action-path distinction; substrate-specific calibration |
| ExpandTopK (more similar neighbors) can reach the missing evidence | **Falsified** | The missing chunk is *dissimilar* to the query (bridge-linked); more neighbors never reach it | Convergent first sighting of the second-hop tax |

The convergence is the point: reranking failures, aggressive-filtering
failures, max-density failures, ExpandTopK failures, and distractor
robustness **all reduce to one geometry** — transformers tolerate
irrelevant context, but are fragile to missing reasoning links. RedHop is
built around that measured geometry.

## APIs grounded in this evidence

- `build_context(strategy = ReasoningPreserving)` → [SECOND_HOP_TAX](SECOND_HOP_TAX.md), [REASONING_PRESERVATION](REASONING_PRESERVATION.md), [CONTEXT_ECONOMICS](CONTEXT_ECONOMICS.md)
- `build_context` as dilution-pruner at large contexts (generic pruning, size-gated) → [CONTEXT_DILUTION](CONTEXT_DILUTION.md)
- `build_context(strategy = DistractorFiltered)` (low threshold only) → [DISTRACTOR_ROBUSTNESS](DISTRACTOR_ROBUSTNESS.md), [CONTEXT_ECONOMICS](CONTEXT_ECONOMICS.md)
- selective reranker escalation (not uniform) → [RERANKING_LIMITS](RERANKING_LIMITS.md)
- conservative adaptive controller (zero-harm, retriever-coupled actions) → [ADAPTIVE_CONTROLLER](ADAPTIVE_CONTROLLER.md), [SUBSTRATE_COUPLING](SUBSTRATE_COUPLING.md)
