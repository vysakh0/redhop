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
| [FRAMEWORK_COMPARISON](FRAMEWORK_COMPARISON.md) | **Confirmed: competitive, not dominant** (Tier 1+3, gpt-4o-mini; latest rerun 2026-06-06) | Head-to-head vs LangChain/LlamaIndex: RedHop leads multi-hop retention by +8 (HotpotQA 80% vs 72%) and beats LangChain on contracts (82 vs 73); the +3 HotpotQA improvement (77→80) lands cleanly on 0.2.1 fixes; CUAD raw-template 4-point gap to LlamaIndex is mechanism-attributed and closeable — see CUAD_RECALL_GAP | `bench/.venv/bin/python bench/compare.py` (retention) · `bench/.venv/bin/python bench/tier3.py --n 150` (answers) |
| [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) | **Confirmed** (mechanism + workaround, n=300) | The CUAD 4-point gap to LlamaIndex on bench/compare.py is **BM25 template-boilerplate dilution** from CUAD's identical 24-word template across every query. A 6-line query preprocessor that strips the template lifts ≥0.8 retention 82% → 88%, overtaking LlamaIndex by 2. Also resolves the apparent "Rust/Python parity gap" as a metric bug (Vec vs set span_recall). | `cargo run -p redhop-examples --example cuad_query_preprocessing --release` · `--example cuad_chunk_strategy_sweep --release` |
| [CUAD_PRF_NULL](CUAD_PRF_NULL.md) | **Null result / falsified** (n=300, 2 param sweeps) | Tested PRF on top of the template-stripping fix to push past 88%. Monotonic degradation across all parameter cells; best PRF cell was −0.4 vs the stripped baseline, default was −3.7. **Mechanism:** unweighted PRF picks corpus-boilerplate terms (`agreement, party, shall, distributor`) as expansion candidates and re-injects the dilution template-stripping just removed. **Predicts unweighted PRF will fail on any boilerplate-heavy corpus** (legal, medical, regulatory, support tickets). Second independent PRF falsification — see the RM3 entry in the registry below for the first | `cargo run -p redhop-examples --example cuad_prf --release` |
| [QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md) | **Confirmed** (cross-workload probe, 3 workloads × n=300) | Heuristic detects CUAD-shape templated queries (CUAD share 0.66, fires) and stays quiet on diverse natural-language QA (HotpotQA share 0.00, MuSiQue share 0.12, neither fires). Both failure modes (false positive on diverse, false negative on canonical) ruled out at this sample. Ships `redhop::analyze_query_set` + `drop_template_terms` so the CUAD_RECALL_GAP pattern is self-discoverable on new workloads | `cargo run -p redhop-examples --example query_set_analyzer_probe --release` |
| [EVALUATE_API](EVALUATE_API.md) | **Shipped** (Rust + Python + Node, 10 + 11 + 9 tests) | In-process `evaluate(query, ctx, gold)` returns an EvalReport blending self-eval (mean_grounding, evidence_density, low_confidence, …) with optional gold-relative metrics (context_recall, context_precision, answer_token_recall). Same primitives the runtime uses for its Decision Report — refraction not independent measurement, so eval and runtime never disagree. Closes the A/B step in detect → strip → A/B (CUAD_RECALL_GAP + QUERY_SET_ANALYZER). | unit tests in `crates/redhop/src/context/eval.rs::tests` (cargo test -p redhop --lib context::eval) |
| [GLOBAL_DENSE](GLOBAL_DENSE.md) | **Confirmed** (semantic-mismatch probe, n=25, no LLM) | On paraphrase/synonym queries BM25's pool misses the answer, so local rerank ≈ BM25 (recall@1 32% vs 20%); global dense scores every chunk → **88% / 96% recall@1/@3**. Shipped `retrieval="semantic"` (exact cosine, no ANN) for bounded synonym-heavy corpora; lexical stays the default | `bench/.venv/bin/python bench/semantic_modes.py` |
| [SPEED_VS_FRAMEWORKS](SPEED_VS_FRAMEWORKS.md) | **Largely falsified as a speed claim** (no LLM) | Like-for-like: lexical (BM25 all) — no speed advantage, all ≤0.25s; semantic (embed all) — RedHop rerank *slower* to set up (51s vs ~7s) but faster warm (~4ms vs ~17ms). "0.02s vs 7s" was lexical-vs-vector defaults, not an engine win. Speed isn't the advantage | `bench/.venv/bin/python bench/speed_compare.py` |
| [SEMANTIC_MISMATCH](SEMANTIC_MISMATCH.md) | **Confirmed value / falsified trigger** (controlled + natural, Tier 1+3) | Dense helps semantic-heavy queries (+0.16 F1); no cheap escalation *trigger* exists (overlap + BM25 margin/entropy both null). Maps the lexical↔semantic boundary; resolution is local rerank (below) | `cargo run -p redhop-examples --example semantic_natural --features onnx --release` |
| [LOCAL_RERANK](LOCAL_RERANK.md) | **Confirmed — ships as `retrieval="hybrid"`** (global HotpotQA, Tier 1+3, n=400) | BM25 prunes → dense reranks only the pool: matches *global* dense on recall (0.80) / answers (0.54 F1) while embedding only the pool — so it **scales to large local corpora with no vector DB** (the agent/folder case). Briefly dropped, then restored as the `hybrid` tier (vs `semantic` = global dense for small sets). | `cargo run -p redhop-examples --example semantic_local_rerank --features onnx --release` |
| [SEMANTIC_ZERO_DEP](SEMANTIC_ZERO_DEP.md) | **Partially confirmed / 3 sub-hyps falsified** (global HotpotQA, n=400) | The lightweight-semantic frontier: a zero-dep corpus-graph **second-order local rerank** lifts semantic recall@3 0.49→0.56 (0.59 idf), but it *and* pretrained static embeddings plateau in the same ~0.56–0.59 band — both non-contextual, so neither replaces BGE (0.80). The free corpus-graph reranker ≥ a 30MB static table. Falsified: MaxSim, RM3, static-as-dense | `cargo run -p redhop-examples --example export_rerank_pool --release` + `bench/.venv/bin/python python/eval/static_rerank.py` |
| [DENSE_RERANK_CEILING](DENSE_RERANK_CEILING.md) | **Confirmed ceiling / falsified fixed-knob rescue** (global HotpotQA, n=400, 100% multi-hop) | Dense local rerank's 0.80 plateau *is* the second-hop tax (148 in-pool misses); a fixed-β reasoning linkage-rescue nets ~zero (+0.001 best, then monotone harm). Oracle per-query β recovers 18% — "no cheap trigger" reappears on the dense substrate; breaking 0.80 needs per-query adaptivity or iterative conditioning (both ruled out) | `cargo run -p redhop-examples --example semantic_reasoning_rerank --features onnx --release` |
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
| MaxSim late-interaction (ColBERT-style) beats centroid on a corpus-graph reranker | **Falsified** | Semantic recall@3 *fell* 0.563→0.531; per-term best-match over sparse graph vectors rewards any doc with one loosely-related term — late interaction needs many query tokens, questions don't have them | Centroid (aggregated context) kept as the zero-dep rerank scorer ([SEMANTIC_ZERO_DEP](SEMANTIC_ZERO_DEP.md)) |
| RM3 / pseudo-relevance feedback lifts lexical recall (semantic-mismatch workload) | **Falsified** | Monotonically harmful (semantic R@3 0.49→0.41); λ=1 (no feedback) is optimal. Low first-pass precision (R@1≈0.31) → feedback built on distractors | Don't bolt PRF onto BM25 here; another instance of the second-hop/relevance-feedback tax ([SEMANTIC_ZERO_DEP](SEMANTIC_ZERO_DEP.md)) |
| Unweighted PRF on top of template-stripping pushes CUAD past 88% (boilerplate-heavy corpus) | **Falsified** (2026-06-06, n=300, 2 param sweeps) | Monotonic degradation: best PRF cell −0.4 vs B, default −3.7. Top expansion terms by frequency are legal-contract boilerplate (`agreement, party, shall, distributor`) — re-injecting them re-introduces the dilution template-stripping just removed | **General rule:** unweighted PRF will fail on boilerplate-heavy corpora (legal, medical, regulatory, support tickets); the subtraction-at-the-query-boundary win is not symmetric with addition. Second independent PRF falsification, different mechanism than the RM3 entry above ([CUAD_PRF_NULL](CUAD_PRF_NULL.md)) |
| Static embeddings are a lighter drop-in for the dense reranker | **Falsified** | potion / static-retrieval-mrl reach only ~0.56–0.57 ALL — below BM25, far below BGE 0.80, same band as the zero-dep corpus graph; both are non-contextual | BGE's edge is contextualization; keep BM25 default + corpus-graph as the free tier ([SEMANTIC_ZERO_DEP](SEMANTIC_ZERO_DEP.md)) |
| A fixed-knob reasoning linkage-rescue lifts dense local rerank past 0.80 | **Falsified** | Best fixed β = +0.001 then monotone harm; the 18% recoverable headroom needs per-query β (oracle) | The second-hop tax caps dense too; no cheap trigger ([DENSE_RERANK_CEILING](DENSE_RERANK_CEILING.md)) |

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
