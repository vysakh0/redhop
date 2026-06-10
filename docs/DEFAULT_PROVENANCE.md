# Default Provenance Audit

Every numeric default in `DocumentConfig` + `ContextConfig` traced to its
justification. Last walked: 2026-06-04 (post-0.2.0).

Each entry is classified as:

- 🟢 **measured**: backed by a specific finding in `docs/findings/` with
  CIs, n, and a reproduce command.
- 🟡 **convention**: sensible but no specific RedHop measurement, matches
  industry defaults or follows from a related finding.
- 🔵 **by-design**: value chosen for an architectural reason, not a
  measurement (e.g. "opt-in floor" or "API alignment").
- 🟠 **model-specific**: the calibration was against a specific LLM /
  embedder, may want re-validation as the model lineup shifts.

## `DocumentConfig`

| Field | Default | Provenance | Notes |
| --- | --- | --- | --- |
| `target_tokens` | **128** | 🟢 [`CHUNK_GRANULARITY.md`](findings/CHUNK_GRANULARITY.md) | Sweep across budgets × datasets vs LangChain / LlamaIndex. 128 lifts multi-hop ≥0.8 evidence retention 54→77% under tight budgets, ties at large. |
| `max_tokens` | **256** | 🔵 derived (2 × `target_tokens`) | No standalone finding. Gives the chunker headroom for a long sentence rather than splitting it. Re-evaluate if `target_tokens` moves. |
| `overlap_sentences` | **1** | 🟡 convention | Light overlap prevents boundary-effects on retrieval. No standalone RedHop finding. Matches the LangChain `SentenceTextSplitter` default. |
| `candidate_k` | **20** | 🟡 convention | Standard "top-k retrieval" depth. Most strategies prune down from here. Raising it costs more BM25 work but more candidates to re-rank. No measurement says 20 is optimal, it's the LangChain/LlamaIndex norm. |
| `rerank_pool` | **50** | 🟡 convention | Pool depth the cross-encoder reranker reorders when one is attached. Matches `candidate_pool` for the hybrid tier. |
| `min_candidates` | **0** | 🔵 by-design | Opt-in floor: when the primary retriever returns fewer results than this, a BM25 fallback tops up. Default off because the strict-superset contract restored in 0.1.3 already returns the right behavior for most callers. |
| `code_neighbors_default` | **1** | 🟢 0.1.4 citation-ergonomics work | Code chunks are fixed-token windows. A `def` line often sits in a different chunk from its body. ±1 neighbor expansion makes `context()` on code hits include the implementation by default. Set to `0` to disable. |
| `prose_heading_default` | **true** | 🟢 0.1.4 citation-ergonomics work | Attach the section's opening (heading) chunk to each cited chunk that carries `metadata["heading"]`. Lossless for the LLM. Off-by-default would mean a chunk deep in `## Refunds → ### Eligibility` arrives without its section title. |
| `retrieval_mode` | **Lexical** | 🔵 by-design | BM25 is the zero-dependency baseline. Opt into `Hybrid` / `Dense` when you want semantic recall, at the cost of an ONNX model download. |
| `embedder_dim` | **384** | 🟡 convention | Matches `bge-small`, the default model when one is requested. Override required when bringing your own model with a different output dim. |

## `ContextConfig`

| Field | Default | Provenance | Notes |
| --- | --- | --- | --- |
| `token_budget` | **8192** | 🔵 by-design | Aligned to the Python binding's long-standing default in 0.2.0 (was 2048 in Rust pre-0.2). Sized for modern LLMs (≥32k usable context). Set lower for tight-budget targets. |
| `strategy` (top-level) | **Auto** | 🟢 [`CONTEXT_DILUTION.md`](findings/CONTEXT_DILUTION.md) | Size-gated decision: passthrough for small contexts, prune for large/diluted. |
| `strategy` (`Document` override) | **ReasoningPreserving** | 🟢 [`REASONING_PRESERVATION.md`](findings/REASONING_PRESERVATION.md), [`SECOND_HOP_TAX.md`](findings/SECOND_HOP_TAX.md) | Drops unlinked junk only. Rescues low-relevance chunks linked to a seed (the second-hop bridge). Aggressive relevance pruning hurts multi-hop. |
| `distractor_min_grounding` | **0.10** | 🟡 convention ("low absolute bar") | Documented intent: "only near-zero-overlap junk is below it". No specific measurement says 0.10 is optimal. Raising it gets aggressive (drops valid evidence), lowering it lets junk through. |
| `redundancy_max_cosine` | **0.92** | 🟡 convention | Cosine ceiling above which a chunk is treated as a near-duplicate of another in the set. 0.92 is the standard "near-dup" cutoff for sentence-embedding cosines. No RedHop-specific measurement. |
| `link_min_jaccard` | **0.12** | 🟡 convention | Jaccard floor for treating two chunks as linked (the bridge signal `ReasoningPreserving` uses to rescue a second hop). No standalone finding. 0.12 was an early empirical choice that stuck. |
| `auto_passthrough_max_tokens` | **1500** | 🟠 [`CONTEXT_DILUTION.md`](findings/CONTEXT_DILUTION.md) (calibrated against **gpt-4o-mini**) | The size crossover where pruning starts recovering accuracy. Conservative lower edge of the measured-benefit band. **Model-specific:** gpt-4o-mini's dilution profile may differ from gpt-4o, claude-haiku, etc. Recalibrating against current frontier models is worth it for 0.3. |
| `low_confidence_max_grounding` | **0.10** | 🔵 derived (same as `distractor_min_grounding`) | If every selected chunk is at-or-below the distractor bar, the retrieval was weak: surface as `report.low_confidence_retrieval = true`. |
| `analyzer` | **minimal raw analyzer** (tokenize → lowercase → ASCII fold) | 🟢 [`RAW_ANALYZER.md`](findings/RAW_ANALYZER.md) | Default flipped in 0.3.2: no stemming, no stopwords, no CamelCase split. Measured vs the previous English-Snowball default: CUAD +5, MuSiQue +7, HotpotQA tied, 1.5-2.5× faster. Opt back in with `language="english"` or any of the 18 Snowball builtins via `Document::with_analyzer`. |

## `report.diagnosis` hint thresholds

Constants that gate the closed hint registry in
`crates/redhop/src/context/diagnosis.rs`. All 🟡 convention. Picked to
fire on the documented failure shapes in `docs/CHOOSING_A_CONFIG.md`,
not chosen by a measurement sweep.

| Constant | Default | Provenance | Notes |
| --- | --- | --- | --- |
| `VOCAB_MISMATCH_MIN_SHARE` | **0.5** | 🟡 convention | "Half or more of the query terms missed" is a strong vocab signal. Lower and the hint will fire on partially-matched queries. |
| `VOCAB_MISMATCH_MIN_TERMS` | **2** | 🟡 convention | Below 2 query terms, "fraction missed" isn't meaningful. |
| `DF_RATIO_LOW_DISCRIMINATION` | **0.25** | 🟡 convention | A term in >25% of chunks carries little BM25 signal. Loose proxy for `analyze_query_set`'s 80% DF threshold which only makes sense over a query set. |
| `LOW_DISCRIMINATION_MIN_TERMS` | **8** | 🟡 convention | Short queries don't have boilerplate dilution; the templated-query failure shape is about long boilerplate-heavy queries. |
| `LOW_DISCRIMINATION_MIN_SHARE` | **0.6** | 🟡 convention | At least 60% of query terms must be low-discrimination before flagging the query as boilerplate-shaped. |
| `UNDERDETERMINED_MAX_TERMS` | **2** | 🟡 convention | The polysemy failure shape from CHOOSING_A_CONFIG ("'vendor'", "'settle'") is single-word. |
| `UNDERDETERMINED_MAX_SPREAD` | **0.15** | 🟡 convention | Relative spread of the top scores. Below 15% means the candidates are nearly tied. **Mode-dependent:** reasoned about on BM25 score distributions. Dense cosines compress into a narrower band, so the hint may over-fire under `Hybrid` / `Dense`. The re-validation sweep should run per retrieval mode. |
| `UNDERDETERMINED_MIN_CANDIDATES` | **5** | 🟡 convention | Below 5, "spread is flat" can just mean "small pool". |
| `SCORE_SPREAD_TOP_K` | **10** | 🟡 convention | Spread window. Matches `candidate_k`'s default ballpark. |
| `SUMMARY_MIN_QUERIES` | **20** | 🟡 convention | `summarize_diagnoses` returns `sample_too_small` below this. A workload audit on <20 queries can swing on a single outlier; better to ask the user to collect more than emit a noisy recommendation. |
| `DOMINANT_HINT_SHARE` | **0.20** | 🟡 convention | A hint must fire on at least 20% of queries before the summary will name it as the workload focus. Below the threshold, no failure shape "dominates" and the summary falls through to `weak_retrieval` or `healthy`. |
| `WEAK_RETRIEVAL_MIN_RATE` | **0.30** | 🟡 convention | The fallback gate: 30%+ of queries are empty or low-confidence but no specific shape leads. Surfaces the "your corpus does not cover these questions" case. |
| `TOP_TERMS_CAP` | **20** | 🟡 convention | Cap on the number of zero-match terms returned in the summary. Enough to be actionable as a `Vocabulary` seed, bounded to keep the JSON small. |

## Defaults flagged for re-validation in 0.3

1. **`auto_passthrough_max_tokens = 1500`**: calibrated against
   gpt-4o-mini. Frontier models (gpt-4o, claude-sonnet, gemini-2-pro)
   may have different dilution profiles. A re-run of the
   `CONTEXT_DILUTION` sweep against ≥3 current models would tell us
   whether 1500 still sits at the conservative-low edge or has drifted.

2. **`candidate_k = 20`**: never measured directly. A small sweep over
   {10, 20, 40, 80} on the existing HotpotQA / MuSiQue corpora would
   either confirm 20 or surface a better default. Cheap to run.

3. **`distractor_min_grounding = 0.10`** + **`link_min_jaccard = 0.12`**:
   set empirically and never re-measured. A grid sweep over each
   independently (other held at default) would yield a defensible
   number per finding.

4. **`overlap_sentences = 1`**: never measured. A 0-vs-1-vs-2 sweep
   across the existing benchmarks would settle whether the boundary-
   effects justification holds for RedHop's chunker shapes.

5. **`report.diagnosis` hint thresholds**: nine 🟡 constants in the
   diagnosis module (see the "hint thresholds" table above). A grid
   sweep over each (others held at default) against the existing
   CUAD / HotpotQA / MuSiQue corpora would surface the false-positive
   and false-negative rates per hint, turning the constants 🟢. Cheap
   to run: no LLM judge needed since we score against gold-evidence
   recall and a corpus where we already know which queries are
   templated, polysemous, or paraphrased. **Fold the four
   `summarize_diagnoses` constants (`SUMMARY_MIN_QUERIES`,
   `DOMINANT_HINT_SHARE`, `WEAK_RETRIEVAL_MIN_RATE`, `TOP_TERMS_CAP`)
   into the same sweep** since the workload-audit precision depends on
   the per-query hint precision.

## Defaults that don't need re-validation

The 🟢-marked rows are anchored in findings docs with reproduce
commands. The 🔵-marked rows are architectural choices (the value
follows from the design, not a measurement). Neither needs scheduled
re-validation, only re-validation when the underlying mechanism
changes (e.g. if `Document::context` adds a new auto-expansion mode,
`code_neighbors_default` should be re-evaluated alongside it).

## How to extend this doc

- New default added: add a row classifying it as 🟢 / 🟡 / 🔵 / 🟠.
- Default value changed: update the "Default" column and the
  "Provenance" column with the finding that drove the change (CHANGELOG
  reference is fine).
- Finding re-run with new model / new sample size: bump the "Notes"
  column and remove the 🟠 if the model-specificity concern is
  resolved.
