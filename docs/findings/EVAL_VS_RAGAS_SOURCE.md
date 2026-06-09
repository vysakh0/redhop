# RedHop's Tier-2 vs Ragas — read-the-source comparison

> **Status: honest gap analysis.** Earlier docs in this series compared
> RedHop and Ragas at the architecture level. This one compares them by
> READING the metric implementations side by side
> (`explodinggradients/ragas/src/ragas/metrics/_*.py`). Where the
> mechanisms diverge — and where they don't — is documented below
> without spin.

## TL;DR — three big mechanical gaps, two intentional, one to fix

1. **Faithfulness statement extraction.** Ragas's "extract claims"
   step uses few-shot prompting + structured output (`PydanticPrompt`)
   to produce a JSON-list of de-pronouned, self-contained statements.
   RedHop's Phase-6 extraction prompt is plain text, no few-shot, no
   structured-output forcing. **Likely produces noisier claim lists
   on real LLM runs.** Worth upgrading.
2. **Answer relevancy.** Ragas measures relevancy by reverse-
   generating questions from the answer and computing **cosine
   similarity** against the original question (embedding-based). RedHop
   uses a direct "does this answer address this question" judgement.
   Completely different mechanism. **Ragas's is more grounded;
   RedHop's is cheaper.**
3. **Aspect critique.** Ragas binarizes (`yes=1`/`no=0` verdict from
   the LLM) and supports `strictness >= 3` majority-voting. RedHop
   asks for a continuous `[0, 1]` score. **Different by design; not
   strictly better either way.**

The rest of this doc walks through each metric.

## Faithfulness

| | Ragas | RedHop (Phase 6 path) |
|---|---|---|
| **Steps** | 2-pass | 2-pass (matches) |
| **Pass 1 — extract** | `StatementGeneratorPrompt`: 1-shot example, asks for de-pronouned full sentences, JSON list via `PydanticPrompt` | Plain prompt: "List the atomic factual claims this ANSWER makes. One claim per line." No examples, no structured output. |
| **Pass 2 — verify** | `NLIStatementPrompt`: per-statement verdict `{statement, reason, verdict ∈ {0, 1}}` via JSON. Includes 2 few-shot examples. | Per-claim "Reply with a single number 0..1" — continuous score, no reason field, no examples. |
| **Score** | `faithful_statements / total_statements` (count of `verdict==1`) | `mean(per_claim_score)`; "supported" count = scores ≥ 0.5 |
| **Cost** | 1 + 1 LLM calls (extract + verify-all-as-batch). Both calls produce structured JSON the LLM has to commit to. | 1 + N calls. Each verification is a separate prompt. **More calls per evaluate.** |

**Gaps that matter:**

- **Few-shot examples.** Ragas's `StatementGeneratorPrompt` has a worked
  example (Einstein → 4 self-contained statements). Without one,
  small LLMs (especially gpt-4o-mini) may under-decompose or include
  pronouns. RedHop should add a 1-2 shot example.
- **JSON-structured output.** Ragas forces a typed `PydanticPrompt`
  contract; the parser knows exactly what to expect. RedHop relies on
  "one claim per line" which a chatty model can break by including
  commentary.
- **Batched verification.** Ragas sends all statements in one
  `NLIStatementInput` and the LLM returns per-statement verdicts in
  one response. RedHop calls the judge N times — one per claim.
  **For an answer with 5 claims, that's 5 LLM calls instead of 1.**
- **Threshold for "supported".** Ragas binarizes per-statement
  (verdict=0/1). RedHop uses a 0.5 threshold on the continuous score.
  Equivalent in expectation when the LLM is well-calibrated; not
  always equivalent in practice.

**Recommendation:** add few-shot examples to the extraction prompt;
keep the per-claim verification (the single-claim prompt is simpler
to debug than a batched JSON one). The N-calls cost is real but the
`CachedJudge` mitigates it on re-runs.

## Answer relevancy

| | Ragas | RedHop |
|---|---|---|
| **Mechanism** | Reverse-generate N questions FROM the answer (`strictness=3` by default), then cosine-similarity each against the original question via an embedder. | Direct LLM judgement: "Does this ANSWER directly address this QUESTION? Reply 0..1." |
| **Side detection** | Yes — separate `noncommittal: bool` from the same prompt. Multiplies score by `int(not all_noncommittal)`. | None. A noncommittal answer that happens to mention query terms can score high. |
| **Cost** | 1 LLM call (n=3 generations sampled) + embedder calls (cosine sim). | 1 LLM call. No embeddings. |
| **Dependencies** | Needs both an `MetricWithLLM` AND `MetricWithEmbeddings`. | Just the Judge. |

**This is a real design difference, not a parity gap.**

Ragas's approach: a relevant answer should generate a question close
to the original (semantic embedding distance is the measure). It also
explicitly catches noncommittal answers like "I don't know" and zeros
them out.

RedHop's approach: ask the LLM directly. Simpler, but loses the
noncommittal detection.

**Recommendation:** add a noncommittal check. We could either bolt it
onto the relevancy prompt ("If the answer is evasive or noncommittal,
score 0") or run a small second prompt. The reverse-question + cosine
machinery requires an embedder, which would be a real dep — keep the
direct-LLM path but cover the noncommittal failure mode.

## Answer correctness

Ragas's `AnswerCorrectness` is the most sophisticated metric in the
library. It's a weighted blend of TWO sub-metrics:

1. **Factuality (75% by default).** Reuses the `StatementGenerator`
   prompt to extract statements from BOTH the answer and the reference.
   Then classifies each statement into TP/FP/FN via a separate
   `CorrectnessClassifier` prompt. Computes
   F1 = `2·TP / (2·TP + FP + FN)`.
2. **Semantic similarity (25%).** Cosine similarity (or cross-encoder
   score) between answer and reference embeddings via
   `AnswerSimilarity`.

RedHop's `correctness_judged` is one prompt: "Does the GENERATED
ANSWER convey the same facts as the REFERENCE ANSWER? Reply 0..1."

**Gaps:**

- **TP/FP/FN decomposition.** Ragas surfaces actionable detail —
  "which specific statements were missed (FN) vs hallucinated (FP)."
  Our metric is a single number.
- **F-beta with caller-tunable β.** Ragas lets the caller weight
  recall vs precision via β. Use case: "missing facts hurt more than
  extra facts" (high β favors recall) vs "extra facts hurt more"
  (low β).
- **Semantic similarity component.** Ragas's embedding-cosine catches
  paraphrase the way a binary verdict won't.

**Recommendation:** the single-prompt path is probably "good enough"
for most users — Ragas's machinery is impressive but costs 3-4 LLM
calls + an embedding model. A future RedHop optimization could add a
2-pass mode (statements + classification) similar to Phase 6
faithfulness. The semantic similarity component would require an
embedder; that's a real dependency expansion we should probably defer.

## Aspect critique

| | Ragas | RedHop |
|---|---|---|
| **Verdict type** | Binary: `Yes=1` / `No=0`. | Continuous `[0, 1]`. |
| **Strictness** | `strictness >= 3` triggers majority vote over N runs (default 1). | Single run, single score. |
| **Output shape** | Verdict + reason (LLM provides both). | Just a score. |
| **Built-in aspects** | 5 presets: `harmfulness`, `maliciousness`, `coherence`, `correctness` (yes, ragas has BOTH this and the `_answer_correctness.py` one), `conciseness`. | None — caller defines all aspects. |
| **Polarity** | All ragas presets phrased so "yes" (1) means the PROBLEM is present (e.g. harmfulness=1 means harmful). | Caller-controlled via `high_is_good`; report is polarity-corrected so high = good across the report. |

**This is a real design difference, not a parity gap.**

Ragas's binary-with-majority-vote is closer to how human evaluators
work — a panel decides yes/no per dimension. RedHop's continuous
score is closer to how an LLM "feels" about the question; it doesn't
require multiple samples.

**Practical implication:** Ragas's binary scores have higher
variance per-call (any one verdict is 0 or 1) but the majority-vote
smooths it. RedHop's continuous score has lower per-call variance
but no built-in self-consistency.

**Recommendation:** consider adding an optional `strictness=N` to
`Aspect` that runs N judge calls and averages (continuous) or
majority-votes (binary). Off by default for cost reasons.

## What Ragas has that RedHop doesn't (yet)

From the metric file listing:

- `_context_entities_recall.py` — entity-extraction-based recall
- `_context_precision.py` (LLM-judged) and `_context_recall.py` —
  alternative chunk-relevance signals beyond set membership
- `_noise_sensitivity.py` — robustness to distractor chunks
- `_factual_correctness.py` — newer fact-extraction-and-verify metric
- `_summarization.py` — purpose-built summarization eval
- `_topic_adherence.py` — multi-turn topic-drift detector
- `_tool_call_*.py` — agent-tool eval
- `_sql_semantic_equivalence.py`, `_chrf_score.py`, `_bleu_score.py`,
  `_rouge_score.py`, `_string.py` — string-similarity metrics
- `_domain_specific_rubrics.py`, `_instance_specific_rubrics.py` —
  rubric-driven scoring
- `_multi_modal_*.py` — image/multimodal
- `_nv_metrics.py` — NVIDIA's NeMo metrics
- `_goal_accuracy.py`, `_simple_criteria.py` — others

Most of these are either out-of-scope for RedHop (multi-modal,
SQL-specific, multi-turn agents) or thin variations on what we
already have (BLEU/ROUGE/CHRF would be one-day adds via the
analyzer's tokens).

**What's worth porting:**

- `noise_sensitivity` — robustness to distractor chunks is something
  RedHop's `low_confidence_retrieval` partially covers but doesn't
  quantify the same way.
- `context_entities_recall` — entity-based recall complements our
  token-overlap `answer_token_recall`.

## What RedHop has that Ragas doesn't

For symmetry — these are mentioned in `EVAL_RAGAS_PARITY.md` and the
source confirms them:

- `overall` composite score (Ragas leaves the caller to blend).
- `mean_grounding`, `evidence_density`, `retained_evidence_ratio`,
  `second_hop_rescues`, `low_confidence_retrieval`,
  `estimated_waste_tokens` — none of these have Ragas analogs.
  They're computed from the runtime's Decision Report; Ragas operates
  on a generic `(question, answer, retrieved_contexts, reference)`
  tuple and doesn't have access to RedHop's internals.
- Deterministic Tier-1 (`_lexical` fields) — Ragas has BLEU/ROUGE/CHRF
  but doesn't surface them as faithfulness/relevancy proxies the same
  way we do.
- `CachedJudge` — Ragas has no built-in caching layer; users add
  their own.
- Cross-binding (Python + Node) — Ragas is Python-only.

## Honest summary

| Category | Verdict |
|---|---|
| **API ergonomics** | Comparable. Ragas is `evaluate(dataset, metrics)` with a HuggingFace `Dataset`; RedHop is `evaluate(query, ctx, answer=, judge=)` per-call. Different idioms. |
| **Faithfulness mechanism** | Ragas more sophisticated (few-shot + structured output + batched verify). **RedHop should add few-shot examples to extraction.** |
| **Relevancy mechanism** | Different design (Ragas: reverse-generate + cosine; RedHop: direct LLM). RedHop should add a noncommittal-answer check. |
| **Correctness mechanism** | Ragas significantly more sophisticated (TP/FP/FN + similarity blend + caller-tunable β). RedHop's single-prompt is functional but a future 2-pass mode would close the gap. |
| **Aspect critique** | Different design (Ragas: binary + majority vote; RedHop: continuous + caller-defined). Neither strictly better. |
| **Built-in metric set** | Ragas has ~20 more specialized metrics. Most are out-of-scope for RedHop or thin variations. |
| **Self-eval / Decision Report integration** | RedHop only. Ragas's metrics are purely answer-quality. |
| **Calibration data** | Neither published as far as I can find. Both produce numbers; neither has "this is the absolute truth" calibration against human judgment for a published benchmark. |

## Open work items from this audit

1. Add a worked example to `CLAIM_EXTRACTION_PROMPT` and switch to a
   numbered-list output format the parser is more robust to.
2. Add a noncommittal-answer check to `relevancy_judged` (either
   bolted onto the existing prompt or as a small separate one).
3. Consider a 2-pass mode for `correctness_judged` (claim
   extraction + TP/FP/FN classification, F-beta blend). Probably
   gated behind a flag like `decompose_correctness=True` mirroring
   Phase 6.
4. Optional `strictness=N` on `Aspect` for self-consistency.
5. Add `noise_sensitivity` and `context_entities_recall` as
   future metrics.

Filing these as future-phase work, not blocking the current ship.

## See also

- [`EVAL_RAGAS_PARITY.md`](EVAL_RAGAS_PARITY.md) — architecture-level
  comparison (this doc is the source-level companion).
- [`EVAL_JUDGED_CALIBRATION.md`](EVAL_JUDGED_CALIBRATION.md) — the
  bench infrastructure that can run RedHop and Ragas side-by-side on
  the same workload.
- Ragas sources at `https://github.com/explodinggradients/ragas/tree/main/src/ragas/metrics`.
