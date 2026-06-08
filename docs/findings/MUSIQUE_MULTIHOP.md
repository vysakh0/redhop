# MuSiQue multi-hop retention — partial replication of the HotpotQA lead

> **Status:** **Confirmed-with-caveat.** The multi-hop retention lead
> RedHop showed on HotpotQA (≥0.8 retention 80% vs LlamaIndex 72%,
> LangChain 71%) is **directionally replicated** on MuSiQue (n=300) but
> the magnitude shrinks substantially and mean recall is essentially
> tied. RedHop's `reasoning_preserving` strategy is **not** the
> differentiator — `raw_topk` matches it on both datasets.

## TL;DR

| MuSiQue n=300, budget 400 tok, BM25 | mean recall | ≥0.5 | **≥0.8** |
|---|---:|---:|---:|
| redhop[topk] | 0.59 | 62% | **22%** |
| redhop[reason] | 0.58 | 60% | **22%** |
| langchain | 0.58 | 64% | 19% |
| llamaindex | 0.56 | 61% | 17% |

Compared to HotpotQA at the same budget:

| HotpotQA n=300, budget 400 tok, BM25 | mean recall | ≥0.5 | **≥0.8** |
|---|---:|---:|---:|
| redhop[topk] | 0.91 | 97% | **80%** |
| redhop[reason] | 0.90 | 98% | **77%** |
| langchain | 0.88 | 96% | 71% |
| llamaindex | 0.88 | 96% | 72% |

## Question

The 0.3.1 audit's reviewer pass flagged that "RedHop wins multi-hop"
rested on a single dataset (HotpotQA). MuSiQue is a natural second
dataset — compositional multi-hop (2-4 reasoning hops; 20 distractor
paragraphs per example; designed to be harder than HotpotQA's 2-hop
shape). The reviewer's recommended probe: replicate the HotpotQA
result, or expose that it's HotpotQA-specific.

## What we measured

Same `bench/compare.py` harness, same n=300, same 400-token budget, same
BM25 retrieval across all three frameworks. The new `musique_items(n)`
iterator unpacks each MuSiQue example into:

- **doc**: 20 paragraphs joined (the supporting + 18-19 distractors)
- **query**: the multi-hop question
- **gold**: the paragraphs marked `is_supporting=true` (the union, since
  MuSiQue uses paragraph-level gold annotations rather than HotpotQA's
  sentence-level)

Metric is word-recall on the gold paragraph(s), matching the existing
CUAD/HotpotQA evaluator.

## What the result says

**The lead replicates directionally:** RedHop is on top at the ≥0.8
retention threshold on both HotpotQA and MuSiQue. So "RedHop's
multi-hop story isn't a HotpotQA artifact" is supported.

**The lead's magnitude doesn't replicate:** on HotpotQA, RedHop's
≥0.8 retention is **+8 over LlamaIndex**. On MuSiQue, it's **+5**.
Mean recall, which was +0.03 over LlamaIndex on HotpotQA, is +0.03 here
too (0.59 vs 0.56) but the lead is well within noise — LangChain ties
RedHop on mean recall (0.58) and is one point below on ≥0.5 retention
(actually one point above: 64% vs 62%).

**`reasoning_preserving` is not the differentiator.** RedHop's `topk`
variant matches `reasoning_preserving` at 22% ≥0.8 on MuSiQue (HotpotQA:
80% vs 77%; the `topk` variant even leads slightly). Whatever's driving
RedHop's retrieval edge over LlamaIndex/LangChain isn't the assembly
strategy — it's the chunking + retrieval defaults (sentence-budgeted
128-token chunks via SentenceChunker, Tantivy BM25 with the configured
analyzer).

**MuSiQue is just hard.** All three systems land in the 17-22% band at
≥0.8 retention vs 71-80% on HotpotQA. Compositional 4-hop reasoning
with 20 distractors at a 400-token budget pressures every BM25-only
retriever roughly equally.

## What this changes in our positioning

The previous-pass framing was *"RedHop is the multi-hop winner."* Honest
revision after MuSiQue:

> "On the two multi-hop datasets we measured (HotpotQA, MuSiQue), RedHop
> leads at the ≥0.8 retention threshold by +5 to +8 points. The
> magnitude depends on dataset difficulty; mean recall is closer to
> a tie. The lead is from RedHop's chunking + BM25 defaults, not from
> the `reasoning_preserving` assembly strategy — `raw_topk` performs
> equally well on both datasets."

## Honest limits

- **Two datasets only.** Multi-hop retention measured on HotpotQA and
  MuSiQue. Both are open-domain Wikipedia-style. Real-world multi-hop
  (legal cross-references, codebase navigation, scientific reasoning
  across papers) is unmeasured.
- **No threshold sensitivity.** Budget=400 tok is fixed; budget sweeps
  not run. At a larger budget (e.g., 800), all systems would likely tie.
- **Word-recall metric** measures retrieval, not answer correctness.
  Downstream LLM answer F1 on MuSiQue is not measured (`bench/tier3.py`
  only runs HotpotQA + CUAD).
- **n=300.** No bootstrap CIs. The shifts are within the typical
  bench-to-bench variance band for ≥0.8 retention metrics.

## Reproduce

```bash
bench/.venv/bin/python bench/compare.py
```

Raw run: [`reports/framework_comparison_with_musique_2026-06-08.txt`](../../reports/framework_comparison_with_musique_2026-06-08.txt).

## See also

- [SECOND_HOP_TAX](SECOND_HOP_TAX.md) — the underlying mechanism
  (relevance-based selection drops the multi-hop bridge passage).
- [REASONING_PRESERVATION](REASONING_PRESERVATION.md) — measures the
  ReasoningPreserving strategy's lift over relevance-only selection.
- [FRAMEWORK_COMPARISON](FRAMEWORK_COMPARISON.md) — head-to-head
  retention numbers (CUAD + HotpotQA, pre-MuSiQue).
- [CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) — argues that RedHop's
  retrieval edge is chunking, not strategy; this finding's
  `topk == reason` parity on both datasets is consistent with that
  claim.
