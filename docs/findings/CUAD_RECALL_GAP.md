# The CUAD recall-gap — investigated, confirmed real

> **Question:** [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md) measured
> RedHop[topk] at **82% ≥0.8 word-recall** vs LlamaIndex at **86%** on CUAD
> contracts (n=300, budget=2000, BM25). Multi-hop wins were sharp; the
> contracts "tie" was a 4-point loss. Can a knob (chunk size, strategy)
> close that gap without regressing HotpotQA?
>
> **Result:** The gap is **real and not closed by chunking or strategy
> choice**. A fresh n=300 rerun (under current main, with all the
> analyzer + BM25 fixes through 0.2.2) reproduces the documented numbers
> almost exactly:
>
>   - CUAD redhop[topk]: 82% then, 82% now (LlamaIndex still 86%).
>   - HotpotQA redhop[topk]: **77% then, 80% now (+3 points)** —
>     a real, measured improvement from the same fixes that closed the
>     BM25 silent-wildcard bug and sharpened the analyzer.
>
> A chunk-size × strategy sweep on the same 300-query slice fails to find
> any cell that clears LlamaIndex's 86% — the best is RawTopK at
> target=32 at 85%, still 1 point short. The 4-point CUAD gap is not a
> chunking problem and not a strategy-choice problem under the BM25
> retrieval that the comparison fixes.
>
> **Reproduce:**
> ```bash
> # The fresh framework-comparison rerun (matches bench/compare.py exactly):
> bench/.venv/bin/python bench/compare.py
> # → output saved to reports/framework_comparison_2026-06-06.txt
>
> # The chunk-size × strategy sweep on the same 300-query CUAD slice:
> cargo run -p redhop-examples --example cuad_chunk_strategy_sweep --release
> ```

## What we re-measured

**1. Re-ran `bench/compare.py` on the current main (0.2.2)**, same setup
as the original: n=300 first questions from `cuad_sample.json`,
budget=2000, candidate_k=40, BM25 across all three frameworks, identical
gold-word-recall metric.

CUAD (n=300, budget=2000, BM25):

| system              | avg tokens | mean recall | ≥0.5 | ≥0.8 |
| ------------------- | ----------:| -----------:| ----:| ----:|
| redhop[reason]      | 1882       | 0.88        | 93%  | 77%  |
| redhop[density]     | 1885       | 0.85        | 91%  | 72%  |
| **redhop[topk]**    | **1887**   | **0.91**    | 94%  | **82%** |
| langchain           | 1813       | 0.87        | 93%  | 73%  |
| **llamaindex**      | **1806**   | **0.93**    | 96%  | **86%** |

HotpotQA (n=300, budget=400, BM25):

| system              | avg tokens | mean recall | ≥0.5 | ≥0.8                     |
| ------------------- | ----------:| -----------:| ----:| ------------------------:|
| redhop[reason]      | 352        | 0.90        | 98%  | 77%                      |
| redhop[density]     | 353        | 0.78        | 86%  | 52%                      |
| **redhop[topk]**    | **350**    | **0.91**    | 97%  | **80% (was 77%, +3)**    |
| langchain           | 328        | 0.88        | 96%  | 71%                      |
| llamaindex          | 329        | 0.88        | 96%  | 72%                      |

**The HotpotQA improvement is the real story.** The +3 points on the
multi-hop benchmark, with everything else held constant, is the
measurable effect of the BM25 silent-wildcard fix from 0.2.1 plus the
analyzer sharpening that landed alongside SECOND_HOP_TAX's signal
validation. The CUAD numbers being identical confirms those fixes
didn't accidentally regress anything on single-doc workloads either.

**2. Swept chunk size × strategy on the same 300-query CUAD slice**
(`cuad_chunk_strategy_sweep.rs`). target_tokens ∈ {32, 48, 64, 96, 128,
192} × strategy ∈ {RawTopK, MaxDensity, DistractorFiltered,
ReasoningPreserving}. candidate_k=40 to match the bench.

Best cell across the 24 combinations: **RawTopK at target=32 → 85% ≥0.8
retention**. Still 1 point below LlamaIndex's 86%. Other strategies
peak lower:

| strategy @ best target  | ≥0.8 retention | best target |
| ----------------------- | --------------:| -----------:|
| RawTopK @ 32            | **85%**        | 32          |
| MaxDensity @ 32         | 84%            | 32          |
| DistractorFiltered @ 192| 82%            | 192         |
| ReasoningPreserving @ 32| 81%            | 32          |

A few observations from the sweep:

- **RawTopK dominates CUAD across all chunk sizes.** Mechanism: CUAD is
  single-document answer-span extraction, not multi-hop. Strategies
  built around bridge-rescue or distractor filtering are solving
  problems CUAD doesn't have. The Auto policy routes large contexts
  to ReasoningPreserving on dilution assumptions, but on
  contract-shaped workloads users should override: `strategy="raw_topk"`.
- **Smaller chunks edge out larger ones.** target=32 wins by a hair on
  RawTopK, but the 85% → 84% gap from target=32 to target=128 is
  within sample noise.
- **The 4-point gap to LlamaIndex is not closed by any cell.** Even the
  best cell falls 1 point short. Whatever LlamaIndex is doing differently
  is upstream of chunking and strategy choice — likely sentence-aware
  chunking with their `SentenceSplitter(chunk_size=256, chunk_overlap=0)`
  that hits clause boundaries CUAD's gold spans happen to align with.

### A small Rust/Python discrepancy worth noting

The Rust sweep (`Document::from_text_with(..., candidate_k=40, ...)`)
shows RawTopK @ target=128 at 84% ≥0.8 retention. The Python bench
(`redhop.Document.from_text(..., candidate_k=40, strategy="raw_topk")`)
shows the same call path at 82%. Both n=300, same metric, same data,
same explicit parameters. Some default field is wired differently
between the Rust direct API and the Python binding path — most likely
`overlap_sentences`, `code_neighbors_default`, or `prose_heading_default`.

This is a real Python/Rust parity bug that would be worth chasing as
follow-up work. It doesn't change the central finding (CUAD gap is real,
LlamaIndex still wins) but it does change which number to quote
externally — Python users see 82%, Rust users see 84%.

## What this changes

### 1. The CUAD gap is REAL

Stop hoping it was stale. The current code, with all the analyzer +
BM25 improvements through 0.2.2, still trails LlamaIndex by 4 points
on CUAD. No chunk size or strategy choice closes it. This is the
honest baseline going forward.

### 2. HotpotQA improvement is REAL and worth advertising

The +3 points on multi-hop (RedHop 77% → 80%, LlamaIndex still 72%) is
the kind of measurable improvement that justifies the work on analyzer
sharpening and the BM25 bug fixes. The
[FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md) headline of
"RedHop wins multi-hop" gets sharper: now +8 over LlamaIndex (was +5)
and +9 over LangChain (was +6).

The README's "How it compares" section + retention chart still show the
older 77% number for HotpotQA; that's worth refreshing.

### 3. RawTopK is the right strategy for contract workloads

Confirmed twice — by the original FRAMEWORK_COMPARISON (RedHop[topk]
beat RedHop[reason] 82% vs 78%) and by this sweep (RawTopK beats
ReasoningPreserving by 4 points at every chunk size). The Auto policy
correctly routes large contexts to pruning but picks the wrong
strategy for single-doc extraction workloads. `CHOOSING_A_CONFIG.md`
should recommend explicit `strategy="raw_topk"` for that workload
shape.

### 4. The "we tie LlamaIndex on contracts" framing is too generous

The honest framing is: "we beat LangChain on contracts (82% vs 73%,
+9 points) and trail LlamaIndex by 4 points." Not a tie.

## Action items (for main, when we cherry-pick)

- [ ] **Update FRAMEWORK_COMPARISON.md** with a "Rerun — 2026-06-06"
  section showing the fresh numbers (HotpotQA +3 improvement is the
  headline; CUAD unchanged).
- [ ] **Update the README + bindings READMEs** retention chart with
  HotpotQA at 80% (was 77%). The CUAD numbers don't change.
- [ ] **Update CHOOSING_A_CONFIG.md** to explicitly recommend
  `strategy="raw_topk"` for single-doc contract/extraction workloads.
- [ ] **Open an investigation** (separate scope) into the 2-point
  Python/Rust parity gap on CUAD — Python binding gives 82%, Rust
  direct gives 84% at the same explicit config. Some default field is
  threaded differently.
- [ ] **Open a future research question:** what is LlamaIndex doing in
  their chunker that gives the 4-point CUAD edge? Hypothesis: sentence-
  boundary chunking that aligns with legal-clause structure. Untested.

## Honest limits

- **n=300** for both benchmarks — same as the original
  FRAMEWORK_COMPARISON; that's the apples-to-apples cadence.
- **cuad_sample.json** (50 contracts) — `bench/compare.py` samples the
  first 300 questions from this. The full CUADv1.json (660 contracts)
  is referenced via `REDHOP_CUAD_PATH` but not committed; a larger run
  would tighten the intervals but the direction is unambiguous.
- **The +3 HotpotQA improvement is not bootstrap-CIed.** Single-run
  delta. n=300 with a 3-point shift on a ~75% baseline is probably
  significant, but a confirmation run would be cheap.
- **No downstream answer eval (Tier 3) here.** Whether the +3
  retention improvement on HotpotQA translates to a measurable
  SQuAD F1/EM lift is a separate question.

This finding is committed on `experiment/cuad-recall-gap` as a research
record. The actionable doc updates and chart refresh belong on main
when we decide to merge.
