# The CUAD recall-gap — investigating the 82%-vs-86% LlamaIndex headline

> **Question:** [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md) measured
> RedHop[topk] at **82% ≥0.8 word-recall** vs LlamaIndex at **86%** on CUAD
> contracts (n=300, budget=2000, BM25). Multi-hop wins were sharp; the
> contracts "tie" was a 4-point loss. Can a knob (chunk size, strategy)
> close that gap without regressing HotpotQA?
>
> **Result:** Two findings, one of which is uncomfortable to write.
> (a) The documented baseline appears to be **stale** — on a 50-contract
> subset under the current main, default chunker + RawTopK gets **85% ≥0.8
> retention** at 1908 tokens, not 82%. (b) Sweeping chunk size on the
> current main shows target=192 + RawTopK at **85.9%** — essentially
> matching LlamaIndex (86%), but only +0.9 over the default (128). The
> "default chunker is optimal" finding from CHUNK_GRANULARITY holds; the
> "we're 4 points behind on CUAD" framing in FRAMEWORK_COMPARISON.md
> needs a refresh.
>
> **Reproduce:**
> ```bash
> cargo run -p redhop-examples --example cuad_chunk_strategy_sweep --release
> ```

## Setup

Same harness shape as `eval_cuad_documents.rs` (the existing CUAD eval),
sweeping two dimensions:

- `target_tokens` ∈ {32, 48, 64, 96, 128, 192}  (max = 2×target)
- `strategy` ∈ {RawTopK, MaxDensity, DistractorFiltered, ReasoningPreserving}

Metric: word-recall of the gold answer span against the assembled context
(`span_recall` from the existing harness — robust to chunk boundaries
splitting long clauses).

Sample: **50 contracts** (cuad_sample.json). FRAMEWORK_COMPARISON.md used
n=300, so the comparison is qualitative; magnitudes within ~3 points are
sample-noise.

## Numbers

### RawTopK (the headline strategy for CUAD)

| target_tokens | mean recall | ≥0.5 | ≥0.8       | avg tokens used |
| ------------- | -----------:| ----:|-----------:| ---------------:|
| 32            | 0.877       | 93%  | 77%        | 1199            |
| 48            | 0.889       | 94%  | 80%        | 1398            |
| 64            | 0.894       | 94%  | 81%        | 1582            |
| 96            | 0.905       | 95%  | 83%        | 1836            |
| **128**       | **0.918**   | 96%  | **85%**    | 1908            |
| **192**       | **0.917**   | 95%  | **85.9%**  | 1887            |

### Other strategies (for completeness)

| strategy @ best target | ≥0.8 retention | best target |
| ---------------------- | --------------:| -----------:|
| RawTopK                | **85.9%**      | 192         |
| MaxDensity             | 83%            | 96          |
| DistractorFiltered     | 84%            | 192         |
| ReasoningPreserving    | 82%            | 192         |

### Comparison context

| system                              | ≥0.8 retention | avg tokens used |
| ----------------------------------- | --------------:| ---------------:|
| LlamaIndex (FRAMEWORK_COMPARISON)   | 86%            | 1806            |
| **RedHop default (128) + RawTopK** (this sweep, n=50) | **85%** | 1908 |
| RedHop target=192 + RawTopK (this sweep, n=50) | **85.9%** | 1887 |
| LangChain (FRAMEWORK_COMPARISON)    | 73%            | 1813            |
| RedHop[topk] (FRAMEWORK_COMPARISON, n=300) | 82%   | 1894            |

## What this changes

### 1. The 82% baseline appears stale

This sweep's default-config RedHop[topk] at 85% is 3 points above the
documented number. Possible explanations:

- **Sample-size difference** (n=50 vs n=300). With CUAD's heterogeneity
  (contracts vary widely in length/clause structure), a 50-contract slice
  can plausibly swing ±3 points from the n=300 mean.
- **Real improvement** between when FRAMEWORK_COMPARISON.md was generated
  and current main — the analyzer was sharpened (Snowball stemming
  validated in SECOND_HOP_TAX), the BM25 silent-wildcard bug was fixed
  in 0.2.1, default chunker was already at 128 by then. Any of these
  could explain a couple of points.
- Both.

The honest answer needs an n=300 rerun under current main to
re-establish the baseline. For now: **don't quote 82% as the current
RedHop CUAD number; quote 85% (current sweep, n=50) with the n=50
caveat, or rerun the full framework comparison.**

### 2. Chunking isn't really the gap

The chunk-size sweep shows the classic U-shape: 32-tokens too granular
(0.877 recall), 192-tokens fine, plateau around 96-192. The default
(128) is approximately optimal; target=192 buys +0.9 points which is
within sample noise at n=50. CHUNK_GRANULARITY's central finding
("128-token default is robust") holds on a second corpus.

### 3. RawTopK is the right strategy choice for CUAD

Across all chunk sizes, RawTopK dominates on CUAD:

- RawTopK @ 192:                85.9% ≥0.8
- DistractorFiltered @ 192:     84.0% ≥0.8
- MaxDensity @ 96:              83.0% ≥0.8
- ReasoningPreserving @ 192:    82.0% ≥0.8

This is mechanism-consistent: CUAD is single-document answer-span
extraction, not multi-hop. ReasoningPreserving's bridge-rescue is
solving a problem CUAD doesn't have — it's preserving links to chunks
that don't matter for span retention. The Auto policy correctly routes
CUAD's large contexts to ReasoningPreserving on the assumption that
dilution is the threat, but on CUAD the bridge-rescue mechanism doesn't
add value (no bridge entities; one document; single-hop).

This is genuinely actionable: **on a CUAD-shaped workload (single-doc
contracts with answer-span queries), users should pin
`strategy = "raw_topk"` rather than `"auto"`**. The Auto default isn't
wrong — it's the right hedge over an unknown workload — but a known
contract workload should override.

## What this doesn't change

- The "RedHop beats LangChain on contracts" headline from
  FRAMEWORK_COMPARISON (73% LangChain vs 85% RedHop). LangChain's gap
  is much larger than LlamaIndex's, and that hasn't moved.
- The "RedHop wins multi-hop" headline from FRAMEWORK_COMPARISON
  (HotpotQA 77% vs LangChain 71% vs LlamaIndex 72%). That story stands.
- The default chunker calibration from CHUNK_GRANULARITY. Confirmed
  again on a second corpus (CUAD): target=128 is robust, peak±20%.

## Honest limits

- **n=50** vs FRAMEWORK_COMPARISON's n=300. Direction is clear; exact
  point estimates are noisy.
- **No HotpotQA cross-check** here. The CHUNK_GRANULARITY sweep already
  showed target=128 optimal on HotpotQA; moving to target=192 to chase
  +0.9 CUAD points might give back HotpotQA. Not worth it.
- **Word-recall metric** is exactly what FRAMEWORK_COMPARISON used;
  apples-to-apples on that axis.
- **No downstream answer eval (Tier 3).** The Tier-1 retention story
  changes; whether that translates to better SQuAD F1/EM is a separate
  question and would need LLM judges to answer.

## Action items

- [ ] **Update FRAMEWORK_COMPARISON.md** with a "Status update — 2026-06-06"
  noting the n=50 spot-check shows the gap is ~1 point not 4, and either
  rerun n=300 or add the caveat.
- [ ] **Update CHOOSING_A_CONFIG.md** to recommend explicit
  `strategy = "raw_topk"` for single-doc contract / extraction workloads.
- [ ] Recommend new contributors run `cargo run -p redhop-examples
  --example cuad_chunk_strategy_sweep --release` if they want a quick
  CUAD-shape calibration check on their own data.

This finding is committed on `experiment/cuad-recall-gap` as a research
record. The actionable changes (FRAMEWORK_COMPARISON refresh, CHOOSING
config note) belong on main when we decide to merge.
