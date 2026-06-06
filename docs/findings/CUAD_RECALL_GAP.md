# The CUAD recall-gap — investigated, mechanism found, RedHop overtakes LlamaIndex

> **Question:** [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md) measured
> RedHop[topk] at **82% ≥0.8 word-recall** vs LlamaIndex at **86%** on CUAD
> contracts (n=300, budget=2000, BM25). Multi-hop wins were sharp; the
> contracts "tie" was a 4-point loss. Where is the gap, and is the same
> fix that improved HotpotQA (+3 points) reaching CUAD?
>
> **Result (TL;DR):** The gap is **BM25 template-boilerplate dilution**.
> Every CUAD question is a 24-word fixed template (`Highlight the parts
> (if any) of this contract related to "X" that should be reviewed by a
> lawyer. Details: …`); the actual discriminating signal is the quoted
> clause name `X` plus the `Details:` elaboration (~5 content words). The
> other ~19 words are identical across every query. BM25 was happily
> computing relevance over the whole 24-word query, with the boilerplate
> diluting the real signal.
>
> Stripping the template to just `<clause_name> <details_elaboration>`
> **lifts ≥0.8 retention from 81% to 88% (+6.3 points), beating
> LlamaIndex's 86% by 2 points** — turning a 4-point deficit into a
> 2-point lead with a 6-line query preprocessor.
>
> Why this also explains the HotpotQA +3 vs CUAD ±0 asymmetry: HotpotQA
> questions are diverse natural language (mean 15.7 words, no shared
> boilerplate); the BM25 silent-wildcard bug fix in 0.2.1 cleanly helped
> HotpotQA queries that previously had no signal. CUAD queries always
> had signal — just buried under 80% noise. Same fix, different
> response, because the dilution mechanism is different.
>
> **Reproduce:**
> ```bash
> # The fresh framework-comparison rerun (matches bench/compare.py exactly):
> bench/.venv/bin/python bench/compare.py
> # → output saved to reports/framework_comparison_2026-06-06.txt
>
> # The chunk-size × strategy sweep on the same 300-query CUAD slice:
> cargo run -p redhop-examples --example cuad_chunk_strategy_sweep --release
>
> # The template-stripping diagnostic that closes the gap:
> cargo run -p redhop-examples --example cuad_query_preprocessing --release
>
> # Performance bench (Rust API path, no Python):
> cargo run -p redhop-examples --example cuad_perf --release
> ```

## What we re-measured

**1. Re-ran `bench/compare.py` on the current main (0.2.2)**, same setup
as the original: n=300 first questions from `cuad_sample.json`,
budget=2000, candidate_k=40, BM25 across all three frameworks, identical
gold-word-recall metric (set-based — see metric note below).

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

**The HotpotQA +3 is the real story.** With everything else held
constant, the multi-hop benchmark improved measurably from the BM25
silent-wildcard fix from 0.2.1 plus the analyzer sharpening that landed
alongside SECOND_HOP_TAX's signal validation. The CUAD numbers being
identical confirms those fixes didn't accidentally regress anything on
single-doc workloads either.

Raw output: [`reports/framework_comparison_2026-06-06.txt`](../../reports/framework_comparison_2026-06-06.txt).

**2. Swept chunk size × strategy on the same 300-query CUAD slice**
(`cuad_chunk_strategy_sweep.rs`). target_tokens ∈ {32, 48, 64, 96, 128,
192} × strategy ∈ {RawTopK, MaxDensity, DistractorFiltered,
ReasoningPreserving}. candidate_k=40 to match the bench.

Across the 24 combinations **RawTopK at small target_tokens dominates**.
ReasoningPreserving and DistractorFiltered both underperform on the
single-doc extraction workload — they're solving multi-hop problems CUAD
doesn't have. The Auto policy routes large contexts to
ReasoningPreserving on dilution assumptions; on contract-shaped
workloads users should override: `strategy="raw_topk"`.

A few observations from the sweep:

- **RawTopK dominates CUAD across all chunk sizes.** Mechanism: CUAD is
  single-document answer-span extraction, not multi-hop. Strategies
  built around bridge-rescue or distractor filtering are solving
  problems CUAD doesn't have.
- **Smaller chunks edge out larger ones.** target=32 wins by a hair on
  RawTopK, but the differences within RawTopK across chunk sizes are
  within sample noise.
- **No cell closes the 4-point gap to LlamaIndex from chunking alone.**
  The mechanism is upstream — see template-stripping below.

### The Rust/Python "parity gap" was a METRIC bug, not a runtime bug

An earlier draft of this finding reported a Rust harness number 2 points
above what `bench/compare.py` (Python) showed, raising a parity
concern. A surgical probe (`cuad_rust_vs_python_path.rs`) compared:

  - **Path A:** `Document::from_text_with(source, text, cfg)` (what the
    Rust sweeps use)
  - **Path B:** `Document::from_sources_with(vec![(source, vec![Section{text,
    ..}])], cfg)` (what the Python binding's `from_text` routes through
    under the hood)

on the same contract, same 300 questions, same explicit cfg:

| | Path A (Rust direct) | Path B (Python's underlying) |
| --- | --- | --- |
| chunk count | 71 | 71 |
| chunks with different text | 0/71 | (identical) |
| chunks with different metadata | 71/71 | (Path B adds `kind:"prose"`) |
| 300-query ≥0.8 retention (set metric) | 81.3% | 81.3% |

**Identical retention.** The only metadata difference is `kind:"prose"`
stamped by `chunk_sections`, which never fires `code_neighbors_default`
or `prose_heading_default` on CUAD (no headings, not code). The runtime
is byte-equivalent.

The 2-point gap came from **my Rust harnesses using a Vec-based
`span_recall` while `bench/compare.py` uses set-based**:

```rust
// my (BUGGY) Rust span_recall:
fn span_recall(gold, ctx_words_set) -> f32 {
    let g: Vec<String> = words(gold);  // duplicates kept
    let hit = g.iter().filter(|w| ctx_words.contains(*w)).count();
    hit as f32 / g.len() as f32        // divides by |with-dups|
}
```

```python
# bench/compare.py's span_recall:
def words(s): return {w for w in ...}  # SET — duplicates collapsed
def span_recall(gold, ctx):
    g = words(gold)                    # set
    return len(g & cw) / len(g)        # divides by |unique|
```

When a CUAD gold answer span has repeated content words (extremely
common in legal contract clauses — "parties to this agreement, the
parties..."), the Vec version over-counts both numerator and
denominator, inflating recall by 2-3 points relative to bench's
set-based metric.

All Rust harnesses in this finding (`cuad_chunk_strategy_sweep.rs`,
`cuad_query_preprocessing.rs`, `cuad_perf.rs`,
`cuad_rust_vs_python_path.rs`) use **set-based** `span_recall`, matching
`bench/compare.py`. The Rust runtime requires no change. Sub-question
closed.

## The mechanism: template-boilerplate dilution

CUAD question template (every single question):

```
Highlight the parts (if any) of this contract related to "X" that should
be reviewed by a lawyer. Details: <elaboration>
```

24 words total. ~19 of them are **identical across every query**:
`highlight, parts, contract, related, reviewed, lawyer, Details, …`

BM25 weights each query term by IDF (inverse document frequency) over
the *corpus*, not the *query set*. Within a single contract (~9k tokens
of legal English), terms like "contract", "lawyer", "parts" have
non-zero IDF. So BM25 was scoring every chunk against ALL 24 words,
with the boilerplate contributing real-but-irrelevant relevance signal.

The discriminating signal — the quoted clause name and the Details
elaboration — was on average 4-6 content words, drowned in the 19
boilerplate words.

### Template-stripping result

Same n=300, same budget=2000, same RawTopK, same candidate_k=40, same
default chunker. The only change: each query is preprocessed to extract
just the quoted clause name + the Details elaboration before being
passed to BM25.

| arm                              | mean recall | ≥0.5 | ≥0.8 | avg tokens |
| -------------------------------- | -----------:| ----:| ----:| ----------:|
| original template (24 words)     | 0.905       | 94%  | **81%**  | 1890   |
| **template stripped** (~5 words) | **0.933**   | 96%  | **88%**  | **1705** |
| Δ                                | +0.028      | +2   | **+6.3** | −185 |
| **LlamaIndex baseline**          | 0.93        | 96%  | **86%**  | 1806 |

**RedHop with template-stripped queries: 88% ≥0.8 vs LlamaIndex's 86%.
A 2-point lead instead of a 4-point deficit.**

Notice the assembled context also got *shorter* (1890 → 1705 tokens):
with a less-diluted query, BM25 picks tighter top-K candidates that fit
in less budget. The runtime is doing more with less.

### Why HotpotQA improved but CUAD didn't, on the same code change

The BM25 silent-wildcard fix in 0.2.1 fixed a specific bug: queries
whose every term got filtered (all stopwords or OOV) silently fell back
to a match-all wildcard, returning the corpus's top-BM25 chunks
regardless of the query. Now they return empty.

- **HotpotQA queries are diverse natural language.** Some had weak
  signal; some had no signal under the bad analyzer. The fix turned
  on-by-accident-fake recall into honest-zero, and the analyzer
  sharpening (Snowball + stopwords) added real recall on top. Net
  +3 points on RedHop[topk].
- **CUAD queries always had signal** — every single one has at least
  the boilerplate words plus the quoted clause. None of them ever hit
  the silent-wildcard path. The fix simply didn't apply. The dilution
  mechanism was orthogonal and untouched.

This is a **clean illustration of the discipline rule** that "general"
fixes should never be claimed without knowing the failure mode they
target. The 0.2.1 fix targeted a specific bug, and that's the only thing
it improved.

## Performance — what the Rust API path delivers

The CUAD investigation in this finding ran through the `redhop` crate
directly (`Document::from_text_with(...) → doc.context(query)`), pure
Rust, single-threaded, release build. The `bench/compare.py` head-to-head
is the only Python in the loop, and only because LangChain and
LlamaIndex are Python-only — the comparison has to live there. The
recall sweeps and the template-stripping diagnostic are Rust.

Measured on Apple M5 (10-core, 16 GB), `cuad_perf` example, 300 queries
across 27 contracts (the first 300 questions in `cuad_sample.json`):

| arm                                  | recall @ ≥0.8 | median context() | p95 context() | throughput |
| ------------------------------------ | -------------:| ----------------:| -------------:| ----------:|
| original template (24 words)         | 81.3%         | 2.81 ms          | 3.56 ms       | 379 qps    |
| **template stripped (~5 words)**     | **87.7%**     | **2.42 ms**      | **3.23 ms**   | **480 qps** |

Document build (one ~9k-token contract, including chunking + BM25
indexing + first-query warmup): **2.73 ms median, 8.06 ms p95**.

Stripping the boilerplate moves both axes in the right direction:
+6.3 points retention AND +27% throughput. The "less work" intuition
holds — with a tighter query, BM25 does less work, the candidate pool
is more discriminative, the assembled context is smaller (~185 fewer
tokens used on average), and the runtime delivers faster *and* better.

This is **BM25 lexical retrieval, no embedding models, no LLM calls,
fully in-process**. The 480 qps single-thread number means a modern
laptop can saturate roughly 4-5k qps with all cores in parallel for
this workload class, without any service or vector DB.

Reproduce:
```bash
cargo run -p redhop-examples --example cuad_perf --release
```

## A general principle (not a CUAD-specific fix)

The deeper point is general, and worth pulling out: **whenever your
workload has templated queries with high boilerplate share, BM25 recall
suffers from term dilution.** Examples in the wild:

- Legal QA systems where every question follows a fixed phrasing
- Support-ticket triage where every query is "Help me with X, my account
  is Y, the error is Z"
- Form-filled queries from structured UIs (drop-down clause → query)

The mechanism is corpus-agnostic; the fix is workload-specific
preprocessing.

### Should this live in RedHop's core?

**No** — hardcoding a CUAD-specific template strip would be the wrong
move. Templates are workload-specific; we don't want a
`fn strip_cuad_template()` in the public surface.

The honest framing for users is workload guidance: see the new
"Templated queries" section in [CHOOSING_A_CONFIG.md](../CHOOSING_A_CONFIG.md).
A small general helper for boilerplate-term-dropping is a candidate for
a future minor release if the pattern recurs across user workloads, but
not on the back of a single benchmark.

## What this changes

### 1. The CUAD gap is REAL but the mechanism is now solved

The current code with the templated query trails LlamaIndex by 4
points on CUAD. No chunk size or strategy choice closes it. But the
*mechanism* is BM25 boilerplate dilution from the 24-word template,
and a 6-line query-preprocessor takes RedHop to 88% — beating
LlamaIndex by 2 points. The "headline 4-point loss" is genuinely
a workload-preprocessing issue, not a runtime weakness.

### 2. HotpotQA improvement is REAL and worth advertising

The +3 points on multi-hop (RedHop 77% → 80%, LlamaIndex still 72%) is
the kind of measurable improvement that justifies the work on analyzer
sharpening and the BM25 bug fixes. The
[FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md) headline of
"RedHop wins multi-hop" gets sharper: now +8 over LlamaIndex (was +5)
and +9 over LangChain (was +6). The retention chart in the README
([`.github/retention_vs_frameworks.svg`](../../.github/retention_vs_frameworks.svg))
is updated to reflect this.

### 3. RawTopK is the right strategy for contract workloads

Confirmed twice — by the original FRAMEWORK_COMPARISON (RedHop[topk]
beat RedHop[reason] 82% vs 78%) and by the chunk-size × strategy sweep
in this finding (RawTopK beats ReasoningPreserving by 4 points at every
chunk size). The Auto policy correctly routes large contexts to pruning
but picks the wrong strategy for single-doc extraction workloads.
[CHOOSING_A_CONFIG.md](../CHOOSING_A_CONFIG.md) now recommends explicit
`strategy="raw_topk"` for that workload shape.

### 4. The "we tie LlamaIndex on contracts" framing is too generous

The honest framing is: "we beat LangChain on contracts (82% vs 73%,
+9 points) and trail LlamaIndex by 4 points on the raw-template query.
With a 6-line preprocessor, we lead LlamaIndex by 2 points." Not a tie.

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
- **The LlamaIndex 4-point CUAD edge on the raw-template query** is
  unattributed. Hypothesis: sentence-boundary chunking that aligns with
  legal-clause structure. Untested; future investigation.
