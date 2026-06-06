# Chunk-boundary fragmentation on CUAD — NULL RESULT (closes an open hypothesis from CUAD_RECALL_GAP)

> **Status:** **Null result / hypothesis falsified.** The "LlamaIndex
> wins CUAD because of sentence-aware chunking" hypothesis listed as
> "Untested" in [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) is falsified.
> RedHop's default chunker (`SentenceChunker`) is **not** fragmenting
> CUAD gold answer spans across chunk boundaries: **100% of gold spans
> are already contained in a single chunk at ≥0.8 coverage; 99.7% at
> ≥0.95**. The remaining 4-point CUAD gap to LlamaIndex on the raw
> template query is *not* chunking.
>
> **What this reframes:** the path from "all gold words exist in some
> chunk of the document" (100%) to "all gold words present in the
> assembled context" (82%) is closed by **retrieval ranking + budget
> pressure**, not by improving the chunker. That's the lever to pull
> next if anyone wants to push past 88% (with template stripping) on
> CUAD specifically.

## Question

[CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) closed the apples-to-apples 4-point
deficit to LlamaIndex by stripping CUAD's fixed 24-word query template
(82% → 88%, +2 over LlamaIndex). But the action items section left one
hypothesis explicitly open:

> "Open a future research question: what is LlamaIndex doing in their
> chunker that gives the 4-point CUAD edge? Hypothesis: sentence-
> boundary chunking that aligns with legal-clause structure. Untested."

The hypothesis was based on LlamaIndex defaulting to
`SentenceSplitter(chunk_size=256, chunk_overlap=0)`. The intuition:
sentence-aware chunking would put clause boundaries on chunk boundaries,
which would put gold-answer spans cleanly inside individual chunks,
which would mean the gold-bearing chunk has full gold-word coverage and
is more likely to retrieve.

When investigated, the premise was already false: RedHop's default
`Document::from_text_with(...)` wires up `SentenceChunker` —
sentence-aware, token-budgeted (target=128, max=256,
overlap_sentences=0). So the gap is **not** "naive token-window vs
sentence-aware splitting." Both runtimes are sentence-aware.

The reframed question: **does our specific sentence splitter
(`unicode-segmentation`'s UAX-#29-based
`split_sentence_bound_indices`) fragment CUAD gold spans across chunk
boundaries in ways an alternative splitter wouldn't?**

## Diagnostic, not a fix

Rather than build a new chunker speculatively and measure end-to-end
retention, this finding ships a **diagnostic that tests the hypothesis
directly**. The harness lives at
[`crates/examples/examples/cuad_chunk_fragmentation.rs`](../../crates/examples/examples/cuad_chunk_fragmentation.rs).

For each CUAD gold answer span:

1. Tokenize the span into a set of unique content words (same `words()`
   helper used by `bench/compare.py` and the other CUAD harnesses, so
   the metric is apples-to-apples with `CUAD_RECALL_GAP`).
2. For every chunk in the source document, compute the chunk's recall
   of the span: `|chunk_words ∩ gold_words| / |gold_words|`.
3. Sort chunks by recall descending. The **primary** chunk is the
   single best — top-1. The top-1 coverage tells us how much of the
   gold span fits in one chunk.

If chunking were fragmenting spans, top-1 coverage would be low
(< 0.5 on a substantial fraction). If the spans are already inside
single chunks, top-1 coverage will be high (≥ 0.8 on most spans).

Same setup as the other CUAD findings: n=300, `cuad_sample.json`,
default `DocumentConfig`. No retrieval, no LLM — pure
span-vs-chunk-boundary geometry.

## Results

| metric | value |
| ------ | -----:|
| n queries analyzed | 300 |
| mean top-1 chunk coverage of gold span | **0.999** |
| mean top-2 chunk coverage | 0.999 |
| % gold spans with top-1 coverage ≥ 0.8 | **100.0%** |
| % gold spans with top-1 coverage ≥ 0.95 | **99.7%** |
| % gold spans with top-2 coverage ≥ 0.8 | 100.0% |
| mean primary-chunk token count | 113 |
| mean gold-span unique-token count | 24.8 |

Distribution of top-1 chunk coverage on the 300-question slice:

| band | count | share |
| ---- | ----:|:------ |
| ≥ 0.95 | 299 | 99.7% |
| 0.80 – 0.95 | 1 | 0.3% |
| 0.50 – 0.80 | 0 | 0.0% |
| 0.20 – 0.50 | 0 | 0.0% |
| < 0.20 | 0 | 0.0% |

The single span that landed in the 0.80–0.95 band is well above the
≥0.8 threshold `bench/compare.py` measures against. Even on the worst
case in the slice, the chunker isn't the problem.

## Mechanism: gold spans are short, chunks are larger

The numbers tell the story plainly. Mean gold-span unique-token count
is **24.8**; mean primary-chunk token count is **113**. The gold span
is roughly *one-fifth* the size of the chunk that holds it. There is
ample room for any natural sentence-aware splitter to keep the entire
span inside one chunk.

The 4-point gap to LlamaIndex on `bench/compare.py`'s ≥0.8 retention
must therefore live at one of the *next* stages:

- **BM25 retrieval ranking.** The chunk containing the gold span
  exists in the index with full gold-word coverage; it just isn't
  always being scored high enough to land inside `candidate_k=40` or
  the assembly budget.
- **Budget pressure during assembly.** The gold-bearing chunk gets
  retrieved but trimmed; another chunk takes its budget.
- **Score noise from boilerplate dilution.** Already mechanism-known
  and closeable; see [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md). The
  template strip moves 82% → 88% because it raises the relative score
  of the gold-bearing chunk, not because it changes the chunks.

## What this rules in / out

**Rules out:**

- Building a sentence-aware chunker variant as a CUAD recall lift.
  We already have one; it's already keeping gold spans inside chunks.
- Tuning `target_tokens` / `max_tokens` / `overlap_sentences` as a
  CUAD-specific knob. The chunker's outputs are not where the spans
  are escaping.
- The LlamaIndex-chunker theory of the 4-point CUAD gap.
  `SentenceSplitter(chunk_size=256, chunk_overlap=0)` does what our
  `SentenceChunker(target=128, max=256, overlap_sentences=0)` already
  does. If LlamaIndex still edges us by 4 on the raw template query,
  the difference is elsewhere.

**Rules in (as the next plausible levers if a future investigator
wants to push past 88% on CUAD specifically):**

- **Retrieval-side fixes** — sub-IDF reweighting of high-frequency
  domain boilerplate (CUAD's per-chunk vocabulary has many low-IDF
  legal-terms; the gold span often has a couple of medium-IDF
  discriminators). This is corpus-side; might be a static analyzer
  enhancement.
- **Budget-aware assembly** — if the gold-bearing chunk reliably has
  high BM25 score but gets squeezed out by other high-scoring chunks
  in the top-K, a more conservative `auto` policy could keep it.
- **Larger candidate_k** on contract-shape workloads — would surface
  the chunk even when its rank is, say, 25–40 instead of 1–10.
- **Reranker on `retrieval="hybrid"`** — semantic rerank over the
  BM25 pool. Cost: latency + a dense model in the loop.

None of these are claimed to work; they're listed as *plausible*
candidates a future probe could test. The same discipline that produced
this null result applies: measure first, ship only on evidence.

## Honest limits

- **One workload, 300 questions.** The mechanism prediction
  (chunker keeps a 25-word span inside a 113-token chunk) generalizes,
  but the magnitudes don't. A workload with very long gold spans
  (multi-paragraph evidence) could behave differently and is not
  tested here.
- **Diagnostic, not benchmark.** We measured chunk-vs-span geometry,
  not end-to-end retention. The conclusion "the chunker isn't the
  gap" is robust because we measured the chunker's *output*; the
  conclusion "you should pull lever X next" is mechanism-predicted,
  not measured. A future positive finding would still need its own
  evidence.
- **No multi-paragraph chunker variant tested.** A paragraph-then-
  sentence two-stage splitter wasn't measured here; the result implies
  it would be a no-op on CUAD, but the prediction is untested.
- **The single 0.80–0.95 outlier** wasn't inspected. With n=300, that
  one case sits well above the ≥0.8 cutoff for the bench's recall
  metric — not investigation-worthy at this scale.

## Reproduce

```bash
cargo run -p redhop-examples --example cuad_chunk_fragmentation --release
```

Runs in well under a second; no models, no embeddings, no retrieval.
Same `cuad_sample.json` as the other CUAD findings.

## What this changes

Nothing in the runtime, no public API. This is a research record —
a piece of evidence that **the 4-point CUAD gap to LlamaIndex on the
raw template query is not chunk-boundary fragmentation**. The
template-strip recipe in [CUAD_RECALL_GAP](CUAD_RECALL_GAP.md) plus
`analyze_query_set` / `drop_template_terms`
([QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md)) plus `evaluate`
([EVALUATE_API](EVALUATE_API.md)) remain the actionable surface.

If anyone (human or AI agent) lands here looking for "how do I push
CUAD past 88%": the chunker is *not* the lever. Read the "Rules in"
list above and pick something testable.
