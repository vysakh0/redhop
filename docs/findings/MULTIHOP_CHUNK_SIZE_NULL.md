# Chunk-size sweep on multi-hop — the MuSiQue gap is NOT chunking

> **Status:** **Null result, falsifies the leading hypothesis.** The
> reigning theory from [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md)
> was that RedHop's MuSiQue underperformance vs LangChain hybrid
> (26% vs 39% ≥0.8) came from RedHop's 128-token chunks fragmenting
> compositional bridge passages. Direct test (chunk_size ∈ {128, 256,
> 384, 512}): **bigger chunks make RedHop worse, not better**, on
> both HotpotQA and MuSiQue. The MuSiQue gap is somewhere else in
> the stack.

## TL;DR

| MuSiQue ≥0.8, RedHop hybrid | chunks/doc | retention | p50 ms |
|---|---:|---:|---:|
| chunk_size=128 (default) | 16.1 | **26%** | 426 |
| chunk_size=256 | 7.0 | 14% | 430 |
| chunk_size=384 | 4.5 | 22% | 432 |
| chunk_size=512 | 3.5 | 7% | 380 |

| HotpotQA ≥0.8, RedHop hybrid | chunks/doc | retention | p50 ms |
|---|---:|---:|---:|
| chunk_size=128 (default) | 10.2 | **83%** | 266 |
| chunk_size=256 | 4.6 | 61% | 242 |
| chunk_size=384 | 3.1 | 61% | 252 |
| chunk_size=512 | 2.3 | 17% | 198 |

Bigger chunks regress retention on both datasets. The 128-token
default is well-tuned for RedHop's whole pipeline.

## What we expected

LangChain's hybrid wins MuSiQue by 13 points (39% vs RedHop's 26%
≥0.8). LangChain chunks via `RecursiveCharacterTextSplitter(chunk_size=1024
chars)` ≈ ~256 tokens. The hypothesis was that RedHop's 128-token chunks
split compositional 2-4 hop bridge passages too finely, and that
matching LangChain's chunk size would close the gap.

## What we got

The opposite. **RedHop with chunk_size=256 lifts only to 14% ≥0.8 on
MuSiQue — 12 points below default and worse than `langchain` BM25
baseline (22%).** Same pattern on HotpotQA: 256 drops 22 points
(83% → 61%). At chunk_size=512 both datasets fall off a cliff
(HotpotQA 17%, MuSiQue 7%).

The cliff at 512 has a likely mechanism: when chunks/doc drops to
~2-3, a single retrieved chunk overflows the 400-token budget,
filling the context with one chunk and dropping the gold-bearing
chunks that live elsewhere. This is the budget-vs-coverage tradeoff
[CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) named — too few, too big
loses.

## Why the smaller-is-better pattern holds across budgets

Mechanism: with 16 chunks per MuSiQue doc at 128 tokens, BM25 can
*select* among them — pull the 2-3 most query-relevant 128-token
chunks rather than 1-2 256-token chunks. The dense reranker then
has 16 candidates to rerank vs 7. More candidates means more chances
to surface the bridge passage. This compounds with the budget: 16
chunks × ~128 tokens caps at 2048 tokens of possible context to
choose from; at 7 × 256 only ~1800.

The chunk_size=128 default isn't an arbitrary number — it's RedHop's
chunker's sweet spot for the SentenceChunker + Tantivy BM25 + 400-token
budget combination. Changing one of those three (e.g., switching to a
different chunker, or to a larger budget) might invert this.

## So why does LangChain hybrid still win MuSiQue?

The MuSiQue gap (RedHop hybrid 26% vs LangChain hybrid 39%) is **not**
caused by chunk size. The remaining candidates:

1. **Chunker strategy** (not size): LangChain's
   RecursiveCharacterTextSplitter splits on character boundaries at
   typical separators (`\n\n`, `\n`, `. `, ` `); RedHop's
   SentenceChunker is sentence-aware and packs to a token budget.
   The character-vs-sentence boundary choice could yield more
   coherent multi-paragraph windows for compositional reasoning.
2. **BM25 implementation**: RedHop uses Tantivy with English
   stemming; LangChain uses BM25Okapi (rank_bm25); LlamaIndex has
   its own. Stemming + tokenization details differ.
3. **Chunk overlap**: RedHop's `chunk_overlap` defaults to 1 sentence;
   LangChain's `RecursiveCharacterTextSplitter` defaults to 40 chars
   in the bench config. Different overlap profiles → different
   chances of catching a bridge in a single chunk.
4. **Coupling**: a specific (chunker, BM25) pair might cohere in
   ways that swapping one breaks.

The next probe (`bench/multihop_constant_chunking_probe.py`) holds
chunking constant and varies only the BM25 retriever — which should
narrow the remaining candidates.

## What this changes

- **RedHop's 128-token default isn't a tuning bug** — sweeping doesn't
  improve it. We can stop second-guessing the chunk_size choice on
  multi-hop workloads.
- **The "RedHop loses MuSiQue because of chunking" line in the
  competitor finding overstates** — it's *some part of* RedHop's stack,
  but not the chunk *size*. Updated MULTIHOP_HYBRID_COMPETITORS to
  reflect the null result on chunk size and point at the constant-
  chunking probe as the next step.

## Honest limits

- **n=100.** Same caveat as the parent probes.
- **One chunker tested** (RedHop's SentenceChunker at varying sizes).
  Other chunker *strategies* not measured here — that's the
  constant-chunking probe's job.
- **Fixed budget** (400 tok) and **fixed candidate_k** (20). At a
  larger budget the bigger-chunks-collapse pattern might attenuate.

## Reproduce

```bash
bench/.venv/bin/python bench/multihop_chunk_size_sweep.py
```

Raw run: [`reports/multihop_chunk_size_sweep_2026-06-08.txt`](../../reports/multihop_chunk_size_sweep_2026-06-08.txt).

## See also

- [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md) — the
  measurement that motivated this probe.
- [CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) — earlier finding that
  chunk size matters more than strategy. This one extends that to
  hybrid + multi-hop: the lever is still chunk size, but the
  optimum is RedHop's existing default, not a larger one.
