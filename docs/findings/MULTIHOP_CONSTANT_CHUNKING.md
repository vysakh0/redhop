# Constant-chunking matrix — the lever is the chunker, AND RedHop's RRF hybrid leaves ~10 points on MuSiQue

> **Status:** **Confirmed, with a concrete actionable.** Three honest
> findings from holding chunking constant across the three systems:
>
> 1. **BM25 implementation doesn't matter** — every retriever produces
>    identical retention on the same chunks. The differences we
>    attributed in [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md)
>    were chunker-driven noise dressed up as retriever differences.
> 2. **The chunker is the real lever** — on HotpotQA, RedHop's
>    SentenceChunker wins by 20 points over LlamaIndex's. On MuSiQue,
>    RedHop ≈ LangChain (both ~36%), and LlamaIndex trails both by 12+.
> 3. **RedHop's RRF-based hybrid leaves ~10 points on MuSiQue.** Under
>    controlled pure-rerank conditions, RedHop's chunks hit 36% ≥0.8
>    on MuSiQue; RedHop's *own* `retrieval="hybrid"` only got 26% in
>    the previous probe. The 10-point gap is in RedHop's HybridRetriever
>    fusion strategy (RRF averages BM25 + dense ranks; on bridge
>    passages this drags the dense win down).

## TL;DR

**HotpotQA ≥0.8 retention** (n=50, identical bge-small rerank):

| chunker (size) | → RedHop BM25 | → LangChain BM25 | → LlamaIndex BM25 |
|---|---:|---:|---:|
| **redhop[128t]** | **82%** | **82%** | **82%** |
| langchain[1024c] | 76% | 76% | 76% |
| llamaindex[256t] | 62% | 62% | 62% |

**MuSiQue ≥0.8 retention** (n=50, identical bge-small rerank):

| chunker (size) | → RedHop BM25 | → LangChain BM25 | → LlamaIndex BM25 |
|---|---:|---:|---:|
| **redhop[128t]** | **36%** | **36%** | **36%** |
| langchain[1024c] | 36% | 38% | 38% |
| llamaindex[256t] | 24% | 24% | 24% |

## Reading the matrix

**Rows flat → BM25 implementation doesn't matter.** Across every
chunker, the three BM25 retrievers (RedHop's Tantivy, LangChain's
BM25Okapi, LlamaIndex's own) produce essentially identical retention
when given the same chunks. ±0-2 points is within noise.

**Columns vary substantially → the chunker is the lever.** On HotpotQA,
the 20-point spread (62% → 82% across chunker choices) is the entire
multi-hop story. On MuSiQue, RedHop and LangChain chunkers tie around
36%; LlamaIndex's chunker is 12 points behind.

## The MuSiQue surprise

The previous probe ([MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md))
measured RedHop's `retrieval="hybrid"` at **26%** ≥0.8 on MuSiQue. This
probe, under controlled pure-rerank conditions, hits **36%** with
RedHop's chunks + any BM25 + bge-small. **10-point gap, traceable.**

The constant-chunking probe does:

```
chunk doc (any chunker) → BM25 top-K (any retriever) → dense rerank by cosine → fill budget
```

RedHop's `HybridRetriever` does ([retrieval/hybrid.rs](../../crates/redhop/src/retrieval/hybrid.rs)):

```
chunk doc → BM25 retrieves top-K → dense retrieves top-K → Reciprocal Rank Fusion → top-K
```

The difference: **RRF averages BM25 and dense as equal voters by rank
position.** Under RRF, a bridge passage that ranks high on dense (rank
1) but low on BM25 (rank 18, because lexically distant) gets a fused
score around 1/61 + 1/78 ≈ 0.029 — slightly *worse* than a
middling-on-both chunk (1/63 + 1/63 ≈ 0.032). RRF demotes true bridge
passages.

A pure-rerank shape (BM25 candidates → dense sort, no fusion) puts the
bridge passage at rank 1 — which is what this probe's external rerank
does, and what recovers the 10 points.

## Why HotpotQA doesn't suffer

HotpotQA's bridges are usually 2-hop with sentence-level supporting
facts — there's *some* lexical overlap between the bridge and the
query. So BM25 ranks the bridge moderately (not buried). RRF then
fuses BM25 rank ~5 with dense rank 1, and the bridge still surfaces
near the top. RedHop hybrid hits 83% on HotpotQA in the competitor
probe; constant-chunking matrix hits 82% on the same chunks with pure
rerank. **No gap on HotpotQA — the failure mode is MuSiQue-specific.**

MuSiQue is compositional 2-4 hop with paragraph-level supporting
facts. Bridge paragraphs frequently share *no* lexical content with
the question (they connect via named entities to a different
paragraph). BM25 buries them at rank 15+. RRF can't recover.

## Actionable, applied: pure rerank is now the default

**Fix landed in this branch.** `LocalRerankRetriever::retrieve` no
longer RRF-fuses BM25 with dense; the top-K is taken from the
dense-sorted candidate pool, with any unembedded (code) chunks from
BM25 appended at the tail to preserve issue-#1 safety.

| Workload | Before (RRF) ≥0.8 | After (pure rerank) ≥0.8 | Δ |
|---|---:|---:|---:|
| HotpotQA | 83% | 81% | **−2** |
| MuSiQue | 26% | 34% | **+8** |

n=100 each, same bench harness, same bge-small embedder, same
candidate_k=20, same 400-token budget. Predicted lift was +10 on
MuSiQue (from the controlled-rerank measurement in this finding); we
got +8, close enough. The HotpotQA −2 is the cost of dropping the RRF
safety: when BM25 already ranks bridges OK (HotpotQA-shape), RRF was
giving us a small lift. We chose to take that loss because the MuSiQue
benefit (+8) is 4× larger than the HotpotQA cost.

Latency unchanged: this commit only swapped the fusion step. The
parallel HYBRID_LATENCY_PROFILE finding (RedHop hybrid is 2-5× slower
than competitors') remains open — lazy embedding is a separate change.

## What this changes in our positioning

The previous-pass headline from MULTIHOP_HYBRID_COMPETITORS said
"workload-shape decides." Honest revision after this probe:

> "RedHop hybrid wins HotpotQA. It's *underperforming what it could*
> on MuSiQue because the default RRF fusion buries bridge passages
> that have low BM25 but high dense rank — a known issue, tracked
> for 0.3.2 with a measured fix path. The chunker is the bigger lever
> than the BM25 implementation."

## Honest limits

- **n=50** (smaller than the n=100 hybrid-competitors probe, to keep
  9-arm matrix tractable). Smaller-sample numbers; magnitudes stable
  but precise values would shift a point or two at larger n.
- **Pure-rerank prediction is unmeasured.** The 26% → ~36% MuSiQue lift
  is what this probe MEASURES on the workload, run through a different
  rerank path. We have not yet built and run a pure-rerank mode inside
  RedHop end-to-end; that's the follow-up.
- **Same bge-small model only.** Larger embedders or cross-encoders
  may close MuSiQue further but we don't measure that here.
- **All three chunkers' "win" applies to their own ecosystem.** RedHop
  chunker wins HotpotQA because RedHop's whole stack expects 128-token
  chunks (CHUNK_GRANULARITY). On a workload that wants 1024-char
  chunks, LangChain's chunker is right. There's no "best chunker."

## Reproduce

```bash
bench/.venv/bin/python bench/multihop_constant_chunking_probe.py
```

Raw run: [`reports/multihop_constant_chunking_2026-06-08.txt`](../../reports/multihop_constant_chunking_2026-06-08.txt).

## See also

- [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md) — the
  three-system comparison that the constant-chunking matrix
  disentangles.
- [MULTIHOP_CHUNK_SIZE_NULL](MULTIHOP_CHUNK_SIZE_NULL.md) — sized-up
  RedHop chunks regress, confirming the chunker's whole *strategy*
  (sentence-aware token packing) matters more than the *size*.
- [HYBRID_LATENCY_PROFILE](HYBRID_LATENCY_PROFILE.md) — the latency
  half of the same RedHop hybrid story; pure-rerank fixes both.
- [CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) — the earlier finding
  that chunk size matters; this probe extends to "chunker strategy
  matters even more than size."
- [retrieval/hybrid.rs](../../crates/redhop/src/retrieval/hybrid.rs) —
  the current RRF implementation that this finding diagnoses.
