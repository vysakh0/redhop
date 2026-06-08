# Multi-hop hybrid retrieval — RedHop vs LangChain vs LlamaIndex with the same dense model

> **Status:** **Confirmed-mixed; primary cause fixed in 0.3.1.**
> Apples-to-apples six-arm comparison (RedHop / LangChain / LlamaIndex
> × {BM25, BM25+bge-small rerank}) on HotpotQA + MuSiQue, n=100,
> identical budget and candidate_k.
>
> **Original measurement (RRF-fused hybrid):**
>
> - HotpotQA: RedHop hybrid 83%, LangChain 77%, LlamaIndex 67% (no lift).
> - MuSiQue: LangChain 39%, LlamaIndex 31%, **RedHop 26%** (worst).
>
> The MuSiQue gap was traced to RedHop's RRF fusion in
> [`LocalRerankRetriever`](../../crates/redhop/src/retrieval/local_rerank.rs)
> ([MULTIHOP_CONSTANT_CHUNKING](MULTIHOP_CONSTANT_CHUNKING.md) probe).
> The fix landed in this branch (pure rerank now default).
>
> **Re-measured after fix (pure rerank, same probe):**
>
> - HotpotQA: RedHop hybrid **81%**, LangChain 77%, LlamaIndex 67%. Still
>   winning, lead narrowed by 2pt (the cost of dropping RRF safety).
> - MuSiQue: LangChain 39%, RedHop **34%** (was 26%, **+8 from fix**),
>   LlamaIndex 31%. Closed most of the gap; LangChain still leads by 5
>   on MuSiQue.
>
> Latency unchanged by this fix (240-467ms p50 RedHop vs 60-100ms p50
> competitors). The latency profile remains the open follow-up
> ([HYBRID_LATENCY_PROFILE](HYBRID_LATENCY_PROFILE.md)).

## TL;DR

**HotpotQA (n=100, budget=400 tok, candidate_k=20):**

| arm | mean recall | ≥0.5 | **≥0.8** | p50 ms |
|---|---:|---:|---:|---:|
| redhop[topk] BM25 baseline | 0.89 | 95% | 71% | 2.8 |
| **redhop[topk] hybrid (bge-small)** | **0.93** | **97%** | **83%** | 240 |
| langchain BM25 baseline | 0.86 | 97% | 69% | 0.4 |
| langchain + bge-small rerank | 0.92 | 98% | 77% | 71 |
| llamaindex BM25 baseline | 0.87 | 94% | 67% | 1.7 |
| llamaindex + bge-small rerank | 0.86 | 97% | 67% | 61 |

**MuSiQue (n=100, budget=400 tok, candidate_k=20):**

| arm | mean recall | ≥0.5 | **≥0.8** | p50 ms |
|---|---:|---:|---:|---:|
| redhop[topk] BM25 baseline | 0.61 | 66% | 23% | 3.3 |
| redhop[topk] hybrid (bge-small) | 0.65 | 74% | 26% | 467 |
| langchain BM25 baseline | 0.60 | 68% | 22% | 0.5 |
| **langchain + bge-small rerank** | **0.71** | **82%** | **39%** | 100 |
| llamaindex BM25 baseline | 0.56 | 59% | 16% | 2.8 |
| llamaindex + bge-small rerank | 0.67 | 75% | 31% | 87 |

## Question

[MULTIHOP_HYBRID](MULTIHOP_HYBRID.md) showed RedHop's
`retrieval="hybrid"` lifts HotpotQA ≥0.8 retention from 71% to 83% — a
substantial +12. The honest follow-up the 0.3.1 audit reviewer flagged:
**is the +12 a property of dense rerank applied to the bridge-passage
problem, or specific to RedHop's hybrid implementation?**

If dense rerank is the lever, LangChain and LlamaIndex with the same
dense model should also lift roughly to ~83% on HotpotQA. If RedHop's
BM25 candidate selection is producing a better pool for the reranker,
the competitors should plateau earlier.

## Probe design

Apples-to-apples: hold the dense rerank step **identical** across all
three systems. Each system uses its own chunker + BM25 to produce a
top-K candidate pool; we then rerank those K with the same bge-small
embedding via sentence-transformers, and fill the same 400-token
budget from the reranked order. Identical metric (gold-span word
recall) across all six arms.

What varies between systems:

- **Chunker.** RedHop: SentenceChunker, 128-token target. LangChain:
  RecursiveCharacterTextSplitter, ~256-token chunks (1024 chars). LlamaIndex:
  SentenceSplitter, 256-token chunks.
- **BM25 implementation.** RedHop: Tantivy with English-stemming
  analyzer. LangChain: in-memory BM25Okapi. LlamaIndex: their own
  BM25Retriever.

What's identical:

- **Dense embedder** (bge-small-en-v1.5).
- **Rerank strategy** (cosine over top-K).
- **Candidate-pool size** (k=20).
- **Budget** (400 tokens).
- **Metric** (word-recall on gold).

## What the result actually says

### HotpotQA: RedHop hybrid is structurally stronger

RedHop's hybrid reaches **83% ≥0.8** retention while LangChain hybrid
reaches **77%** and LlamaIndex hybrid stays flat at **67%** — *no lift
over LlamaIndex's own BM25 baseline.* The +12 on HotpotQA is therefore
**not** just "dense rerank is good"; it's "RedHop's BM25 candidate pool
is producing the right paragraphs for the dense reranker to find."

Why LlamaIndex's hybrid shows zero lift is worth investigating but is
out of scope here — possibly their 256-token SentenceSplitter
fragments bridge passages so the dense reranker can't see them as
single coherent units.

### MuSiQue: the picture flips

LangChain's hybrid reaches **39% ≥0.8** while RedHop hybrid only
manages **26%** and LlamaIndex hybrid **31%**. *RedHop is the worst
hybrid on MuSiQue.* The compositional multi-hop shape (2-4 reasoning
hops, 20 distractor paragraphs) appears to reward larger chunks
(LangChain's ~256-token chunks) over RedHop's 128-token chunks. With
smaller chunks the multi-paragraph reasoning bridge that connects two
hops gets split, and even a strong reranker can't reconstruct it.

This is the cleanest evidence we've measured that **chunking is
workload-specific** — the same library winning a small-chunk dataset
(HotpotQA) loses a larger-chunk one (MuSiQue), and vice versa.

### Latency: RedHop's hybrid is 2-5× slower than competitors'

RedHop hybrid: 240ms (HotpotQA), 467ms (MuSiQue) p50. LangChain
hybrid: 71ms / 100ms. LlamaIndex hybrid: 61ms / 87ms.

The likely cause: RedHop's hybrid tier uses ONNX runtime via the
`ort` crate; the competitors' arms here use sentence-transformers
(PyTorch). For warm queries with a small candidate pool, the
sentence-transformers path appears to be faster on Apple Silicon. The
fixed ONNX session overhead may dominate at this scale.

This is a real concern worth tracking. Possible next steps: profile
the ORT call to identify the overhead; batch the embeddings across
candidates (we may be doing one-call-per-candidate); revisit the model
loading path for cold-start cost.

## What this changes in our positioning

The MULTIHOP_HYBRID writeup said:

> "Switch to `retrieval="hybrid"` for a substantial retention lift (+12
> on HotpotQA at our measured budget)."

Honest revision after this competitor probe:

> "On HotpotQA-shaped multi-hop (short Wikipedia paragraphs, 2-hop
> reasoning), RedHop's `retrieval="hybrid"` is the strongest hybrid
> we've measured (83% ≥0.8 vs LangChain hybrid 77%, LlamaIndex hybrid
> 67%). On MuSiQue-shaped compositional multi-hop (longer paragraphs,
> 2-4 hop reasoning), LangChain's hybrid wins (39% vs RedHop 26%) —
> our chunking choice doesn't fit that shape. Hybrid latency in RedHop
> is currently 2-5× competitors'; that's a follow-up to investigate.
> Test on your own workload before committing."

## Honest limits

- **n=100.** Smaller than the n=300 BM25-baseline numbers. Hybrid arms
  are slow; n=300 would have run ~30+ min.
- **One dense model only** (bge-small). Larger models might shift the
  ranking — bge-base could close the MuSiQue gap, or could shift the
  HotpotQA winner. Untested.
- **The competitor "hybrid" we built isn't necessarily what users would
  ship.** Real-world LangChain hybrid often uses EnsembleRetriever with
  reciprocal rank fusion, not simple rerank. We picked the simpler
  shape to hold the dense step constant; the *idiomatic* hybrid for
  each library might score differently.
- **Chunking is uncontrolled.** Each system uses its own chunker, which
  is part of what we want to compare — but it confounds the result. A
  fairer-fairer comparison would hold chunking constant and only vary
  the retriever/reranker; that requires a sharable chunk format.
- **No comparison vs hosted services** (Cohere Rerank, OpenAI
  Embeddings, Voyage, etc.). Probably the strongest available rerankers
  are out of this probe's scope.

## Reproduce

```bash
bench/.venv/bin/python bench/multihop_hybrid_competitors_probe.py
```

Raw run: [`reports/multihop_hybrid_competitors_2026-06-08.txt`](../../reports/multihop_hybrid_competitors_2026-06-08.txt).

## See also

- [MULTIHOP_HYBRID](MULTIHOP_HYBRID.md) — the previous finding that
  established RedHop's +12 hybrid lift on HotpotQA but did not
  control for "is this a property of dense rerank?"
- [MUSIQUE_MULTIHOP](MUSIQUE_MULTIHOP.md) — the BM25-only multi-hop
  comparison that motivated this probe.
- [CHUNK_GRANULARITY](CHUNK_GRANULARITY.md) — the prior finding that
  chunk size matters more than strategy. The MuSiQue result here is
  consistent: a different chunk size (LangChain's ~256-token) wins on
  a different workload shape.
- [SECOND_HOP_TAX](SECOND_HOP_TAX.md) — the bridge-passage failure
  mode the dense rerank addresses.
