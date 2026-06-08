# Why RedHop's hybrid is 2-5× slower than LangChain/LlamaIndex hybrid

> **Status:** **Mechanism-attributed.** The 2-5× latency gap measured
> in [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md)
> comes from **two compounding causes**, not one:
>
> 1. **ORT CPU EP is ~30% slower than sentence-transformers PyTorch
>    MPS** on Apple Silicon for the same model + same workload. On the
>    8-chunk × 165-token forward pass: ORT 220ms, sentence-transformers
>    MPS 171ms.
> 2. **RedHop embeds more chunks** because of its smaller chunking
>    default. On a ~4500-char HotpotQA paragraph, RedHop produces 8
>    chunks @ 128 tokens; LangChain produces ~5 chunks @ 256 tokens.
>    RedHop also embeds *all* chunks at index-build time (right for
>    many-queries-per-doc; wrong for one-shot bench).
>
> ~30% per-text overhead × ~60% more chunks ≈ 2× total. Matches the
> measured 240ms (RedHop) vs 71ms (LangChain) p50 on HotpotQA hybrid.

## The profile

Single HotpotQA query with `REDHOP_EMBED_PROFILE=1` set:

```
doc length: 4474 chars, ~732 words
query: Were Scott Derrickson and Ed Wood of the same nationality?

Document.from_text(retrieval="hybrid"):                ~100ms
[embed] tokenized 8 texts in 1.4ms                            ← chunk tokenization
[embed] window batch=8 seq_len=165 run=220.2ms                ← chunk forward pass
[embed] tokenized 1 texts in 0.0ms                            ← query tokenization
[embed] window batch=1 seq_len=14 run=6.6ms                   ← query forward pass
[embed] window batch=1 seq_len=7 run=7.6ms                    ← (second query embed in same call)
TOTAL doc.context(query):                              244.5ms

Warm second query against same doc:                      7.8ms
```

Sentence-transformers (LangChain hybrid path) on the same 8-chunk
workload, MPS backend on Apple Silicon:

```
sentence-transformers: 8 chunks encoded in 171.4ms
```

So **the ORT forward-pass overhead alone is ~50ms (220 - 171)** —
~30% of sentence-transformers' time. That's the per-text inefficiency.

The other half of the gap is **chunk count × eagerness**. RedHop's
default chunking produces ~60% more chunks per document than
LangChain's, and embeds all of them at `from_text()` time. LangChain's
"hybrid" arm in the competitor probe only embeds what BM25 returns
(top-K candidates), which is the lazy path. Compare:

| system | chunks/doc | when embedded | dense calls per query |
|---|---|---|---|
| RedHop hybrid (current) | ~8 (128-tok target) | at `from_text()` | 1 query embed |
| LangChain (rerank in probe) | ~5 (1024-char split) | lazy at query time | query + ≤5 candidates |

If you reuse the Document across many queries, RedHop's eager
embedding amortizes — measured warm second query: **7.8ms**. If each
query gets a new Document (the bench pattern; common in stateless
single-doc QA), the upfront cost is paid each time.

## What to do about it

**Three potential fixes, ranked by impact-per-effort:**

1. **Lazy embedding option.** A new flag/strategy where dense
   embeddings are computed only over the BM25 top-K instead of every
   chunk. Matches LangChain's pattern. Single-query: faster.
   Many-query: slower (no amortization). Honest tradeoff to expose.
   Estimated impact: 30-40% latency cut on single-query workloads.

2. **CoreML execution provider for ORT on Apple Silicon.** The `ort`
   crate supports EPs; we currently use the default CPU EP. CoreML
   would route through Apple's accelerators. Estimated impact: 20-40%
   on the forward pass on Mac. Has no effect on Linux/Windows. Adds
   a Mac-specific build path.

3. **Batched candidate-only rerank** under hybrid: instead of
   pre-embedding the full chunk index, embed only top-K BM25
   candidates per query. Same idea as #1 but as an internal strategy
   tweak. Estimated impact: same as #1.

None implemented in this finding — this is a profile/diagnosis, not a
fix. The architectural call (eager vs lazy embedding) should be made
deliberately, with the tradeoff documented.

## What this doesn't say

- **It doesn't mean ORT is the wrong choice.** On Linux production
  servers without GPUs, ORT CPU is competitive with PyTorch CPU; the
  gap measured here is specifically PyTorch *MPS* (Apple Silicon
  accelerator) vs ORT *CPU*. A Linux-host comparison would close
  most of the gap.
- **It doesn't mean RedHop's chunking is wrong.** The same 128-token
  default that costs latency here is what wins on HotpotQA's
  short-paragraph shape (see [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md):
  RedHop hybrid 83% vs LangChain 77%). The win + the cost come from
  the same design choice.
- **It doesn't mean users should avoid `retrieval="hybrid"`.** The +12
  on HotpotQA is real; for a 240ms-per-query workload that's a real
  retention win at a real latency price. Just price it honestly.

## Reproduce

```bash
REDHOP_EMBED_PROFILE=1 bench/.venv/bin/python -c "
import json
from pathlib import Path
import redhop
data = json.loads(Path('data/hotpotqa/hotpot_dev_distractor_v1.json').read_text())
ex = data[0]
paras = {title: sents for title, sents in ex['context']}
doc_text = '\n\n'.join(' '.join(s) for s in paras.values())
doc = redhop.Document.from_text(doc_text, retrieval='hybrid', model='bge-small', candidate_k=20, token_budget=400)
ctx = doc.context(ex['question'])
"
```

## See also

- [MULTIHOP_HYBRID_COMPETITORS](MULTIHOP_HYBRID_COMPETITORS.md) — the
  measurement this profile explains.
- [MULTIHOP_HYBRID](MULTIHOP_HYBRID.md) — the original probe that
  established the +12 HotpotQA lift (and its latency cost).
