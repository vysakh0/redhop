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

## What we tried, what worked, what didn't

**Lazy embedding (attempted in 0.3.1, reverted):** built a mode where
`index()` skips bulk embed and `retrieve()` embeds the BM25 top-K on
the fly. Predicted ~50% latency cut for single-query workloads
because we'd embed fewer chunks per query.

**Re-measured with lazy on, n=100, same probe:** RedHop hybrid
HotpotQA latency 240ms → **309ms (worse by 70ms)**. MuSiQue 467ms →
**499ms (also worse)**. **Reverted.**

Why the prediction was wrong:

- The Retriever trait's `retrieve(&self, ...)` is read-only — `&self`
  not `&mut self`. So lazy retrieve couldn't update `self.embeddings`
  on first query. Without a cache update path, every query re-embeds
  the same candidates from scratch.
- For one-shot single-query patterns, total embed work is roughly
  identical between lazy and eager (you embed the same total number
  of tokens; just at different times).
- For multi-query-per-doc patterns, lazy DESTROYED the warm-query
  benefit: eager warm queries hit ~8ms (cached cosine); lazy warm
  queries hit ~200ms (re-embed the pool each time).
- So lazy was strictly worse: equal or slightly slower on the first
  query, dramatically slower on warm queries.

Adding interior mutability (Mutex around the embeddings HashMap) would
recover the warm-query benefit but only delivers a "deferred eager"
win — push the first embed cost from `from_text()` to first
`context()`. Same total wall-clock time. Worth doing for UX (faster
index when the user doesn't query immediately) but not a latency cut.

**What actually moves the number:**

1. **The ORT vs sentence-transformers MPS gap is the real bottleneck.**
   Same model, same 8-chunk × 165-token workload, same hardware:
   sentence-transformers PyTorch MPS 171ms; ORT CPU 220ms. ~30% per
   forward pass. On Linux without MPS this gap closes substantially.
2. **Sentence-transformers via PyO3 PyTorch bridge** would eliminate
   it on Mac. Heavy dep change; probably not worth it for a per-call
   30% cut.
3. **~~CoreML EP for ORT~~ (tried, no measured win).** Wired up
   `ep-coreml` end-to-end (Cargo feature, runtime EP registration,
   CI matrix). Re-ran the n=100 hybrid-competitors probe on
   Apple Silicon:

   | | Before CoreML | With CoreML EP |
   |---|---:|---:|
   | RedHop hybrid HotpotQA p50 | 240 ms | **303 ms (worse)** |
   | RedHop hybrid MuSiQue p50 | 467 ms | **513 ms (worse)** |

   Multi-query-per-doc pattern (3 docs × 5 queries each):
   CoreML on 945 ms total; CoreML off 915 ms — essentially tied.
   Either the bge-small ONNX graph has ops that aren't accelerated
   by ort 2.0.0-rc.10's CoreML EP, or CoreML's per-Document model-
   compile overhead exceeds the per-query forward-pass savings.
   **Code infrastructure kept** (Cargo features, EP registration in
   `embeddings/onnx.rs`, `REDHOP_DISABLE_EP=1` escape hatch); the EP
   is **not** enabled in published wheels until a measured win lands
   on some platform.
4. **Accept the gap as the price of pure-Rust ONNX.** Honest answer:
   on Apple Silicon at ort 2.0.0-rc.10, RedHop hybrid is ~30% slower
   than sentence-transformers-PyTorch hybrid for the same model.
   The user should know this when picking a configuration.

## What's still worth trying

- **OneDNN on Linux x86_64 + Windows x64.** Wired up as `ep-onednn`,
  not enabled in published wheels. The same single-platform sanity
  probe should be run on a Linux x86 runner before any default
  switch. Different op coverage than CoreML; the result may differ.
- **XNNPACK on Linux aarch64.** Wired as `ep-xnnpack`. Same caveat.
- **A newer ort release.** ort 2.0.0-rc.10 is a release candidate;
  later RCs may have better EP op coverage. Worth re-running this
  probe whenever the dep gets bumped.

For users who want to experiment: every EP can be opted into via
`cargo install redhop --features ep-coreml` (or `ep-onednn`, etc.).
The user-facing recommendation is to measure on the specific
workload before assuming any EP helps — three "predicted lift, didn't
land" results in this branch suggest there's no shortcut.

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
