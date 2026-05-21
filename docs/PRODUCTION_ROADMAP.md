# NeoRAG Production Runtime Roadmap

**Premise.** The architecture is done. The moat is **calibrated adaptive
retrieval control** — deciding *when not to spend compute*. Every item
below serves retrieval quality, retrieval economics, or operational
usefulness. Nothing below invents new retrieval math, graph transforms,
RL, or autonomous orchestration.

The single empirical fact that anchors all of it:

> Uniform cross-encoder reranking buys +0.046 recall lift.
> Selective adaptive escalation buys +0.112 — **more than double** —
> while reranking only ~44% of queries. 63% of reranks change nothing.
> Harmful-intervention lift is ≈0.

NeoRAG's value is *not* "rerankers are good." It's "most reranking is
waste, and we can tell which." Every engineering decision optimizes
that.

---

## 1. Implementation roadmap (sequencing)

Ordered by **value ÷ risk**, where risk includes "can we validate it
hermetically in CI without downloading multi-GB models."

| Order | Phase | Why here | Hermetic? |
| ----- | ----- | -------- | --------- |
| **1** | **D — Observability** | Names the moat in a form buyers can see. Pure assembly of data we already produce. Zero external deps. | ✅ fully |
| **2** | **E — Cost/Quality** | Quantifies the moat ("2.5× quality at 0.44× cost"). Extends the calibration harness. | ✅ fully |
| **3** | **C — PDF/ingestion diagnostics** | Highest commercial value; the diagnostics are text-only so no PDF dep needed in Rust. | ✅ mostly |
| **4** | **A — Real embeddings** | Unblocks everything quality-wise, but needs model files + an inference runtime. | ⚠️ needs models |
| **5** | **B — Cross-encoder runtime** | Depends on A's ONNX infra; the highest-leverage latency work. | ⚠️ needs models |

**Rationale.** D and E are buildable and testable today and they make
the existing wins legible and quantified — which is what tells us
whether A/B/C are even worth their cost. C's *diagnostics* are text-only
(the value-add is detecting corruption, not parsing PDFs), so it's
hermetic even though PDF *parsing* stays out of Rust. A unblocks B (they
share the ONNX runtime and tokenizer). A and B are gated on a model
runtime that can't be exercised in a no-network sandbox, so they're
designed carefully and validated against fixtures + a feature-gated
integration test that runs only where model files exist.

---

## 2. Dependency recommendations

Concrete crates, with the reason and the fallback.

| Need | Recommended | Why | Fallback / note |
| ---- | ----------- | --- | --------------- |
| ONNX inference | **`ort` v2** (ONNX Runtime bindings) | Mature, CPU SIMD via the ONNX Runtime CPU EP, optional CUDA/CoreML/DirectML execution providers, int8 quant support | feature-gated; not in default build |
| Tokenization | **`tokenizers`** (HF Rust) | exact parity with the Python tokenizers the models were trained with; handles WordPiece/BPE/Unigram | already in the original tech-stack spec |
| Pure-Rust inference (optional) | **`candle-core` + `candle-transformers`** | no C++/system deps, good for `cargo install` ergonomics | slower than `ort` on CPU; narrower model coverage; feature-gated |
| Embedding cache | **`moka`** (sync + future-aware concurrent cache) | bounded, TTL, weight-based eviction, lock-free reads | hand-rolled `lru` if we want zero deps |
| Hashing for cache keys | **`ahash`** or `std` `DefaultHasher` | fast non-crypto hashing of text → cache key | already use FNV-1a inline; fine to keep |
| PDF text extraction | **none in Rust** | parsing stays Python (or `pdftotext`/poppler shell-out in an ingestion tool) | the boundary from `INTEROPERABILITY.md` holds |
| HTML report | **none** (hand-rolled templating) | self-contained single-file HTML, inline CSS, minimal vanilla JS; no build step | `askama` only if templates grow unwieldy |
| ANN at scale | **`usearch`** (already flagged) | HNSW, memory-mapped, quantization | only when FlatVectorIndex's O(n) scan becomes the bottleneck (>~50k vectors) |
| SIMD cosine | **none yet** | `ort` does the heavy matmuls; cosine over normalized vectors autovectorizes well | profile before hand-writing `wide`/`std::simd` |

**Principle:** every heavyweight dep (`ort`, `candle`, `usearch`) is
behind a Cargo feature flag. The default `cargo build` stays
dependency-light and fully hermetic. The hashing-TF embedder remains the
always-available baseline.

---

## 3. Runtime architecture proposal

No new core abstractions. The existing traits (`EmbeddingProvider`,
`Reranker`, `Retriever`, `DiagnosticsEngine`, `RegimeClassifier`,
`Policy`, `Actuator`) are the seams. We add *implementations* and *one
new crate*.

```
crates/
  embeddings/        NEW
    HashingEmbedder           (moved from calibration; the no-dep baseline)
    OnnxEmbedder              (feature "onnx")    — BGE / E5 / jina / mxbai
    CandleEmbedder            (feature "candle")  — pure-Rust alternative
    CachedEmbedder<E>         wraps any provider with a moka cache
    ModelRegistry             loads + warms + shares Arc<Session>
  reranking/         (existing) +
    OnnxCrossEncoder          (feature "onnx") — implements Reranker
  diagnostics/       (existing) +
    ingestion/                NEW tier: OCR-noise, dup, boilerplate, fragmentation
  observability/     NEW
    RetrievalTrace            serializable journey of one RetrievalState
    trace recorders           hook into the orchestrator (no behavior change)
    renderers                 cli / json / html
  calibration/       (existing) +
    cost model + ROI + latency-quality Pareto with a cost axis
```

Key invariants preserved:
- **Conservative policy untouched.** Embeddings and rerankers get
  *faster/better*; the decision of *whether* to escalate is unchanged.
- **Interpretability untouched.** Every new metric is a named, traced
  number, never a hidden heuristic.
- **Calibration discipline untouched.** New backends are evaluated by the
  same harness (recall lift, ECE, regret, bootstrap stability) before
  any default changes.

---

## 4. Benchmark plan

Two harnesses, two questions.

**(a) Throughput/latency** — extend `neorag-benchmarks` (criterion):
- `embed_batch` throughput (texts/sec) per backend, per batch size.
- `cross_encoder_rerank` latency p50/p99 per candidate-count.
- `cosine_search` over FlatVectorIndex vs usearch at 1k/10k/100k vectors.

**(b) Retrieval quality + economics** — extend `neorag-calibration`:
- **Embedder bake-off:** hashing-TF vs BGE-small vs E5-small on the
  HotpotQA/MuSiQue adaptive eval. Report a single table:

  | embedder | recall lift | classifier accuracy | ECE | embed latency p50 | mem |
  | -------- | ----------- | ------------------- | --- | ----------------- | --- |

- **Selective-escalation economics:** for each policy setting, report
  `(mean recall lift, mean rerank calls, mean latency)` and the derived
  ROI (§10). The headline output is the cost-quality Pareto.

Both harnesses already exist in skeleton; this is extension, not
new infrastructure.

---

## 5. Performance optimization opportunities

Ranked by expected impact on a real deployment:

1. **Selective escalation itself (already shipped).** The biggest "perf"
   win is *not running* the cross-encoder on 56% of queries. Frame every
   cost number against the uniform-rerank baseline.
2. **Embed-once-at-ingest.** Chunk embeddings computed at index time,
   persisted, never recomputed at query time. Only the *query* is
   embedded online. (The current dense path already supports this; make
   it the documented default.)
3. **Embedding cache.** Query embeddings cached by text hash. High hit
   rate on repeated/templated queries (enterprise FAQ, dashboards).
4. **Batched ONNX inference.** One session call for N texts. 5–20×
   throughput vs per-text calls.
5. **Int8 quantized models.** ~2–4× faster CPU inference, <1% quality
   loss on retrieval embedders typically. Feature-gated, opt-in.
6. **HNSW (usearch) above ~50k vectors.** FlatVectorIndex's exact scan
   is faster below that; don't pay HNSW build cost prematurely.
7. **Cosine autovectorization.** Profile first. Hand-SIMD only if a flame
   graph says cosine is hot — unlikely once `ort` dominates the budget.

**Anti-optimization:** do not hand-roll SIMD, do not add HNSW, do not add
quantization *before* the benchmark harness shows they're the bottleneck.
Measure, then optimize.

---

## 6. Enterprise PDF strategy

**The boundary holds: Rust does not parse PDFs.** Parsing is a Python
(or `pdftotext`) job that emits clean text. NeoRAG's value on messy
corpora is the *diagnostics*, which operate on text regardless of source.

Build an **ingestion-diagnostics tier** in `neorag-diagnostics`
(text-only, hermetic):

| Metric | Detects | Cheap signal |
| ------ | ------- | ------------ |
| `ocr_noise_score` | scanned/OCR'd garbage | ratio of non-dictionary tokens, broken intra-word chars, isolated single chars |
| `duplicate_ratio` | repeated sections, copy-paste | shingle/minhash near-dup detection across chunks |
| `boilerplate_ratio` | headers/footers/page numbers | lines/short-spans repeated across many chunks |
| `fragmentation_score` | mid-sentence chunk breaks | fraction of chunks that start lowercase / end without terminal punctuation |
| `table_noise_score` | flattened tables | high digit/delimiter density, low sentence structure |

These produce **ingestion warnings**, surfaced through the existing
diagnostics warning channel. They can gate retrieval behavior
*conservatively*: e.g. high `ocr_noise_score` → emit a warning and
optionally fall back to a more robust chunker. They do **not** introduce
a new regime or a new controller — they extend the diagnostics surface.

This is the highest commercial-value direction precisely because it's
where production RAG silently fails and nobody has visibility. NeoRAG's
diagnostics-first design is already the right tool; we just point it at
ingestion.

---

## 7. Embedding backend design

The `EmbeddingProvider` trait already exists in core with batch `embed`,
`dim`, and `name`. We add implementations, not trait changes.

```rust
// neorag-embeddings, feature "onnx"
pub struct OnnxEmbedder {
    session: Arc<ort::Session>,
    tokenizer: tokenizers::Tokenizer,
    config: EmbedderConfig,
}

pub struct EmbedderConfig {
    pub pooling: Pooling,          // Mean | Cls
    pub normalize: bool,           // L2 normalize output
    pub query_prefix: String,      // E5 needs "query: "; BGE often ""
    pub doc_prefix: String,        // E5 needs "passage: "
    pub max_seq_len: usize,        // truncate to model context
    pub dim: usize,
}
```

Per-model notes (all are BERT-family, ONNX-exportable via `optimum`):
- **BGE-small/base** (`BAAI/bge-*`): CLS or mean pooling, no prefix for
  passages; query instruction optional. 384/768 dim.
- **E5-small/base** (`intfloat/e5-*`): **requires** `"query: "` /
  `"passage: "` prefixes — wrong prefixes silently tank recall. 384/768.
- **jina** (`jinaai/jina-embeddings-v2-*`): mean pooling, 512–8192 ctx.
- **mxbai** (`mixedbread-ai/mxbai-embed-*`): CLS pooling, 1024 dim.

`CachedEmbedder<E: EmbeddingProvider>` wraps any backend:
```rust
pub struct CachedEmbedder<E> {
    inner: E,
    cache: moka::sync::Cache<u64, Embedding>,  // key = hash(prefix + text)
}
```

Validation strategy in a no-network sandbox: ship the config + pooling +
prefix logic with unit tests on *synthetic* hidden-state tensors (verify
mean/CLS pooling and L2 norm math), plus a `#[ignore]`-by-default
integration test that runs only when `NEORAG_ONNX_MODEL` env var points
at a real model file. CI without models still passes; CI with models
validates end-to-end.

---

## 8. Cross-encoder runtime design

The `Reranker` trait already exists. Add one implementation.

```rust
// neorag-reranking, feature "onnx"
pub struct OnnxCrossEncoder {
    session: Arc<ort::Session>,
    tokenizer: tokenizers::Tokenizer,
    max_seq_len: usize,
    batch_size: usize,
}
```

Flow: for `(query, candidates)`, form `[query, passage]` token pairs,
tokenize+pad to a batch, one ONNX call → relevance logits, sort
descending, truncate to `top_k`.

Optimizations that matter *because escalation is selective*:
- **CPU-first.** The ONNX Runtime CPU EP with int8 quantization is the
  default. GPU (CUDA/CoreML EP) is feature-gated for high-QPS deploys.
- **Intra-query batching.** Since the controller only escalates on ~44%
  of queries and reranks a handful of candidates each, batch *within* a
  query first. Cross-query batching (a windowed micro-batcher) is a later
  optimization with a real latency/throughput tradeoff — design it,
  don't build it until QPS demands it.
- **Warmup.** Run one dummy inference at load to trigger allocation/JIT
  so the first real query isn't a cold-start outlier.
- **Shared session.** `Arc<Session>` across worker threads;
  `ort` sessions are `Send + Sync`.

The key product framing: **a fast cross-encoder makes selective
escalation cheap enough that the +0.066 adaptive advantage is nearly
free.** Latency is the lever that turns the quality win into an economic
win.

---

## 9. Observability architecture

The named "killer feature." Pure assembly of data we already produce —
`RetrievalState`, `TakenAction`, `DiagnosticsReport`, `RegimeDistribution`,
`QueryOutcome`. No behavior change to the controller.

```
neorag-observability
  RetrievalTrace          serde-serializable record of one query's journey:
                            query, per-iteration {diagnostics, regime dist,
                            confidence, policy decision + rationale,
                            action taken, latency, cost}, final evidence
  TraceRecorder           thin hook the orchestrator calls; off by default,
                            zero overhead when disabled
  renderers::cli          the existing ASCII style, per-query
  renderers::json         machine-readable, one JSON object per query
  renderers::html         self-contained single-file report:
                            - regime distribution (bar)
                            - escalation decisions (timeline)
                            - useful vs wasted reranks (the headline)
                            - calibration curve (reliability diagram)
                            - latency breakdown (per stage)
                            - intervention regret summary
```

The HTML report is the artifact a buyer opens to *see* why retrieval
behaved as it did: "you ran 1,000 queries, escalated 440, 203 of those
escalations changed the evidence set, here's the calibration curve, here's
where compute went." That story is the product.

Design constraints:
- **Opt-in, zero-cost-when-off.** Tracing is a feature/flag; the hot path
  pays nothing when disabled.
- **No JS framework.** Single HTML file, inline CSS, optional vanilla JS
  for collapsible sections. Emails cleanly, archives cleanly, diffs
  cleanly.
- **Same numbers everywhere.** CLI, JSON, and HTML render identical
  underlying metrics. No view-specific recomputation.

---

## 10. Cost / quality evaluation methodology

A cost model assigns a unit cost to each action, then every analysis the
calibration crate already does gains a cost axis.

```rust
pub struct CostModel {
    pub cost_per_query_embed: f32,    // online query embedding
    pub cost_per_retrieval: f32,      // BM25 / ANN lookup
    pub cost_per_rerank_candidate: f32, // cross-encoder pair scoring
    pub latency_per_rerank_candidate_ms: f32,
}
```

Two headline metrics:

1. **Selective-escalation ROI** =
   `(adaptive recall lift) / (adaptive rerank cost)`
   compared to
   `(uniform recall lift) / (uniform rerank cost)`.
   From current data: adaptive ≈ +0.112 at ~0.44 reranks/query; uniform
   ≈ +0.046 at 1.0 reranks/query. ROI ratio ≈ `(0.112/0.44) / (0.046/1.0)`
   ≈ **5.5×**. NeoRAG delivers ~2.5× the quality at ~0.44× the cost — a
   ~5× efficiency multiple.

2. **Cost-quality Pareto frontier.** Extend the existing Pareto renderer
   with a cost axis: x = mean rerank cost (or latency), y = mean recall
   lift, one point per policy setting. The conservative settings should
   sit on the upper-left (high lift, low cost) frontier. This is the plot
   that proves "do less, get more."

Headline framing for any deployment:

> "NeoRAG matched uniform-rerank recall while reranking 44% of queries
> — a 56% reduction in cross-encoder compute — and actually *exceeded*
> uniform-rerank quality by firing on the right queries."

That sentence is the commercial pitch, and it's measured, not claimed.

---

## What we explicitly will NOT do

- No new retrieval math, graph transforms, topology, or trajectory work.
- No RL controller. The rule-based conservative policy stays; it's
  interpretable and it works.
- No agent framework, no autonomous planning.
- No PDF parser in Rust.
- No premature SIMD / HNSW / quantization — measure first.
- No default dependency on heavyweight runtimes — all feature-gated.

## Suggested first build

**Phases D + E together** (observability + cost/quality). Reasons:
- Fully hermetic — buildable and testable in this sandbox today.
- Directly visualizes and quantifies the moat (selective escalation
  economics), which is the thing the user identified as most important.
- Produces the measurement substrate that the A/B/C benchmarks plug
  into — so it's the right *foundation*, not just the easy win.

A and B need model files and an inference runtime that a no-network
sandbox can't exercise end-to-end; they'll ship with feature-gated
backends + fixture-based unit tests + `#[ignore]` integration tests that
activate only where models exist. C's diagnostics are hermetic and can
follow D+E immediately.
