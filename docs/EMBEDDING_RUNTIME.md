# Embedding & Cross-Encoder Runtime (Phases A + B)

Real model backends for RedHop, behind feature flags. The default build
stays dependency-light and fully offline; the `onnx` feature adds
ONNX-Runtime-backed BGE/E5 embeddings and a cross-encoder reranker.

## What shipped

| Component | Crate | Always available? | Status |
| --------- | ----- | ----------------- | ------ |
| `pooling` (mean/CLS + L2, mask-aware) | `redhop::embeddings` | feature `semantic` | unit-tested |
| `EmbedderConfig` (BGE/E5/mxbai presets, prefixes) | `redhop::embeddings` | feature `semantic` | unit-tested |
| `HashingProvider` (zero-dep TF baseline) | `redhop::embeddings` | feature `semantic` | unit-tested |
| `CachedEmbedder<E>` (bounded LRU) | `redhop::embeddings` | feature `semantic` | unit-tested |
| `OnnxEmbedder` (BGE/E5/jina/mxbai) | `redhop::embeddings` | feature `semantic` | **compile-verified** vs `ort` 2.0.0-rc.10 |
| `apply_scores` (rerank decision logic) | `redhop::reranking` | feature `semantic` | unit-tested |
| `OnnxCrossEncoder` (ms-marco MiniLM etc.) | `redhop::reranking` | feature `semantic` | **compile-verified** |
| `bench_embedder` / `compare_embedders` | `redhop-calibration` | ✅ | unit-tested |
| criterion embedding benches | `redhop-benchmarks` | ✅ | runs |

"Compile-verified" means `cargo check --features semantic` passes against
the pinned `ort` and the ONNX Runtime binary downloads and links. It
does **not** mean end-to-end-validated: that needs a real model file
(BGE-small etc.), which the build sandbox does not carry. The error-prone
*math* (mask-aware mean pooling, E5 prefixes, L2 normalization, rerank
score application) is fully unit-tested without a model, so the only
unvalidated surface is tokenization + tensor-shape glue.

## Design discipline

- **Feature-gated.** `ort` + `tokenizers` are optional deps behind the
  `onnx` feature. `cargo build` (no features) never compiles them and
  works offline.
- **No new abstractions.** Both backends implement the existing
  `EmbeddingProvider` / `Reranker` traits. Nothing new in `redhop-core`.
- **Math separated from inference.** The ONNX modules delegate every
  numeric operation to the hermetic `pooling` / `apply_scores`
  functions, shrinking the unverifiable surface to glue.
- **Cache composes.** `CachedEmbedder<E>` wraps *any* provider, so the
  hashing baseline and the ONNX backend both get caching for free.

## Dependency notes (hard-won)

The `ort` 2.x release-candidate line is finicky about transitive
constraints. What works:

```toml
ort = { version = "=2.0.0-rc.10", default-features = false,
        features = ["std", "download-binaries"] }
tokenizers = { version = "0.20", default-features = false,
               features = ["onig"] }
```

- **Pin `ort` exactly** (`=2.0.0-rc.10`). A range pulls a newer RC whose
  `ndarray` constraint (`NdFloat` behind the `std` feature) fails to
  resolve.
- **Enable `ort`'s `std` feature** — `commit_from_file` is gated behind
  it; without `std` you get "no method named `commit_from_file`".
- **Avoid the `ndarray` feature.** We construct tensors from plain
  `([batch, seq], Vec<i64>)` pairs (`ToShape` is implemented for
  `[usize; N]`), so we never touch `ndarray` and dodge its version
  churn.
- `download-binaries` fetches a prebuilt ONNX Runtime; for air-gapped
  builds use `load-dynamic` and point at a system `libonnxruntime`.

## Getting models

RedHop ships no models. Export or download ONNX variants:

```bash
# Option A: HuggingFace ONNX exports (many models ship model.onnx + tokenizer.json)
huggingface-cli download BAAI/bge-small-en-v1.5 --include "onnx/*" "tokenizer.json"

# Option B: optimum export
optimum-cli export onnx --model intfloat/e5-small-v2 e5-small-onnx/
optimum-cli export onnx --model cross-encoder/ms-marco-MiniLM-L-6-v2 ce-onnx/
```

Then:

```rust
# #[cfg(feature = "semantic")]
# {
use redhop::embeddings::{OnnxEmbedder, EmbedderConfig};
use redhop::reranking::OnnxCrossEncoder;

// BGE-small: CLS pooling, normalize, 384-dim.
let embedder = OnnxEmbedder::load("bge/model.onnx", "bge/tokenizer.json",
                                  EmbedderConfig::bge(384))?;

// E5 needs prefixes — build a query embedder and a doc embedder.
let e5_query = OnnxEmbedder::load("e5/model.onnx", "e5/tokenizer.json",
                                  EmbedderConfig::e5(384, "query: "))?;
let e5_doc   = OnnxEmbedder::load("e5/model.onnx", "e5/tokenizer.json",
                                  EmbedderConfig::e5(384, "passage: "))?;

let reranker = OnnxCrossEncoder::load("ce/model.onnx", "ce/tokenizer.json", 512)?;
# }
```

## Wiring it as the `Document` semantic tier

This is the dependency cost of RedHop's **semantic retrieval tier**. The default
`Document` is BM25 — zero model, fully offline. To get semantic recall you opt in:
enable the `semantic` feature, **download a model** (above), and inject the embedder.

```rust
# #[cfg(feature = "semantic")]
# fn demo() -> redhop::core::Result<()> {
use std::sync::Arc;
use redhop::core::EmbeddingProvider;
use redhop::document::{Document, DocumentConfig, RetrievalMode};
use redhop::embeddings::{OnnxEmbedder, EmbedderConfig};

let embedder: Arc<dyn EmbeddingProvider> =
    Arc::new(OnnxEmbedder::load("bge/model.onnx", "bge/tokenizer.json", EmbedderConfig::bge(384))?);

let cfg = DocumentConfig {
    retrieval_mode: RetrievalMode::Hybrid { candidate_pool: 50 },
    ..Default::default()
};
let mut doc = Document::from_text_with("doc.txt", "…", cfg)?.with_embedder(embedder);
let _ctx = doc.context("a paraphrased / semantic query")?;
# Ok(()) }
```

BM25 and global dense retrieve independently over the whole corpus (each
returning `candidate_pool` candidates), and the two ranked lists are
**RRF-fused** (k=60). No vector DB, no ANN — exact brute-force cosine over
every chunk, traded off against the recall lift fusion delivers on bounded
corpora (`docs/findings/MUSIQUE_RECALL_GAP.md`). Selecting `Hybrid` (or
`Dense`) without an embedder is a clear error, so the model dependency is
explicit, never implicit.

For very-large corpora where the global cosine is prohibitive, the previous
BM25-prune-then-dense-rerank composition is preserved as
[`LocalRerankRetriever`](../crates/redhop/src/retrieval/local_rerank.rs) and
can be assembled manually from the public retrieval surface. See
`docs/findings/LOCAL_RERANK.md`. Runnable example:
`cargo run -p redhop-examples --example semantic_local_rerank --features onnx`.
The tier trade-offs (and where the free tiers fall short) are measured in
`docs/findings/SEMANTIC_ZERO_DEP.md` and `docs/findings/LOCAL_RERANK.md`.

## Benchmark plan (run on a real box)

The harness is in place; the numbers below are what you fill in.

**(a) Embedding bake-off** (`redhop-calibration::embedder_bench`):

```rust
let cmp = compare_embedders(
    Arc::new(HashingProvider::with_dim(256)),  // baseline
    Arc::new(CachedEmbedder::new(bge_embedder, 50_000)),  // candidate
    &labeled_corpus, &chunk_texts, /*top_k*/ 4,
).await?;
println!("{}", render_comparison(&cmp));
```

Produces a table of `recall · query_embed_us · bytes/vec` for both
arms plus the recall delta and latency multiple. Run it on the
HotpotQA/MuSiQue `LabeledCorpus` (from the loaders) to get the real
quality lift of BGE/E5 over the hashing baseline.

**(b) Throughput/latency** (`redhop-benchmarks`, criterion):

```bash
cargo bench -p redhop-benchmarks --bench embeddings
```

Hermetic numbers today (this machine, hashing baseline):
- `hashing_embed_512`: ~323 µs for 512 texts (~0.63 µs/text).
- `cached_embed_512_all_hit`: dominated by the cache probe, not
  embedding.

Add the ONNX arm on a real box by extending the bench (which already
pulls in `redhop` with the `semantic` feature) with a loaded model.

## Expected outcomes (hypotheses to falsify with real numbers)

1. **BGE/E5 beats hashing-TF on paraphrase recall.** The hashing
   baseline has zero paraphrase capability (it's lexical); a real
   embedder should lift `mean_recall` on the loaders' corpora,
   especially on queries whose gold chunks share *meaning* but not
   *words*.
2. **The recall lift widens the adaptive advantage.** Better embeddings
   improve the semantic-tier diagnostics, which sharpens regime
   classification, which should make selective escalation fire more
   precisely. Re-run the adaptive eval with the ONNX embedder and
   compare `fraction_useful` to the hashing-baseline run.
3. **Caching dominates query latency on repeated workloads.** Enterprise
   query distributions are heavy-tailed; the `CachedEmbedder` hit rate
   should be high enough that mean query-embed latency approaches the
   cache-probe cost, not the model-inference cost.
4. **Selective escalation makes the cross-encoder affordable.** With the
   controller firing the ONNX cross-encoder on only ~44% of queries
   (per `docs/findings/REAL_WORKLOAD.md`), the cross-encoder's high per-call
   latency is paid only where it earns recall — the economics
   become real latency numbers.

These are deployment-machine experiments. The runtime is built; the
numbers are a data run away.
