//! Embedder bake-off harness.
//!
//! Compares two [`EmbeddingProvider`]s on the same labeled corpus along
//! the three axes that matter for a deployment decision: **retrieval
//! quality** (gold-chunk recall), **latency** (embed wall-clock), and
//! **memory** (bytes per vector). The control arm is normally the
//! zero-dep hashing baseline; the treatment arm is a real model (BGE/E5
//! via ONNX). The harness is model-agnostic — it takes any two
//! providers — so the same code runs the synthetic comparison here and
//! the real bake-off on a machine with model files.
//!
//! It deliberately does *not* exercise the adaptive controller: this
//! measures the **embedding backend in isolation**, so an embedder
//! change can be attributed cleanly before it interacts with policy.

use std::sync::Arc;
use std::time::Instant;

use redhop::core::{ChunkId, Embedding, EmbeddingProvider, Query, Result, VectorIndex};
use redhop::storage::FlatVectorIndex;
use serde::{Deserialize, Serialize};

use crate::dataset::LabeledCorpus;

/// Per-embedder bake-off result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderResult {
    /// Provider name.
    pub name: String,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Mean gold-chunk recall@k over the corpus's queries.
    pub mean_recall: f32,
    /// Total wall-clock spent embedding queries + corpus, in ms.
    pub embed_latency_ms: f32,
    /// Mean query-embed latency, in microseconds.
    pub query_embed_us: f32,
    /// Bytes per stored vector (`dim * 4` for f32).
    pub bytes_per_vector: usize,
    /// Number of queries evaluated.
    pub n_queries: usize,
    /// Number of corpus chunks indexed.
    pub n_chunks: usize,
}

/// Comparison of two embedders on identical data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderComparison {
    /// Control arm (typically the hashing baseline).
    pub baseline: EmbedderResult,
    /// Treatment arm (typically a real model).
    pub candidate: EmbedderResult,
    /// `candidate.mean_recall - baseline.mean_recall`. The headline.
    pub recall_delta: f32,
    /// `candidate.query_embed_us / baseline.query_embed_us`. Latency
    /// cost multiple of the candidate.
    pub latency_multiple: f32,
}

/// Run a single embedder over a labeled corpus and measure recall +
/// latency. `top_k` is the retrieval cutoff for the recall metric.
///
/// The corpus chunks must carry text on `gold_chunk_ids`-referenced
/// documents; here we treat each `LabeledQuery`'s gold ids as the
/// relevant set and measure how many land in the dense top-k. The
/// chunk pool is built from `corpus.docs` (one chunk per doc by id),
/// so the gold ids must reference doc-derived chunk ids — which is how
/// the loaders emit them.
pub async fn bench_embedder(
    provider: Arc<dyn EmbeddingProvider>,
    corpus: &LabeledCorpus,
    chunk_texts: &[(ChunkId, String)],
    top_k: usize,
) -> Result<EmbedderResult> {
    let dim = provider.dim();
    let start = Instant::now();

    // Embed + index the corpus chunks.
    let texts: Vec<String> = chunk_texts.iter().map(|(_, t)| t.clone()).collect();
    let chunk_vecs = provider.embed(&texts).await?;
    let mut index = FlatVectorIndex::new(dim);
    for ((id, _), v) in chunk_texts.iter().zip(chunk_vecs.into_iter()) {
        index.add(id.clone(), v)?;
    }

    // Embed queries (timed separately for the per-query latency figure).
    let q_texts: Vec<String> = corpus.queries.iter().map(|q| q.text.clone()).collect();
    let q_start = Instant::now();
    let q_vecs = provider.embed(&q_texts).await?;
    let q_elapsed = q_start.elapsed();

    // Recall@k.
    let mut total_recall = 0f32;
    let mut counted = 0usize;
    for (q, qv) in corpus.queries.iter().zip(q_vecs.iter()) {
        if q.gold_chunk_ids.is_empty() {
            continue;
        }
        let hits = index.search(qv, top_k)?;
        let retrieved: Vec<&ChunkId> = hits.iter().map(|(id, _)| id).collect();
        let found = q
            .gold_chunk_ids
            .iter()
            .filter(|g| retrieved.contains(g))
            .count();
        total_recall += found as f32 / q.gold_chunk_ids.len() as f32;
        counted += 1;
    }
    let mean_recall = if counted > 0 {
        total_recall / counted as f32
    } else {
        0.0
    };

    let total_elapsed = start.elapsed();
    Ok(EmbedderResult {
        name: provider.name().to_string(),
        dim,
        mean_recall,
        embed_latency_ms: total_elapsed.as_secs_f32() * 1000.0,
        query_embed_us: if !corpus.queries.is_empty() {
            q_elapsed.as_secs_f32() * 1e6 / corpus.queries.len() as f32
        } else {
            0.0
        },
        bytes_per_vector: dim * std::mem::size_of::<f32>(),
        n_queries: corpus.queries.len(),
        n_chunks: chunk_texts.len(),
    })
}

/// Compare two embedders on the same corpus.
pub async fn compare_embedders(
    baseline: Arc<dyn EmbeddingProvider>,
    candidate: Arc<dyn EmbeddingProvider>,
    corpus: &LabeledCorpus,
    chunk_texts: &[(ChunkId, String)],
    top_k: usize,
) -> Result<EmbedderComparison> {
    let b = bench_embedder(baseline, corpus, chunk_texts, top_k).await?;
    let c = bench_embedder(candidate, corpus, chunk_texts, top_k).await?;
    let latency_multiple = if b.query_embed_us > 0.0 {
        c.query_embed_us / b.query_embed_us
    } else {
        0.0
    };
    Ok(EmbedderComparison {
        recall_delta: c.mean_recall - b.mean_recall,
        latency_multiple,
        baseline: b,
        candidate: c,
    })
}

/// Render a comparison as an ASCII table.
pub fn render_comparison(cmp: &EmbedderComparison) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:<14} {:>6} {:>10} {:>14} {:>12}\n",
        "embedder", "dim", "recall", "q_embed_us", "bytes/vec"
    ));
    s.push_str(&"─".repeat(60));
    s.push('\n');
    for r in [&cmp.baseline, &cmp.candidate] {
        s.push_str(&format!(
            "{:<14} {:>6} {:>10.3} {:>14.1} {:>12}\n",
            r.name, r.dim, r.mean_recall, r.query_embed_us, r.bytes_per_vector
        ));
    }
    s.push_str(&format!(
        "\nrecall delta (candidate − baseline): {:+.3}\n",
        cmp.recall_delta
    ));
    s.push_str(&format!(
        "latency multiple (candidate / baseline): {:.1}×\n",
        cmp.latency_multiple
    ));
    s
}

// Silence unused import of Query/Embedding in non-test builds; they are
// part of the public signatures via the trait.
#[allow(dead_code)]
fn _sig_anchor(_q: Query, _e: Embedding) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::LabeledQuery;
    use crate::embedder::HashingEmbedder;
    use redhop::core::{Document, RetrievalRegime};

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
    }

    /// Adapter making the sync calibration HashingEmbedder satisfy the
    /// async EmbeddingProvider trait, for the test.
    struct HashAsync(HashingEmbedder, usize);
    #[async_trait::async_trait]
    impl EmbeddingProvider for HashAsync {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
            Ok(texts.iter().map(|t| self.0.embed(t)).collect())
        }
        fn dim(&self) -> usize {
            self.1
        }
        fn name(&self) -> &'static str {
            "hashing"
        }
    }

    #[test]
    fn bench_runs_and_reports_recall() {
        rt().block_on(async {
            // Two docs; query gold-matches doc-a.
            let chunk_texts = vec![
                (
                    ChunkId::new("a"),
                    "rust memory safety ownership borrow checker".to_string(),
                ),
                (
                    ChunkId::new("b"),
                    "baking sourdough bread flour yeast water".to_string(),
                ),
            ];
            let mut q = LabeledQuery::new("q1", "rust ownership borrow", RetrievalRegime::Easy);
            q.gold_chunk_ids = vec![ChunkId::new("a")];
            let corpus = LabeledCorpus {
                docs: vec![Document::new("a", ""), Document::new("b", "")],
                queries: vec![q],
            };
            let dim = 128;
            let provider: Arc<dyn EmbeddingProvider> =
                Arc::new(HashAsync(HashingEmbedder::with_dim(dim), dim));
            let r = bench_embedder(provider, &corpus, &chunk_texts, 1)
                .await
                .unwrap();
            // The rust query should retrieve the rust chunk at top-1.
            assert_eq!(r.mean_recall, 1.0);
            assert_eq!(r.bytes_per_vector, dim * 4);
            assert_eq!(r.n_chunks, 2);
        });
    }

    #[test]
    fn comparison_computes_delta() {
        rt().block_on(async {
            let chunk_texts = vec![
                (ChunkId::new("a"), "alpha beta gamma".to_string()),
                (ChunkId::new("b"), "delta epsilon zeta".to_string()),
            ];
            let mut q = LabeledQuery::new("q1", "alpha beta", RetrievalRegime::Easy);
            q.gold_chunk_ids = vec![ChunkId::new("a")];
            let corpus = LabeledCorpus {
                docs: vec![],
                queries: vec![q],
            };
            let p1: Arc<dyn EmbeddingProvider> =
                Arc::new(HashAsync(HashingEmbedder::with_dim(64), 64));
            let p2: Arc<dyn EmbeddingProvider> =
                Arc::new(HashAsync(HashingEmbedder::with_dim(64), 64));
            let cmp = compare_embedders(p1, p2, &corpus, &chunk_texts, 1)
                .await
                .unwrap();
            // Identical providers → zero delta.
            assert_eq!(cmp.recall_delta, 0.0);
        });
    }
}
