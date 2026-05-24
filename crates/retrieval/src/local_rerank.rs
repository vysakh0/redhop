//! Local dense rerank — "lexical topology first, semantic refinement second".
//!
//! [`LocalRerankRetriever`] composes a lexical first stage (BM25) with a dense
//! second stage: BM25 prunes the whole corpus to a candidate pool of
//! `candidate_pool` chunks, then *only that pool* is reordered by cosine of the
//! query embedding against precomputed chunk embeddings. This matches a global
//! dense retriever's recall on the validated workloads while touching the dense
//! model on a bounded pool and needing **no ANN / vector index**.
//! See `docs/findings/LOCAL_RERANK.md`.
//!
//! It is **embedder-agnostic**: it takes any [`EmbeddingProvider`], so the ONNX
//! dependency lives at the *construction site* (the caller builds the embedder),
//! never in this crate.
//!
//! [`EmbeddingProvider`]: redhop_core::EmbeddingProvider

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use redhop_core::{
    Chunk, EmbeddingProvider, Error, Query, RetrievalMethod, RetrievalResult, Retriever, Score,
    ScoreBreakdown,
};

use crate::bm25::Bm25Retriever;

const EMBED_BATCH: usize = 64;

/// BM25 candidate generation + local dense rerank. The dense model only ever
/// scores the BM25 candidate pool, never the whole corpus.
pub struct LocalRerankRetriever {
    bm25: Bm25Retriever,
    embedder: Arc<dyn EmbeddingProvider>,
    /// BM25 prune depth — how many lexical candidates the dense stage reorders.
    candidate_pool: usize,
    /// Precomputed chunk embeddings, keyed by chunk id.
    embeddings: HashMap<String, redhop_core::Embedding>,
}

impl LocalRerankRetriever {
    /// Construct a local-rerank retriever over a fresh BM25 index. `candidate_pool`
    /// is the BM25 prune depth (e.g. 50) that the dense stage reorders; it should
    /// be ≥ the final `top_k` you intend to request.
    pub fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        candidate_pool: usize,
    ) -> redhop_core::Result<Self> {
        Ok(Self {
            bm25: Bm25Retriever::new()?,
            embedder,
            candidate_pool: candidate_pool.max(1),
            embeddings: HashMap::new(),
        })
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-9)
}

#[async_trait]
impl Retriever for LocalRerankRetriever {
    async fn index(&mut self, chunks: &[Chunk]) -> redhop_core::Result<()> {
        self.bm25.index(chunks).await?;
        // Precompute chunk embeddings (batched) so retrieval only embeds the query.
        for batch in chunks.chunks(EMBED_BATCH) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let embs = self.embedder.embed(&texts).await?;
            for (c, e) in batch.iter().zip(embs) {
                self.embeddings.insert(c.id.as_str().to_string(), e);
            }
        }
        Ok(())
    }

    async fn retrieve(
        &self,
        query: &Query,
        top_k: usize,
    ) -> redhop_core::Result<Vec<RetrievalResult>> {
        // Lexical prune: fetch the candidate pool (at least as deep as top_k).
        let pool = self.candidate_pool.max(top_k.max(1));
        let cand = self.bm25.retrieve(query, pool).await?;
        if cand.is_empty() {
            return Ok(cand);
        }
        // Query embedding: reuse a precomputed one, else embed the query text.
        let qe = match &query.embedding {
            Some(e) => e.clone(),
            None => self
                .embedder
                .embed(std::slice::from_ref(&query.text))
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    Error::Embedding("embedder returned no vector for the query".into())
                })?,
        };
        // Local refinement: reorder only the pool by cosine.
        let mut scored: Vec<RetrievalResult> = cand
            .into_iter()
            .filter_map(|mut r| {
                let emb = self.embeddings.get(r.chunk.id.as_str())?;
                let s = cosine(qe.as_slice(), emb.as_slice());
                r.score = Score {
                    value: s,
                    method: RetrievalMethod::Rerank,
                };
                r.breakdown = ScoreBreakdown {
                    dense: Some(s),
                    ..Default::default()
                };
                Some(r)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .value
                .partial_cmp(&a.score.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k.max(1));
        Ok(scored)
    }

    fn name(&self) -> &'static str {
        "local_rerank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{Embedding, TokenCount};

    /// Deterministic stub: 3-dim presence vector over {alpha, beta, gamma}. No
    /// model, so the rerank path is testable without ONNX.
    struct StubEmbedder;

    #[async_trait]
    impl EmbeddingProvider for StubEmbedder {
        async fn embed(&self, texts: &[String]) -> redhop_core::Result<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|t| {
                    Embedding::from(vec![
                        t.matches("alpha").count() as f32,
                        t.matches("beta").count() as f32,
                        t.matches("gamma").count() as f32,
                    ])
                })
                .collect())
        }
        fn dim(&self) -> usize {
            3
        }
        fn name(&self) -> &'static str {
            "stub"
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn chunks() -> Vec<Chunk> {
        vec![
            Chunk::new("a", "alpha alpha alpha", "doc", TokenCount(3)),
            Chunk::new("b", "beta beta", "doc", TokenCount(2)),
            Chunk::new("c", "gamma gamma gamma", "doc", TokenCount(3)),
        ]
    }

    #[test]
    fn reorders_pool_by_query_embedding() {
        rt().block_on(async {
            let mut r = LocalRerankRetriever::new(Arc::new(StubEmbedder), 10).unwrap();
            r.index(&chunks()).await.unwrap();
            // Lexically matches all three; the embedding points at "gamma".
            let q =
                Query::new("alpha beta gamma").with_embedding(Embedding::from(vec![0.0, 0.0, 1.0]));
            let res = r.retrieve(&q, 3).await.unwrap();
            assert_eq!(res[0].chunk.id.as_str(), "c");
            assert_eq!(res[0].score.method, RetrievalMethod::Rerank);
        });
    }

    #[test]
    fn embeds_query_when_absent() {
        rt().block_on(async {
            let mut r = LocalRerankRetriever::new(Arc::new(StubEmbedder), 10).unwrap();
            r.index(&chunks()).await.unwrap();
            // No query embedding -> the retriever embeds the text itself; the
            // repeated "beta" makes the query vector lean toward chunk "b".
            let q = Query::new("alpha beta gamma beta");
            let res = r.retrieve(&q, 1).await.unwrap();
            assert_eq!(res[0].chunk.id.as_str(), "b");
        });
    }
}
