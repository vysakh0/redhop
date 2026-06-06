//! Hybrid retrieval — composition over an arbitrary number of sub-retrievers.
//!
//! The hybrid retriever fans out the query to every configured sub-retriever
//! (in parallel via `tokio::join_all`), pulls a generous `candidate_k` per
//! retriever, and fuses with the configured [`FusionStrategy`] into a final
//! `top_k`.
//!
//! Why fan out? Lexical and dense retrievers reliably catch different kinds
//! of evidence: lexical wins on rare-term recall and exact phrase matches,
//! dense wins on paraphrase. The hybrid retriever is the cheapest way to
//! capture both without committing to a single signal.

use std::sync::Arc;

use crate::core::{Chunk, Error, Query, RetrievalResult, Retriever};
use async_trait::async_trait;
use futures::future::try_join_all;

use crate::retrieval::fusion::{reciprocal_rank_fusion, weighted_sum_fusion, FusionStrategy};

/// A hybrid retriever combining multiple sub-retrievers via fusion.
pub struct HybridRetriever {
    retrievers: Vec<Arc<dyn Retriever>>,
    strategy: FusionStrategy,
    candidate_k: usize,
    /// Used only by `FusionStrategy::WeightedSum`. Must match
    /// `retrievers.len()`.
    weights: Vec<f32>,
}

impl HybridRetriever {
    /// Construct a hybrid retriever using RRF.
    pub fn rrf(retrievers: Vec<Arc<dyn Retriever>>, candidate_k: usize) -> Self {
        let n = retrievers.len();
        Self {
            retrievers,
            strategy: FusionStrategy::ReciprocalRank,
            candidate_k,
            weights: vec![1.0; n],
        }
    }

    /// Construct a hybrid retriever using weighted-sum fusion.
    pub fn weighted(
        retrievers: Vec<Arc<dyn Retriever>>,
        weights: Vec<f32>,
        candidate_k: usize,
    ) -> crate::core::Result<Self> {
        if retrievers.len() != weights.len() {
            return Err(Error::InvalidConfig(
                "weights length must match retrievers length".into(),
            ));
        }
        if weights.iter().sum::<f32>() <= 0.0 {
            return Err(Error::InvalidConfig("weights must sum to > 0".into()));
        }
        Ok(Self {
            retrievers,
            strategy: FusionStrategy::WeightedSum,
            candidate_k,
            weights,
        })
    }
}

#[async_trait]
impl Retriever for HybridRetriever {
    /// Surface the first sub-retriever's chunk-embedding cache, if any.
    /// Lets `Document::embedded_chunks()` find embeddings cached inside a
    /// dense sub-retriever even when the outermost retriever is a
    /// composition. The convention is "first non-None wins" — in practice
    /// at most one sub-retriever (the dense one) carries embeddings, so
    /// the order-sensitivity is academic.
    fn embeddings(&self) -> Option<&std::collections::HashMap<String, crate::core::Embedding>> {
        for r in &self.retrievers {
            if let Some(map) = r.embeddings() {
                return Some(map);
            }
        }
        None
    }

    async fn index(&mut self, _chunks: &[Chunk]) -> crate::core::Result<()> {
        // Hybrid is a composition; indexing happens on the sub-retrievers
        // directly. We intentionally do not duplicate writes here because
        // the sub-retrievers may have wildly different ingest costs (e.g.
        // a remote dense service vs. a local BM25 index) and the caller
        // is best placed to decide how to coordinate them.
        Err(Error::InvalidConfig(
            "HybridRetriever does not own ingest; call index() on sub-retrievers".into(),
        ))
    }

    async fn retrieve(
        &self,
        query: &Query,
        top_k: usize,
    ) -> crate::core::Result<Vec<RetrievalResult>> {
        let k = self.candidate_k.max(top_k).max(1);
        let futures = self.retrievers.iter().map(|r| {
            let r = r.clone();
            let q = query.clone();
            async move { r.retrieve(&q, k).await }
        });
        let lists: Vec<Vec<RetrievalResult>> = try_join_all(futures).await?;

        let fused = match self.strategy {
            FusionStrategy::ReciprocalRank => reciprocal_rank_fusion(&lists, 60.0, top_k),
            FusionStrategy::WeightedSum => weighted_sum_fusion(&lists, &self.weights, top_k),
        };
        Ok(fused)
    }

    fn name(&self) -> &'static str {
        "hybrid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    struct MockRetriever {
        name: &'static str,
        results: Vec<RetrievalResult>,
    }

    #[async_trait]
    impl Retriever for MockRetriever {
        async fn retrieve(
            &self,
            _q: &Query,
            top_k: usize,
        ) -> crate::core::Result<Vec<RetrievalResult>> {
            Ok(self.results.iter().take(top_k).cloned().collect())
        }
        async fn index(&mut self, _c: &[Chunk]) -> crate::core::Result<()> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            self.name
        }
    }

    fn r(id: &str, method: RetrievalMethod, score: f32) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(ChunkId::new(id), id, "doc", TokenCount(1)),
            score: Score {
                value: score,
                method,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn hybrid_fuses_lists() {
        rt().block_on(async {
            let lex: Arc<dyn Retriever> = Arc::new(MockRetriever {
                name: "lex",
                results: vec![
                    r("a", RetrievalMethod::Lexical, 3.0),
                    r("b", RetrievalMethod::Lexical, 2.0),
                ],
            });
            let dense: Arc<dyn Retriever> = Arc::new(MockRetriever {
                name: "dense",
                results: vec![
                    r("b", RetrievalMethod::Dense, 0.9),
                    r("c", RetrievalMethod::Dense, 0.8),
                ],
            });
            let h = HybridRetriever::rrf(vec![lex, dense], 5);
            let fused = h.retrieve(&Query::new("q"), 3).await.unwrap();
            // `b` appears in both lists; it must rank ahead of `c` (one list).
            let ids: Vec<_> = fused
                .iter()
                .map(|r| r.chunk.id.as_str().to_string())
                .collect();
            let bi = ids.iter().position(|x| x == "b").unwrap();
            let ci = ids.iter().position(|x| x == "c").unwrap();
            assert!(bi < ci);
        });
    }
}
