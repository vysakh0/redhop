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
//! [`EmbeddingProvider`]: crate::core::EmbeddingProvider

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use crate::core::{
    Chunk, EmbeddingProvider, Error, Query, RetrievalMethod, RetrievalResult, Retriever, Score,
    ScoreBreakdown,
};

use crate::retrieval::bm25::Bm25Retriever;
use crate::retrieval::fusion::reciprocal_rank_fusion;

/// A chunk routed to lexical-only retrieval (code): under the hybrid tier it's
/// ranked by BM25 and never embedded, since general embedders are weak on code.
fn is_code(c: &Chunk) -> bool {
    c.metadata.get("kind").and_then(|v| v.as_str()) == Some("code")
}

/// BM25 candidate generation + local dense rerank. The dense model only ever
/// scores the BM25 candidate pool, never the whole corpus.
pub struct LocalRerankRetriever {
    bm25: Bm25Retriever,
    /// Embeds the passages (chunks) at index time.
    embedder: Arc<dyn EmbeddingProvider>,
    /// Optional separate embedder for the query side. `None` reuses `embedder`
    /// (symmetric models like BGE/MiniLM). Set it for asymmetric models that apply
    /// different query vs passage handling — e.g. E5's `query:` / `passage:`
    /// prefixes, where both sides share weights but differ in prefix.
    query_embedder: Option<Arc<dyn EmbeddingProvider>>,
    /// BM25 prune depth — how many lexical candidates the dense stage reorders.
    candidate_pool: usize,
    /// Precomputed chunk embeddings, keyed by chunk id.
    embeddings: HashMap<String, crate::core::Embedding>,
    /// When true, skip the BM25 prune and cosine the query against **every**
    /// chunk (exact, brute-force — still no ANN). Recovers pure-paraphrase
    /// queries that share no terms with the answer, at O(N) cosine per query;
    /// for *bounded* corpora only (at scale, use a real vector store).
    global: bool,
    /// Chunks retained for the global path (so it can score chunks BM25 would
    /// never surface). Only populated when `global` is set.
    chunks: Vec<Chunk>,
}

impl LocalRerankRetriever {
    /// Construct a local-rerank retriever over a fresh BM25 index. `candidate_pool`
    /// is the BM25 prune depth (e.g. 50) that the dense stage reorders; it should
    /// be ≥ the final `top_k` you intend to request. The one embedder is used for
    /// both passages and queries (symmetric models).
    pub fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        candidate_pool: usize,
    ) -> crate::core::Result<Self> {
        Ok(Self {
            bm25: Bm25Retriever::new()?,
            embedder,
            query_embedder: None,
            candidate_pool: candidate_pool.max(1),
            embeddings: HashMap::new(),
            global: false,
            chunks: Vec::new(),
        })
    }

    /// Switch to **global** dense: cosine the query against every chunk
    /// embedding (exact brute force, no BM25 prune, no ANN). Use for
    /// paraphrase/synonym-heavy bounded corpora where the answer may share no
    /// terms with the query; O(N) cosine per query.
    pub fn global(mut self) -> Self {
        self.global = true;
        self
    }

    /// Embed the query: reuse a precomputed embedding, else embed the text with
    /// the query-side embedder (falling back to the passage embedder).
    async fn embed_query(&self, query: &Query) -> crate::core::Result<crate::core::Embedding> {
        match &query.embedding {
            Some(e) => Ok(e.clone()),
            None => {
                let q_embedder = self.query_embedder.as_ref().unwrap_or(&self.embedder);
                q_embedder
                    .embed(std::slice::from_ref(&query.text))
                    .await?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        Error::Embedding("embedder returned no vector for the query".into())
                    })
            }
        }
    }

    /// Like [`LocalRerankRetriever::new`] but with a **separate query embedder**.
    /// `passage_embedder` embeds the chunks at index time; `query_embedder` embeds
    /// the query at retrieval time — for asymmetric models (e.g. E5, which needs a
    /// `passage:` prefix on documents and a `query:` prefix on queries).
    pub fn new_with_query_embedder(
        passage_embedder: Arc<dyn EmbeddingProvider>,
        query_embedder: Arc<dyn EmbeddingProvider>,
        candidate_pool: usize,
    ) -> crate::core::Result<Self> {
        Ok(Self {
            bm25: Bm25Retriever::new()?,
            embedder: passage_embedder,
            query_embedder: Some(query_embedder),
            candidate_pool: candidate_pool.max(1),
            embeddings: HashMap::new(),
            global: false,
            chunks: Vec::new(),
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
    async fn index(&mut self, chunks: &[Chunk]) -> crate::core::Result<()> {
        self.bm25.index(chunks).await?;
        // The global path scores chunks BM25 would never surface, so retain them.
        if self.global {
            self.chunks = chunks.to_vec();
        }
        // Reuse any embedding already carried on a chunk (e.g. loaded from a
        // persisted index); only embed the ones that lack one. This makes
        // re-indexing a mostly-unchanged corpus incremental — we pay the
        // embedding cost only for new/changed chunks.
        let to_embed: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| {
                if let Some(e) = &c.embedding {
                    self.embeddings.insert(c.id.as_str().to_string(), e.clone());
                    return false;
                }
                // Hybrid (pool path): code is retrieved lexically, so skip embedding
                // it. The global/semantic path still embeds everything.
                if !self.global && is_code(c) {
                    return false;
                }
                true
            })
            .collect();
        // Precompute the rest in a single call so the embedder can batch them
        // globally (it length-sorts internally to minimize padding waste).
        if !to_embed.is_empty() {
            let texts: Vec<String> = to_embed.iter().map(|c| c.text.clone()).collect();
            let embs = self.embedder.embed(&texts).await?;
            for (c, e) in to_embed.iter().zip(embs) {
                self.embeddings.insert(c.id.as_str().to_string(), e);
            }
        }
        Ok(())
    }

    fn embeddings(&self) -> Option<&HashMap<String, crate::core::Embedding>> {
        Some(&self.embeddings)
    }

    async fn retrieve(
        &self,
        query: &Query,
        top_k: usize,
    ) -> crate::core::Result<Vec<RetrievalResult>> {
        let qe = self.embed_query(query).await?;

        // Global dense: cosine the query against EVERY chunk (no BM25 prune), so
        // a pure-paraphrase answer that shares no terms with the query is still
        // reachable. Exact brute force — no ANN.
        if self.global {
            let mut scored: Vec<RetrievalResult> = self
                .chunks
                .iter()
                .filter_map(|c| {
                    let emb = self.embeddings.get(c.id.as_str())?;
                    let s = cosine(qe.as_slice(), emb.as_slice());
                    let mut r = RetrievalResult::new(
                        c.clone(),
                        Score {
                            value: s,
                            method: RetrievalMethod::Rerank,
                        },
                    );
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
            return Ok(scored);
        }

        // Lexical prune: fetch the candidate pool (already ranked by BM25).
        let pool = self.candidate_pool.max(top_k.max(1));
        let cand = self.bm25.retrieve(query, pool).await?;
        if cand.is_empty() {
            return Ok(cand);
        }

        // Any pool chunk without an embedding is lexical-only (code) — it must
        // survive on its BM25 rank rather than being dropped.
        let has_lexical_only = cand
            .iter()
            .any(|r| !self.embeddings.contains_key(r.chunk.id.as_str()));

        // Dense reranking of the embeddable (prose/data) subset of the pool.
        let mut dense: Vec<RetrievalResult> = cand
            .iter()
            .filter_map(|r| {
                let emb = self.embeddings.get(r.chunk.id.as_str())?;
                let s = cosine(qe.as_slice(), emb.as_slice());
                let mut d = r.clone();
                d.score = Score {
                    value: s,
                    method: RetrievalMethod::Rerank,
                };
                d.breakdown = ScoreBreakdown {
                    dense: Some(s),
                    ..Default::default()
                };
                Some(d)
            })
            .collect();
        dense.sort_by(|a, b| {
            b.score
                .value
                .partial_cmp(&a.score.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if !has_lexical_only {
            // Pure prose pool: dense reranking, as before.
            dense.truncate(top_k.max(1));
            return Ok(dense);
        }
        if dense.is_empty() {
            // Pure lexical pool (all code): BM25 ranking stands.
            let mut out = cand;
            out.truncate(top_k.max(1));
            return Ok(out);
        }
        // Mixed: fuse the BM25 ranking (keeps code chunks) with the dense
        // reranking (prose/data) via reciprocal rank fusion.
        Ok(reciprocal_rank_fusion(&[cand, dense], 60.0, top_k.max(1)))
    }

    fn name(&self) -> &'static str {
        "local_rerank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Embedding, TokenCount};

    /// Deterministic stub: 3-dim presence vector over {alpha, beta, gamma}. No
    /// model, so the rerank path is testable without ONNX.
    struct StubEmbedder;

    #[async_trait]
    impl EmbeddingProvider for StubEmbedder {
        async fn embed(&self, texts: &[String]) -> crate::core::Result<Vec<Embedding>> {
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
    fn code_chunks_stay_lexical_and_survive_in_hybrid() {
        rt().block_on(async {
            let mut r = LocalRerankRetriever::new(Arc::new(StubEmbedder), 10).unwrap();
            let mut code = Chunk::new("code", "alpha gamma", "main.py", TokenCount(2));
            code.metadata
                .insert("kind".into(), serde_json::Value::String("code".into()));
            let mut prose = Chunk::new("prose", "beta beta", "notes.md", TokenCount(2));
            prose
                .metadata
                .insert("kind".into(), serde_json::Value::String("prose".into()));
            r.index(&[code, prose]).await.unwrap();

            // Code was never embedded; prose was.
            let embs = r.embeddings().unwrap();
            assert!(embs.get("code").is_none(), "code should not be embedded");
            assert!(embs.get("prose").is_some(), "prose should be embedded");

            // A query matching the code chunk's terms still returns it (BM25 → RRF),
            // even though it has no embedding.
            let res = r.retrieve(&Query::new("alpha gamma"), 5).await.unwrap();
            assert!(
                res.iter().any(|x| x.chunk.id.as_str() == "code"),
                "lexical-only code chunk must survive hybrid retrieval"
            );
        });
    }

    #[test]
    fn reuses_precomputed_chunk_embeddings() {
        rt().block_on(async {
            let mut r = LocalRerankRetriever::new(Arc::new(StubEmbedder), 10).unwrap();
            // Chunk "a" carries an embedding the StubEmbedder would never produce
            // for "alpha alpha alpha" (it'd be [3,0,0]). If index() reused it, the
            // cached vector is this one; if it re-embedded, it'd be [3,0,0].
            let mut cs = chunks();
            cs[0].embedding = Some(Embedding::from(vec![9.0, 9.0, 9.0]));
            r.index(&cs).await.unwrap();
            let cached = r.embeddings().expect("dense retriever caches embeddings");
            assert_eq!(cached.get("a").unwrap().as_slice(), &[9.0, 9.0, 9.0]);
            // A chunk without a preset embedding is still computed normally.
            assert_eq!(cached.get("b").unwrap().as_slice(), &[0.0, 2.0, 0.0]);
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

    /// Embeds every text to a fixed vector — stands in for an asymmetric query
    /// embedder whose output is decided by the embedder, not the query text.
    struct ConstEmbedder(Vec<f32>);

    #[async_trait]
    impl EmbeddingProvider for ConstEmbedder {
        async fn embed(&self, texts: &[String]) -> crate::core::Result<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding::from(self.0.clone()))
                .collect())
        }
        fn dim(&self) -> usize {
            self.0.len()
        }
        fn name(&self) -> &'static str {
            "const"
        }
    }

    #[test]
    fn separate_query_embedder_drives_the_query_side() {
        rt().block_on(async {
            // Passages embedded by presence (StubEmbedder); the query embedder is a
            // *different* one that ignores the text and points at "gamma".
            let mut r = LocalRerankRetriever::new_with_query_embedder(
                Arc::new(StubEmbedder),
                Arc::new(ConstEmbedder(vec![0.0, 0.0, 1.0])),
                10,
            )
            .unwrap();
            r.index(&chunks()).await.unwrap();
            // Query text leans "alpha", but the query embedder forces gamma — so
            // chunk "c" wins, proving the query side used the query embedder.
            let q = Query::new("alpha beta gamma");
            let res = r.retrieve(&q, 1).await.unwrap();
            assert_eq!(res[0].chunk.id.as_str(), "c");
        });
    }
}
