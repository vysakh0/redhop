//! Local dense rerank — "lexical topology first, semantic refinement second".
//!
//! [`LocalRerankRetriever`] composes a lexical first stage (BM25) with a dense
//! second stage: BM25 prunes the whole corpus to a candidate pool of
//! `candidate_pool` chunks, then that pool is reordered by cosine of the query
//! embedding against precomputed chunk embeddings, and the two rankings are
//! **RRF-fused** to produce the final hybrid order. RRF guarantees the
//! documented contract that hybrid is at least as large as either tier on its
//! own — a chunk BM25 ranked highly never silently disappears just because the
//! dense model didn't surface it. Matches a global dense retriever's recall on
//! the validated workloads while touching the dense model on a bounded pool
//! and needing **no ANN / vector index**.
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

        // Dense reranking of the embeddable (prose/data) subset of the pool.
        // Code chunks (no embedding) are absent here; they survive in `cand`.
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

        if dense.is_empty() {
            // Pure lexical pool (all code): BM25 ranking stands.
            let mut out = cand;
            out.truncate(top_k.max(1));
            return Ok(out);
        }
        // RRF-fuse the BM25 ranking with the dense reranking. This preserves
        // the hybrid contract: a chunk ranked highly by BM25 is never dropped
        // just because the dense model demoted it (issue #1). The previous
        // prose-pool shortcut (return dense alone) silently lost BM25-only
        // hits whenever cosine pushed them past `top_k`.
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
            // Lexically matches all three; the embedding points at "gamma" —
            // c is rank-1 in dense (only positive cosine). RRF puts c near
            // the top, but the exact rank depends on BM25 micro-stats which
            // can shift as the analyzer evolves; pin the structural invariant
            // instead: top result was RRF-fused (method=Hybrid + fused
            // breakdown) and c is in the result set.
            let q =
                Query::new("alpha beta gamma").with_embedding(Embedding::from(vec![0.0, 0.0, 1.0]));
            let res = r.retrieve(&q, 3).await.unwrap();
            assert!(
                res.iter().any(|x| x.chunk.id.as_str() == "c"),
                "c (the dense favorite) must be in the top-3, got {:?}",
                res.iter().map(|x| x.chunk.id.as_str()).collect::<Vec<_>>()
            );
            assert_eq!(res[0].score.method, RetrievalMethod::Hybrid);
            assert!(
                res[0].breakdown.fused.is_some(),
                "RRF must populate the fused score on every result"
            );
        });
    }

    /// Regression for issue #1: hybrid must RRF-fuse BM25 + dense rather than
    /// use dense-only. Pre-fix the "pure-prose pool" branch returned
    /// `dense.truncate(k)`, silently discarding the BM25 ranking. RRF restores
    /// the documented hybrid contract that BM25 signal contributes to every
    /// result and that the result count never drops below lexical's.
    #[test]
    fn hybrid_fuses_bm25_with_dense_when_dense_disagrees() {
        rt().block_on(async {
            let mut r = LocalRerankRetriever::new(Arc::new(StubEmbedder), 10).unwrap();
            // Five prose chunks all matching "alpha" to varying degrees; the
            // dense embedder points at "gamma" so the cosine ranking is
            // anti-correlated with the BM25 ranking — the worst case for
            // dense-only retrieval.
            let cs = vec![
                Chunk::new("a0", "alpha alpha alpha gamma", "doc", TokenCount(4)),
                Chunk::new("a1", "alpha alpha gamma gamma", "doc", TokenCount(4)),
                Chunk::new("a2", "alpha gamma gamma gamma", "doc", TokenCount(4)),
                Chunk::new("a3", "alpha alpha", "doc", TokenCount(2)),
                Chunk::new("a4", "alpha", "doc", TokenCount(1)),
            ];
            r.index(&cs).await.unwrap();
            let q = Query::new("alpha").with_embedding(Embedding::from(vec![0.0, 0.0, 1.0]));

            // Pre-fix structural marker #1: method was `Rerank` (dense-only).
            // Post-fix: every result is `Hybrid` (RRF-fused).
            let hyb = r.retrieve(&q, 3).await.unwrap();
            assert!(!hyb.is_empty(), "hybrid must not be empty when pool is non-empty");
            for r in &hyb {
                assert_eq!(
                    r.score.method,
                    RetrievalMethod::Hybrid,
                    "issue #1: every hybrid result must carry the Hybrid method (RRF), \
                     got {:?} for {}",
                    r.score.method,
                    r.chunk.id.as_str(),
                );
                assert!(
                    r.breakdown.fused.is_some(),
                    "issue #1: every hybrid result must have a fused RRF score, missing on {}",
                    r.chunk.id.as_str(),
                );
            }

            // Count parity across top_k values: RRF preserves the candidate
            // count from the pool, never truncating below what lexical would
            // return on the same indexed corpus.
            for k in [1usize, 2, 3, 5] {
                let lex = r.bm25.retrieve(&q, k).await.unwrap();
                let hyb = r.retrieve(&q, k).await.unwrap();
                assert_eq!(
                    hyb.len(),
                    lex.len(),
                    "issue #1: top_k={}: hybrid count {} != lexical count {}",
                    k,
                    hyb.len(),
                    lex.len(),
                );
            }
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
            // Query text would rank "alpha" highly under BM25, but the query
            // embedder ignores the text and forces dense alignment with chunk
            // "c" (the only one whose vector matches [0,0,1]). The final
            // ranking is RRF-fused (BM25 + dense), so we don't assert top-1
            // — we assert the *dense* score on c was the strongest, which is
            // the directly-observable proof that the const query embedder fed
            // the query side rather than the passage embedder.
            let q = Query::new("alpha beta gamma");
            let res = r.retrieve(&q, 3).await.unwrap();
            let c = res
                .iter()
                .find(|r| r.chunk.id.as_str() == "c")
                .expect("c must be in the top-3 results");
            let c_dense = c.breakdown.dense.expect("c has a dense breakdown score");
            for r in &res {
                if r.chunk.id.as_str() == "c" {
                    continue;
                }
                let other = r.breakdown.dense.expect("each result has a dense score");
                assert!(
                    c_dense > other,
                    "c's dense score ({}) must exceed {}'s ({}) — proves the const \
                     query embedder pointed at gamma",
                    c_dense,
                    r.chunk.id.as_str(),
                    other,
                );
            }
        });
    }
}
