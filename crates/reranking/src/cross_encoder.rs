//! Cross-encoder reranking.
//!
//! A cross-encoder scores a `(query, passage)` pair jointly, which is
//! far more accurate than the bi-encoder cosine used for first-stage
//! retrieval — and far more expensive. That cost asymmetry is exactly
//! why RedHop's selective escalation matters: the cross-encoder should
//! fire only on the queries the controller judges worth it.
//!
//! This module splits the cheap, testable decision logic
//! ([`apply_scores`]) from the model-dependent ONNX inference
//! ([`OnnxCrossEncoder`], behind the `onnx` feature). The scoring
//! arithmetic — pairing scores back to candidates, sorting, truncating
//! — is unit-tested without a model.

use redhop_core::{RetrievalMethod, RetrievalResult, Score};

/// Given candidates and a parallel vector of cross-encoder relevance
/// scores, produce the reranked top-`k`.
///
/// `scores[i]` is the relevance of `candidates[i]`. The function tags
/// each result with [`RetrievalMethod::Rerank`], records the score in
/// the breakdown, sorts descending, and truncates. Pure and
/// deterministic — this is the part that must never be wrong, so it is
/// tested independently of any model.
pub fn apply_scores(
    mut candidates: Vec<RetrievalResult>,
    scores: &[f32],
    top_k: usize,
) -> Vec<RetrievalResult> {
    debug_assert_eq!(candidates.len(), scores.len());
    for (r, &s) in candidates.iter_mut().zip(scores.iter()) {
        r.breakdown.rerank = Some(s);
        r.score = Score {
            value: s,
            method: RetrievalMethod::Rerank,
        };
    }
    candidates.sort_by(|a, b| {
        b.score
            .value
            .partial_cmp(&a.score.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(top_k);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use redhop_core::{Chunk, RetrievalMethod, Score, ScoreBreakdown, TokenCount};

    fn r(id: &str, retrieval_score: f32) -> RetrievalResult {
        RetrievalResult {
            chunk: Chunk::new(id, id, "doc", TokenCount(1)),
            score: Score {
                value: retrieval_score,
                method: RetrievalMethod::Dense,
            },
            breakdown: ScoreBreakdown::default(),
        }
    }

    #[test]
    fn reorders_by_cross_encoder_score() {
        // First-stage order: a > b > c. Cross-encoder disagrees: c best.
        let cands = vec![r("a", 0.9), r("b", 0.8), r("c", 0.7)];
        let ce_scores = [0.1, 0.5, 0.95];
        let out = apply_scores(cands, &ce_scores, 3);
        assert_eq!(out[0].chunk.id.as_str(), "c");
        assert_eq!(out[1].chunk.id.as_str(), "b");
        assert_eq!(out[2].chunk.id.as_str(), "a");
        assert_eq!(out[0].score.method, RetrievalMethod::Rerank);
        assert_eq!(out[0].breakdown.rerank, Some(0.95));
    }

    #[test]
    fn truncates_to_top_k() {
        let cands = vec![r("a", 0.1), r("b", 0.2), r("c", 0.3), r("d", 0.4)];
        let ce = [0.4, 0.3, 0.2, 0.1];
        let out = apply_scores(cands, &ce, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk.id.as_str(), "a"); // highest CE score
        assert_eq!(out[1].chunk.id.as_str(), "b");
    }
}

#[cfg(feature = "onnx")]
mod onnx_impl {
    use std::path::Path;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;
    use redhop_core::{Error, Query, Reranker, Result, RetrievalResult};
    use tokenizers::Tokenizer;

    use super::apply_scores;

    /// ONNX cross-encoder reranker (e.g.
    /// `cross-encoder/ms-marco-MiniLM-L-6-v2`).
    ///
    /// Scores each `(query, passage)` pair jointly. CPU-first; the
    /// session is internally thread-safe and shared behind a `Mutex`
    /// because `Session::run` takes `&mut self` in this `ort` line.
    ///
    /// Selective use is the whole point — pair this with the adaptive
    /// controller's `EscalateReranker` action so it fires only on the
    /// queries that warrant the cost.
    pub struct OnnxCrossEncoder {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
        max_seq_len: usize,
        has_token_type_ids: bool,
    }

    impl OnnxCrossEncoder {
        /// Load a cross-encoder model + tokenizer from disk.
        pub fn load(
            model_path: impl AsRef<Path>,
            tokenizer_path: impl AsRef<Path>,
            max_seq_len: usize,
        ) -> Result<Self> {
            let session = Session::builder()
                .map_err(|e| Error::Reranking(format!("ort builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| Error::Reranking(format!("ort opt: {e}")))?
                .commit_from_file(model_path.as_ref())
                .map_err(|e| Error::Reranking(format!("ort load: {e}")))?;
            let has_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");
            let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
                .map_err(|e| Error::Reranking(format!("tokenizer load: {e}")))?;
            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                max_seq_len,
                has_token_type_ids,
            })
        }

        fn score_pairs(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
            if passages.is_empty() {
                return Ok(Vec::new());
            }
            // Encode (query, passage) pairs. tokenizers supports pair
            // encoding via a tuple input.
            let pairs: Vec<(String, String)> = passages
                .iter()
                .map(|p| (query.to_string(), p.clone()))
                .collect();
            let encodings = self
                .tokenizer
                .encode_batch(pairs, true)
                .map_err(|e| Error::Reranking(format!("tokenize: {e}")))?;

            let batch = encodings.len();
            let seq_len = encodings
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0)
                .min(self.max_seq_len)
                .max(1);

            let mut input_ids = vec![0i64; batch * seq_len];
            let mut attention_mask = vec![0i64; batch * seq_len];
            let token_type_ids = {
                let mut tt = vec![0i64; batch * seq_len];
                for (b, enc) in encodings.iter().enumerate() {
                    let types = enc.get_type_ids();
                    let n = types.len().min(seq_len);
                    for t in 0..n {
                        tt[b * seq_len + t] = types[t] as i64;
                    }
                }
                tt
            };
            for (b, enc) in encodings.iter().enumerate() {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                let n = ids.len().min(seq_len);
                for t in 0..n {
                    input_ids[b * seq_len + t] = ids[t] as i64;
                    attention_mask[b * seq_len + t] = mask[t] as i64;
                }
            }

            let shape = [batch, seq_len];
            let ids_t = Tensor::from_array((shape, input_ids))
                .map_err(|e| Error::Reranking(format!("ids tensor: {e}")))?;
            let mask_t = Tensor::from_array((shape, attention_mask))
                .map_err(|e| Error::Reranking(format!("mask tensor: {e}")))?;

            let mut session = self.session.lock().unwrap();
            let outputs = if self.has_token_type_ids {
                let tt_t = Tensor::from_array((shape, token_type_ids))
                    .map_err(|e| Error::Reranking(format!("tt tensor: {e}")))?;
                session
                    .run(ort::inputs![
                        "input_ids" => ids_t,
                        "attention_mask" => mask_t,
                        "token_type_ids" => tt_t,
                    ])
                    .map_err(|e| Error::Reranking(format!("ort run: {e}")))?
            } else {
                session
                    .run(ort::inputs![
                        "input_ids" => ids_t,
                        "attention_mask" => mask_t,
                    ])
                    .map_err(|e| Error::Reranking(format!("ort run: {e}")))?
            };

            // Cross-encoder logits: [batch, 1] (regression) or
            // [batch, num_labels]. We take the first logit per row as the
            // relevance score (ms-marco MiniLM is single-logit).
            let (_name, value) = outputs
                .iter()
                .next()
                .ok_or_else(|| Error::Reranking("no model output".into()))?;
            let (out_shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Reranking(format!("extract: {e}")))?;
            let cols = if out_shape.len() >= 2 {
                out_shape[1] as usize
            } else {
                1
            };
            let scores = (0..batch).map(|b| data[b * cols]).collect();
            Ok(scores)
        }
    }

    #[async_trait]
    impl Reranker for OnnxCrossEncoder {
        async fn rerank(
            &self,
            query: &Query,
            candidates: Vec<RetrievalResult>,
            top_k: usize,
        ) -> Result<Vec<RetrievalResult>> {
            if candidates.is_empty() {
                return Ok(candidates);
            }
            let passages: Vec<String> = candidates.iter().map(|r| r.chunk.text.clone()).collect();
            let scores = self.score_pairs(&query.text, &passages)?;
            Ok(apply_scores(candidates, &scores, top_k))
        }

        fn name(&self) -> &'static str {
            "onnx_cross_encoder"
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_impl::OnnxCrossEncoder;
