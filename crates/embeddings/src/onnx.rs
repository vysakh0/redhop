//! ONNX-Runtime embedding backend (feature `onnx`).
//!
//! A real [`EmbeddingProvider`] for BERT-family sentence embedders
//! (BGE / E5 / jina / mxbai) exported to ONNX. It does the minimum:
//! tokenize a batch, run the session, pool, normalize. All math is
//! delegated to [`crate::pooling`], which is unit-tested without a
//! model — so the only model-dependent surface here is tokenization +
//! tensor shapes + the session call.
//!
//! ## Loading
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # {
//! use redhop_embeddings::{OnnxEmbedder, EmbedderConfig};
//! // BGE-small: CLS pooling, normalize, 384-dim.
//! let embedder = OnnxEmbedder::load(
//!     "bge-small-en-v1.5/model.onnx",
//!     "bge-small-en-v1.5/tokenizer.json",
//!     EmbedderConfig::bge(384),
//! ).unwrap();
//! # }
//! ```
//!
//! For E5, build two embedders sharing intent but different prefixes:
//! one with `EmbedderConfig::e5(384, "query: ")` for queries and one
//! with `"passage: "` for documents.
//!
//! ## Threading note
//!
//! `embed` runs inference synchronously inside the async method. ONNX
//! inference is CPU-bound; for high concurrency, batch aggressively
//! (one `embed` call with many texts) rather than many concurrent
//! single-text calls. The session is internally thread-safe.

use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use redhop_core::{Embedding, EmbeddingProvider, Error, Result};
use tokenizers::Tokenizer;

use crate::config::EmbedderConfig;
use crate::pooling::{l2_normalize, pool};

/// Texts per model run. With length-sorted batching the padding within a window
/// is tight, so this is purely a throughput/latency knob, not a waste lever.
const SUB_BATCH: usize = 64;

/// ONNX-Runtime sentence embedder.
pub struct OnnxEmbedder {
    // `Session::run` takes `&mut self` in this ort line; a Mutex gives
    // us `&self`-shaped access for the EmbeddingProvider trait while
    // keeping the session thread-safe to share.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: EmbedderConfig,
    has_token_type_ids: bool,
}

impl OnnxEmbedder {
    /// Load a model + tokenizer from disk.
    ///
    /// `token_type_ids` support is auto-detected from the model's input
    /// names; BGE/E5 BERT exports usually take `token_type_ids`,
    /// RoBERTa-based ones do not.
    pub fn load(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: EmbedderConfig,
    ) -> Result<Self> {
        // Intra-op thread count for the CPU EP. ORT defaults to ~single-threaded
        // under `ort` here, which makes batched embedding far slower than it needs
        // to be. Measured on this `ort`/ONNX-Runtime build, embedding throughput
        // *improves* with all cores (10 threads ~51s vs 4 threads ~72s for the
        // rerank embed-all), so default to the core count. `REDHOP_ONNX_INTRA_THREADS`
        // overrides for other hardware/runtime versions.
        let intra_threads = std::env::var("REDHOP_ONNX_INTRA_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        let session = Session::builder()
            .map_err(|e| Error::Embedding(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Embedding(format!("ort opt level: {e}")))?
            .with_intra_threads(intra_threads)
            .map_err(|e| Error::Embedding(format!("ort intra threads: {e}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| Error::Embedding(format!("ort load: {e}")))?;

        let has_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Embedding(format!("tokenizer load: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
            has_token_type_ids,
        })
    }

    /// Embed many texts efficiently. Tokenize once, then run the model in
    /// **length-sorted sub-batches** so each batch pads only to its own longest
    /// item. Unsorted batching pads every text in a batch to the single longest
    /// chunk in it — measured ~45% wasted forward-pass compute on real corpora,
    /// because one long chunk drags the whole batch's sequence length up. Sorting
    /// by length first reclaims that. Results are returned in the caller's order.
    fn embed_many(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let profile = std::env::var("REDHOP_EMBED_PROFILE").is_ok();
        let t_tok = std::time::Instant::now();
        let prefixed: Vec<String> = texts.iter().map(|t| self.config.apply_prefix(t)).collect();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|e| Error::Embedding(format!("tokenize: {e}")))?;
        if profile {
            eprintln!(
                "[embed] tokenized {} texts in {:.1}ms",
                texts.len(),
                t_tok.elapsed().as_secs_f64() * 1e3
            );
        }

        // Group similar lengths together; longest-first so peak memory is hit
        // early and the many short batches run fast.
        let mut order: Vec<usize> = (0..encodings.len()).collect();
        let len_of = |i: usize| encodings[i].get_ids().len().min(self.config.max_seq_len);
        order.sort_by_key(|&i| std::cmp::Reverse(len_of(i)));

        let mut out: Vec<Option<Embedding>> = (0..texts.len()).map(|_| None).collect();
        for window in order.chunks(SUB_BATCH) {
            let embs = self.run_window(&encodings, window, profile)?;
            for (&orig, e) in window.iter().zip(embs) {
                out[orig] = Some(e);
            }
        }
        Ok(out.into_iter().map(|e| e.expect("every index filled")).collect())
    }

    /// Run the model over one sub-batch — the indices `idxs` into `encodings` —
    /// padded to that sub-batch's own max length. Returns embeddings in `idxs`
    /// order. Pooling + normalization happen here.
    fn run_window(
        &self,
        encodings: &[tokenizers::Encoding],
        idxs: &[usize],
        profile: bool,
    ) -> Result<Vec<Embedding>> {
        let batch = idxs.len();
        let seq_len = idxs
            .iter()
            .map(|&i| encodings[i].get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_seq_len)
            .max(1);

        let mut input_ids = vec![0i64; batch * seq_len];
        let mut attention_mask = vec![0i64; batch * seq_len];
        let token_type_ids = vec![0i64; batch * seq_len];
        for (b, &i) in idxs.iter().enumerate() {
            let enc = &encodings[i];
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
            .map_err(|e| Error::Embedding(format!("ids tensor: {e}")))?;
        let mask_t = Tensor::from_array((shape, attention_mask.clone()))
            .map_err(|e| Error::Embedding(format!("mask tensor: {e}")))?;

        let t_run = std::time::Instant::now();
        let mut session = self.session.lock().unwrap();
        let outputs = if self.has_token_type_ids {
            let tt_t = Tensor::from_array((shape, token_type_ids))
                .map_err(|e| Error::Embedding(format!("tt tensor: {e}")))?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "token_type_ids" => tt_t,
                ])
                .map_err(|e| Error::Embedding(format!("ort run: {e}")))?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                ])
                .map_err(|e| Error::Embedding(format!("ort run: {e}")))?
        };

        if profile {
            eprintln!(
                "[embed] window batch={batch} seq_len={seq_len} run={:.1}ms",
                t_run.elapsed().as_secs_f64() * 1e3
            );
        }
        // First output = token-level last_hidden_state [batch, seq, hidden].
        let (_name, value) = outputs
            .iter()
            .next()
            .ok_or_else(|| Error::Embedding("model produced no outputs".into()))?;
        let (out_shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(format!("extract: {e}")))?;

        if out_shape.len() != 3 {
            return Err(Error::Embedding(format!(
                "expected [batch, seq, hidden] output, got shape {out_shape:?}"
            )));
        }
        let out_seq = out_shape[1] as usize;
        let hidden = out_shape[2] as usize;

        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            let row_off = b * out_seq * hidden;
            let row = &data[row_off..row_off + out_seq * hidden];
            let mut mask_f = vec![0f32; out_seq];
            let mask_base = b * seq_len;
            for t in 0..out_seq.min(seq_len) {
                mask_f[t] = attention_mask[mask_base + t] as f32;
            }
            let mut v = pool(row, out_seq, hidden, &mask_f, self.config.pooling);
            if self.config.normalize {
                l2_normalize(&mut v);
            }
            results.push(Embedding(v));
        }
        Ok(results)
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.embed_many(texts)
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn name(&self) -> &'static str {
        "onnx"
    }
}
