//! Curated model registry + HuggingFace auto-download for the dense tier.
//!
//! Removes the "export your own ONNX" friction: a user names a model
//! (`"bge-small"`) and we fetch a pre-built ONNX export from a pinned,
//! curated HuggingFace repo, cache it, and hand back local paths + the right
//! [`EmbedderConfig`]. Downloads use `hf-hub` (gated behind the `onnx` feature),
//! cache in the standard HF cache, and honor explicit paths / offline use:
//! the caller can always bypass this and pass `embedder_model`/`embedder_tokenizer`
//! directly.

use std::path::PathBuf;

use redhop_core::{Error, Result};

use crate::config::EmbedderConfig;
use crate::pooling::Pooling;

/// A known embedding model: where to fetch it and how to configure it.
struct ModelSpec {
    /// HuggingFace repo id holding a pre-built ONNX export.
    repo: &'static str,
    /// Pinned revision (commit SHA recommended for reproducibility; `"main"`
    /// is a pragmatic default until SHAs are pinned).
    revision: &'static str,
    /// Every file to fetch — the ONNX graph, any external-weight `*.onnx_data`
    /// companion, and the tokenizer. All land co-located in the cache snapshot.
    files: &'static [&'static str],
    /// The ONNX graph file (handed to `OnnxEmbedder`).
    onnx_file: &'static str,
    /// The tokenizer file.
    tokenizer_file: &'static str,
    dim: usize,
    pooling: Pooling,
    /// Query/passage prefixes. Equal (usually empty) ⇒ symmetric model (BGE,
    /// MiniLM); different ⇒ asymmetric (E5) and we build two embedders.
    query_prefix: &'static str,
    passage_prefix: &'static str,
}

/// The default rerank model: a single self-contained int8 ONNX file
/// (fastest to index, smallest download) — see `docs/findings/SPEED_VS_FRAMEWORKS.md`.
pub const DEFAULT_MODEL: &str = "bge-small";

const REGISTRY: &[(&str, ModelSpec)] = &[
    (
        "bge-small",
        ModelSpec {
            repo: "Qdrant/bge-small-en-v1.5-onnx-Q",
            // Pinned for reproducibility (was "main"). Bump deliberately.
            revision: "52398278842ec682c6f32300af41344b1c0b0bb2",
            files: &["model_optimized.onnx", "tokenizer.json"],
            onnx_file: "model_optimized.onnx",
            tokenizer_file: "tokenizer.json",
            dim: 384,
            pooling: Pooling::Cls,
            query_prefix: "",
            passage_prefix: "",
        },
    ),
    (
        "bge-base",
        ModelSpec {
            repo: "Qdrant/bge-base-en-v1.5-onnx-Q",
            revision: "738cad1c108e2f23649db9e44b2eab988626493b",
            files: &["model_optimized.onnx", "tokenizer.json"],
            onnx_file: "model_optimized.onnx",
            tokenizer_file: "tokenizer.json",
            dim: 768,
            pooling: Pooling::Cls,
            query_prefix: "",
            passage_prefix: "",
        },
    ),
];

/// Names of the models RedHop can fetch by name.
pub fn available_models() -> Vec<&'static str> {
    REGISTRY.iter().map(|(n, _)| *n).collect()
}

// ── Cross-encoder rerankers ───────────────────────────────────────────────────

/// A known cross-encoder reranker: where to fetch its ONNX export + tokenizer.
struct RerankerSpec {
    repo: &'static str,
    revision: &'static str,
    files: &'static [&'static str],
    onnx_file: &'static str,
    tokenizer_file: &'static str,
    /// Token cap per `(query, passage)` pair.
    max_seq_len: usize,
}

/// The default cross-encoder: the MS-MARCO MiniLM reranker, the robust
/// second-stage winner in our method comparison.
pub const DEFAULT_RERANKER: &str = "cross-encoder";

const RERANKER_REGISTRY: &[(&str, RerankerSpec)] = &[(
    "cross-encoder",
    RerankerSpec {
        // Pre-built ONNX export of cross-encoder/ms-marco-MiniLM-L-6-v2.
        repo: "Xenova/ms-marco-MiniLM-L-6-v2",
        revision: "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        files: &["onnx/model.onnx", "tokenizer.json"],
        onnx_file: "onnx/model.onnx",
        tokenizer_file: "tokenizer.json",
        max_seq_len: 512,
    },
)];

/// Names of the cross-encoder rerankers RedHop can fetch by name.
pub fn available_rerankers() -> Vec<&'static str> {
    RERANKER_REGISTRY.iter().map(|(n, _)| *n).collect()
}

/// A resolved, on-disk cross-encoder ready for `OnnxCrossEncoder::load`.
pub struct ResolvedReranker {
    /// Local path to the ONNX graph file.
    pub model_path: PathBuf,
    /// Local path to the tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Token cap per `(query, passage)` pair.
    pub max_seq_len: usize,
}

/// Resolve a cross-encoder name to local files, downloading from HuggingFace if
/// absent. Honors the standard HF cache, like [`resolve_model`].
pub fn resolve_reranker(name: &str) -> Result<ResolvedReranker> {
    let spec = RERANKER_REGISTRY
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s)
        .ok_or_else(|| {
            Error::Reranking(format!(
                "unknown reranker '{name}'. Known: {}.",
                available_rerankers().join(", ")
            ))
        })?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(false)
        .build()
        .map_err(|e| Error::Reranking(format!("hf-hub init failed: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        spec.repo.to_string(),
        hf_hub::RepoType::Model,
        spec.revision.to_string(),
    ));

    let mut model_path = None;
    let mut tokenizer_path = None;
    for f in spec.files {
        let p = repo.get(f).map_err(|e| {
            Error::Reranking(format!(
                "failed to fetch {}/{f} ({e}). For offline use, pre-download the model.",
                spec.repo
            ))
        })?;
        if *f == spec.onnx_file {
            model_path = Some(p);
        } else if *f == spec.tokenizer_file {
            tokenizer_path = Some(p);
        }
    }

    Ok(ResolvedReranker {
        model_path: model_path.expect("onnx_file is in files"),
        tokenizer_path: tokenizer_path.expect("tokenizer_file is in files"),
        max_seq_len: spec.max_seq_len,
    })
}

/// A resolved, on-disk model ready to hand to [`crate::OnnxEmbedder::load`].
pub struct ResolvedModel {
    /// Local path to the ONNX graph file.
    pub model_path: PathBuf,
    /// Local path to the tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Output embedding dimensionality.
    pub dim: usize,
    /// Token pooling strategy for this model.
    pub pooling: Pooling,
    /// Query-side prefix (empty for symmetric models like BGE/MiniLM).
    pub query_prefix: String,
    /// Passage-side prefix (empty for symmetric models).
    pub passage_prefix: String,
}

impl ResolvedModel {
    /// True when query and passage need different prefixes (E5-style) — build
    /// a separate query-side embedder.
    pub fn is_asymmetric(&self) -> bool {
        self.query_prefix != self.passage_prefix
    }

    /// An [`EmbedderConfig`] for one side, with the model's pooling + the given prefix.
    pub fn config(&self, prefix: &str) -> EmbedderConfig {
        let mut c = EmbedderConfig::bge(self.dim); // bge = CLS + L2 + no prefix
        c.pooling = self.pooling;
        c.prefix = prefix.to_string();
        c
    }
}

/// Resolve a model name to local files, downloading from HuggingFace if absent.
///
/// Honors the standard HF cache (`HF_HOME` / `~/.cache/huggingface`). For
/// offline use, either pre-warm the cache or bypass this entirely by passing
/// explicit `embedder_model`/`embedder_tokenizer` paths.
pub fn resolve_model(name: &str) -> Result<ResolvedModel> {
    let spec = REGISTRY
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s)
        .ok_or_else(|| {
            Error::Embedding(format!(
                "unknown model '{name}'. Known models: {}. Or pass explicit \
                 embedder_model/embedder_tokenizer paths to use any ONNX model.",
                available_models().join(", ")
            ))
        })?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(false)
        .build()
        .map_err(|e| Error::Embedding(format!("hf-hub init failed: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        spec.repo.to_string(),
        hf_hub::RepoType::Model,
        spec.revision.to_string(),
    ));

    let mut model_path = None;
    let mut tokenizer_path = None;
    for f in spec.files {
        let p = repo.get(f).map_err(|e| {
            Error::Embedding(format!(
                "failed to fetch {}/{f} ({e}). For offline/air-gapped use, pre-download \
                 the model or pass explicit embedder_model/embedder_tokenizer paths.",
                spec.repo
            ))
        })?;
        if *f == spec.onnx_file {
            model_path = Some(p);
        } else if *f == spec.tokenizer_file {
            tokenizer_path = Some(p);
        }
    }

    Ok(ResolvedModel {
        model_path: model_path.expect("onnx_file is in files"),
        tokenizer_path: tokenizer_path.expect("tokenizer_file is in files"),
        dim: spec.dim,
        pooling: spec.pooling,
        query_prefix: spec.query_prefix.to_string(),
        passage_prefix: spec.passage_prefix.to_string(),
    })
}
