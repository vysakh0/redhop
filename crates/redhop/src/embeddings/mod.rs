//! # redhop-embeddings
//!
//! Pluggable embedding backends implementing
//! [`crate::core::EmbeddingProvider`].
//!
//! Two always-available, zero-heavyweight-dep pieces:
//!
//! - [`HashingProvider`] — deterministic feature-hashing TF baseline.
//!   The control arm for embedder bake-offs and the default for
//!   tests / CI / cold-start.
//! - [`CachedEmbedder`] — bounded LRU wrapper over any provider.
//!
//! Plus the pooling/normalization math ([`pooling`]) and model config
//! ([`config`]) that the transformer backends share — kept hermetic and
//! unit-tested so the error-prone parts (mask-aware mean pooling, E5
//! prefixes, L2 normalization) are validated without a model.
//!
//! ## The `onnx` feature
//!
//! Enabling `--features onnx` adds [`onnx::OnnxEmbedder`], a real
//! ONNX-Runtime backend for BGE / E5 / jina / mxbai. It pulls in `ort`
//! and `tokenizers`; it is **not** in the default build, so a
//! dependency-light, fully offline build stays possible. The ONNX glue
//! delegates all math to [`pooling`], so the only feature-gated,
//! model-dependent surface is the tokenize + session-run + tensor-shape
//! handling.
//!
//! ## Philosophy
//!
//! RedHop is a retrieval *controller/runtime*, not an embedding library.
//! These backends exist so the controller has real vectors to reason
//! over — they are not a model zoo. One strong default (BGE/E5 via ONNX)
//! plus a no-dep baseline is the whole scope.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod config;
pub mod hashing;
pub mod pooling;

#[cfg(feature = "semantic")]
pub mod onnx;

#[cfg(feature = "semantic")]
pub mod registry;

pub use cache::CachedEmbedder;
pub use config::EmbedderConfig;
pub use hashing::HashingProvider;
pub use pooling::{l2_normalize, pool, Pooling};

#[cfg(feature = "semantic")]
pub use onnx::OnnxEmbedder;

#[cfg(feature = "semantic")]
pub use registry::{
    available_models, available_rerankers, resolve_model, resolve_reranker, ResolvedModel,
    ResolvedReranker, DEFAULT_MODEL, DEFAULT_RERANKER,
};
