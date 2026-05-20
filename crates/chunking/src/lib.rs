//! # neorag-chunking
//!
//! Chunking strategies built on top of [`neorag_core::Chunker`].
//!
//! NeoRAG ships three chunkers covering the practical quality/cost spectrum:
//!
//! - [`FixedChunker`] — deterministic token-window chunking. Cheapest and
//!   reproducible, used as a baseline.
//! - [`SentenceChunker`] — sentence-segmented, token-budgeted chunking. The
//!   right default for most retrieval workloads.
//! - [`adaptive::AdaptiveChunker`] — a foundation chunker that combines
//!   sentence segmentation with lightweight cohesion/density heuristics. The
//!   target architecture for evidence-aware chunking; ships with conservative
//!   heuristics today and is designed for future entropy/topic-purity
//!   extensions.
//!
//! Chunkers are paired with a pluggable [`TokenizerBackend`]. A built-in
//! [`tokenizer::WhitespaceTokenizer`] is provided as a zero-dependency
//! default; integrations with HuggingFace `tokenizers` or `tiktoken-rs` can be
//! added behind feature flags without changing the chunker API.
//!
//! [`TokenizerBackend`]: neorag_core::TokenizerBackend

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adaptive;
pub mod fixed;
pub mod sentence;
pub mod tokenizer;

pub use adaptive::AdaptiveChunker;
pub use fixed::FixedChunker;
pub use sentence::SentenceChunker;
pub use tokenizer::WhitespaceTokenizer;
