//! Embedder configuration.

use crate::pooling::Pooling;
use serde::{Deserialize, Serialize};

/// Configuration for a transformer embedding backend.
///
/// The fields here are exactly the knobs that, set wrong, silently tank
/// retrieval quality:
///
/// - **`prefix`** — E5 *requires* `"query: "` / `"passage: "` prefixes;
///   omitting them or using the wrong one collapses recall. BGE uses an
///   empty passage prefix and an optional query instruction. Because the
///   [`EmbeddingProvider`][ep] trait does not distinguish query from
///   document, the pattern is: construct one embedder with the query
///   prefix and one with the passage prefix, sharing the underlying
///   model session.
/// - **`pooling`** — mean vs CLS is model-specific (see [`Pooling`]).
/// - **`normalize`** — must be `true` for cosine == dot-product.
/// - **`max_seq_len`** — inputs longer than the model context must be
///   truncated or the session errors.
///
/// [ep]: redhop_core::EmbeddingProvider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Output embedding dimensionality (384 for *-small, 768 for *-base,
    /// 1024 for mxbai).
    pub dim: usize,
    /// Token pooling strategy.
    pub pooling: Pooling,
    /// Whether to L2-normalize the pooled vector.
    pub normalize: bool,
    /// String prepended to every input before tokenization. Empty for
    /// BGE passages; `"query: "` / `"passage: "` for E5.
    pub prefix: String,
    /// Maximum sequence length; inputs are truncated to this many tokens.
    pub max_seq_len: usize,
}

impl EmbedderConfig {
    /// BGE-small / bge-base style: CLS pooling, normalize, no prefix.
    pub fn bge(dim: usize) -> Self {
        Self {
            dim,
            pooling: Pooling::Cls,
            normalize: true,
            prefix: String::new(),
            max_seq_len: 512,
        }
    }

    /// E5-small / e5-base style: mean pooling, normalize, explicit
    /// prefix. Pass `"query: "` for a query embedder, `"passage: "` for
    /// a document embedder.
    pub fn e5(dim: usize, prefix: impl Into<String>) -> Self {
        Self {
            dim,
            pooling: Pooling::Mean,
            normalize: true,
            prefix: prefix.into(),
            max_seq_len: 512,
        }
    }

    /// mixedbread mxbai style: CLS pooling, normalize, 1024-dim.
    pub fn mxbai() -> Self {
        Self {
            dim: 1024,
            pooling: Pooling::Cls,
            normalize: true,
            prefix: String::new(),
            max_seq_len: 512,
        }
    }

    /// Apply the configured prefix to an input string.
    pub fn apply_prefix(&self, text: &str) -> String {
        if self.prefix.is_empty() {
            text.to_string()
        } else {
            format!("{}{}", self.prefix, text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e5_prefix_is_applied() {
        let cfg = EmbedderConfig::e5(384, "query: ");
        assert_eq!(cfg.apply_prefix("hello"), "query: hello");
        assert_eq!(cfg.pooling, Pooling::Mean);
    }

    #[test]
    fn bge_has_no_prefix_and_cls_pooling() {
        let cfg = EmbedderConfig::bge(384);
        assert_eq!(cfg.apply_prefix("hello"), "hello");
        assert_eq!(cfg.pooling, Pooling::Cls);
        assert!(cfg.normalize);
    }
}
