//! Core data types shared across RedHop crates.
//!
//! These types are deliberately small, owned, and `Serialize` / `Deserialize`
//! so they can cross language boundaries (Python / Node) without surprises.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stable identifier for a chunk within a RedHop corpus.
///
/// Chunks are identified by an opaque string rather than an integer so that
/// hashes, UUIDs, or source-derived keys can all be used. Identifiers are
/// expected to be unique within a single index instance.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub String);

impl ChunkId {
    /// Construct a new chunk identifier from any string-like value.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ChunkId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ChunkId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Number of tokens in a piece of text, as reported by a [`TokenizerBackend`].
///
/// Wrapped in a newtype to prevent accidental confusion with byte/char counts.
///
/// [`TokenizerBackend`]: crate::core::traits::TokenizerBackend
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TokenCount(pub usize);

impl TokenCount {
    /// Underlying integer count.
    pub fn value(self) -> usize {
        self.0
    }
}

/// A single dense vector embedding.
///
/// RedHop is not an embedding library; callers supply vectors produced by
/// whatever model they prefer. The dimensionality is checked at retrieval
/// time by the dense index (see [`Error::DimensionMismatch`]).
///
/// [`Error::DimensionMismatch`]: crate::Error::DimensionMismatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Number of dimensions in this embedding.
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the embedding is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Borrow the underlying vector.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(v: Vec<f32>) -> Self {
        Self(v)
    }
}

/// Arbitrary user metadata attached to chunks and documents.
///
/// Kept as JSON values (rather than a typed bag) so that downstream
/// applications can store anything from page numbers to author IDs without
/// requiring RedHop to know about it.
pub type ChunkMetadata = HashMap<String, serde_json::Value>;

/// A piece of indexable evidence.
///
/// A `Chunk` is the unit that retrievers operate over. Chunks are produced by
/// a [`Chunker`] from input [`Document`]s and may optionally carry a
/// precomputed [`Embedding`].
///
/// Chunks are intentionally **owned** rather than borrowed: the cost of a
/// `String` clone per chunk is negligible relative to retrieval work, and
/// owned data dramatically simplifies async pipelines and FFI bindings.
///
/// [`Chunker`]: crate::core::traits::Chunker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable identifier for this chunk.
    pub id: ChunkId,
    /// Raw text content.
    pub text: String,
    /// Source document identifier (path, URL, hash, …).
    pub source: String,
    /// Token count as reported by the configured tokenizer.
    pub token_count: TokenCount,
    /// Caller-supplied metadata; may include offsets, page numbers, headings.
    pub metadata: ChunkMetadata,
    /// Optional precomputed embedding. Dense retrievers require this to be
    /// populated; lexical retrievers ignore it.
    pub embedding: Option<Embedding>,
}

impl Chunk {
    /// Construct a new chunk with no metadata and no embedding.
    pub fn new<I: Into<ChunkId>, T: Into<String>, S: Into<String>>(
        id: I,
        text: T,
        source: S,
        token_count: TokenCount,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            source: source.into(),
            token_count,
            metadata: ChunkMetadata::default(),
            embedding: None,
        }
    }

    /// Attach an embedding to this chunk and return it.
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attach metadata to this chunk and return it.
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// An input document, prior to chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Source identifier (path, URL, hash, …).
    pub source: String,
    /// Raw text.
    pub text: String,
    /// Optional caller-supplied metadata; propagated to every chunk derived
    /// from this document.
    pub metadata: ChunkMetadata,
}

impl Document {
    /// Construct a new document with no metadata.
    pub fn new<S: Into<String>, T: Into<String>>(source: S, text: T) -> Self {
        Self {
            source: source.into(),
            text: text.into(),
            metadata: ChunkMetadata::default(),
        }
    }
}

/// A sentence segmentation result.
///
/// Carries both the sentence text and its byte offsets in the originating
/// document, which downstream chunkers use to reassemble groups of sentences
/// without re-running segmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentence {
    /// Sentence text.
    pub text: String,
    /// Byte offset of the sentence in the source string.
    pub start: usize,
    /// Byte offset one past the end of the sentence in the source string.
    pub end: usize,
}

/// A user query.
///
/// Currently just text plus an optional precomputed embedding, but kept as a
/// struct so additional signals (filters, intent labels, …) can be added
/// without breaking the retriever trait signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Raw query text.
    pub text: String,
    /// Optional precomputed embedding for dense retrieval.
    pub embedding: Option<Embedding>,
    /// Optional `top_k` override; retrievers may treat this as a hint.
    pub top_k: Option<usize>,
}

impl Query {
    /// Construct a text-only query.
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            embedding: None,
            top_k: None,
        }
    }

    /// Attach an embedding and return the query.
    pub fn with_embedding(mut self, e: Embedding) -> Self {
        self.embedding = Some(e);
        self
    }

    /// Set `top_k` and return the query.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
}

impl From<&str> for Query {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// A retrieval score with provenance.
///
/// Scores are kept as raw `f32`s — different retrievers report wildly
/// different ranges (BM25 is unbounded, cosine is `[-1, 1]`, RRF is small
/// positive) — and the [`RetrievalMethod`] tag lets fusion/reranking code
/// reason about that without normalizing destructively.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Score {
    /// Numeric score; higher is better.
    pub value: f32,
    /// Which method produced this score.
    pub method: RetrievalMethod,
}

/// Per-method scores that contributed to a single fused [`RetrievalResult`].
///
/// `lexical` / `dense` / `rerank` are populated as different stages of the
/// pipeline run; missing stages are simply `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Lexical (BM25) score, if any.
    pub lexical: Option<f32>,
    /// Dense (cosine) score, if any.
    pub dense: Option<f32>,
    /// Fused score (RRF, weighted, …), if any.
    pub fused: Option<f32>,
    /// Reranker score, if any.
    pub rerank: Option<f32>,
}

/// Which retrieval/reranking method produced a score.
///
/// Used both as a tag on [`Score`] and as a routing key inside hybrid
/// retrievers. Variants are explicit rather than open-ended to keep the
/// FFI surface stable.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum RetrievalMethod {
    /// BM25 / lexical retrieval.
    Lexical,
    /// Dense vector retrieval.
    Dense,
    /// A hybrid of lexical + dense, fused via RRF or weighted sum.
    Hybrid,
    /// Output of a reranker stage.
    Rerank,
    /// Caller-provided / external retriever.
    External,
}

/// A single retrieval result.
///
/// `score` is the *primary* score used for ordering. `breakdown` records what
/// each stage contributed so diagnostics and rerankers can reason about it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// The chunk that was retrieved.
    pub chunk: Chunk,
    /// Primary score used for ranking.
    pub score: Score,
    /// Per-stage score breakdown.
    pub breakdown: ScoreBreakdown,
}

impl RetrievalResult {
    /// Construct a new retrieval result with an empty breakdown.
    pub fn new(chunk: Chunk, score: Score) -> Self {
        Self {
            chunk,
            score,
            breakdown: ScoreBreakdown::default(),
        }
    }
}

