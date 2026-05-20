//! BM25 retrieval backed by Tantivy.
//!
//! Uses an in-memory RAM directory so the retriever is fully embeddable —
//! no on-disk state to manage in the typical library use case. A future
//! constructor variant can take a [`tantivy::directory::MmapDirectory`] for
//! persistent indices without changing the trait surface.

use std::sync::Arc;

use async_trait::async_trait;
use neorag_core::{
    Chunk, ChunkId, Error, Query, RetrievalMethod, RetrievalResult, Retriever, Score,
    ScoreBreakdown, TokenCount,
};
use parking_lot::RwLock;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, FAST, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, TantivyDocument};

const WRITER_HEAP_BYTES: usize = 64 * 1024 * 1024;

struct Schema_ {
    schema: Schema,
    id_field: Field,
    text_field: Field,
    source_field: Field,
    token_count_field: Field,
}

impl Schema_ {
    fn build() -> Self {
        let mut sb = Schema::builder();
        let id_field = sb.add_text_field("id", STRING | STORED);
        let text_field = sb.add_text_field("text", TEXT | STORED);
        let source_field = sb.add_text_field("source", STRING | STORED);
        let token_count_field = sb.add_u64_field("token_count", STORED | FAST);
        Self {
            schema: sb.build(),
            id_field,
            text_field,
            source_field,
            token_count_field,
        }
    }
}

/// BM25 retriever over an in-memory Tantivy index.
pub struct Bm25Retriever {
    inner: Arc<RwLock<Inner>>,
}

struct Inner {
    schema: Schema_,
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
}

impl Bm25Retriever {
    /// Construct a new in-memory BM25 retriever.
    pub fn new() -> neorag_core::Result<Self> {
        let schema = Schema_::build();
        let index = Index::create_in_ram(schema.schema.clone());
        let writer: IndexWriter = index
            .writer(WRITER_HEAP_BYTES)
            .map_err(|e| Error::Storage(format!("tantivy writer: {e}")))?;
        let reader = index
            .reader()
            .map_err(|e| Error::Storage(format!("tantivy reader: {e}")))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(Inner {
                schema,
                index,
                writer,
                reader,
            })),
        })
    }
}

#[async_trait]
impl Retriever for Bm25Retriever {
    async fn index(&mut self, chunks: &[Chunk]) -> neorag_core::Result<()> {
        let inner = self.inner.clone();
        let chunks = chunks.to_vec();
        // Tantivy's indexing is CPU-bound; do it on a blocking worker.
        tokio::task::spawn_blocking(move || -> neorag_core::Result<()> {
            let mut g = inner.write();
            let s = &g.schema;
            for c in &chunks {
                let d: TantivyDocument = doc!(
                    s.id_field => c.id.as_str().to_string(),
                    s.text_field => c.text.clone(),
                    s.source_field => c.source.clone(),
                    s.token_count_field => c.token_count.value() as u64,
                );
                g.writer
                    .add_document(d)
                    .map_err(|e| Error::Storage(format!("tantivy add: {e}")))?;
            }
            g.writer
                .commit()
                .map_err(|e| Error::Storage(format!("tantivy commit: {e}")))?;
            g.reader
                .reload()
                .map_err(|e| Error::Storage(format!("tantivy reload: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::Storage(format!("join: {e}")))??;
        Ok(())
    }

    async fn retrieve(
        &self,
        query: &Query,
        top_k: usize,
    ) -> neorag_core::Result<Vec<RetrievalResult>> {
        let inner = self.inner.clone();
        let text = query.text.clone();
        let top_k = top_k.max(1);
        let results: Vec<RetrievalResult> =
            tokio::task::spawn_blocking(move || -> neorag_core::Result<Vec<RetrievalResult>> {
                let g = inner.read();
                let searcher = g.reader.searcher();
                let qp = QueryParser::for_index(&g.index, vec![g.schema.text_field]);
                let q = qp
                    .parse_query(&sanitize_query(&text))
                    .map_err(|e| Error::Retrieval(format!("parse: {e}")))?;
                let hits = searcher
                    .search(&q, &TopDocs::with_limit(top_k))
                    .map_err(|e| Error::Retrieval(format!("search: {e}")))?;
                let mut out = Vec::with_capacity(hits.len());
                for (score, address) in hits {
                    let d: TantivyDocument = searcher
                        .doc(address)
                        .map_err(|e| Error::Retrieval(format!("doc fetch: {e}")))?;
                    let id = field_text(&d, g.schema.id_field).unwrap_or_default();
                    let text = field_text(&d, g.schema.text_field).unwrap_or_default();
                    let source = field_text(&d, g.schema.source_field).unwrap_or_default();
                    let tokens = field_u64(&d, g.schema.token_count_field).unwrap_or(0) as usize;
                    let chunk = Chunk::new(ChunkId::new(id), text, source, TokenCount(tokens));
                    let breakdown = ScoreBreakdown {
                        lexical: Some(score),
                        ..Default::default()
                    };
                    out.push(RetrievalResult {
                        chunk,
                        score: Score {
                            value: score,
                            method: RetrievalMethod::Lexical,
                        },
                        breakdown,
                    });
                }
                Ok(out)
            })
            .await
            .map_err(|e| Error::Retrieval(format!("join: {e}")))??;
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "bm25"
    }
}

/// Strip Tantivy query-parser metacharacters so user input is treated as a
/// plain bag-of-words. Lexical operators are useful but should be opt-in via
/// a future structured-query API rather than implicit in free-text queries.
fn sanitize_query(s: &str) -> String {
    const META: &[char] = &[
        '+', '-', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/', '|',
        '&',
    ];
    let cleaned: String = s
        .chars()
        .map(|c| if META.contains(&c) { ' ' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "*".to_string()
    } else {
        trimmed.to_string()
    }
}

fn field_text(d: &TantivyDocument, f: Field) -> Option<String> {
    use tantivy::schema::Value;
    d.get_first(f)
        .and_then(|v| v.as_str().map(|s| s.to_string()))
}

fn field_u64(d: &TantivyDocument, f: Field) -> Option<u64> {
    use tantivy::schema::Value;
    d.get_first(f).and_then(|v| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neorag_core::Document;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn mk(id: &str, text: &str) -> Chunk {
        Chunk::new(id, text, "doc", TokenCount(text.split_whitespace().count()))
    }

    #[test]
    fn bm25_indexes_and_retrieves() {
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();
            r.index(&[
                mk("c1", "the quick brown fox jumps over the lazy dog"),
                mk("c2", "rust is a systems programming language"),
                mk("c3", "retrieval evidence density matters for QA"),
            ])
            .await
            .unwrap();
            let results = r.retrieve(&Query::new("rust language"), 3).await.unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].chunk.id.as_str(), "c2");
            assert_eq!(results[0].score.method, RetrievalMethod::Lexical);
        });
    }

    #[test]
    fn sanitize_strips_metachars() {
        assert_eq!(sanitize_query("foo:bar +baz"), "foo bar  baz");
        assert_eq!(sanitize_query(" "), "*");
    }

    #[test]
    fn _silence_unused_doc() {
        let _ = Document::new("d", "t");
    }
}
