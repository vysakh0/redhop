//! BM25 retrieval backed by Tantivy.
//!
//! Uses an in-memory RAM directory so the retriever is fully embeddable —
//! no on-disk state to manage in the typical library use case. A future
//! constructor variant can take a [`tantivy::directory::MmapDirectory`] for
//! persistent indices without changing the trait surface.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use crate::core::{
    Chunk, ChunkId, Error, Query, RetrievalMethod, RetrievalResult, Retriever, Score,
    ScoreBreakdown, TokenCount,
};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, STORED, STRING,
};
use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter,
    TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer,
};
use tantivy::{doc, Index, IndexReader, IndexWriter, TantivyDocument};

const WRITER_HEAP_BYTES: usize = 64 * 1024 * 1024;

/// Custom analyzer name we register on the BM25 index: a Snowball-Porter2
/// pipeline (lowercase → strip overlong tokens → English stem). Picked to match
/// the grounding/Jaccard scorers in `crate::context` so a chunk's BM25 score
/// and its post-retrieval grounding agree on what "the same term" means —
/// otherwise queries like `"compression"` never match a chunk containing
/// `compress_video`, even though the grounding scorer would treat them as
/// identical.
const STEM_ANALYZER: &str = "redhop_en_stem";

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
        // Custom analyzer (`STEM_ANALYZER`) instead of plain `TEXT` so the
        // BM25 tokenization stems morphological variants. Registered against
        // the Index in `Bm25Retriever::new`.
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer(STEM_ANALYZER)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_opts = TextOptions::default()
            .set_indexing_options(text_indexing)
            .set_stored();
        let text_field = sb.add_text_field("text", text_opts);
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
    pub fn new() -> crate::core::Result<Self> {
        let schema = Schema_::build();
        let index = Index::create_in_ram(schema.schema.clone());
        // Register the stemming analyzer the schema references. Built once
        // per index (cheap — just composes filters); the analyzer applies on
        // both indexing and query-parsing for the `text` field. The
        // stopword and stemmer steps share their word lists with the
        // grounding scorer in `crate::context` so the two layers agree on
        // what a query consists of.
        let stopwords: Vec<String> = crate::context::STOPWORDS
            .iter()
            .map(|s| s.to_string())
            .collect();
        let stem_analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            // Camel/Pascal-case split BEFORE LowerCaser so we still have the
            // case signal to split on. Emits both the original token and the
            // pieces, so users querying `compress` find chunks containing
            // `compressVideo`, and `http` finds `HTTPResponse`.
            .filter(CamelCaseSplitter)
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(stopwords))
            .filter(Stemmer::new(Language::English))
            .build();
        index.tokenizers().register(STEM_ANALYZER, stem_analyzer);
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
    async fn index(&mut self, chunks: &[Chunk]) -> crate::core::Result<()> {
        let inner = self.inner.clone();
        let chunks = chunks.to_vec();
        // Tantivy's indexing is CPU-bound; do it on a blocking worker.
        tokio::task::spawn_blocking(move || -> crate::core::Result<()> {
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
    ) -> crate::core::Result<Vec<RetrievalResult>> {
        let inner = self.inner.clone();
        let text = query.text.clone();
        let top_k = top_k.max(1);
        let results: Vec<RetrievalResult> =
            tokio::task::spawn_blocking(move || -> crate::core::Result<Vec<RetrievalResult>> {
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

/// Reduce arbitrary natural-language text to a clean bag of word tokens, so
/// Tantivy's `QueryParser` never sees its query meta-syntax. We keep only
/// alphanumerics (Unicode) and collapse everything else to whitespace; the
/// parser then ORs the resulting terms over the text field. This is what keeps
/// `doc.context("Highlight the parts (if any)… “requirements”…")` from being a
/// parse error — a real natural-language query must never crash internal
/// retrieval. Ranking is unchanged (same content terms reach the index).
fn sanitize_query(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        "*".to_string()
    } else {
        collapsed
    }
}

/// Split a token on case transitions, in the standard "word-delimiter"
/// fashion used by Lucene/Elasticsearch. Returns the pieces in source order;
/// callers decide whether to also keep the original.
///
/// - `compressVideo`  → `["compress", "Video"]`           (lower→upper)
/// - `HTTPResponse`   → `["HTTP", "Response"]`            (upper→upper→lower)
/// - `URL`            → `["URL"]`                          (no transitions)
/// - `iPhone`         → `["i", "Phone"]`                  (single-char pieces are
///                                                         dropped later by length /
///                                                         stopword filters)
fn case_split_pieces(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    for i in 0..chars.len() {
        let c = chars[i];
        if i > 0 {
            let prev = chars[i - 1];
            let camel = prev.is_lowercase() && c.is_uppercase();
            // Acronym tail: HTTPResponse → split before R because R is the
            // last upper of a run that's followed by a lowercase.
            let acronym = prev.is_uppercase()
                && c.is_uppercase()
                && chars.get(i + 1).is_some_and(|n| n.is_lowercase());
            if (camel || acronym) && !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        }
        current.push(c);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Tantivy token filter: when an input token has internal case transitions,
/// emit the original token AND each split piece at the same position. The
/// original keeps queries that exact-match the identifier working (`compressvideo`
/// after lowercasing); the pieces make `compress` and `video` queries hit the
/// same chunk. Pieces share the original's source offsets — they describe the
/// same span of source text.
#[derive(Clone)]
struct CamelCaseSplitter;

impl TokenFilter for CamelCaseSplitter {
    type Tokenizer<T: Tokenizer> = CamelCaseSplitterTokenizer<T>;
    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        CamelCaseSplitterTokenizer { inner: tokenizer }
    }
}

#[derive(Clone)]
struct CamelCaseSplitterTokenizer<T> {
    inner: T,
}

impl<T: Tokenizer> Tokenizer for CamelCaseSplitterTokenizer<T> {
    type TokenStream<'a> = CamelCaseSplitterStream<T::TokenStream<'a>>;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CamelCaseSplitterStream {
            tail: self.inner.token_stream(text),
            queued: Vec::new(),
            current: Token::default(),
        }
    }
}

struct CamelCaseSplitterStream<S: TokenStream> {
    tail: S,
    /// Tokens generated from the current input token that we haven't emitted
    /// yet. Stored in REVERSE emission order so `pop()` yields them in order.
    queued: Vec<Token>,
    current: Token,
}

impl<S: TokenStream> TokenStream for CamelCaseSplitterStream<S> {
    fn advance(&mut self) -> bool {
        if let Some(t) = self.queued.pop() {
            self.current = t;
            return true;
        }
        if !self.tail.advance() {
            return false;
        }
        let orig = self.tail.token().clone();
        let pieces = case_split_pieces(&orig.text);
        if pieces.len() <= 1 {
            // No internal case transitions — pass the token through unchanged.
            self.current = orig;
            return true;
        }
        // Emit the original first; queue the pieces (in reverse, so `pop()`
        // yields them in source order). All share the original's offsets +
        // position — they're synonyms at the same conceptual location.
        self.current = orig.clone();
        for piece in pieces.into_iter().rev() {
            let mut t = orig.clone();
            t.text = piece;
            self.queued.push(t);
        }
        true
    }

    fn token(&self) -> &Token {
        &self.current
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current
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
    use crate::core::Document;

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
    fn sanitize_reduces_to_word_bag() {
        // Meta-chars and punctuation collapse to single spaces (no parser syntax).
        assert_eq!(sanitize_query("foo:bar +baz"), "foo bar baz");
        // Natural-language query with quotes/parens/smart-quotes never crashes.
        assert_eq!(
            sanitize_query("Highlight parts (if any) of “Exclusivity”."),
            "Highlight parts if any of Exclusivity"
        );
        assert_eq!(sanitize_query("   "), "*");
    }

    #[test]
    fn stopwords_are_stripped_consistently_with_grounding() {
        // The bug this regresses: BM25 kept "the"/"is"/"what" while the
        // grounding scorer (`crate::context::terms`) dropped them. On small
        // in-process corpora the IDF stats can't reliably suppress them, so
        // adding stopwords to a query could dilute or shift the ranking
        // away from the chunk the grounding scorer considers most relevant.
        // After registering the same stopword list on the BM25 analyzer,
        // a stopword-padded query ranks identically to the content-only
        // query against any non-pathological corpus.
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();
            r.index(&[
                mk("hit", "the refund window is thirty days from purchase"),
                mk("miss1", "shipping takes two business days from order"),
                mk("miss2", "warranty extends for one year after delivery"),
            ])
            .await
            .unwrap();

            // Bare keywords.
            let bare = r.retrieve(&Query::new("refund window"), 3).await.unwrap();
            // Same query padded with stopwords every BM25-with-stopwords pipeline
            // would otherwise score differently.
            let padded = r
                .retrieve(&Query::new("what is the refund window"), 3)
                .await
                .unwrap();

            assert!(!bare.is_empty(), "bare query must hit");
            assert!(!padded.is_empty(), "padded query must hit");
            assert_eq!(
                bare[0].chunk.id.as_str(),
                padded[0].chunk.id.as_str(),
                "stopword-padded query must rank the same chunk first as the bare query; \
                 got bare={:?} padded={:?}",
                bare.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
                padded.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn morphological_query_variants_match_via_stemming() {
        // The bug this regresses: querying `compression` against a chunk that
        // contains `compress_video` returned nothing because the default
        // Tantivy `TEXT` analyzer didn't stem (so `compression` ≠ `compress`).
        // After registering the Snowball English stemmer, both stem to
        // `compress` and match. Mirrors the user-reported symptom from a real
        // Rust source file indexed with redhop 0.1.2.
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();
            r.index(&[
                mk(
                    "video",
                    "pub async fn compress_video(file_path: &str, quality: &str)",
                ),
                mk(
                    "convert",
                    "pub async fn convert_video(file_path: &str, format: &str)",
                ),
                mk("unrelated", "the quick brown fox jumps over the lazy dog"),
            ])
            .await
            .unwrap();

            for query in ["compress", "compression", "compresses", "compressing"] {
                let hits = r.retrieve(&Query::new(query), 3).await.unwrap();
                assert!(
                    hits.iter().any(|h| h.chunk.id.as_str() == "video"),
                    "query {query:?} must find the compress_video chunk via stemming; \
                     got hits: {:?}",
                    hits.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
                );
            }
        });
    }

    #[test]
    fn case_split_pieces_handles_camel_pascal_and_acronyms() {
        assert_eq!(case_split_pieces("compressVideo"), vec!["compress", "Video"]);
        assert_eq!(case_split_pieces("HTTPResponse"), vec!["HTTP", "Response"]);
        assert_eq!(case_split_pieces("XMLParser"), vec!["XML", "Parser"]);
        assert_eq!(case_split_pieces("URL"), vec!["URL"]);
        assert_eq!(case_split_pieces("iPhone"), vec!["i", "Phone"]);
        assert_eq!(case_split_pieces("alreadylower"), vec!["alreadylower"]);
        assert_eq!(case_split_pieces("ALLUPPER"), vec!["ALLUPPER"]);
        assert_eq!(case_split_pieces(""), Vec::<String>::new());
    }

    #[test]
    fn case_split_makes_camelcase_identifiers_searchable() {
        // The bug this regresses: `compressVideo` (camelCase) tokenized as
        // one term `compressvideo` after lowercasing, so a query for
        // `compress` against a JS/Go/TS codebase using camelCase never
        // matched. The CamelCaseSplitter filter emits `compress` + `video`
        // as additional tokens at the same position.
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();
            r.index(&[
                mk("camel", "function compressVideo(filePath, quality) { ... }"),
                mk("pascal", "class HTTPResponse extends BaseResponse { ... }"),
                mk("snake", "def compress_audio(file_path, bitrate): pass"),
                mk("unrelated", "the quick brown fox jumps over the lazy dog"),
            ])
            .await
            .unwrap();

            // camelCase: `compress` finds `compressVideo`.
            let r1 = r.retrieve(&Query::new("compress"), 5).await.unwrap();
            assert!(
                r1.iter()
                    .any(|h| matches!(h.chunk.id.as_str(), "camel" | "snake")),
                "`compress` query should hit at least the camelCase chunk; got {:?}",
                r1.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );
            assert!(
                r1.iter().any(|h| h.chunk.id.as_str() == "camel"),
                "`compress` query must reach the camelCase compressVideo chunk; got {:?}",
                r1.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );

            // PascalCase + acronym: `http` finds `HTTPResponse`.
            let r2 = r.retrieve(&Query::new("http response"), 5).await.unwrap();
            assert!(
                r2.iter().any(|h| h.chunk.id.as_str() == "pascal"),
                "`http response` must reach the HTTPResponse chunk; got {:?}",
                r2.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );

            // Original full identifier still hits (the splitter preserves it).
            let r3 = r.retrieve(&Query::new("compressVideo"), 5).await.unwrap();
            assert!(
                r3.iter().any(|h| h.chunk.id.as_str() == "camel"),
                "the original camelCase identifier must still match itself; got {:?}",
                r3.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn _silence_unused_doc() {
        let _ = Document::new("d", "t");
    }
}
