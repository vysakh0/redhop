//! BM25 retrieval backed by Tantivy.
//!
//! Uses an in-memory RAM directory so the retriever is fully embeddable —
//! no on-disk state to manage in the typical library use case. A future
//! constructor variant can take a [`tantivy::directory::MmapDirectory`] for
//! persistent indices without changing the trait surface.

use std::sync::Arc;

use crate::core::{
    Chunk, ChunkId, Error, Query, RetrievalMethod, RetrievalResult, Retriever, Score,
    ScoreBreakdown, TokenCount,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use tantivy::collector::TopDocs;
use tantivy::query::{
    BooleanQuery, BoostQuery, Occur, Query as TantivyQuery, QueryParser, TermQuery,
};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, STORED, STRING,
};
use tantivy::tokenizer::{
    AsciiFoldingFilter, Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer,
    StopWordFilter, TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer,
};
use tantivy::Term;
use tantivy::{doc, Index, IndexReader, IndexWriter, TantivyDocument};

const WRITER_HEAP_BYTES: usize = 64 * 1024 * 1024;

struct Schema_ {
    schema: Schema,
    id_field: Field,
    text_field: Field,
    source_field: Field,
    heading_field: Field,
    token_count_field: Field,
}

impl Schema_ {
    /// Build the schema, routing the three searchable text fields
    /// (`text` / `source` / `heading`) to the analyzer registered under
    /// `analyzer_name` on the Tantivy index. The caller is responsible for
    /// having registered that analyzer (see [`Bm25Retriever::with_analyzer`]).
    fn build(analyzer_name: &str) -> Self {
        let mut sb = Schema::builder();
        let id_field = sb.add_text_field("id", STRING | STORED);
        // Same analyzer for all three searchable fields so the query parser
        // can fold a single query into matches against any of them — a query
        // for `auth.rs` reaches a chunk via its `source` field even when the
        // chunk text itself never mentions the filename.
        let analyzed = |stored: bool| -> TextOptions {
            let indexing = TextFieldIndexing::default()
                .set_tokenizer(analyzer_name)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions);
            let mut opts = TextOptions::default().set_indexing_options(indexing);
            if stored {
                opts = opts.set_stored();
            }
            opts
        };
        let text_field = sb.add_text_field("text", analyzed(true));
        // `source` is the file path — keep STORED for citation round-trips,
        // but ALSO analyze it (separate STRING was exact-match-only before).
        let source_field = sb.add_text_field("source", analyzed(true));
        // Heading from chunk metadata (markdown ## headings, code symbols).
        // Not stored (citations carry it via metadata) — index-only.
        let heading_field = sb.add_text_field("heading", analyzed(false));
        let token_count_field = sb.add_u64_field("token_count", STORED | FAST);
        Self {
            schema: sb.build(),
            id_field,
            text_field,
            source_field,
            heading_field,
            token_count_field,
        }
    }
}

/// Per-field BM25 query boosts for the three searchable fields
/// (`text` / `source` / `heading`).
///
/// The default ([`FieldWeights::uniform`], all `1.0`) reproduces the
/// equal-weight behavior **bit-for-bit**: a boost of `1.0` is the identity
/// (Tantivy multiplies the parser boost into the score), and the retriever
/// skips the `set_field_boost` call entirely when a weight is `1.0`, so the
/// default query path is literally unchanged.
///
/// Non-uniform weights are a **domain lever**, not a universal win. In a
/// near-duplicate / title-heavy corpus (a product catalog, an API surface
/// where the symbol name disambiguates) boosting the field that carries the
/// discriminative token (often `heading`) lifts strict set-coverage; but
/// over-boosting starves recall — the brand index is a lever, not free lift.
/// Sweep on a held-out set with your own eval before shipping a weight. See
/// `docs/findings/CATALOG_REGIME.md` and
/// `docs/CHOOSING_A_CONFIG.md` (Choosing field weights).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldWeights {
    /// Boost on the chunk body (`text`). Carries recall.
    pub text: f32,
    /// Boost on the source path (`source`).
    pub source: f32,
    /// Boost on the section heading / symbol name (`heading`). The usual
    /// lever for near-duplicate variant families.
    pub heading: f32,
}

impl FieldWeights {
    /// All-`1.0` weights — bit-for-bit the equal-weight default behavior.
    pub const fn uniform() -> Self {
        Self {
            text: 1.0,
            source: 1.0,
            heading: 1.0,
        }
    }

    /// `true` when every weight is `1.0` (the default query path, no boosts
    /// applied).
    pub fn is_uniform(&self) -> bool {
        self.text == 1.0 && self.source == 1.0 && self.heading == 1.0
    }
}

impl Default for FieldWeights {
    fn default() -> Self {
        Self::uniform()
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
    field_weights: FieldWeights,
    /// Registered analyzer name — used to fetch the query-side tokenizer for
    /// the bag-of-grams (union) path.
    analyzer_name: String,
    /// When `true`, a query word that analyzes into multiple tokens is run as
    /// a boolean union of per-token term queries instead of a phrase. Set from
    /// [`crate::analyzer::Analyzer::union_subword_query`]. See
    /// [`crate::analyzer::CharNgramAnalyzer`].
    union_subword: bool,
}

/// Build redhop's standard analyzer pipeline as a Tantivy `TextAnalyzer`.
///
/// The pipeline (shared by every [`crate::analyzer::SnowballAnalyzer`]
/// language variant): `SimpleTokenizer` → `RemoveLongFilter(40)` →
/// `CamelCaseSplitter` → `AsciiFoldingFilter` → `LowerCaser` →
/// `StopWordFilter(stopwords)` → `Stemmer(language)`.
///
/// Exposed as a public helper returning a concrete `TextAnalyzer` so the
/// generic intermediate `TextAnalyzerBuilder<...>` type — which carries the
/// crate-private `CamelCaseSplitter` filter family — stays inside this
/// module. Callers in `crate::analyzer` just get a built analyzer.
pub fn build_redhop_pipeline(stopwords: Vec<String>, language: Language) -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(CamelCaseSplitter)
        .filter(AsciiFoldingFilter)
        .filter(LowerCaser)
        .filter(StopWordFilter::remove(stopwords))
        .filter(Stemmer::new(language))
        .build()
}

/// The minimal Tantivy pipeline used by [`crate::analyzer::RawAnalyzer`]:
/// `SimpleTokenizer` → `AsciiFoldingFilter` → `LowerCaser`. No CamelCase
/// splitting, no RemoveLongFilter, no stopword filtering, no stemming.
/// Use when warm-query latency matters more than inflectional recall —
/// see `docs/findings/FRAMEWORK_MULTIQUERY.md` for the measured tradeoff.
pub fn build_raw_pipeline() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(AsciiFoldingFilter)
        .filter(LowerCaser)
        .build()
}

/// Build the character n-gram pipeline used by
/// [`crate::analyzer::CharNgramAnalyzer`]: a [`CharNgramTokenizer`] that
/// lowercases, collapses every run of non-`[a-z0-9]` characters to a single
/// space, pads the result with a leading and trailing space, and emits every
/// character n-gram of length `n_min..=n_max` over that normalized string.
///
/// The padding makes word-boundary grams (e.g. `" la"`, `"ys "`) first-class,
/// so a leading or trailing subword still matches — the lever that recovers a
/// transcription-typo'd token (`"lays"` vs `"1ays"` still share `"ays"`,
/// `"ays "`) that token-exact BM25 scores at zero. No model, no dependency.
///
/// This is a recall booster for short, noisy tokens, NOT a drop-in retriever:
/// at large near-duplicate corpus sizes the dense gram vocabulary collides and
/// recall inverts (see `docs/findings/CATALOG_REGIME.md`). Position
/// it in a hybrid with word-BM25, not standalone.
pub fn build_char_ngram_pipeline(n_min: usize, n_max: usize) -> TextAnalyzer {
    TextAnalyzer::builder(CharNgramTokenizer { n_min, n_max }).build()
}

impl Bm25Retriever {
    /// Construct a new in-memory BM25 retriever using the default English
    /// Snowball analyzer (preserves the 0.1.4 behavior bit-for-bit). For a
    /// different language or a custom pipeline, see
    /// [`Bm25Retriever::with_analyzer`].
    pub fn new() -> crate::core::Result<Self> {
        Self::with_analyzer(crate::analyzer::default_english())
    }

    /// Construct a new in-memory BM25 retriever using the supplied analyzer.
    ///
    /// The analyzer drives BOTH indexing and query-parsing for the three
    /// searchable fields (`text`, `source`, `heading`) — so a chunk's BM25
    /// score and the grounding scorer in `crate::context` agree on what
    /// counts as "the same term" only when both layers use the same
    /// analyzer. `Document::with_analyzer` (C4) wires this end-to-end.
    pub fn with_analyzer(
        analyzer: Arc<dyn crate::analyzer::Analyzer>,
    ) -> crate::core::Result<Self> {
        let schema = Schema_::build(analyzer.name());
        let index = Index::create_in_ram(schema.schema.clone());
        // Register the analyzer's pipeline under its `name()`. Cheap — built
        // once at construction; reused for every index and query.
        index
            .tokenizers()
            .register(analyzer.name(), analyzer.build_text_analyzer());
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
                field_weights: FieldWeights::uniform(),
                analyzer_name: analyzer.name().to_string(),
                union_subword: analyzer.union_subword_query(),
            })),
        })
    }

    /// Set per-field query boosts (`text` / `source` / `heading`).
    ///
    /// Defaults to [`FieldWeights::uniform`] (all `1.0`), which is bit-for-bit
    /// the equal-weight behavior. Pass non-uniform weights only when a
    /// held-out sweep shows a lift on your workload — over-boosting starves
    /// recall (see [`FieldWeights`]). Cheap to call: it only stores the
    /// weights, applied at query time.
    ///
    /// ```no_run
    /// # fn main() -> redhop::Result<()> {
    /// use redhop::retrieval::{Bm25Retriever, FieldWeights};
    /// let r = Bm25Retriever::new()?
    ///     .with_field_weights(FieldWeights { text: 1.0, source: 1.0, heading: 2.0 });
    /// # let _ = r; Ok(()) }
    /// ```
    pub fn with_field_weights(self, weights: FieldWeights) -> Self {
        self.inner.write().field_weights = weights;
        self
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
                // Extract heading from metadata if present (markdown headings,
                // code symbol names). Indexed so a query for the heading text
                // surfaces the chunk even when the chunk body doesn't repeat it.
                let heading = c
                    .metadata
                    .get("heading")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("");
                let d: TantivyDocument = doc!(
                    s.id_field => c.id.as_str().to_string(),
                    s.text_field => c.text.clone(),
                    s.source_field => c.source.clone(),
                    s.heading_field => heading.to_string(),
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
                // Some queries reduce to nothing after the analyzer pipeline
                // — `""`, `"   "`, `"!!!???"`, or `"the and is of"` (all
                // stopwords) all produce a parse error from Tantivy ("Only
                // excluding terms given") or — worse — would parse to a `*`
                // wildcard (match-everything) if sanitize_query fell back to
                // that. Treating no-signal queries as empty results is the
                // only sane behavior. Caught by quality_suite::t25 (empty +
                // all-stopword) and t51 (whitespace + punctuation only).
                let parsed_text = sanitize_query(&text);
                if parsed_text.is_empty() {
                    return Ok(Vec::new());
                }
                let fw = g.field_weights;
                let fields = [
                    g.schema.text_field,
                    g.schema.source_field,
                    g.schema.heading_field,
                ];
                let hits = if g.union_subword {
                    // Bag-of-grams union path (subword analyzers such as
                    // CharNgramAnalyzer). A query word here analyzes into many
                    // character grams; the default QueryParser would fold them
                    // into a phrase (AND-at-slot), which a transcription typo
                    // always breaks (it shares only SOME grams). So OR every
                    // gram across all three fields as Should clauses, scored by
                    // BM25 — any shared gram contributes, partial overlap ranks.
                    let mut analyzer =
                        g.index.tokenizers().get(&g.analyzer_name).ok_or_else(|| {
                            Error::Retrieval(format!(
                                "tokenizer '{}' not registered",
                                g.analyzer_name
                            ))
                        })?;
                    let weights = [fw.text, fw.source, fw.heading];
                    let mut clauses: Vec<(Occur, Box<dyn TantivyQuery>)> = Vec::new();
                    let mut ts = analyzer.token_stream(&parsed_text);
                    ts.process(&mut |tok| {
                        for (fi, &field) in fields.iter().enumerate() {
                            let term = Term::from_field_text(field, &tok.text);
                            let tq: Box<dyn TantivyQuery> =
                                Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
                            let clause: Box<dyn TantivyQuery> = if weights[fi] != 1.0 {
                                Box::new(BoostQuery::new(tq, weights[fi]))
                            } else {
                                tq
                            };
                            clauses.push((Occur::Should, clause));
                        }
                    });
                    if clauses.is_empty() {
                        return Ok(Vec::new());
                    }
                    let bq = BooleanQuery::new(clauses);
                    searcher
                        .search(&bq, &TopDocs::with_limit(top_k))
                        .map_err(|e| Error::Retrieval(format!("search: {e}")))?
                } else {
                    // Default phrase path via the multi-field QueryParser. Equal
                    // weights let BM25's TF-IDF settle the ranking rather than
                    // baking a prior; a caller can override per field via
                    // `with_field_weights` for near-duplicate / title-heavy
                    // corpora (a weight of 1.0 is the identity and is skipped,
                    // so the default path is bit-for-bit unchanged).
                    let mut qp = QueryParser::for_index(&g.index, fields.to_vec());
                    if fw.text != 1.0 {
                        qp.set_field_boost(g.schema.text_field, fw.text);
                    }
                    if fw.source != 1.0 {
                        qp.set_field_boost(g.schema.source_field, fw.source);
                    }
                    if fw.heading != 1.0 {
                        qp.set_field_boost(g.schema.heading_field, fw.heading);
                    }
                    let q = match qp.parse_query(&parsed_text) {
                        Ok(q) => q,
                        Err(e) => {
                            let msg = e.to_string();
                            if msg.contains("Only excluding terms") || msg.contains("empty query") {
                                return Ok(Vec::new());
                            }
                            return Err(Error::Retrieval(format!("parse: {e}")));
                        }
                    };
                    searcher
                        .search(&q, &TopDocs::with_limit(top_k))
                        .map_err(|e| Error::Retrieval(format!("search: {e}")))?
                };
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

/// Reduce arbitrary natural-language text to a string Tantivy's `QueryParser`
/// can parse without crashing. Only the chars QueryParser actually treats as
/// syntax are replaced with a space; everything else (dots, hyphens, slashes,
/// @, /, etc.) is preserved so the analyzer can tokenize as it would on
/// indexed text — keeping query and index in lockstep. The uppercase boolean
/// keywords `AND`/`OR`/`NOT` are neutralized by lowercasing the whole input
/// (the analyzer's `LowerCaser` would have done it anyway).
fn sanitize_query(s: &str) -> String {
    // Tantivy QueryParser meta-chars (per the docs: bool ops `+`/`-`, field
    // selector `:`, wildcards `*`/`?`, fuzzy/boost `~`/`^`, escape `\`, range
    // brackets `[`/`]`/`{`/`}`, grouping `(`/`)`, and phrase quotes `"`).
    const META: &[char] = &[
        '+', '-', ':', '*', '?', '^', '~', '\\', '(', ')', '[', ']', '{', '}', '"', '<', '>',
    ];
    let cleaned: String = s
        .chars()
        .map(|c| if META.contains(&c) { ' ' } else { c })
        .collect();
    let lowered = cleaned.to_lowercase();
    // Empty fallback is **empty**, not `"*"`. A wildcard would silently turn
    // a no-signal query (whitespace, punctuation, all-stopword) into a
    // match-everything — wrong behavior, and the caller (`retrieve`) now
    // short-circuits to an empty result when this returns empty.
    lowered.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split a token on case and letter/digit transitions, in the standard
/// "word-delimiter" fashion used by Lucene/Elasticsearch. Returns the pieces
/// in source order; callers decide whether to also keep the original.
///
/// - `compressVideo`  → `["compress", "Video"]` (lower→upper)
/// - `HTTPResponse`   → `["HTTP", "Response"]` (upper→upper→lower)
/// - `parseV2`        → `["parse", "V", "2"]` (lower→upper, then letter→digit)
/// - `Phi3`           → `["Phi", "3"]` (letter→digit)
/// - `URL`            → `["URL"]` (no transitions)
/// - `iPhone`         → `["i", "Phone"]` (single-char pieces are dropped later
///   by the length / stopword filters)
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
            // Letter ↔ digit transition: `parseV2` → `parse` + `V` + `2`,
            // `Phi3` → `Phi` + `3`. Catches version suffixes and id-style
            // identifiers a plain camelCase split would miss.
            let letter_to_digit = prev.is_alphabetic() && c.is_ascii_digit();
            let digit_to_letter = prev.is_ascii_digit() && c.is_alphabetic();
            if (camel || acronym || letter_to_digit || digit_to_letter) && !current.is_empty() {
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
///
/// Used internally by [`build_redhop_pipeline`] which returns a concrete
/// `TextAnalyzer` — this keeps the generic intermediate type
/// `TextAnalyzerBuilder<...>` out of the public surface.
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

/// Lowercase `text`, collapse every maximal run of non-`[a-z0-9]` characters
/// to a single space, and pad with one leading + one trailing space. The
/// output is pure ASCII (`a-z`, `0-9`, and single spaces), so a char index is
/// also a byte offset. Mirrors the char-n-gram TF-IDF normalizer the
/// catalog-regime evidence used (drops non-ASCII rather than folding it — the
/// regime is transliterated/ASCII noise; fold upstream if you need accents).
fn normalize_for_char_ngram(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push(' ');
    let mut prev_space = true;
    for ch in text.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    if !out.ends_with(' ') {
        out.push(' ');
    }
    out
}

/// Tantivy tokenizer emitting character n-grams (length `n_min..=n_max`) over
/// the normalized form of the input (see [`normalize_for_char_ngram`]). Used
/// by [`build_char_ngram_pipeline`]. All grams share `position = 0` (matching
/// Tantivy's own `NgramTokenizer`); offsets are byte ranges into the
/// normalized string (used only for highlighting, never for BM25 scoring).
#[derive(Clone)]
struct CharNgramTokenizer {
    n_min: usize,
    n_max: usize,
}

struct CharNgramTokenStream {
    tokens: Vec<Token>,
    idx: usize,
    current: Token,
}

impl Tokenizer for CharNgramTokenizer {
    type TokenStream<'a> = CharNgramTokenStream;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> CharNgramTokenStream {
        let norm = normalize_for_char_ngram(text);
        // Pure ASCII after normalization, so byte length == char count and a
        // char index is a valid byte offset.
        let bytes = norm.as_bytes();
        let len = bytes.len();
        let mut tokens = Vec::new();
        for n in self.n_min..=self.n_max {
            if len < n {
                continue;
            }
            for i in 0..=(len - n) {
                let gram = &norm[i..i + n];
                tokens.push(Token {
                    offset_from: i,
                    offset_to: i + n,
                    position: 0,
                    text: gram.to_string(),
                    position_length: 1,
                });
            }
        }
        CharNgramTokenStream {
            tokens,
            idx: 0,
            current: Token::default(),
        }
    }
}

impl TokenStream for CharNgramTokenStream {
    fn advance(&mut self) -> bool {
        if self.idx < self.tokens.len() {
            self.current = self.tokens[self.idx].clone();
            self.idx += 1;
            true
        } else {
            false
        }
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
    fn sanitize_strips_only_query_parser_metachars() {
        // The chars QueryParser would otherwise treat as syntax are replaced
        // with spaces; everything else passes through so the analyzer can
        // tokenize it the same way it did the indexed text. Output is
        // lowercased so uppercase boolean keywords (AND/OR/NOT) become
        // ordinary words.
        assert_eq!(sanitize_query("foo:bar +baz"), "foo bar baz");
        // Natural-language query with quotes/parens/smart-quotes never crashes.
        // Smart quotes (U+201C/D) are NOT in META — they pass through.
        assert_eq!(
            sanitize_query("Highlight parts (if any) of “Exclusivity”."),
            "highlight parts if any of “exclusivity”."
        );
        // No-signal queries (whitespace, punctuation, all-stopword) now
        // collapse to empty string, not `"*"`. The retriever caller
        // short-circuits to empty results when sanitize_query returns
        // empty — see the `if parsed_text.is_empty()` guard in
        // `Bm25Retriever::retrieve`. Previously the `"*"` fallback would
        // silently match every document, which is the opposite of what a
        // user typing accidental whitespace expects.
        assert_eq!(sanitize_query("   "), "");
        assert_eq!(sanitize_query(""), "");
        // All-META input also collapses to empty.
        assert_eq!(sanitize_query("()[]{}"), "");
        assert_eq!(sanitize_query("???"), "");
        // Mixed META + non-META: `?` is META, `!` is not — `!` stays as a
        // sub-word token that the analyzer's stopword/length filters drop
        // downstream, so retrieve still short-circuits to empty results.
        assert_eq!(sanitize_query("!!!???"), "!!!");

        // Punctuation that's NOT QueryParser syntax is preserved — the
        // analyzer will tokenize on it later as it does for indexed text.
        // Pre-fix these all degraded into single-char tokens that the length
        // filter dropped.
        assert_eq!(sanitize_query("v1.2.3"), "v1.2.3");
        assert_eq!(sanitize_query("e-mail templates"), "e mail templates");
        assert_eq!(sanitize_query(".NET runtime"), ".net runtime");
        assert_eq!(sanitize_query("api/v1/users"), "api/v1/users");
        assert_eq!(sanitize_query("@username"), "@username");
        // Uppercase boolean keywords are neutralized.
        assert_eq!(sanitize_query("foo AND bar"), "foo and bar");
    }

    #[test]
    fn multi_field_search_reaches_file_paths_and_headings() {
        // The bug this regresses: BM25 only searched the `text` field, so a
        // query for the filename `auth.rs` returned nothing unless the file's
        // content also mentioned that string. After indexing `source` and
        // `heading` through the same analyzer, the file is reachable by name
        // and by section heading too.
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();

            // Chunk text intentionally has NOTHING in common with the
            // filename or the heading we'll query for.
            let mut by_path = Chunk::new(
                "by_path",
                "validate the supplied credentials and issue a token",
                "src/auth.rs",
                TokenCount(8),
            );
            by_path
                .metadata
                .insert("heading".into(), serde_json::json!("fn login"));
            let mut by_heading = Chunk::new(
                "by_heading",
                "delegate to the upstream identity provider",
                "src/handlers/users.rs",
                TokenCount(6),
            );
            by_heading
                .metadata
                .insert("heading".into(), serde_json::json!("Refund window"));
            let unrelated = mk("unrelated", "the quick brown fox jumps over the lazy dog");

            r.index(&[by_path, by_heading, unrelated]).await.unwrap();

            // Query for the filename: only `by_path`'s `source` matches.
            let r1 = r.retrieve(&Query::new("auth"), 5).await.unwrap();
            assert!(
                r1.iter().any(|h| h.chunk.id.as_str() == "by_path"),
                "filename query must reach the chunk via its source path; got {:?}",
                r1.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );

            // Query for the heading: only `by_heading`'s heading metadata matches.
            let r2 = r.retrieve(&Query::new("refund window"), 5).await.unwrap();
            assert!(
                r2.iter().any(|h| h.chunk.id.as_str() == "by_heading"),
                "heading query must reach the chunk via its heading metadata; got {:?}",
                r2.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );
        });
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
                padded
                    .iter()
                    .map(|h| h.chunk.id.as_str())
                    .collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn queries_that_reduce_to_empty_return_empty_results_not_an_error() {
        // After the analyzer pipeline (stopword filter, length filter, etc.)
        // some queries have zero positive terms, which Tantivy's QueryParser
        // surfaces as a hard error ("Only excluding terms given"). The
        // retriever traps that case and returns an empty result — a
        // no-signal query is not a crash. Caught by quality_suite::t25.
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();
            r.index(&[mk("0", "the refund window is thirty days from purchase")])
                .await
                .unwrap();
            for q in ["", "   ", "the and is of in or", "a"] {
                let res = r.retrieve(&Query::new(q), 5).await;
                assert!(res.is_ok(), "query {q:?} crashed: {res:?}");
            }
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
    fn case_split_pieces_handles_camel_pascal_acronyms_and_digits() {
        assert_eq!(
            case_split_pieces("compressVideo"),
            vec!["compress", "Video"]
        );
        assert_eq!(case_split_pieces("HTTPResponse"), vec!["HTTP", "Response"]);
        assert_eq!(case_split_pieces("XMLParser"), vec!["XML", "Parser"]);
        assert_eq!(case_split_pieces("URL"), vec!["URL"]);
        assert_eq!(case_split_pieces("iPhone"), vec!["i", "Phone"]);
        assert_eq!(case_split_pieces("alreadylower"), vec!["alreadylower"]);
        assert_eq!(case_split_pieces("ALLUPPER"), vec!["ALLUPPER"]);
        assert_eq!(case_split_pieces(""), Vec::<String>::new());
        // Letter ↔ digit transitions.
        assert_eq!(case_split_pieces("parseV2"), vec!["parse", "V", "2"]);
        assert_eq!(case_split_pieces("Phi3"), vec!["Phi", "3"]);
        assert_eq!(case_split_pieces("v2API"), vec!["v", "2", "API"]);
        assert_eq!(case_split_pieces("gpt4o"), vec!["gpt", "4", "o"]);
    }

    #[test]
    fn digit_boundary_split_makes_versioned_identifiers_searchable() {
        // The bug this regresses: an identifier like `parseV2` or `Phi3` was
        // one indivisible lowercase token (`parsev2`/`phi3`) after lowercase,
        // so a query for the base name (`parse`, `Phi`) didn't match. With
        // the letter/digit split rule each component is its own searchable
        // token.
        rt().block_on(async {
            let mut r = Bm25Retriever::new().unwrap();
            r.index(&[
                mk("v2", "fn parseV2(input: &str) -> Result<()>"),
                mk("v3", "pub fn parseV3(input: &str) -> Result<()>"),
                mk("model", "loaded model: Phi3-mini-4k-instruct"),
                mk("unrelated", "the quick brown fox jumps over the lazy dog"),
            ])
            .await
            .unwrap();
            // Base name of a versioned identifier finds both versions.
            let r1 = r.retrieve(&Query::new("parse"), 5).await.unwrap();
            let ids1: Vec<_> = r1.iter().map(|h| h.chunk.id.as_str()).collect();
            assert!(
                ids1.contains(&"v2") && ids1.contains(&"v3"),
                "`parse` must reach parseV2 + parseV3 via digit-split; got {ids1:?}"
            );
            // Brand-with-version (Phi3) findable by brand.
            let r2 = r.retrieve(&Query::new("phi"), 5).await.unwrap();
            assert!(
                r2.iter().any(|h| h.chunk.id.as_str() == "model"),
                "`phi` must reach the Phi3 chunk via digit-split; got {:?}",
                r2.iter().map(|h| h.chunk.id.as_str()).collect::<Vec<_>>(),
            );
        });
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
    fn field_weights_uniform_is_bit_for_bit_the_default() {
        // The zero-regression contract: FieldWeights::uniform() must produce
        // exactly the same ranking AND scores as the no-weights default,
        // because a boost of 1.0 is skipped entirely (never reaches Tantivy).
        rt().block_on(async {
            let chunks = [
                mk("a", "rust is a systems programming language"),
                mk("b", "python is a scripting language"),
                mk("c", "go has built-in concurrency support"),
            ];
            let mut base = Bm25Retriever::new().unwrap();
            base.index(&chunks).await.unwrap();
            let mut uniform = Bm25Retriever::new()
                .unwrap()
                .with_field_weights(FieldWeights::uniform());
            uniform.index(&chunks).await.unwrap();

            for q in ["rust language", "python", "concurrency support"] {
                let a = base.retrieve(&Query::new(q), 3).await.unwrap();
                let b = uniform.retrieve(&Query::new(q), 3).await.unwrap();
                let ids_a: Vec<_> = a.iter().map(|h| h.chunk.id.as_str()).collect();
                let ids_b: Vec<_> = b.iter().map(|h| h.chunk.id.as_str()).collect();
                assert_eq!(
                    ids_a, ids_b,
                    "uniform weights must match default order for {q:?}"
                );
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(
                        x.score.value, y.score.value,
                        "uniform weights must match default scores bit-for-bit for {q:?}"
                    );
                }
            }
        });
    }

    #[test]
    fn field_weights_heading_boost_changes_ranking() {
        // A heavy heading boost lifts a chunk that matches only in its heading
        // above a chunk that out-scores it on raw text-field term frequency —
        // the near-duplicate-catalog lever (boost the field carrying the
        // discriminative brand/title token).
        rt().block_on(async {
            let mut by_heading = Chunk::new("h", "generic filler words here", "doc", TokenCount(4));
            by_heading
                .metadata
                .insert("heading".into(), serde_json::json!("acme"));
            // High term frequency for "acme" in the body, nothing in heading.
            let by_text = mk("t", "acme acme acme acme acme product line");

            // Default weights: the high-TF text chunk wins.
            let mut base = Bm25Retriever::new().unwrap();
            base.index(&[by_heading.clone(), by_text.clone()])
                .await
                .unwrap();
            let unboosted = base.retrieve(&Query::new("acme"), 2).await.unwrap();
            assert_eq!(
                unboosted[0].chunk.id.as_str(),
                "t",
                "without a boost the high-frequency text chunk ranks first; got {:?}",
                unboosted
                    .iter()
                    .map(|h| h.chunk.id.as_str())
                    .collect::<Vec<_>>(),
            );

            // Heavy heading boost flips it: the heading-matching chunk leads.
            let mut boosted_r = Bm25Retriever::new()
                .unwrap()
                .with_field_weights(FieldWeights {
                    text: 1.0,
                    source: 1.0,
                    heading: 20.0,
                });
            boosted_r.index(&[by_heading, by_text]).await.unwrap();
            let boosted = boosted_r.retrieve(&Query::new("acme"), 2).await.unwrap();
            assert_eq!(
                boosted[0].chunk.id.as_str(),
                "h",
                "a heading boost must lift the heading-matching chunk to #1; got {:?}",
                boosted
                    .iter()
                    .map(|h| h.chunk.id.as_str())
                    .collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn char_ngram_analyzer_recovers_a_typo_word_bm25_misses() {
        use std::sync::Arc;
        rt().block_on(async {
            let chunks = [
                mk("lays", "lays classic salted potato chips"),
                mk("kurkure", "kurkure masala munch namkeen"),
                mk("bingo", "bingo mad angles tomato"),
            ];

            // Word-BM25 (default): a transcription-typo'd token (`lays` ->
            // `1ays`) shares no whole token, so it cannot reach the chunk.
            let mut word = Bm25Retriever::new().unwrap();
            word.index(&chunks).await.unwrap();
            let word_hits = word.retrieve(&Query::new("1ays"), 3).await.unwrap();
            assert!(
                word_hits.iter().all(|h| h.chunk.id.as_str() != "lays"),
                "word-BM25 should miss the typo'd token (no exact overlap); got {:?}",
                word_hits
                    .iter()
                    .map(|h| h.chunk.id.as_str())
                    .collect::<Vec<_>>(),
            );

            // Char-ngram analyzer: subword grams bridge the typo, no model.
            let mut ng = Bm25Retriever::with_analyzer(Arc::new(
                crate::analyzer::CharNgramAnalyzer::default(),
            ))
            .unwrap();
            ng.index(&chunks).await.unwrap();
            let ng_hits = ng.retrieve(&Query::new("1ays"), 3).await.unwrap();
            assert_eq!(
                ng_hits[0].chunk.id.as_str(),
                "lays",
                "char-ngram must recover the typo'd token; got {:?}",
                ng_hits
                    .iter()
                    .map(|h| h.chunk.id.as_str())
                    .collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn _silence_unused_doc() {
        let _ = Document::new("d", "t");
    }
}
