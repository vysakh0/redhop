# Design: pluggable lexical analyzer

**Status**: **shipped** on `main`. Queued for the next release (0.2.0:
`ContextConfig` and `DocumentConfig` grew new required fields — callers
constructing those via struct field literals from outside the crate need
to add `analyzer: ...`; callers using `..Default::default()` are
unaffected).

## What this solves

Through 0.1.3-0.1.4 a class of silent-search-miss bugs surfaced where
BM25's tokenization and the grounding scorer's notion of "the same term"
disagreed (stemming on one side but not the other; stopwords on one
side; camelCase split on one side; ASCII fold on one side). Each one
was fixed by hand. The plugin closes that class structurally: one
`Analyzer` drives both layers, so they cannot drift.

The plugin also surfaces a public extension point for non-English
content. Before 0.1.5 a user who wanted German morphology had to fork
the crate and edit `crates/redhop/src/retrieval/bm25.rs::Bm25Retriever::new`
*and* the grounding scorer separately — and accept silent drift between
them. That's gone.

## API surface

`crate::analyzer` (in `crates/redhop/src/analyzer.rs`):

```rust
pub trait Analyzer: Send + Sync + std::fmt::Debug {
    /// Identifier used to register the analyzer against Tantivy's tokenizer
    /// manager. Must be unique per implementation.
    fn name(&self) -> &str;

    /// Build the Tantivy `TextAnalyzer` for BM25 indexing + query parsing.
    /// Called once when a `Bm25Retriever` is constructed with this analyzer.
    fn build_text_analyzer(&self) -> tantivy::tokenizer::TextAnalyzer;

    /// Tokenize + normalize text into search terms. Default impl runs the
    /// analyzer from `build_text_analyzer()` and collects its output, so
    /// the BM25 side and the grounding side share a single source of
    /// truth. Override only if a custom analyzer can produce the term
    /// list more cheaply than running the Tantivy pipeline.
    fn tokens(&self, text: &str) -> Vec<String> { /* default impl */ }
}

/// Snowball Porter2 stemmer over any of `rust_stemmers`' 18 languages.
/// Pipeline: SimpleTokenizer → RemoveLong(40) → CamelCaseSplitter →
/// AsciiFolding → LowerCaser → StopWordFilter(<lang>) → Snowball(<lang>).
pub struct SnowballAnalyzer { /* … */ }

impl SnowballAnalyzer {
    /// English builtin — ships with the curated stopword list from
    /// `crate::context::STOPWORDS`.
    pub fn english() -> Self;

    /// One pre-baked constructor per Snowball language (`german`,
    /// `french`, `spanish`, `italian`, `portuguese`, `dutch`, `russian`,
    /// `swedish`, `norwegian`, `danish`, `finnish`, `romanian`,
    /// `hungarian`, `turkish`, `arabic`, `greek`, `tamil`). All default
    /// to **empty** stopword lists — see Calibration disclaimer below.
    pub fn german() -> Self;
    pub fn french() -> Self;
    // … 15 more

    /// Route a language name (case-insensitive) to its builtin. Used by
    /// the string-routed binding surfaces (Python `language=`, Node
    /// `language`). Returns `None` for unknown names so callers can
    /// surface a clean error rather than silently falling back.
    pub fn by_name(name: &str) -> Option<Self>;

    /// Attach a stopword list (lowercase + folded) on top of the
    /// language's stemmer.
    pub fn with_stopwords(self, stopwords: Vec<String>) -> Self;
}

/// Process-wide cached `Arc<dyn Analyzer>` pointing at
/// `SnowballAnalyzer::english()`. Used as the default in
/// `ContextConfig::analyzer` (always populated).
pub fn default_english() -> Arc<dyn Analyzer>;
```

## Wiring

### Document API

```rust
let mut doc = redhop::Document::from_text("d", "ich habe Bücher")?
    .with_analyzer(Arc::new(SnowballAnalyzer::german()));
```

`Document::with_analyzer(Arc<dyn Analyzer>)` mirrors `with_embedder` — it
sets `self.analyzer` (drives the retrievers) AND
`self.cfg.context.analyzer` (drives the grounding scorer) in lockstep,
and resets the lazily-built BM25 index since analyzer choice is fixed
at index time.

### `Bm25Retriever`

```rust
let bm25 = Bm25Retriever::with_analyzer(analyzer.clone())?;
// `Bm25Retriever::new()` is sugar over with_analyzer(default_english()).
```

### Grounding scorer

`crate::context::terms(text, &dyn Analyzer)` collects the analyzer's
output for the grounding-scorer pass. `build_context` /
`analyze_context` read the analyzer from `cfg.context.analyzer` (always
populated; default = `default_english()`).

### Cross-binding (Python / Node)

Rust trait objects don't cross FFI cleanly, so the bindings expose a
**string-routed** view:

```python
doc = redhop.Document.from_text(text, options=redhop.DocumentOptions(language="german"))
```

```javascript
const doc = Document.fromText(text, { language: "german" });
```

Both map to a `SnowballAnalyzer::by_name(...)` lookup in `LoadOptions`.
Unknown names surface as an `Err`/`ValueError`/`Error` — no silent
fallback.

Custom analyzers implemented in Python or Node (via pyo3 `PyAny`-based
dispatch / napi callbacks) are deliberately deferred — per-token FFI
dispatch would be 100× slower than the Rust-side fold. Implement the
trait in Rust if you need a CJK tokenizer or a custom pipeline.

## Pinning tests

- `crates/redhop/src/analyzer.rs` (mod tests) — 10 unit tests: English
  stems, English stopwords, German morphology, French inflections,
  ASCII folding across languages, camelCase split, `by_name` routing
  (case-insensitive + unknown name), `default_english` Arc identity,
  `tokens` ⇔ `build_text_analyzer` agreement, and a parametrized smoke
  across all 18 builtins.
- `crates/redhop/tests/quality_suite.rs::t41`-`t45` — end-to-end
  behavior: German `Bücher`↔`Buch` via `with_analyzer(German)`, French
  `manger`↔`mange`, both-layers-swapped proof, unknown-language error,
  and per-Document analyzer isolation.
- `python/tests/test_analyzer.py` — 24 cases through the pyo3 boundary
  (caught a real binding bug on first run: `from_chunks` was silently
  dropping `language=`).
- `nodejs/test/analyzer.cjs` — 24 assertions through the napi boundary.

## Calibration disclaimer

Pre-baked non-English builtins get you the Snowball Porter2 stemmer for
that language, full stop. There is **no eval corpus** for non-English —
ranking quality on a real German legal contract is not measured to
match English Snowball's on a real English one. The pinning tests prove
morphology unifies (`Bücher` ↔ `Buch`); they don't prove ranking
quality on a domain corpus.

For demanding use cases, supply a custom `Analyzer` impl and benchmark
on your own corpus.

## What's out of scope

- **Python/Node users implementing custom analyzers** — FFI overhead
  per token. Implement in Rust.
- **Calibrated stopword lists per language** — empty by default; users
  can supply their own via `SnowballAnalyzer::with_stopwords`.
- **CJK word segmentation** — needs a separate tokenizer family
  (lindera / jieba / kuromoji); not shipped. Implement the trait
  yourself and attach via `Document::with_analyzer`.
- **Language auto-detection** — mixed-language content needs explicit
  per-chunk tagging by the loader.
