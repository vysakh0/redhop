# Design: pluggable lexical analyzer

**Status**: **IMPLEMENTED** on `main`. Queued for the next release.
Targets 0.2.0 because `ContextConfig` and `DocumentConfig` grew new
required fields — callers constructing those structs via field literals
from outside the crate need to add `analyzer: ...`.
**Scope**: cross-binding extension surface for the lexical analyzer (the
tokenizer + filter pipeline that drives both BM25 retrieval AND the
grounding scorer's term extraction).

## What today is broken

The promise in 0.1.4's `docs/LANGUAGE.md` — "the bones are there, when a
user needs German morphology they plug in here" — is misleading. There IS
no public extension point for the lexical analyzer. To swap the Snowball
stemmer from English to German, a user has to **fork the redhop crate**
and edit `crates/redhop/src/retrieval/bm25.rs::Bm25Retriever::new`. That's
not "extensible."

Worse, even with a forked redhop, the grounding scorer in
`crate::context::normalize` hardcodes the English stemmer separately. The
two layers (retrieval + grounding) would drift unless the user edits both.

Compare to the **dense** path, which IS publicly extensible:

```rust
let mut doc = Document::from_text("d", "…")?
    .with_embedder(Arc::new(my_custom_embedder));
```

The lexical path needs the same shape.

## API shape

```rust
//! New module: crates/redhop/src/analyzer.rs

/// Lexical analyzer plugin point. The default is English Snowball Porter2
/// (`EnglishAnalyzer`). Swap to another via `Document::with_analyzer`.
///
/// One `Analyzer` drives BOTH the BM25 retrieval pipeline AND the grounding
/// scorer's term extraction so the two layers stay in lockstep (the recurring
/// "same tokenizer/scorer contract" we kept fixing through 0.1.3-0.1.4).
pub trait Analyzer: Send + Sync + std::fmt::Debug {
    /// Identifier used to register the analyzer against Tantivy's tokenizer
    /// manager. Must be unique per implementation.
    fn name(&self) -> &str;

    /// Tokenize + normalize text into search terms. Used by the grounding
    /// scorer in `crate::context::normalize`. Must produce the same terms
    /// that `build_text_analyzer().token_stream(text)` would, so BM25 and
    /// grounding agree on what "the same term" means.
    fn tokens(&self, text: &str) -> Vec<String>;

    /// Build the Tantivy `TextAnalyzer` for the BM25 index. Called once
    /// per `Bm25Retriever::with_analyzer`. Default impl: compose a pipeline
    /// from `tokens()` so callers only override one method. Custom impls
    /// may override to use Tantivy's built-in filters (faster).
    fn build_text_analyzer(&self) -> tantivy::tokenizer::TextAnalyzer;
}

/// The 0.1.4 default — pre-baked English pipeline.
/// SimpleTokenizer → RemoveLong(40) → CamelCaseSplitter → AsciiFolding
/// → LowerCaser → StopWordFilter(English) → Snowball English stemmer.
pub struct EnglishAnalyzer { /* … */ }

/// Snowball Porter2 stemmer over any of `rust-stemmers`' 17 languages.
/// Stopwords are caller-supplied (we ship empty lists by default for
/// non-English — callers can supply their own).
pub struct SnowballAnalyzer {
    language: rust_stemmers::Algorithm,
    stopwords: HashSet<String>,
    name: String,
    /// Same filter chain as EnglishAnalyzer (camelCase split + ASCII fold
    /// + lowercase + stopword + Snowball with `language`).
}

// Builtins — pre-baked configs for the most common Snowball languages,
// with EMPTY stopword lists (gap honestly documented — callers can supply
// their own via `SnowballAnalyzer::with_stopwords`).
impl SnowballAnalyzer {
    pub fn french() -> Self { /* Algorithm::French + [] stopwords */ }
    pub fn german() -> Self { /* Algorithm::German + [] */ }
    pub fn spanish() -> Self { /* Algorithm::Spanish + [] */ }
    pub fn italian() -> Self { /* Algorithm::Italian + [] */ }
    pub fn portuguese() -> Self { /* Algorithm::Portuguese + [] */ }
    pub fn dutch() -> Self { /* Algorithm::Dutch + [] */ }
    pub fn russian() -> Self { /* Algorithm::Russian + [] */ }
    pub fn swedish() -> Self { /* Algorithm::Swedish + [] */ }
    pub fn norwegian() -> Self { /* Algorithm::Norwegian + [] */ }
    pub fn danish() -> Self { /* Algorithm::Danish + [] */ }
    pub fn romanian() -> Self { /* Algorithm::Romanian + [] */ }
    pub fn hungarian() -> Self { /* Algorithm::Hungarian + [] */ }
    pub fn turkish() -> Self { /* Algorithm::Turkish + [] */ }
    pub fn arabic() -> Self { /* Algorithm::Arabic + [] */ }
    pub fn greek() -> Self { /* Algorithm::Greek + [] */ }
    pub fn tamil() -> Self { /* Algorithm::Tamil + [] */ }

    /// Caller-supplied stopword list (must already be lowercase + folded).
    pub fn with_stopwords(mut self, stopwords: Vec<String>) -> Self { /* … */ }
}
```

## Wiring

### Document API

```rust
let mut doc = Document::from_text("d", "Bücher gelesen")?
    .with_analyzer(Arc::new(SnowballAnalyzer::german()));
```

`Document::with_analyzer(Arc<dyn Analyzer>)` mirrors `with_embedder` —
attaches an analyzer that gets passed to:
- `Bm25Retriever::with_analyzer(analyzer)` at index build time
- `ContextConfig::analyzer` for the grounding scorer

If not called, the default is `EnglishAnalyzer` (0.1.4 behavior preserved).

### Bm25Retriever

Add a new constructor:
```rust
pub fn with_analyzer(analyzer: Arc<dyn Analyzer>) -> Result<Self> {
    // Register analyzer.build_text_analyzer() against the index's
    // tokenizer manager under analyzer.name().
}
```

Keep `Bm25Retriever::new()` as a sugar that calls
`Bm25Retriever::with_analyzer(Arc::new(EnglishAnalyzer::new()))`.

### Grounding scorer

`crate::context::normalize` becomes a method on the analyzer (delegated):

```rust
// Before:
fn normalize(w: &str) -> Option<String> { /* hardcoded English */ }

// After:
fn normalize_with(w: &str, analyzer: &dyn Analyzer) -> Option<String> {
    let tokens = analyzer.tokens(w);
    tokens.into_iter().next()  // pre-tokenized input → single term
}

// `terms(text, analyzer)` becomes:
fn terms(text: &str, analyzer: &dyn Analyzer) -> HashSet<String> {
    analyzer.tokens(text).into_iter().collect()
}
```

`build_context` / `analyze_context` get `cfg.analyzer` from `ContextConfig`
(new field, default = `Arc::new(EnglishAnalyzer::new())`).

### Cross-binding

For `pyo3` / `napi-rs`: Rust trait objects don't cross FFI cleanly, so the
bindings get a **string-routed** view:

```python
doc = redhop.Document.from_text(text, language="german")
```

```javascript
const doc = Document.fromText(text, { language: "german" });
```

Maps to one of the builtin `SnowballAnalyzer::<language>()` constructors.

Truly-custom analyzers (a Python user implementing the trait via pyo3's
`PyAny`-based dispatch) are deliberately deferred — Python-side method
dispatch per token would be 100× slower than the Rust-side fold.
Documented as a future "advanced extension" once a real user needs it.

## What we get (and don't)

**Get**:
- Cross-binding language selection by name (`language="german"`, etc.).
- Rust-side custom analyzer trait (anyone can implement `Analyzer`).
- BM25 and grounding stay in lockstep — same trait drives both.
- Backward-compatible: `with_analyzer` is additive; existing callers see
  no change.

**Don't get** (deferred):
- Python/Node users implementing custom analyzers (FFI overhead per token).
- Calibrated stopword lists per language — we ship empty lists; users can
  supply their own. Calibration needs eval corpora per language.
- CJK word segmentation — needs a separate tokenizer family
  (lindera / jieba / kuromoji). Different scope.
- A `language` auto-detection step. Mixed-language content needs explicit
  per-chunk tagging by the loader; explicit > magic.

## Execution plan (3-5 commits)

1. **C1**: define the `Analyzer` trait + `EnglishAnalyzer` + `SnowballAnalyzer` in
   `crates/redhop/src/analyzer.rs`. No wiring yet — just the surface. No
   behavior change to existing code paths.

2. **C2**: refactor `Bm25Retriever` to take `Arc<dyn Analyzer>`. Keep
   `Bm25Retriever::new()` as a sugar over `with_analyzer(EnglishAnalyzer)`.
   No external behavior change.

3. **C3**: refactor `crate::context::normalize` + `terms` to use the
   analyzer from `ContextConfig`. Add `ContextConfig::analyzer` (default
   = English). No external behavior change.

4. **C4**: wire `Document::with_analyzer`. Add `DocumentConfig::analyzer`
   so the loaders can configure it via `LoadOptions::language: Option<String>`.

5. **C5**: bindings + tests + docs.
   - Python: `language: Option<String>` arg on every `from_*` constructor.
   - Node: `language: Option<String>` field on `Options`.
   - quality_suite tests for German `Bücher` → `Buch`, French
     `courait` → `court`, etc. with the new analyzer.
   - Update `docs/LANGUAGE.md` to point at the new public API.

Each commit keeps `cargo test --workspace`, `clippy -- -D warnings`, and
`fmt --check` green.

## Open questions before coding

1. **Do we want `with_analyzer` to take `Arc<dyn Analyzer>` or
   `Box<dyn Analyzer>`?** `Arc` mirrors `with_embedder` and lets the
   analyzer be cheap to clone across BM25 / grounding / reranker. Probably
   `Arc`.

2. **Should `ContextConfig::analyzer` be `Option<Arc<dyn Analyzer>>`
   (defaults to `None` → callers wire English internally) or
   `Arc<dyn Analyzer>` (always populated)?** The latter is cleaner but
   needs a non-`None` default value in `Default` impl. Probably the latter
   with a lazy-static English instance.

3. **Snowball builtins: which 8-10 languages do we pre-bake?** All 17 is
   easy (rust-stemmers ships them) but bloats the API surface. Top 8 by
   user population: english, spanish, french, german, portuguese, italian,
   russian, arabic. Or all 17 if we don't care about API noise.

4. **Cross-binding language string format**: ISO 639-1 (`"en"`, `"de"`)
   or full names (`"english"`, `"german"`)? Snowball's `Algorithm` enum
   uses full names. Match that for consistency.

5. **What's `EnglishAnalyzer::tokens()` doing differently from the rest?**
   It includes the English `STOPWORDS` const from `crate::context`. The
   generic `SnowballAnalyzer` defaults to empty stopwords. So
   `EnglishAnalyzer ≠ SnowballAnalyzer::with_language(English)`. Worth
   documenting; not a problem.

Sign off on the API shape and the open questions, then C1 begins.
