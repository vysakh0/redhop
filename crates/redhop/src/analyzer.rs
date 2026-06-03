//! Lexical analyzer plugin point — the cross-cutting tokenization that drives
//! BOTH [`crate::retrieval::Bm25Retriever`] (via Tantivy's `TextAnalyzer`
//! interface) AND the grounding scorer in [`crate::context`] (via
//! [`Analyzer::tokens`]).
//!
//! ## Why one analyzer drives both layers
//!
//! Through 0.1.3-0.1.4 a series of bugs surfaced where BM25's tokenization
//! and the grounding scorer's notion of "the same term" disagreed (stemming
//! on one side but not the other; stopwords on one side; camelCase split on
//! one side; ASCII fold on one side). Each one was a silent-search-miss.
//! The `Analyzer` trait makes the contract explicit: **a chunk that BM25
//! returns and the grounding scorer evaluates went through the same
//! pipeline.**
//!
//! Concretely, [`Analyzer::tokens`] is implemented in terms of
//! [`Analyzer::build_text_analyzer`]: the BM25 side and the grounding side
//! share a *single source of truth*, the Tantivy `TextAnalyzer`. There is no
//! way for them to drift.
//!
//! ## Default — English Snowball Porter2
//!
//! [`SnowballAnalyzer::english`] preserves the 0.1.4 hardcoded behavior bit-
//! for-bit: SimpleTokenizer → RemoveLong(40) → CamelCaseSplitter →
//! AsciiFolding → LowerCaser → StopWordFilter(English) → Porter2(English).
//!
//! ## Swapping languages
//!
//! [`SnowballAnalyzer`] is generic across the 18 languages
//! [`rust_stemmers`] ships:
//!
//! ```no_run
//! # fn main() -> redhop::Result<()> {
//! use std::sync::Arc;
//! use redhop::analyzer::SnowballAnalyzer;
//!
//! let german = Arc::new(SnowballAnalyzer::german());
//! let mut doc = redhop::Document::from_text("doc", "Bücher und Schriften")?
//!     .with_analyzer(german);
//! # Ok(()) }
//! ```
//!
//! Stopwords are caller-supplied for non-English (we don't ship lists we
//! can't calibrate). The English builtin includes the curated stopword set
//! shared with the grounding scorer; the rest ship with empty stopwords.
//!
//! ## Custom analyzers
//!
//! Implement the trait yourself for non-Snowball pipelines (e.g. a CJK
//! tokenizer like `lindera` or `jieba-rs`). Both methods must use the same
//! tokenization or BM25 / grounding will silently disagree — the default
//! `tokens` impl delegates to `build_text_analyzer` to keep them aligned.

use std::sync::{Arc, OnceLock};

use rust_stemmers::Algorithm;
use tantivy::tokenizer::{Language, TextAnalyzer, TokenStream};

use crate::retrieval::build_redhop_pipeline;

/// Lexical analyzer plugin. Drives BM25 retrieval AND grounding scoring.
///
/// Implementors get the grounding/retrieval parity guarantee for free by
/// letting the default [`Analyzer::tokens`] impl delegate to
/// [`Analyzer::build_text_analyzer`].
pub trait Analyzer: Send + Sync + std::fmt::Debug {
    /// Identifier used to register the analyzer against Tantivy's tokenizer
    /// manager. Must be unique per implementation; recommended convention is
    /// the lowercase language family (`"english"`, `"german"`, …) or
    /// `<vendor>_<name>` for custom analyzers (`"acme_japanese"`).
    fn name(&self) -> &str;

    /// Build the Tantivy `TextAnalyzer` for BM25 indexing + query parsing.
    /// Called once when a [`crate::retrieval::Bm25Retriever`] is constructed
    /// with this analyzer.
    ///
    /// Implementors MUST ensure that running this pipeline over a piece of
    /// text produces the same tokens as [`Analyzer::tokens`] — the default
    /// impl of `tokens` does this for free.
    fn build_text_analyzer(&self) -> TextAnalyzer;

    /// Tokenize + normalize text into search terms. Called by the grounding
    /// scorer in [`crate::context`].
    ///
    /// The default impl runs the analyzer returned by
    /// [`Analyzer::build_text_analyzer`] and collects its token text. This
    /// keeps the BM25 side and the grounding side in lockstep automatically;
    /// overriding it usually isn't necessary.
    fn tokens(&self, text: &str) -> Vec<String> {
        let mut analyzer = self.build_text_analyzer();
        let mut stream = analyzer.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.to_string());
        }
        out
    }
}

// ── SnowballAnalyzer ────────────────────────────────────────────────────────

/// Snowball Porter2-based analyzer, generic across the 18 languages
/// [`rust_stemmers`] supports.
///
/// The pipeline (same for every language) is:
///
/// 1. `SimpleTokenizer` (Unicode whitespace + punctuation boundaries)
/// 2. `RemoveLongFilter(40)`
/// 3. `CamelCaseSplitter` (split camelCase / PascalCase / letter↔digit; emits
///    both the original token and the pieces so `compress` finds `compressVideo`)
/// 4. `AsciiFoldingFilter` (NFKD-fold combining diacritics: `café` → `cafe`)
/// 5. `LowerCaser`
/// 6. `StopWordFilter(stopwords)` (caller-supplied; English builtin includes
///    the curated list from `crate::context::STOPWORDS`)
/// 7. `Stemmer(language)`
///
/// Use the per-language constructors ([`SnowballAnalyzer::english`],
/// [`SnowballAnalyzer::german`], …) or [`SnowballAnalyzer::for_language`]
/// for the generic case. Use [`SnowballAnalyzer::with_stopwords`] to attach a
/// custom stopword list (words must already be lowercase + ASCII-folded).
#[derive(Clone, Debug)]
pub struct SnowballAnalyzer {
    name: String,
    language: Algorithm,
    /// Shared so cloning a `SnowballAnalyzer` is cheap.
    stopwords: Arc<Vec<String>>,
}

impl SnowballAnalyzer {
    /// Generic constructor — used by all the per-language sugar methods.
    /// `stopwords` are taken as-is (no folding / lowercasing applied) so the
    /// caller controls exactly which surface forms are filtered.
    pub fn new(name: impl Into<String>, language: Algorithm, stopwords: Vec<String>) -> Self {
        Self {
            name: name.into(),
            language,
            stopwords: Arc::new(stopwords),
        }
    }

    /// Build with empty stopwords. Equivalent to
    /// `SnowballAnalyzer::new(name, language, vec![])`.
    pub fn for_language(language: Algorithm) -> Self {
        let name = snowball_language_name(language);
        Self::new(name, language, Vec::new())
    }

    /// Replace the stopword list. Words must already be lowercase +
    /// ASCII-folded — the analyzer compares the post-LowerCaser-post-fold
    /// token to the list verbatim.
    pub fn with_stopwords(mut self, stopwords: Vec<String>) -> Self {
        self.stopwords = Arc::new(stopwords);
        self
    }

    /// Look up a builtin by lowercase language name (`"english"`,
    /// `"german"`, `"french"`, …). Returns `None` for unknown names — callers
    /// decide whether to error or fall back to English.
    ///
    /// Recognised names: `arabic, danish, dutch, english, finnish, french,
    /// german, greek, hungarian, italian, norwegian, portuguese, romanian,
    /// russian, spanish, swedish, tamil, turkish` (the 18 Snowball Porter2
    /// languages that ship in `rust_stemmers` + Tantivy).
    pub fn by_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "arabic" => Self::arabic(),
            "danish" => Self::danish(),
            "dutch" => Self::dutch(),
            "english" => Self::english(),
            "finnish" => Self::finnish(),
            "french" => Self::french(),
            "german" => Self::german(),
            "greek" => Self::greek(),
            "hungarian" => Self::hungarian(),
            "italian" => Self::italian(),
            "norwegian" => Self::norwegian(),
            "portuguese" => Self::portuguese(),
            "romanian" => Self::romanian(),
            "russian" => Self::russian(),
            "spanish" => Self::spanish(),
            "swedish" => Self::swedish(),
            "tamil" => Self::tamil(),
            "turkish" => Self::turkish(),
            _ => return None,
        })
    }

    // ── Per-language builtins ──────────────────────────────────────────────

    /// English Snowball Porter2 with the curated stopword list from
    /// `crate::context::STOPWORDS` (shared with the grounding scorer in
    /// 0.1.4 — the same words the distractor/linkage signals drop). The 0.1.4
    /// default; preserves the hardcoded behavior bit-for-bit.
    pub fn english() -> Self {
        static EN_STOPS: OnceLock<Vec<String>> = OnceLock::new();
        let stopwords = EN_STOPS
            .get_or_init(|| {
                crate::context::STOPWORDS
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect()
            })
            .clone();
        Self::new("english", Algorithm::English, stopwords)
    }

    // Per-language builtins — the remaining 17 Snowball Porter2 languages
    // shipped by `rust_stemmers`. Each defaults to an empty stopword list;
    // attach one via [`SnowballAnalyzer::with_stopwords`] if you have a
    // curated list for that language.

    /// Arabic Snowball analyzer (empty stopwords by default).
    pub fn arabic() -> Self {
        Self::new("arabic", Algorithm::Arabic, Vec::new())
    }
    /// Danish Snowball analyzer (empty stopwords by default).
    pub fn danish() -> Self {
        Self::new("danish", Algorithm::Danish, Vec::new())
    }
    /// Dutch Snowball analyzer (empty stopwords by default).
    pub fn dutch() -> Self {
        Self::new("dutch", Algorithm::Dutch, Vec::new())
    }
    /// Finnish Snowball analyzer (empty stopwords by default).
    pub fn finnish() -> Self {
        Self::new("finnish", Algorithm::Finnish, Vec::new())
    }
    /// French Snowball analyzer (empty stopwords by default).
    pub fn french() -> Self {
        Self::new("french", Algorithm::French, Vec::new())
    }
    /// German Snowball analyzer (empty stopwords by default).
    pub fn german() -> Self {
        Self::new("german", Algorithm::German, Vec::new())
    }
    /// Greek Snowball analyzer (empty stopwords by default).
    pub fn greek() -> Self {
        Self::new("greek", Algorithm::Greek, Vec::new())
    }
    /// Hungarian Snowball analyzer (empty stopwords by default).
    pub fn hungarian() -> Self {
        Self::new("hungarian", Algorithm::Hungarian, Vec::new())
    }
    /// Italian Snowball analyzer (empty stopwords by default).
    pub fn italian() -> Self {
        Self::new("italian", Algorithm::Italian, Vec::new())
    }
    /// Norwegian Snowball analyzer (empty stopwords by default).
    pub fn norwegian() -> Self {
        Self::new("norwegian", Algorithm::Norwegian, Vec::new())
    }
    /// Portuguese Snowball analyzer (empty stopwords by default).
    pub fn portuguese() -> Self {
        Self::new("portuguese", Algorithm::Portuguese, Vec::new())
    }
    /// Romanian Snowball analyzer (empty stopwords by default).
    pub fn romanian() -> Self {
        Self::new("romanian", Algorithm::Romanian, Vec::new())
    }
    /// Russian Snowball analyzer (empty stopwords by default).
    pub fn russian() -> Self {
        Self::new("russian", Algorithm::Russian, Vec::new())
    }
    /// Spanish Snowball analyzer (empty stopwords by default).
    pub fn spanish() -> Self {
        Self::new("spanish", Algorithm::Spanish, Vec::new())
    }
    /// Swedish Snowball analyzer (empty stopwords by default).
    pub fn swedish() -> Self {
        Self::new("swedish", Algorithm::Swedish, Vec::new())
    }
    /// Tamil Snowball analyzer (empty stopwords by default).
    pub fn tamil() -> Self {
        Self::new("tamil", Algorithm::Tamil, Vec::new())
    }
    /// Turkish Snowball analyzer (empty stopwords by default).
    pub fn turkish() -> Self {
        Self::new("turkish", Algorithm::Turkish, Vec::new())
    }
}

impl Default for SnowballAnalyzer {
    /// The default analyzer is [`SnowballAnalyzer::english`] — preserves the
    /// 0.1.4 behavior.
    fn default() -> Self {
        Self::english()
    }
}

impl Analyzer for SnowballAnalyzer {
    fn name(&self) -> &str {
        &self.name
    }

    fn build_text_analyzer(&self) -> TextAnalyzer {
        build_redhop_pipeline(
            (*self.stopwords).clone(),
            snowball_to_tantivy(self.language),
        )
    }
}

// ── Shared "default English instance" ──────────────────────────────────────

/// A process-wide cloned `Arc<dyn Analyzer>` pointing at
/// [`SnowballAnalyzer::english`]. Used as the default in
/// [`crate::context::ContextConfig`] and elsewhere — clone is cheap (just
/// an Arc bump), construction happens
/// once.
pub fn default_english() -> Arc<dyn Analyzer> {
    static INSTANCE: OnceLock<Arc<dyn Analyzer>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| Arc::new(SnowballAnalyzer::english()) as Arc<dyn Analyzer>)
        .clone()
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Map `rust_stemmers::Algorithm` to Tantivy's `Language` enum. They both
/// cover the same 18 Snowball languages; this just shuttles between the two
/// representations.
fn snowball_to_tantivy(a: Algorithm) -> Language {
    match a {
        Algorithm::Arabic => Language::Arabic,
        Algorithm::Danish => Language::Danish,
        Algorithm::Dutch => Language::Dutch,
        Algorithm::English => Language::English,
        Algorithm::Finnish => Language::Finnish,
        Algorithm::French => Language::French,
        Algorithm::German => Language::German,
        Algorithm::Greek => Language::Greek,
        Algorithm::Hungarian => Language::Hungarian,
        Algorithm::Italian => Language::Italian,
        Algorithm::Norwegian => Language::Norwegian,
        Algorithm::Portuguese => Language::Portuguese,
        Algorithm::Romanian => Language::Romanian,
        Algorithm::Russian => Language::Russian,
        Algorithm::Spanish => Language::Spanish,
        Algorithm::Swedish => Language::Swedish,
        Algorithm::Tamil => Language::Tamil,
        Algorithm::Turkish => Language::Turkish,
    }
}

/// Default `name()` for a given language. Used by
/// [`SnowballAnalyzer::for_language`].
fn snowball_language_name(a: Algorithm) -> &'static str {
    match a {
        Algorithm::Arabic => "arabic",
        Algorithm::Danish => "danish",
        Algorithm::Dutch => "dutch",
        Algorithm::English => "english",
        Algorithm::Finnish => "finnish",
        Algorithm::French => "french",
        Algorithm::German => "german",
        Algorithm::Greek => "greek",
        Algorithm::Hungarian => "hungarian",
        Algorithm::Italian => "italian",
        Algorithm::Norwegian => "norwegian",
        Algorithm::Portuguese => "portuguese",
        Algorithm::Romanian => "romanian",
        Algorithm::Russian => "russian",
        Algorithm::Spanish => "spanish",
        Algorithm::Swedish => "swedish",
        Algorithm::Tamil => "tamil",
        Algorithm::Turkish => "turkish",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_analyzer_stems_via_tokens() {
        let a = SnowballAnalyzer::english();
        let tokens = a.tokens("running compresses compression");
        // All three should stem to the same root.
        assert!(
            tokens.contains(&"run".to_string()),
            "expected 'run' in tokens, got {tokens:?}"
        );
        assert!(
            tokens.contains(&"compress".to_string()),
            "expected 'compress' in tokens, got {tokens:?}"
        );
    }

    #[test]
    fn english_analyzer_drops_stopwords() {
        let a = SnowballAnalyzer::english();
        let tokens = a.tokens("the quick brown fox");
        assert!(
            !tokens.contains(&"the".to_string()),
            "'the' should be filtered as a stopword"
        );
        assert!(tokens.contains(&"quick".to_string()));
    }

    #[test]
    fn german_analyzer_stems_german_morphology() {
        // Bücher (books, plural) and Buch (book, singular) should share a
        // stem under the German Porter2 algorithm. The English stemmer
        // wouldn't unify them.
        let a = SnowballAnalyzer::german();
        let plural = a.tokens("Bücher");
        let singular = a.tokens("Buch");
        assert_eq!(
            plural, singular,
            "German analyzer should stem 'Bücher' and 'Buch' to the same form; \
             got plural={plural:?} singular={singular:?}"
        );
    }

    #[test]
    fn french_analyzer_stems_french_morphology() {
        // `manger` (to eat) / `mange` (eat-1sg) — common verb inflections.
        // French Porter2 strips `-er` and `-e` to a common stem.
        let a = SnowballAnalyzer::french();
        let inf = a.tokens("manger");
        let pres = a.tokens("mange");
        assert_eq!(
            inf, pres,
            "French analyzer should stem 'manger' and 'mange' to the same form; \
             got infinitive={inf:?} present={pres:?}"
        );
    }

    #[test]
    fn ascii_folding_works_in_all_languages() {
        // café → cafe via AsciiFoldingFilter, regardless of which language's
        // stemmer is selected.
        for analyzer in [
            SnowballAnalyzer::english(),
            SnowballAnalyzer::french(),
            SnowballAnalyzer::german(),
            SnowballAnalyzer::spanish(),
        ] {
            let with_accent = analyzer.tokens("café");
            let without_accent = analyzer.tokens("cafe");
            assert_eq!(
                with_accent, without_accent,
                "{}: 'café' and 'cafe' should produce identical tokens; got {with_accent:?} vs {without_accent:?}",
                analyzer.name()
            );
        }
    }

    #[test]
    fn camelcase_split_in_all_languages() {
        // CamelCaseSplitter applies to every SnowballAnalyzer regardless of
        // language — emits both the original token and the case-split parts.
        let a = SnowballAnalyzer::english();
        let tokens = a.tokens("compressVideo");
        assert!(
            tokens.iter().any(|t| t == "compress"),
            "camelCase split should emit 'compress' alongside; got {tokens:?}"
        );
    }

    #[test]
    fn by_name_routes_to_builtins() {
        for (name, expected_lang) in [
            ("english", Algorithm::English),
            ("german", Algorithm::German),
            ("french", Algorithm::French),
            ("spanish", Algorithm::Spanish),
            ("ENGLISH", Algorithm::English), // case-insensitive
        ] {
            let a = SnowballAnalyzer::by_name(name).unwrap_or_else(|| panic!("by_name({name:?})"));
            assert_eq!(a.language, expected_lang);
        }
        assert!(
            SnowballAnalyzer::by_name("klingon").is_none(),
            "unknown language must return None, not English fallback"
        );
    }

    #[test]
    fn default_english_returns_same_arc() {
        let a = default_english();
        let b = default_english();
        // Both should be Arcs to the same instance (OnceLock-cached).
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn tokens_and_build_text_analyzer_agree() {
        // The contract: tokens() output equals what build_text_analyzer()
        // would emit when fed the same text. The default impl gives this for
        // free by delegating; this test pins it as a guarantee.
        let a = SnowballAnalyzer::english();
        let direct = a.tokens("The Quick Brown Fox jumps over compressVideo");
        let via_analyzer = {
            let mut ta = a.build_text_analyzer();
            let mut stream = ta.token_stream("The Quick Brown Fox jumps over compressVideo");
            let mut out = Vec::new();
            while stream.advance() {
                out.push(stream.token().text.to_string());
            }
            out
        };
        assert_eq!(direct, via_analyzer);
    }

    /// Smoke every advertised Snowball builtin. Catches: (a) a missing
    /// per-language constructor, (b) drift between the by_name() name
    /// list and the actual constructors, (c) a rust-stemmers ABI break
    /// on any single language we don't otherwise exercise.
    ///
    /// We don't assert per-language linguistic correctness here — we
    /// have no eval corpus for most of these languages (see
    /// docs/LANGUAGE.md's calibration disclaimer). What we DO assert:
    /// the builtin is reachable, its analyzer name matches the input
    /// string, and `tokens()` round-trips on a plain ASCII sentence
    /// without panicking. Per-language behavioral tests for the
    /// languages we ship corpora for (German, French) live in
    /// `tests/quality_suite.rs` (T41-T44).
    #[test]
    fn all_18_snowball_builtins_construct_and_tokenize() {
        // Every name advertised in the unknown-language error message
        // (in python/src/lib.rs and nodejs/src/lib.rs) must work here,
        // or callers see "supported: ..., tamil, ..." and then get a
        // None from by_name. The match below is the authoritative list.
        let all = [
            "arabic",
            "danish",
            "dutch",
            "english",
            "finnish",
            "french",
            "german",
            "greek",
            "hungarian",
            "italian",
            "norwegian",
            "portuguese",
            "romanian",
            "russian",
            "spanish",
            "swedish",
            "tamil",
            "turkish",
        ];

        for name in all {
            let a = SnowballAnalyzer::by_name(name)
                .unwrap_or_else(|| panic!("by_name({name:?}) returned None"));
            assert_eq!(a.name(), name, "analyzer name should match input");

            let tokens = a.tokens("the quick brown fox jumps over the lazy dog");
            assert!(
                !tokens.is_empty(),
                "{name}: tokens() should produce non-empty output on a plain ASCII sentence"
            );

            // Build the Tantivy TextAnalyzer too — proves the cross-FFI
            // path is wired, not just the in-Rust tokens() helper.
            let mut ta = a.build_text_analyzer();
            let mut stream = ta.token_stream("hello world");
            let mut got_any = false;
            while stream.advance() {
                got_any = true;
            }
            assert!(
                got_any,
                "{name}: build_text_analyzer() stream should emit at least one token"
            );
        }
    }
}
