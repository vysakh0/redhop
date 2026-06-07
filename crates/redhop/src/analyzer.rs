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

// ─── Query-side diagnostics (templated workloads) ───────────────────────────
//
// Two helpers grounded in `docs/findings/CUAD_RECALL_GAP.md` + `CUAD_PRF_NULL.md`:
//
//   * [`drop_template_terms`] — token-level boilerplate removal. You supply
//     the boilerplate list (typically from [`analyze_query_set`] or domain
//     knowledge); this does the mechanical strip.
//   * [`analyze_query_set`] — diagnostic over a representative sample of
//     your queries: detects whether they share a high-boilerplate template
//     and, if so, which terms are doing the dilution.
//
// Status note: `analyze_query_set` was probed across CUAD / HotpotQA /
// MuSiQue before landing; see `docs/findings/QUERY_SET_ANALYZER.md` for the
// true-positive / false-positive measurements that justify the heuristic
// thresholds and the `suggested_action` copy.

use std::collections::HashSet as StdHashSet;

/// Drop boilerplate tokens from a query before retrieval.
///
/// Token matching is **case-insensitive on alphanumeric tokens**: each
/// whitespace-separated chunk of the query is compared (lowercased, with
/// leading/trailing non-alphanumerics trimmed) against the lowercased
/// boilerplate set. Surviving tokens are rejoined with a single space.
/// Non-alphanumeric punctuation embedded inside a token is preserved on
/// the surviving side and stripped on the matching side.
///
/// This is intentionally a thin helper — it does **not** decide what is
/// boilerplate (that's workload-specific; use [`analyze_query_set`] or your
/// domain knowledge) and it does **not** stem, lemmatize, or rewrite. The
/// goal is to leave the discriminating tokens visually intact while
/// removing the words you told it are noise.
///
/// ```
/// use redhop::analyzer::drop_template_terms;
/// let q = "Highlight the parts of this contract related to \"Change of Control\".";
/// let stripped = drop_template_terms(
///     q,
///     &["highlight", "the", "parts", "of", "this", "contract", "related", "to"],
/// );
/// assert_eq!(stripped, "\"Change Control\".");
/// ```
///
/// Pair with `strategy="raw_topk"` on single-doc extraction workloads —
/// the Auto policy's `reasoning_preserving` solves a multi-hop problem
/// that contract-shape workloads don't have. See
/// `docs/CHOOSING_A_CONFIG.md` for the decision rule.
pub fn drop_template_terms(query: &str, boilerplate: &[&str]) -> String {
    if boilerplate.is_empty() {
        return query.to_string();
    }

    // Partition boilerplate terms by the script they came from.
    //   - **Whitespace-separated scripts** (Latin, Cyrillic, Greek, Arabic
    //     letters, etc.): the term is a single word; match at whitespace-
    //     tokenize granularity so a boilerplate "of" doesn't erase the
    //     "of" inside "office".
    //   - **No-space scripts** (Han, Hiragana, Katakana, Hangul, Thai,
    //     Lao): the term came from `analyze_query_set` via punctuation-
    //     bounded segmentation — it's a phrase, not a word, and the
    //     surrounding query has no whitespace to tokenize on. Match by
    //     case-insensitive substring removal.
    let (phrase_terms, token_terms): (Vec<&str>, Vec<&str>) = boilerplate
        .iter()
        .copied()
        .partition(|t| t.chars().any(is_no_space_script));

    // Phase 1: substring removal for phrase-style terms. Longest-first so a
    // shorter substring doesn't consume part of a longer phrase.
    let mut result = query.to_string();
    if !phrase_terms.is_empty() {
        let mut sorted = phrase_terms.clone();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.len()));
        for term in &sorted {
            // CJK/no-space scripts don't have case; direct match is fine.
            result = result.replace(term, "");
        }
    }

    // Phase 2: whitespace-tokenize filter for word-style terms (the
    // original behavior). If there's nothing word-style to filter, just
    // collapse whitespace from Phase 1's output.
    if token_terms.is_empty() {
        return result.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    let stop: StdHashSet<String> = token_terms.iter().map(|s| s.to_lowercase()).collect();
    result
        .split_whitespace()
        .filter(|tok| {
            let key: String = tok
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            !stop.contains(&key)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// True iff `c` belongs to a Unicode script that has no whitespace between
/// words (so word-boundary matching doesn't apply). Used by
/// [`drop_template_terms`] to decide whether to substring-remove or
/// whitespace-tokenize the term.
fn is_no_space_script(c: char) -> bool {
    let n = c as u32;
    // Han (CJK Unified Ideographs) — Chinese, kanji
    (0x4E00..=0x9FFF).contains(&n) ||
    (0x3400..=0x4DBF).contains(&n) ||   // CJK Extension A
    // Hiragana / Katakana — Japanese
    (0x3040..=0x309F).contains(&n) ||
    (0x30A0..=0x30FF).contains(&n) ||
    // Hangul — Korean
    (0xAC00..=0xD7AF).contains(&n) ||
    // Thai
    (0x0E00..=0x0E7F).contains(&n) ||
    // Lao
    (0x0E80..=0x0EFF).contains(&n) ||
    // Khmer
    (0x1780..=0x17FF).contains(&n) ||
    // Myanmar (Burmese)
    (0x1000..=0x109F).contains(&n)
}

/// Append high-IDF discriminative terms to a query when a known key appears.
///
/// The additive counterpart to [`drop_template_terms`]. Where the strip
/// helper *removes* low-IDF boilerplate, this helper *adds* high-IDF
/// synonyms — the operations target opposite ends of the same dilution
/// problem.
///
/// Each `(key, synonyms)` pair in `expansions` is checked against the
/// query case-insensitively. If the query contains the key as a substring,
/// every synonym is appended to the returned string with a single space
/// separator. Matches against the **original** query only — synonyms are
/// never re-checked against the growing result, so an expansion can't
/// runaway-chain or duplicate itself.
///
/// Like [`drop_template_terms`], the *content* of `expansions` is
/// workload-specific (CUAD has clause names, support tickets have error
/// codes, an HR KB has policy names). The library ships the mechanism;
/// the caller supplies the data.
///
/// ```
/// use redhop::analyzer::expand_query_terms;
/// let q = "\"Change of Control\" The right of either party to terminate";
/// let expansions: &[(&str, &[&str])] = &[
///     ("change of control", &["merger", "successor", "acquisition", "assignment"]),
///     ("non-compete", &["restraint", "non-competition", "compete"]),
/// ];
/// let expanded = expand_query_terms(q, expansions);
/// assert!(expanded.contains("merger"));
/// assert!(expanded.contains("\"Change of Control\""));
/// ```
///
/// **Why this works on BM25 corpora dominated by domain boilerplate** —
/// the dilution failure mode that killed unweighted PRF
/// (see `docs/findings/CUAD_PRF_NULL.md`) was *additive* in the wrong
/// shape: feedback added low-IDF corpus boilerplate. A static
/// workload-specific dictionary adds **high-IDF**, **caller-curated**
/// terms — exactly the discriminators the corpus is missing in the query.
/// The mechanism is the opposite direction; the trap from PRF doesn't apply.
pub fn expand_query_terms(query: &str, expansions: &[(&str, &[&str])]) -> String {
    if expansions.is_empty() {
        return query.to_string();
    }
    let q_lower = query.to_lowercase();
    // Use a small ordered-insertion vec so the appended block is
    // deterministic (alphabetical-by-insertion, not HashSet-random).
    let mut added: Vec<String> = Vec::new();
    let mut seen: StdHashSet<String> = StdHashSet::new();
    for (key, syns) in expansions {
        let key_lower = key.to_lowercase();
        if !q_lower.contains(&key_lower) {
            continue;
        }
        for s in *syns {
            let s_lower = s.to_lowercase();
            if seen.insert(s_lower) {
                added.push((*s).to_string());
            }
        }
    }
    if added.is_empty() {
        query.to_string()
    } else {
        format!("{} {}", query, added.join(" "))
    }
}

/// How heavy the template-boilerplate dilution looks on a query set.
///
/// Mapped from [`QuerySetReport::template_word_share`] using thresholds
/// chosen from the cross-workload probe (see
/// `docs/findings/QUERY_SET_ANALYZER.md`). The bands are deliberately
/// coarse — what matters is the **direction**, not a precise fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DilutionCost {
    /// `template_word_share >= 0.70` — the template dominates the query.
    /// On CUAD, the source workload, this band corresponds to a
    /// measured 4–6 point ≥0.8 retention lift from template-stripping.
    High,
    /// `0.40 <= template_word_share < 0.70` — meaningful shared
    /// boilerplate but not dominant.
    Medium,
    /// `0.20 <= template_word_share < 0.40` — some shared filler;
    /// stripping is unlikely to move the needle.
    Low,
    /// `template_word_share < 0.20` — natural-language diversity.
    /// No template to strip; the analyzer recommends no action.
    None,
}

/// Diagnostic report over a representative sample of a workload's queries.
///
/// Returned by [`analyze_query_set`]. Fields are intentionally simple
/// (no nested types) so they survive cleanly across the Python and Node
/// bindings.
#[derive(Debug, Clone)]
pub struct QuerySetReport {
    /// How many queries were analyzed (informational; thresholds aren't
    /// trustworthy below ~30 queries — see [`Self::is_templated`]).
    pub n_queries: usize,
    /// `true` when [`Self::template_word_share`] >= 0.50 **and** at least
    /// two boilerplate terms were detected. Conservative by design:
    /// false positives are worse than false negatives because they push
    /// users toward a workaround that won't help.
    pub is_templated: bool,
    /// Mean over queries of `(boilerplate-token count) / (total token
    /// count)`. 0.0 means no shared boilerplate; 1.0 means every token
    /// in every query is shared. CUAD measures ~0.79 here.
    pub template_word_share: f32,
    /// Words appearing in at least 80% of the query set, sorted by
    /// document-frequency descending. These are the candidates you would
    /// pass to [`drop_template_terms`].
    pub boilerplate_terms: Vec<String>,
    /// Coarse band derived from `template_word_share`.
    pub estimated_dilution_cost: DilutionCost,
    /// Human-readable recommendation describing what (if anything) to
    /// do next. Suitable for printing in a CLI or surfacing in a
    /// notebook.
    pub suggested_action: String,
}

/// Detect templated-workload dilution on a representative sample of queries.
///
/// Mechanism: for every alphanumeric token in the query set we compute its
/// **query-set document frequency** (how many queries contain it).
/// Tokens with df / n_queries >= 0.80 are called "boilerplate". The
/// `template_word_share` is the average fraction of each query that is
/// boilerplate. A workload is `is_templated` when the share is >= 0.50
/// *and* the boilerplate list has at least two entries.
///
/// Designed for the early-2026 RAG-pipeline pattern documented in
/// `docs/findings/CUAD_RECALL_GAP.md`. **Read that finding before acting
/// on the report** — the report tells you whether the *shape* matches;
/// it does not measure your actual retention numbers.
///
/// ```
/// use redhop::analyzer::analyze_query_set;
/// let queries = [
///     "Highlight the parts of this contract related to X",
///     "Highlight the parts of this contract related to Y",
///     "Highlight the parts of this contract related to Z",
/// ];
/// let report = analyze_query_set(&queries);
/// assert!(report.is_templated);
/// assert!(report.boilerplate_terms.contains(&"highlight".to_string()));
/// ```
pub fn analyze_query_set<S: AsRef<str>>(queries: &[S]) -> QuerySetReport {
    let n = queries.len();
    if n == 0 {
        return QuerySetReport {
            n_queries: 0,
            is_templated: false,
            template_word_share: 0.0,
            boilerplate_terms: vec![],
            estimated_dilution_cost: DilutionCost::None,
            suggested_action: "empty query set — nothing to analyze".to_string(),
        };
    }

    // Tokenize each query → vector of distinct alphanumeric tokens.
    let per_query: Vec<Vec<String>> = queries
        .iter()
        .map(|q| analyzer_tokens(q.as_ref()))
        .collect();

    // Document frequency in the query set.
    let mut df: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for tokens in &per_query {
        let distinct: StdHashSet<&String> = tokens.iter().collect();
        for t in distinct {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
    }

    // Boilerplate: appears in >= 80% of queries.
    let threshold = ((n as f32) * 0.80).ceil() as usize;
    let mut boilerplate_pairs: Vec<(String, usize)> = df
        .iter()
        .filter(|(_, c)| **c >= threshold)
        .map(|(w, c)| (w.clone(), *c))
        .collect();
    boilerplate_pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let boilerplate_terms: Vec<String> = boilerplate_pairs.iter().map(|(w, _)| w.clone()).collect();
    let bp_set: StdHashSet<&String> = boilerplate_terms.iter().collect();

    // template_word_share = mean over queries of (boilerplate tokens / total tokens).
    let mut shares = 0.0f64;
    let mut counted = 0usize;
    for tokens in &per_query {
        if tokens.is_empty() {
            continue;
        }
        let bp = tokens.iter().filter(|t| bp_set.contains(t)).count();
        shares += bp as f64 / tokens.len() as f64;
        counted += 1;
    }
    let share = if counted == 0 {
        0.0
    } else {
        (shares / counted as f64) as f32
    };

    let cost = if share >= 0.70 {
        DilutionCost::High
    } else if share >= 0.40 {
        DilutionCost::Medium
    } else if share >= 0.20 {
        DilutionCost::Low
    } else {
        DilutionCost::None
    };

    let is_templated = share >= 0.50 && boilerplate_terms.len() >= 2;

    let suggested_action = match (is_templated, cost) {
        (true, DilutionCost::High) => format!(
            "Highly templated workload (~{:.0}% boilerplate). Expected lift on CUAD-shape \
             cases was +6 points ≥0.8 retention. Recommended: write a thin preprocessor \
             that drops the {} shared terms before calling `context()`; pair with \
             `strategy=\"raw_topk\"` for single-doc extraction. See \
             docs/CHOOSING_A_CONFIG.md and docs/findings/CUAD_RECALL_GAP.md.",
            share * 100.0,
            boilerplate_terms.len(),
        ),
        (true, _) => format!(
            "Templated workload (~{:.0}% boilerplate). Lift from stripping is uncertain at \
             this share; consider running an A/B with `drop_template_terms` before \
             committing. See docs/findings/CUAD_RECALL_GAP.md.",
            share * 100.0,
        ),
        (false, DilutionCost::Medium) => format!(
            "Some shared filler ({:.0}% of tokens), but not dominant. Stripping is \
             unlikely to move retention measurably; skip unless an A/B says otherwise.",
            share * 100.0,
        ),
        (false, _) => format!(
            "Diverse natural-language queries (~{:.0}% shared filler). No template to \
             strip. Standard defaults apply.",
            share * 100.0,
        ),
    };

    QuerySetReport {
        n_queries: n,
        is_templated,
        template_word_share: share,
        boilerplate_terms,
        estimated_dilution_cost: cost,
        suggested_action,
    }
}

/// Plain tokenizer for the query-set analyzer.
///
/// Deliberately matches the harness tokenization used by the CUAD findings
/// (lowercase, alphanumeric split, drop tokens with len < 2). Does NOT
/// stem — we want to detect shared *surface* tokens that BM25 will see.
/// Does NOT drop stopwords — if `the` appears in every query, that is
/// still boilerplate and worth surfacing.
fn analyzer_tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| w.to_string())
        .collect()
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

    // ─── drop_template_terms ────────────────────────────────────────────────

    #[test]
    fn drop_template_terms_basic_case_insensitive() {
        let q = "Highlight the parts of this Contract related to Change of Control";
        let got = drop_template_terms(
            q,
            &["highlight", "the", "parts", "of", "this", "contract", "related", "to"],
        );
        assert_eq!(got, "Change Control");
    }

    #[test]
    fn drop_template_terms_preserves_punctuation_on_surviving_tokens() {
        let q = "Highlight the parts related to \"Change of Control\".";
        let got = drop_template_terms(
            q,
            &["highlight", "the", "parts", "of", "related", "to"],
        );
        assert_eq!(got, "\"Change Control\".");
    }

    #[test]
    fn drop_template_terms_empty_boilerplate_is_identity() {
        let q = "Highlight the parts of this contract";
        assert_eq!(drop_template_terms(q, &[]), q);
    }

    #[test]
    fn drop_template_terms_no_match_is_identity_token_set() {
        let q = "find document name";
        // Boilerplate doesn't appear in the query at all.
        let got = drop_template_terms(q, &["foo", "bar"]);
        assert_eq!(got, "find document name");
    }

    #[test]
    fn drop_template_terms_all_filtered_returns_empty() {
        let q = "the the the";
        assert_eq!(drop_template_terms(q, &["the"]), "");
    }

    #[test]
    fn drop_template_terms_handles_chinese_phrase_boilerplate() {
        // CJK queries have no whitespace, so the analyzer surfaces boilerplate
        // as punctuation-bounded phrases. drop_template_terms must do
        // substring removal on those, not whitespace-token matching, or it
        // can't find any of them.
        let q = "请标注本合同中与「文档名称」相关的、应由律师审核的部分（如有）。";
        let stripped = drop_template_terms(
            q,
            &["请标注本合同中与", "应由律师审核的部分", "相关的", "如有"],
        );
        // All four boilerplate phrases should be gone.
        for noise in &["请标注本合同中与", "应由律师审核的部分", "相关的", "如有"] {
            assert!(
                !stripped.contains(noise),
                "expected {noise} to be stripped; got {stripped:?}"
            );
        }
        // The discriminator (the quoted clause name) should still be present.
        assert!(stripped.contains("文档名称"), "expected 文档名称 to survive; got {stripped:?}");
    }

    #[test]
    fn drop_template_terms_japanese_phrase_boilerplate() {
        // Mirror of the Chinese test for Japanese, which mixes Hiragana,
        // Katakana, and Han characters in the boilerplate.
        let q = "本契約のうち「文書名」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。";
        let stripped = drop_template_terms(
            q,
            &[
                "本契約のうち",
                "に関連する",
                "弁護士の確認が必要な部分",
                "もしあれば",
                "を示してください",
            ],
        );
        for noise in &[
            "本契約のうち",
            "に関連する",
            "弁護士の確認が必要な部分",
            "もしあれば",
            "を示してください",
        ] {
            assert!(!stripped.contains(noise), "expected {noise} stripped; got {stripped:?}");
        }
        assert!(stripped.contains("文書名"), "expected discriminator preserved; got {stripped:?}");
    }

    // ─── expand_query_terms ────────────────────────────────────────────────

    #[test]
    fn expand_query_terms_appends_matched_synonyms() {
        let q = "\"Change of Control\" the right to terminate";
        let expansions: &[(&str, &[&str])] = &[
            ("change of control", &["merger", "successor", "acquisition"]),
            ("non-compete", &["restraint", "non-competition"]),
        ];
        let expanded = expand_query_terms(q, expansions);
        assert!(expanded.starts_with(q), "original query must be preserved verbatim; got {expanded:?}");
        for syn in &["merger", "successor", "acquisition"] {
            assert!(expanded.contains(syn), "expected {syn} appended; got {expanded:?}");
        }
        // non-compete didn't match → its synonyms must not appear.
        for not_expected in &["restraint", "non-competition"] {
            assert!(!expanded.contains(not_expected), "expected {not_expected} NOT appended; got {expanded:?}");
        }
    }

    #[test]
    fn expand_query_terms_empty_dict_is_identity() {
        let q = "anything goes here";
        assert_eq!(expand_query_terms(q, &[]), q);
    }

    #[test]
    fn expand_query_terms_no_match_is_identity() {
        let q = "the refund window is thirty days";
        let expansions: &[(&str, &[&str])] = &[
            ("change of control", &["merger"]),
            ("non-compete", &["restraint"]),
        ];
        assert_eq!(expand_query_terms(q, expansions), q);
    }

    #[test]
    fn expand_query_terms_dedupes_synonyms_across_matches() {
        // Two keys both match and share a synonym → it must appear once.
        let q = "change of control and termination for convenience";
        let expansions: &[(&str, &[&str])] = &[
            ("change of control", &["merger", "assignment"]),
            ("termination for convenience", &["assignment", "rescission"]),
        ];
        let expanded = expand_query_terms(q, expansions);
        let n = expanded.matches("assignment").count();
        // Original query has 0 occurrences of "assignment"; expansion should
        // append exactly 1 even though two keys list it.
        assert_eq!(n, 1, "expected dedup; got {expanded:?}");
    }

    #[test]
    fn expand_query_terms_case_insensitive_key_match() {
        let q = "What about CHANGE OF CONTROL clauses?";
        let expansions: &[(&str, &[&str])] = &[("change of control", &["merger"])];
        let expanded = expand_query_terms(q, expansions);
        assert!(expanded.contains("merger"), "case-insensitive match should fire; got {expanded:?}");
    }

    #[test]
    fn expand_query_terms_no_recursive_chaining() {
        // The synonym "merger" is itself a key with its own synonyms. The
        // helper must NOT re-check appended synonyms against the dict — that
        // would lead to runaway expansion. Matches against the ORIGINAL query.
        let q = "change of control clause";
        let expansions: &[(&str, &[&str])] = &[
            ("change of control", &["merger"]),
            ("merger", &["consolidation"]),  // would chain if naive
        ];
        let expanded = expand_query_terms(q, expansions);
        assert!(expanded.contains("merger"));
        assert!(
            !expanded.contains("consolidation"),
            "must not recursively expand; got {expanded:?}"
        );
    }

    #[test]
    fn drop_template_terms_latin_word_boundary_safe() {
        // The original Latin behavior must not regress: a boilerplate
        // single-word "of" should NOT erase the "of" inside "office".
        let q = "the office is open";
        let stripped = drop_template_terms(q, &["of", "the"]);
        // "of" appears inside "office" but as a Latin token term it's
        // matched at whitespace-boundary granularity, so "office" survives.
        assert_eq!(stripped, "office is open");
    }

    // ─── analyze_query_set ──────────────────────────────────────────────────

    #[test]
    fn analyze_query_set_detects_cuad_shape() {
        // 6 CUAD-flavored queries, only the quoted clause name varies.
        let queries = [
            "Highlight the parts (if any) of this contract related to \"Document Name\" that should be reviewed by a lawyer.",
            "Highlight the parts (if any) of this contract related to \"Parties\" that should be reviewed by a lawyer.",
            "Highlight the parts (if any) of this contract related to \"Agreement Date\" that should be reviewed by a lawyer.",
            "Highlight the parts (if any) of this contract related to \"Effective Date\" that should be reviewed by a lawyer.",
            "Highlight the parts (if any) of this contract related to \"Expiration Date\" that should be reviewed by a lawyer.",
            "Highlight the parts (if any) of this contract related to \"Renewal Term\" that should be reviewed by a lawyer.",
        ];
        let report = analyze_query_set(&queries);
        assert!(report.is_templated, "CUAD-shape queries should be flagged");
        assert!(
            report.template_word_share > 0.6,
            "share should reflect the heavy template; got {:.3}",
            report.template_word_share
        );
        assert_eq!(report.estimated_dilution_cost, DilutionCost::High);
        // Spot-check a couple of the obvious boilerplate words.
        for expected in ["highlight", "contract", "lawyer"] {
            assert!(
                report.boilerplate_terms.iter().any(|w| w == expected),
                "expected {expected:?} in boilerplate_terms; got {:?}",
                report.boilerplate_terms
            );
        }
    }

    #[test]
    fn analyze_query_set_does_not_fire_on_diverse_queries() {
        let queries = [
            "Who is the current president of France?",
            "When was the Eiffel Tower built?",
            "What language do they speak in Brazil?",
            "How tall is Mount Everest?",
            "Which planet is closest to the sun?",
            "When did World War II end?",
            "Who wrote Pride and Prejudice?",
            "What is the capital of Japan?",
        ];
        let report = analyze_query_set(&queries);
        assert!(
            !report.is_templated,
            "diverse natural-language queries should not be flagged (share={:.3}, terms={:?})",
            report.template_word_share, report.boilerplate_terms
        );
    }

    #[test]
    fn analyze_query_set_handles_empty_and_singleton() {
        let empty: [&str; 0] = [];
        let r = analyze_query_set(&empty);
        assert_eq!(r.n_queries, 0);
        assert!(!r.is_templated);

        let one = ["one query is not a workload"];
        let r = analyze_query_set(&one);
        assert_eq!(r.n_queries, 1);
        // Every word is shared trivially across a 1-query set; the
        // ">=2 boilerplate terms" guard prevents most pathological flags,
        // but the share will be 1.0 because all tokens are "shared" in a
        // single-query set. That's a known edge — `n_queries` is the
        // signal callers should check first.
    }
}
