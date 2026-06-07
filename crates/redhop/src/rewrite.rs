//! Query-side rewrite primitives — the seam where workload-specific
//! preprocessing plugs into the pipeline, with the chain recorded in the
//! [`crate::context::ContextReport`] so every change is auditable.
//!
//! Two implementations ship in-box:
//!
//! - [`Stripper`] — compile a boilerplate list once, then remove the listed
//!   terms from any query token-by-token, using the document's own analyzer
//!   (so a single-letter "of" cannot accidentally erase the "of" inside
//!   "office", and stemming + stopword behavior matches what the BM25 index
//!   actually saw).
//! - [`Vocabulary`] — compile workload-specific equivalence classes once
//!   (clause-name → synonyms, error-code → variants, anything where the
//!   query and the indexed text use different surface forms for the same
//!   concept). Supports bidirectional matching (PTO ↔ paid time off,
//!   ROFR ↔ right of first refusal) — match either side, add the others.
//!
//! Both implement [`QueryRewrite`]. Anything implementing the trait
//! composes into a chain via [`crate::Document::context_with_rewrites`]
//! (or by hand for the low-level path).
//!
//! ## Why a trait instead of bare functions
//!
//! The earlier `drop_template_terms(query, &[&str])` /
//! `expand_query_terms(query, &[(&str, &[&str])])` shape (replaced in
//! 0.3.0) had three flaws this redesign fixes:
//!
//! - **Correctness.** Substring matching (`q_lower.contains(key_lower)`)
//!   fired on partial-word collisions: a vocabulary key `"ip"` matched
//!   `"recipient"`, a stripper key `"of"` would have erased the `"of"`
//!   inside `"office"` without the bolted-on word-boundary fallback. The
//!   trait-based primitives match at **post-analyzer token granularity**,
//!   inheriting the same tokenization the BM25 index used.
//! - **Performance.** A 2000-term vocabulary on a chatbot hot path
//!   re-lowercased and re-walked every entry per request. The compiled
//!   objects do the analyzer pass once at construction.
//! - **Observability.** Bare-function expansion silently appended terms;
//!   the user couldn't see *why* `merger successor` had been added to a
//!   query. Each [`QueryRewrite::apply`] returns a [`RewriteRecord`] that
//!   lands in [`crate::context::ContextReport::query_rewrites`]. The
//!   Decision Report tells you what each stage in the chain did.
//!
//! ## What is deliberately *not* in v1
//!
//! - **No auto-mining of synonyms from the corpus.** PRF / RM3 was
//!   falsified twice ([CUAD_PRF_NULL](../../docs/findings/CUAD_PRF_NULL.md)),
//!   index-side auto-drop was falsified
//!   ([SUB_IDF_AUTO_DROP_NULL](../../docs/findings/SUB_IDF_AUTO_DROP_NULL.md)).
//!   Vocabulary content is supplied, never inferred.
//! - **No vocabulary file format.** Accept `HashMap<String, Vec<String>>`
//!   (or the typed [`Vocabulary::new`] constructors); let the user load
//!   their CSV / Notion / DB in three lines.
//! - **No fuzzy / partial / soft matching.** Exact token-sequence match
//!   only. Fuzzy is a v2 with its own eval; bundling false-positive risk
//!   into the first cut would erode the very thing this redesign fixes.

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analyzer::{default_english, Analyzer};

/// A query-side rewrite step. Implementations transform a query string and
/// emit a [`RewriteRecord`] documenting what changed; the record lands in
/// [`crate::context::ContextReport::query_rewrites`] so the chain is
/// auditable.
pub trait QueryRewrite: Send + Sync {
    /// Short identifier for the report (e.g., `"strip"`, `"vocabulary"`).
    fn name(&self) -> &str;

    /// Apply the rewrite to `query`. Returns the new query string and a
    /// trace record. Implementations should be cheap when nothing matches
    /// (return the query verbatim, with an empty-deltas record).
    fn apply(&self, query: &str) -> RewriteResult;
}

/// Output of one [`QueryRewrite::apply`] or [`Vocabulary::enrich`] step.
#[derive(Debug, Clone)]
pub struct RewriteResult {
    /// The rewritten text — a query (for `apply`) or a chunk
    /// (for `enrich`). The same field serves both directions; the
    /// distinction is signaled by `record.stage` (`"strip"` /
    /// `"vocabulary"` / `"enrich"`).
    pub text: String,
    /// The audit record for this step.
    pub record: RewriteRecord,
}

/// One entry in the Decision Report's query-rewrite trace.
///
/// Surfaces what each stage in the rewrite chain did. The fields are
/// deliberately concrete (rather than a `kind` enum + payload) so the
/// shape JSON-serializes cleanly for binding consumers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewriteRecord {
    /// Stage identifier — matches the [`QueryRewrite::name`] of the
    /// rewrite that emitted this record.
    pub stage: String,
    /// Input query (the string handed to this stage).
    pub from: String,
    /// Output query (the string this stage produced — possibly identical
    /// to `from` if nothing matched).
    pub to: String,
    /// Surface forms this stage *matched* in the input. For
    /// [`Stripper`]: the boilerplate forms it found and removed. For
    /// [`Vocabulary`]: the equivalence-class members that triggered a
    /// match.
    pub matched: Vec<String>,
    /// Surface forms this stage *added* to the query. For [`Vocabulary`]:
    /// the synonyms appended. For [`Stripper`]: empty.
    pub added: Vec<String>,
    /// Surface forms this stage *removed* from the query. For
    /// [`Stripper`]: the boilerplate tokens dropped. For [`Vocabulary`]:
    /// empty.
    pub removed: Vec<String>,
}

// ─── Stripper ───────────────────────────────────────────────────────────────

/// Compiled boilerplate-removal rewrite. Token-level matching using the
/// supplied analyzer, so single-token stripper keys cannot accidentally
/// erase a substring inside a longer word (an "of" stripper does **not**
/// erase the "of" inside "office").
///
/// Typical use is the templated-workload workflow:
///
/// ```no_run
/// # use redhop::{Document, rewrite::{Stripper, Vocabulary, QueryRewrite}};
/// # fn main() -> redhop::Result<()> {
/// let stripper = Stripper::new(&[
///     "highlight", "the", "parts", "if", "any", "of", "this",
///     "contract", "related", "to", "that", "should", "be",
///     "reviewed", "by", "a", "lawyer", "details",
/// ]);
/// let vocabulary = Vocabulary::new(&[
///     ("change of control", &["merger", "successor", "acquisition"][..]),
///     ("non-compete", &["restraint", "non-competition"]),
/// ]);
///
/// let mut doc = Document::from_text("contract.pdf", "…")?;
/// let ctx = doc.context_with_rewrites(
///     "Highlight the parts (if any) of this contract related to \"Change of Control\" …",
///     &[&stripper, &vocabulary],
/// )?;
/// for record in &ctx.report.query_rewrites {
///     println!("{}: {:?} → added {:?}", record.stage, record.matched, record.added);
/// }
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct Stripper {
    /// Original surface forms (for the audit trail).
    surface_forms: Vec<String>,
    /// Each surface form, run through the analyzer at construction time.
    /// `tokens[i]` is the analyzer's token sequence for `surface_forms[i]`.
    /// When the analyzer drops `surface_forms[i]` entirely (a stopword like
    /// `"the"`, or pure punctuation), `tokens[i]` is empty and the matcher
    /// falls back to a normalized surface-form comparison so the user's
    /// boilerplate still strips.
    tokens: Vec<Vec<String>>,
    /// Surface-form fallback key for each boilerplate. Computed by
    /// lowercasing and dropping non-alphanumeric chars (so `"the"`, `"of"`,
    /// `"to"` get a non-empty key that matches the chunk's normalized form
    /// at word-boundary granularity — preserving the word-boundary safety
    /// that the analyzer would have given us if it didn't drop stopwords).
    surface_keys: Vec<String>,
    analyzer: Arc<dyn Analyzer>,
}

impl Stripper {
    /// Compile a boilerplate list using the default English analyzer.
    /// For non-English workloads, use
    /// [`Stripper::with_analyzer`] to attach the analyzer the corresponding
    /// `Document` was built with — keeping the rewrite and the index in
    /// lockstep on what counts as "the same term."
    pub fn new<S: AsRef<str>>(boilerplate: &[S]) -> Self {
        Self::with_analyzer(boilerplate, default_english())
    }

    /// Compile a boilerplate list using the supplied analyzer.
    pub fn with_analyzer<S: AsRef<str>>(boilerplate: &[S], analyzer: Arc<dyn Analyzer>) -> Self {
        let surface_forms: Vec<String> =
            boilerplate.iter().map(|s| s.as_ref().to_string()).collect();
        let tokens: Vec<Vec<String>> = surface_forms.iter().map(|s| analyzer.tokens(s)).collect();
        let surface_keys: Vec<String> = surface_forms.iter().map(|s| normalize_word(s)).collect();
        Self {
            surface_forms,
            tokens,
            surface_keys,
            analyzer,
        }
    }

    /// Number of boilerplate forms registered (informational).
    pub fn len(&self) -> usize {
        self.surface_forms.len()
    }

    /// True iff no boilerplate is registered.
    pub fn is_empty(&self) -> bool {
        self.surface_forms.is_empty()
    }
}

impl QueryRewrite for Stripper {
    fn name(&self) -> &str {
        "strip"
    }

    fn apply(&self, query: &str) -> RewriteResult {
        if self.surface_forms.is_empty() {
            return RewriteResult {
                text: query.to_string(),
                record: RewriteRecord {
                    stage: "strip".to_string(),
                    from: query.to_string(),
                    to: query.to_string(),
                    ..Default::default()
                },
            };
        }

        // Two-tier match. Tier 1: analyzer-stem set, for boilerplate the
        // analyzer keeps (covers stemming so a "highlight" stripper matches
        // a "highlighted" chunk). Tier 2: normalized surface-form set, for
        // boilerplate the analyzer would have dropped — typically English
        // stopwords like "the", "of", "to" — so a user list including
        // those still strips the wrapper.
        let stem_stop: HashSet<&String> = self.tokens.iter().flatten().collect();
        let surface_stop: HashSet<&String> = self
            .surface_keys
            .iter()
            .zip(self.tokens.iter())
            .filter_map(|(key, toks)| {
                if toks.is_empty() && !key.is_empty() {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();

        // Walk the original query in whitespace chunks so case and
        // punctuation on surviving tokens are preserved verbatim.
        let mut kept: Vec<&str> = Vec::new();
        let mut removed_set: HashSet<String> = HashSet::new();
        for chunk in query.split_whitespace() {
            let chunk_tokens = self.analyzer.tokens(chunk);
            let chunk_surface = normalize_word(chunk);

            let stem_match = chunk_tokens.iter().any(|t| stem_stop.contains(t));
            let surface_match = !chunk_surface.is_empty() && surface_stop.contains(&chunk_surface);

            if stem_match || surface_match {
                // Audit which surface form(s) caused the drop.
                for (form, form_tokens) in self.surface_forms.iter().zip(self.tokens.iter()) {
                    if !form_tokens.is_empty()
                        && form_tokens.iter().any(|t| chunk_tokens.contains(t))
                    {
                        removed_set.insert(form.clone());
                    }
                }
                for (form, key) in self.surface_forms.iter().zip(self.surface_keys.iter()) {
                    if !key.is_empty() && key == &chunk_surface {
                        removed_set.insert(form.clone());
                    }
                }
            } else {
                kept.push(chunk);
            }
        }

        let new_text = kept.join(" ");
        let mut removed: Vec<String> = removed_set.into_iter().collect();
        removed.sort();

        RewriteResult {
            text: new_text.clone(),
            record: RewriteRecord {
                stage: "strip".to_string(),
                from: query.to_string(),
                to: new_text,
                matched: removed.clone(),
                added: vec![],
                removed,
            },
        }
    }
}

/// Lowercase + drop non-alphanumeric chars, mirroring what
/// [`Stripper`]'s tier-2 fallback uses for both boilerplate keys and
/// query chunks. Keeping the normalization in one place makes the
/// equality check trivial and the word-boundary guarantee easy to read.
fn normalize_word(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

// ─── Vocabulary ───────────────────────────────────────────────────────────────

/// One equivalence class inside a [`Vocabulary`].
///
/// Internal — exposed only through [`Vocabulary::new`] /
/// [`Vocabulary::bidirectional`].
#[derive(Clone)]
struct VocabularyClass {
    /// Surface forms in their original (case, punctuation) shape, used
    /// for the audit trail.
    members: Vec<String>,
    /// Each `members[i]` tokenized through the analyzer.
    member_tokens: Vec<Vec<String>>,
}

/// Compiled workload-curated equivalence classes. Each entry maps a
/// "key" surface form (or, in bidirectional mode, any member) to a list
/// of surface forms that should be appended to a query whenever a
/// member is found in it.
///
/// **Token-level matching.** All forms (keys, synonyms, query) are
/// tokenized through the same analyzer at compile time. A vocabulary key
/// `"ip"` matches the token `ip`, never the substring inside `recipient`;
/// a multi-token key `"change of control"` matches the analyzer's token
/// sequence for the same phrase in the query (so stemming and stopword
/// handling agree with the BM25 index).
///
/// **Bidirectional.** A symmetric vocabulary entry like
/// `("PTO", &["paid time off", "vacation"])` in
/// [`Vocabulary::bidirectional`] mode matches whichever surface form the
/// user typed and appends the other two. Non-bidirectional mode treats
/// the first form as the only trigger and the rest as the additions
/// (the asymmetric "key → synonyms" shape).
#[derive(Clone)]
pub struct Vocabulary {
    classes: Vec<VocabularyClass>,
    analyzer: Arc<dyn Analyzer>,
    bidirectional: bool,
}

impl Vocabulary {
    /// Compile a vocabulary using the default English analyzer in
    /// asymmetric mode (the first form of each entry is the only
    /// trigger; the rest are appended on match).
    pub fn new<K: AsRef<str>, S: AsRef<str>>(entries: &[(K, &[S])]) -> Self {
        Self::with_options(entries, false, default_english())
    }

    /// Compile a vocabulary in **bidirectional** mode: any member of the
    /// equivalence class is a trigger; matching any one appends the
    /// others.
    pub fn bidirectional<K: AsRef<str>, S: AsRef<str>>(entries: &[(K, &[S])]) -> Self {
        Self::with_options(entries, true, default_english())
    }

    /// Compile a vocabulary with caller-supplied options.
    pub fn with_options<K: AsRef<str>, S: AsRef<str>>(
        entries: &[(K, &[S])],
        bidirectional: bool,
        analyzer: Arc<dyn Analyzer>,
    ) -> Self {
        let classes = entries
            .iter()
            .map(|(k, syns)| {
                let mut members: Vec<String> = vec![k.as_ref().to_string()];
                members.extend(syns.iter().map(|s| s.as_ref().to_string()));
                let member_tokens: Vec<Vec<String>> =
                    members.iter().map(|m| analyzer.tokens(m)).collect();
                VocabularyClass {
                    members,
                    member_tokens,
                }
            })
            .collect();
        Self {
            classes,
            analyzer,
            bidirectional,
        }
    }

    /// Number of equivalence classes (informational).
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// True iff no classes are registered.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Override the analyzer after construction. Re-tokenizes every
    /// member; cheap on a 10-class vocabulary, more expensive on a
    /// 2000-class one — prefer passing the analyzer to
    /// [`Vocabulary::with_options`] in hot paths.
    pub fn with_analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        for class in &mut self.classes {
            class.member_tokens = class.members.iter().map(|m| analyzer.tokens(m)).collect();
        }
        self.analyzer = analyzer;
        self
    }
}

impl Vocabulary {
    /// Shared matching+appending logic used by both query-side
    /// [`QueryRewrite::apply`] (stage `"vocabulary"`) and chunk-side
    /// [`Vocabulary::enrich`] (stage `"enrich"`). Same algorithm, same
    /// audit-record shape — only the `stage` name differs so consumers
    /// can tell at-a-glance which side of the pipeline a record came
    /// from.
    fn run(&self, text: &str, stage: &'static str) -> RewriteResult {
        if self.classes.is_empty() {
            return RewriteResult {
                text: text.to_string(),
                record: RewriteRecord {
                    stage: stage.to_string(),
                    from: text.to_string(),
                    to: text.to_string(),
                    ..Default::default()
                },
            };
        }

        let text_tokens = self.analyzer.tokens(text);

        let mut matched: Vec<String> = Vec::new();
        let mut added: Vec<String> = Vec::new();
        let mut added_seen: HashSet<String> = HashSet::new();

        for class in &self.classes {
            // Which members are eligible to match? In asymmetric mode,
            // only the first (the "key"). In bidirectional mode, all of
            // them.
            let candidate_count = if self.bidirectional {
                class.member_tokens.len()
            } else {
                1
            };

            // Track which member indices fired so we know which OTHER
            // members to add as synonyms (asymmetric mode: all `i > 0`
            // when index 0 fires; bidirectional: all `j != i` for each
            // firing `i`).
            let mut hit_indices: Vec<usize> = Vec::new();
            for i in 0..candidate_count {
                let key_tokens = &class.member_tokens[i];
                if key_tokens.is_empty() {
                    continue;
                }
                if token_subsequence_match(&text_tokens, key_tokens) {
                    hit_indices.push(i);
                }
            }

            if hit_indices.is_empty() {
                continue;
            }

            // Record matches.
            for &i in &hit_indices {
                matched.push(class.members[i].clone());
            }

            // Append additions — every NON-hit member, deduped against
            // earlier additions.
            let hit_set: HashSet<usize> = hit_indices.iter().copied().collect();
            for (i, surface) in class.members.iter().enumerate() {
                if hit_set.contains(&i) {
                    continue;
                }
                if added_seen.insert(surface.to_lowercase()) {
                    added.push(surface.clone());
                }
            }
        }

        let new_text = if added.is_empty() {
            text.to_string()
        } else {
            format!("{} {}", text, added.join(" "))
        };

        RewriteResult {
            text: new_text.clone(),
            record: RewriteRecord {
                stage: stage.to_string(),
                from: text.to_string(),
                to: new_text,
                matched,
                added,
                removed: vec![],
            },
        }
    }

    /// Chunk-side enrichment — the symmetric to [`QueryRewrite::apply`].
    /// Same compiled vocabulary, same matching, but applied at **ingest
    /// time** to the chunk text so opaque coded units (column names,
    /// error codes, API symbols, defined terms) become matchable for
    /// natural-language queries that don't share surface forms with
    /// them.
    ///
    /// **When this earns its keep.** Enrich's value scales with how
    /// short and opaque the retrieval unit is and whether a decoding
    /// dictionary exists:
    ///
    /// ```text
    /// value ∝ (chunk shortness) × (chunk opacity) × (dictionary exists)
    /// ```
    ///
    /// Schema columns (`emp_compensation` → `monthly base salary`), API
    /// symbols, error codes (`ERR_4012` → `payment declined`), defined
    /// terms in contracts, clinical abbreviations (`MI` → `myocardial
    /// infarction`) — all extreme cases of the rule. On long descriptive
    /// prose the operation is redundant (matching already works) and
    /// can dilute if the same boilerplate gets bolted onto every
    /// chunk — same low-IDF failure mode as
    /// [`CUAD_PRF_NULL`](../../docs/findings/CUAD_PRF_NULL.md). See
    /// [`VOCABULARY_ENRICH`](../../docs/findings/VOCABULARY_ENRICH.md)
    /// for the regime rule, use-case ranking, and failure modes.
    ///
    /// **expand vs enrich, the axis.** Query-side
    /// [`QueryRewrite::apply`] patches gaps you *anticipated* (you list
    /// the synonyms). Chunk-side `enrich` raises the content's semantic
    /// floor for queries you *can't* anticipate. The two are different
    /// jobs; pick by whether you can enumerate user phrasings or only
    /// describe what your content is.
    ///
    /// Typical use is at ingest time, before
    /// [`crate::Document::from_chunks`]:
    ///
    /// ```no_run
    /// # use redhop::{Document, Vocabulary};
    /// # fn main() -> redhop::Result<()> {
    /// let vocab = Vocabulary::new(&[
    ///     ("usrSvc", &["user service", "account creation", "signup"][..]),
    ///     ("calcAmt", &["calculate amount", "billing total"]),
    /// ]);
    /// let raw_chunks = vec![
    ///     "fn usrSvc(req: Req) -> Resp { … }".to_string(),
    ///     "fn calcAmt(items: &[Item]) -> Cents { … }".to_string(),
    /// ];
    /// let enriched: Vec<String> = raw_chunks
    ///     .into_iter()
    ///     .map(|c| vocab.enrich(&c).text)
    ///     .collect();
    /// let mut doc = Document::from_chunks(enriched, None)?;
    /// // Now "how do we create accounts?" lights up usrSvc's chunk.
    /// # Ok(()) }
    /// ```
    pub fn enrich(&self, chunk: &str) -> RewriteResult {
        self.run(chunk, "enrich")
    }
}

impl QueryRewrite for Vocabulary {
    fn name(&self) -> &str {
        "vocabulary"
    }

    fn apply(&self, query: &str) -> RewriteResult {
        self.run(query, "vocabulary")
    }
}

/// True iff `needle` appears as a contiguous subsequence of `haystack`.
fn token_subsequence_match(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    for start in 0..=haystack.len() - needle.len() {
        if &haystack[start..start + needle.len()] == needle {
            return true;
        }
    }
    false
}

// ─── chain runner ──────────────────────────────────────────────────────────

/// Run a chain of rewrites in order, returning the final query and the
/// full sequence of records (one per stage). Each stage sees the
/// **output** of the previous stage — `apply` composes left-to-right.
///
/// Used internally by `Document::context_with_rewrites` and the
/// low-level `build_context_with_rewrites`.
pub fn apply_chain(query: &str, rewrites: &[&dyn QueryRewrite]) -> (String, Vec<RewriteRecord>) {
    let mut current = query.to_string();
    let mut trail: Vec<RewriteRecord> = Vec::with_capacity(rewrites.len());
    for rw in rewrites {
        let result = rw.apply(&current);
        trail.push(result.record);
        current = result.text;
    }
    (current, trail)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Stripper ──────────────────────────────────────────────────────────

    #[test]
    fn stripper_drops_listed_tokens_preserving_punctuation() {
        // The stripper strips ANY occurrence of a listed token, including
        // tokens inside quoted phrases. If the user wants `"Change of
        // Control"` preserved verbatim, the boilerplate list must NOT
        // contain `"of"`. Pin the actual contract.
        let s = Stripper::new(&[
            "highlight",
            "the",
            "parts",
            "of",
            "this",
            "contract",
            "related",
            "to",
        ]);
        let r = s.apply("Highlight the parts of this contract related to \"Change of Control\".");
        assert_eq!(r.text, "\"Change Control\".");
        assert_eq!(r.record.stage, "strip");
        assert!(r.record.matched.iter().any(|m| m == "highlight"));
        assert!(r.record.removed.iter().any(|m| m == "contract"));
    }

    #[test]
    fn stripper_preserves_phrase_when_boilerplate_excludes_internal_words() {
        // The flip-side guarantee: when the boilerplate omits a token
        // ("of"), that token survives the strip — including inside a
        // quoted phrase like "Change of Control". Lets users compose a
        // stripper carefully when phrase preservation matters.
        let s = Stripper::new(&[
            "highlight",
            "the",
            "parts",
            "this",
            "contract",
            "related",
            "to",
        ]); // note: no "of"
        let r = s.apply("Highlight the parts of this contract related to \"Change of Control\".");
        assert_eq!(r.text, "of \"Change of Control\".");
    }

    #[test]
    fn stripper_word_boundary_safe_for_short_tokens() {
        // The Latin-script regression test: a boilerplate "of" must NOT
        // erase the "of" inside "office" because the analyzer tokenizes
        // "office" as ["offic"] (post-Porter2), not ["of", "fice"].
        let s = Stripper::new(&["of", "the"]);
        let r = s.apply("the office is open");
        assert_eq!(r.text, "office is open");
    }

    #[test]
    fn stripper_empty_boilerplate_is_identity() {
        let s = Stripper::new::<&str>(&[]);
        let r = s.apply("the office is open");
        assert_eq!(r.text, "the office is open");
        assert!(r.record.removed.is_empty());
    }

    #[test]
    fn stripper_stem_aware_matching() {
        // The analyzer Porter2-stems both sides; a stripper "highlight"
        // matches a query "highlighted" (both stem to "highlight").
        let s = Stripper::new(&["highlight"]);
        let r = s.apply("Highlighted parts of this contract");
        assert!(
            !r.text.to_lowercase().contains("highlighted"),
            "stem match should drop highlighted; got {:?}",
            r.text
        );
    }

    // ─── Vocabulary ──────────────────────────────────────────────────────────

    #[test]
    fn vocabulary_basic_asymmetric_append() {
        let g = Vocabulary::new(&[
            (
                "change of control",
                &["merger", "successor", "acquisition"][..],
            ),
            ("non-compete", &["restraint"]),
        ]);
        let r = g.apply("\"Change of Control\" the right to terminate");
        for syn in &["merger", "successor", "acquisition"] {
            assert!(
                r.text.contains(syn),
                "expected {syn} appended; got {:?}",
                r.text
            );
        }
        assert!(
            !r.text.contains("restraint"),
            "non-compete didn't match — its synonyms must not appear"
        );
        assert!(r.record.matched.iter().any(|m| m == "change of control"));
        assert_eq!(r.record.added, vec!["merger", "successor", "acquisition"]);
    }

    #[test]
    fn vocabulary_short_acronym_does_not_substring_fire() {
        // The exact correctness flaw the redesign fixes: a vocabulary key
        // "ip" must NOT match the word "recipient" (which the old
        // substring-based expand_query_terms DID match).
        let g = Vocabulary::new(&[("ip", &["intellectual property"][..])]);
        let r = g.apply("send it to the recipient by Friday");
        assert_eq!(
            r.text, "send it to the recipient by Friday",
            "ip must not match recipient via substring"
        );
        assert!(r.record.matched.is_empty());
        assert!(r.record.added.is_empty());
    }

    #[test]
    fn vocabulary_short_acronym_matches_when_actually_present_as_token() {
        let g = Vocabulary::new(&[("ip", &["intellectual property"][..])]);
        let r = g.apply("the IP belongs to the licensee");
        assert!(r.text.contains("intellectual property"));
        assert!(r.record.matched.iter().any(|m| m == "ip"));
    }

    #[test]
    fn vocabulary_bidirectional_appends_other_members() {
        let g = Vocabulary::bidirectional(&[("pto", &["paid time off", "vacation"][..])]);

        let r1 = g.apply("How do I file for PTO?");
        // Matched "pto" → both other members appended.
        assert!(r1.text.contains("paid time off"));
        assert!(r1.text.contains("vacation"));
        assert!(r1.record.matched.iter().any(|m| m == "pto"));

        let r2 = g.apply("How do I file for vacation?");
        assert!(r2.text.contains("pto"));
        assert!(r2.text.contains("paid time off"));
        assert!(r2.record.matched.iter().any(|m| m == "vacation"));
    }

    #[test]
    fn vocabulary_non_bidirectional_does_not_fire_from_synonym_side() {
        // Asymmetric (default): only the key (first member) triggers.
        let g = Vocabulary::new(&[("pto", &["paid time off", "vacation"][..])]);
        let r = g.apply("How do I file for vacation?");
        assert!(
            !r.text.contains("pto"),
            "non-bidirectional: matching a synonym should NOT fire; got {:?}",
            r.text
        );
        assert!(r.record.matched.is_empty());
    }

    #[test]
    fn vocabulary_no_recursive_chaining() {
        // A class fires when its key matches the ORIGINAL query, not the
        // growing rewritten string. So if synonyms include something
        // that happens to be a key in a different class, the second
        // class doesn't fire.
        let g = Vocabulary::new(&[
            ("change of control", &["merger"][..]),
            ("merger", &["consolidation"]),
        ]);
        let r = g.apply("change of control clause");
        assert!(r.text.contains("merger"));
        assert!(
            !r.text.contains("consolidation"),
            "must not recursively expand; got {:?}",
            r.text
        );
    }

    #[test]
    fn vocabulary_dedupes_synonyms_across_matches() {
        // Two classes that both fire and share a synonym should append
        // it exactly once.
        let g = Vocabulary::new(&[
            ("change of control", &["merger", "assignment"][..]),
            ("termination for convenience", &["assignment", "rescission"]),
        ]);
        let r = g.apply("change of control and termination for convenience");
        assert_eq!(
            r.text.matches("assignment").count(),
            1,
            "expected dedup; got {:?}",
            r.text
        );
    }

    #[test]
    fn vocabulary_multi_token_key_matches_in_order() {
        let g = Vocabulary::new(&[("change of control", &["merger"][..])]);
        let r = g.apply("the change of ownership and change of control clauses");
        assert!(r.text.contains("merger"));
        assert!(r.record.matched.iter().any(|m| m == "change of control"));
    }

    #[test]
    fn vocabulary_empty_is_identity() {
        let g = Vocabulary::new::<&str, &str>(&[]);
        let r = g.apply("any query at all");
        assert_eq!(r.text, "any query at all");
        assert!(r.record.matched.is_empty());
        assert!(r.record.added.is_empty());
    }

    // ─── Vocabulary.enrich (chunk-side mirror of apply) ────────────────────

    #[test]
    fn enrich_appends_synonyms_to_chunk_text_when_key_matches() {
        let vocab = Vocabulary::new(&[(
            "change of control",
            &["merger", "successor", "acquisition"][..],
        )]);
        let chunk = "Change of Control means the consummation of a transaction.";
        let r = vocab.enrich(chunk);
        assert!(r.text.starts_with(chunk));
        for syn in ["merger", "successor", "acquisition"] {
            assert!(
                r.text.contains(syn),
                "expected {syn} appended; got {}",
                r.text
            );
        }
        assert_eq!(r.record.stage, "enrich");
        assert_eq!(r.record.from, chunk);
        assert_eq!(r.record.to, r.text);
        assert!(r
            .record
            .matched
            .iter()
            .any(|m| m.eq_ignore_ascii_case("change of control")));
        assert!(r.record.added.contains(&"merger".to_string()));
    }

    #[test]
    fn enrich_is_identity_when_no_key_matches() {
        let vocab = Vocabulary::new(&[("pto", &["paid time off"][..])]);
        let chunk = "Distinct text with no vocabulary triggers.";
        let r = vocab.enrich(chunk);
        assert_eq!(r.text, chunk);
        assert!(r.record.matched.is_empty());
        assert!(r.record.added.is_empty());
        assert_eq!(r.record.stage, "enrich");
    }

    #[test]
    fn enrich_short_acronym_token_level_safe() {
        // `"ip"` must NOT enrich `"recipient"` chunks via substring fire.
        let vocab = Vocabulary::new(&[("ip", &["intellectual property"][..])]);
        let r = vocab.enrich("the recipient shall execute this agreement");
        assert!(!r.text.contains("intellectual property"));
        assert!(r.record.matched.is_empty());
    }

    #[test]
    fn enrich_short_acronym_matches_real_token() {
        let vocab = Vocabulary::new(&[("ip", &["intellectual property"][..])]);
        let r = vocab.enrich("Section 8: IP assignment");
        assert!(r.text.contains("intellectual property"));
    }

    #[test]
    fn enrich_empty_vocab_is_identity_with_enrich_stage() {
        let vocab = Vocabulary::new::<&str, &str>(&[]);
        let chunk = "anything at all";
        let r = vocab.enrich(chunk);
        assert_eq!(r.text, chunk);
        assert_eq!(r.record.stage, "enrich");
        assert!(r.record.matched.is_empty());
        assert!(r.record.added.is_empty());
    }

    #[test]
    fn enrich_and_apply_share_logic_but_record_different_stage() {
        // Same vocabulary + same text should produce the same output
        // string from apply() and enrich() — only the `stage` field of
        // the audit record differs. This is the contract the binding
        // tests will lean on.
        let vocab = Vocabulary::new(&[("merger", &["acquisition"][..])]);
        let text = "a merger clause";
        let a = vocab.apply(text);
        let e = vocab.enrich(text);
        assert_eq!(a.text, e.text);
        assert_eq!(a.record.from, e.record.from);
        assert_eq!(a.record.to, e.record.to);
        assert_eq!(a.record.matched, e.record.matched);
        assert_eq!(a.record.added, e.record.added);
        assert_eq!(a.record.stage, "vocabulary");
        assert_eq!(e.record.stage, "enrich");
    }

    // ─── chain runner ──────────────────────────────────────────────────────

    #[test]
    fn chain_composes_strip_then_vocabulary() {
        let stripper = Stripper::new(&["highlight", "the", "parts", "of", "this", "contract"]);
        let vocabulary = Vocabulary::new(&[("change of control", &["merger"][..])]);
        let (final_q, trail) = apply_chain(
            "Highlight the parts of this contract for Change of Control terms",
            &[&stripper, &vocabulary],
        );
        // Strip removed the wrapper words …
        assert!(!final_q.to_lowercase().contains("highlight"));
        // … then vocabulary appended `merger`.
        assert!(final_q.contains("merger"));
        // Trail has both records, in order.
        assert_eq!(trail.len(), 2);
        assert_eq!(trail[0].stage, "strip");
        assert_eq!(trail[1].stage, "vocabulary");
        // Each record's `to` matches the next record's `from`.
        assert_eq!(trail[0].to, trail[1].from);
        // The final stage's `to` is the final query.
        assert_eq!(trail[1].to, final_q);
    }
}
