# Design: `report.diagnosis` — a self-service retrieval diagnostic on the Decision Report

**Status**: proposed, approved for implementation. Target release: 0.3.4
(additive, non-breaking per [API_STABILITY](../API_STABILITY.md): new
fields on `ContextReport` are backward-compatible extensions, and the
rendered string is explicitly unstable).

**Audience**: this spec is written to be picked up cold by an
implementing agent. Every integration point cites a file and line on
`main` as of 2026-06-10 (commit `425e82e`). Verify line numbers before
editing; the structures are the contract, the line numbers are hints.

---

## 1. Problem

RedHop's docs catalog exactly three ways users fail at query time
([CHOOSING_A_CONFIG.md → "Query writing"](../CHOOSING_A_CONFIG.md)):

1. **Vocabulary mismatch** — the query says "cancel and get my money
   back", the contract says "refund" and "termination for convenience".
   Result: empty or near-empty context.
2. **One-word polysemy** — `'vendor'` retrieves §C.10 Vendor Risk
   Management, not §7.2 Limitation of Liability. The query
   under-determines which section the user means.
3. **Templated boilerplate** — a fixed 24-word wrapper dilutes the 5
   signal words because BM25 weights by corpus IDF, not query-set
   frequency (measured on CUAD,
   [CUAD_RECALL_GAP](../findings/CUAD_RECALL_GAP.md)).

In every case, the library's *runtime* answer today is a single boolean:
`low_confidence_retrieval` (`crates/redhop/src/context/mod.rs:968-971`,
fires when the context is empty or every selected chunk is at/below the
grounding bar). The *explanation* and the *fix* live on a docs page the
user has to already know about. This is the most common
"it returned nothing / garbage, why?" support moment, and the engine
already holds every signal needed to answer it at query time.

This feature closes that gap: the Decision Report gains a `diagnosis`
section reporting **facts** about how the query interacted with the
corpus and candidates, plus a small set of **bounded hints** that fire
on the three documented failure shapes and cite the measured finding
that justifies each one.

Positioning fit: "the context layer that shows its work" extends
naturally to "and tells you what it observed when the work went badly".

## 2. Goals

- A user whose `ctx.text()` came back empty or weak can read
  `ctx.report.diagnosis` (or the rendered report) and see *which query
  terms matched nothing in their corpus*, without leaving their REPL.
- The three documented failure shapes each produce a recognizable,
  findings-cited hint.
- Identical surface in Python, Node, and Rust.
- Zero new configuration knobs. Zero behavior change to retrieval or
  assembly. Diagnosis is observability only.

## 3. Non-goals (read these first)

These follow from the project's two hard constraints (bounded
architecture, measure-don't-overclaim):

- **No automatic retry, fallback, or query mutation.** Diagnosis never
  changes what `context()` returns. Acting on the diagnosis is the
  caller's decision. (Auto-retry is planner territory.)
- **No LLM-based diagnosis.** Facts come from set arithmetic on terms
  and scores the engine already computed.
- **No promised lifts in hint text.** A hint may say a mechanism was
  *measured* on a named finding. It must never say "this will improve
  your results."
- **No per-term BM25 score attribution.** Per-term contributions are
  collapsed inside Tantivy before results return
  (`crates/redhop/src/core/types.rs`, `Score { value, method }`).
  Surfacing them would need Tantivy's explain API. Out of scope; noted
  as a possible future extension in §12.
- **No threshold config knobs.** Hint thresholds are module constants
  (§7). They become configurable only if a measurement shows users need
  to tune them.
- **No query-set-level analysis at runtime.** `analyze_query_set`
  (`crates/redhop/src/analyzer.rs:552-656`) stays a batch tool; its 80%
  document-frequency threshold is meaningless for n=1. The per-query
  analogue here is corpus-DF-based term discrimination (§6, fact F4).

## 4. Design decisions, with rationale

**D1 — Facts first, hints bounded.** The struct is organized as
observed facts (term match counts, score spread, candidate counts) plus
a `hints: Vec<DiagnosisHint>` where every hint carries an `evidence`
citation to a findings or docs file. Facts cannot overclaim. Hints are
the only interpretive layer, and the registry (§7) is the complete,
closed list. Adding a hint requires adding a registry entry with a
citation, by construction.

**D2 — Always computed.** Diagnosis runs on every `context()` /
`build_context()` call. Cost is a few set operations over data already
in memory (query terms ≤ dozens, candidates ≤ `candidate_k`). No flag
to discover, no silent-until-broken mode. The corpus vocabulary map
(D4) is built lazily once per `Document`.

**D3 — Same analyzer as everything else.** Term tokenization MUST go
through `cfg.analyzer` via the existing `terms()` helper
(`crates/redhop/src/context/mod.rs:1292`,
`fn terms(text: &str, analyzer: &dyn Analyzer) -> HashSet<String>`).
The [ANALYZER_PLUGIN design doc](ANALYZER_PLUGIN.md) records the
0.1.3-0.1.4 class of silent-miss bugs caused by two layers disagreeing
on what "the same term" means. Diagnosis must not reopen that class: if
the corpus was indexed with the German analyzer, a query for "Buch"
against a corpus containing "Bücher" must NOT report a zero-match.

**D4 — Corpus stats from a lazy vocabulary map on `Document`, not from
Tantivy.** Two candidate sources for "does term X exist in the corpus,
and in how many chunks":

- (a) Tantivy `doc_freq` through `Bm25Retriever`
  (`crates/redhop/src/retrieval/bm25.rs:80-91`, index behind
  `Arc<RwLock<Inner>>`, no public term-stats API today).
- (b) A `HashMap<String, u32>` (analyzed term → number of chunks
  containing it) built in one pass over `Document.chunks`
  (`crates/redhop/src/document/mod.rs:193-195`) using `cfg.analyzer`.

Choose **(b)**: it is engine-agnostic (works identically under
`RetrievalMode::Lexical`, `Hybrid`, and `Dense`), needs no new public
surface on the retriever, cannot drift from the grounding scorer's
tokenization (D3), and the corpus is in-memory by design ("your corpus
fits in memory" is a published fit criterion). Build lazily on first
`context()` call and cache on the `Document` (`context` already takes
`&mut self`, `document/mod.rs:624-626`). Memory cost is one u32 per
distinct term. If `Document` has any chunk-mutating method (verify;
none expected), the cache must be invalidated there.

**D5 — Two-layer computation, post-hoc enrichment.** `build_context`
(`context/mod.rs:701-705`) only sees the query and the retrieved
candidates, not the corpus, and direct `build_context` callers (the
documented multi-source pattern) have no corpus at all. So:

- Layer 1, inside `build_context` / `build_context_expanded`: compute
  candidate-level facts and set `corpus_stats_available = false`.
  Evaluate only the hints that don't need corpus stats.
- Layer 2, inside `Document::context_inner`
  (`document/mod.rs:682-728`): after `build_context` returns, enrich
  the diagnosis with corpus stats and re-evaluate the full hint
  registry. Follow the existing post-hoc mutation precedent,
  `attach_rewrite_trail` (`context/mod.rs:674`).

**D6 — Diagnosis describes the query that was actually retrieved
with.** Under `context_with_rewrites` (`document/mod.rs:669-675`) the
rewritten query is the one sent to retrieval, so `diagnosis.query_terms`
reflect the rewritten query. The existing `query_rewrites` trail on the
report already records original → rewritten, so the user can see both.

**D7 — Bindings follow each language's existing report pattern.**
Python exposes nested report data as dicts (`economics` getter pattern,
`python/src/lib.rs:429-432`), so `diagnosis` is a dict with a list of
dicts for hints. Node exposes the report as a `#[napi(object)]` struct
(`nodejs/src/lib.rs:169-237`), so `Diagnosis` and `DiagnosisHint`
become two new napi object structs and a `diagnosis` field on `Report`.

## 5. Data structures (Rust core)

New module `crates/redhop/src/context/diagnosis.rs`, re-exported from
`context::mod`. All types `Debug + Clone + serde::Serialize`/`Deserialize`
(match whatever derives `ContextReport` carries today).

```rust
/// Query-level facts observed during retrieval and assembly, plus
/// bounded hints that fire on documented failure shapes. Pure
/// observability: nothing here changes what was retrieved or kept.
pub struct Diagnosis {
    /// The query's analyzed terms (deduped, first-occurrence order),
    /// produced by the same analyzer that indexed the corpus.
    pub query_terms: Vec<String>,
    /// Whether corpus-level stats (zero_match_terms, term_stats) were
    /// computed. False when `build_context` was called directly with a
    /// caller-supplied candidate pool and no Document.
    pub corpus_stats_available: bool,
    /// Query terms that appear in zero chunks of the corpus.
    /// Empty when `corpus_stats_available` is false.
    pub zero_match_terms: Vec<String>,
    /// Per-term corpus stats for terms that DO appear (df > 0).
    /// Empty when `corpus_stats_available` is false.
    pub term_stats: Vec<TermStat>,
    /// Query terms that appear in no *retrieved candidate* (they may
    /// still exist elsewhere in the corpus: present but outranked).
    /// Always computed.
    pub terms_unmatched_in_candidates: Vec<String>,
    /// Number of candidates handed to assembly (retrieved.len()).
    pub n_candidates: usize,
    /// Relative score spread across candidates:
    /// (top_score - kth_score) / top_score, over min(n_candidates, 10).
    /// None when n_candidates < 2 or top_score <= 0.
    pub score_spread: Option<f32>,
    /// True when assembly selected zero chunks.
    pub empty_context: bool,
    /// Hints that fired, from the closed registry in this module.
    pub hints: Vec<DiagnosisHint>,
}

pub struct TermStat {
    pub term: String,
    /// Number of corpus chunks containing the term.
    pub df: u32,
    /// df / total corpus chunks, in [0, 1].
    pub df_ratio: f32,
}

pub struct DiagnosisHint {
    pub code: HintCode,
    /// One or two sentences. User-facing prose: no em dashes, no
    /// semicolons (repo style rule). States observations, never
    /// promises improvements.
    pub message: String,
    /// Repo-relative path of the doc or finding that grounds the hint,
    /// e.g. "docs/findings/CUAD_RECALL_GAP.md".
    pub evidence: String,
}

#[non_exhaustive]
pub enum HintCode {
    EmptyContext,
    VocabMismatch,
    LowConfidence,
    LowDiscriminationQuery,
    UnderdeterminedQuery,
}
```

`ContextReport` (`context/mod.rs:339`) gains:

```rust
/// Query-level diagnosis: term/corpus match facts and bounded hints.
pub diagnosis: Diagnosis,
```

Not `Option`: facts are always computable. Partial availability is
expressed by `corpus_stats_available`.

`Document` (`document/mod.rs:193`) gains a private field:

```rust
/// term -> number of chunks containing it, built lazily on first
/// context() call. Uses cfg.analyzer (see ANALYZER_PLUGIN.md: the two
/// layers must not disagree on tokenization).
corpus_vocab: Option<std::collections::HashMap<String, u32>>,
```

## 6. Facts: definitions and computation

All tokenization through `terms(text, cfg.analyzer.as_ref())`
(`context/mod.rs:1292`). For ordered `query_terms`, either generalize
that helper or add an ordered variant in `diagnosis.rs` that calls the
same analyzer; the analyzer call is the part that must be shared.

- **F1 `query_terms`**: analyze `query.text`. Dedupe preserving first
  occurrence.
- **F2 `zero_match_terms` / `term_stats`** (Layer 2 only): look up each
  query term in the corpus vocab map. `df == 0` → `zero_match_terms`,
  else a `TermStat` with `df_ratio = df as f32 / n_corpus_chunks`.
- **F3 `terms_unmatched_in_candidates`** (Layer 1): union the term sets
  of all retrieved candidate chunks (analyzed), subtract from
  `query_terms`. With corpus stats present, a term in F3 but not in F2's
  zero list means "exists in the corpus, outranked in retrieval", which
  is the polysemy signature.
- **F4 term discrimination** (derived, used by hints): a term with
  `df_ratio > DF_RATIO_LOW_DISCRIMINATION` carries little ranking
  signal. This is the per-query analogue of `analyze_query_set`'s
  boilerplate detection.
- **F5 `score_spread`**: over the top `min(n, 10)` candidates by
  `score.value`: `(s_top - s_k) / s_top`. Guard `s_top <= 0` and `n < 2`
  → `None`. A flat spread on a short query means the query did not
  discriminate between candidates.
- **F6 `empty_context`**: `selected.is_empty()` at `make_report` time
  (same place `low_confidence_retrieval` is computed,
  `context/mod.rs:943-1001`).

## 7. The hint registry (closed list)

Module constants in `diagnosis.rs`:

```rust
const VOCAB_MISMATCH_MIN_SHARE: f32 = 0.5;
const VOCAB_MISMATCH_MIN_TERMS: usize = 2;
const DF_RATIO_LOW_DISCRIMINATION: f32 = 0.25;
const LOW_DISCRIMINATION_MIN_TERMS: usize = 8;
const LOW_DISCRIMINATION_MIN_SHARE: f32 = 0.6;
const UNDERDETERMINED_MAX_TERMS: usize = 2;
const UNDERDETERMINED_MAX_SPREAD: f32 = 0.15;
const UNDERDETERMINED_MIN_CANDIDATES: usize = 5;
```

Every constant is a 🟡 convention (no RedHop measurement chose these
exact values). They MUST be added to
[DEFAULT_PROVENANCE.md](../DEFAULT_PROVENANCE.md) as a new "Diagnosis
hint thresholds" section, classified 🟡, with a re-validation entry
(§11). Messages below are normative for tone and content; exact wording
may be polished but must stay observation-only, citation-backed, and
free of em dashes and semicolons.

| # | code | trigger (all conditions AND) | message template | evidence |
|---|------|------------------------------|------------------|----------|
| H1 | `EmptyContext` | `empty_context` | "Assembly selected zero chunks. {n_candidates} candidates were retrieved." (If H2 also fires it carries the why.) | `docs/CHOOSING_A_CONFIG.md` |
| H2 | `VocabMismatch` | `corpus_stats_available`, `query_terms.len() >= VOCAB_MISMATCH_MIN_TERMS`, `zero_match_terms.len() as f32 / query_terms.len() >= VOCAB_MISMATCH_MIN_SHARE`, and (`empty_context` or `low_confidence_retrieval`) | "{k} of {m} query terms appear nowhere in this corpus: {zero_match_terms}. The query and the documents may use different vocabulary. Rephrasing with the documents' own terms is the measured first fix. Dense retrieval (retrieval=\"hybrid\") matches paraphrases by embedding similarity and was measured to lift multi-hop retention on exactly this failure shape." | `docs/findings/MULTIHOP_HYBRID.md` |
| H3 | `LowConfidence` | `low_confidence_retrieval` and not `empty_context` and not H2 | "Every selected chunk is at or below the grounding bar ({low_confidence_threshold}). Retrieval matched something, but weakly. Check diagnosis.term_stats for which terms carried the match." | `docs/CHOOSING_A_CONFIG.md` |
| H4 | `LowDiscriminationQuery` | `corpus_stats_available`, `query_terms.len() >= LOW_DISCRIMINATION_MIN_TERMS`, share of `term_stats` with `df_ratio > DF_RATIO_LOW_DISCRIMINATION` is `>= LOW_DISCRIMINATION_MIN_SHARE` | "{k} of {m} query terms appear in more than {pct}% of chunks and carry little ranking signal. If your queries follow a fixed template, this is the boilerplate-dilution shape measured on CUAD. analyze_query_set on a sample of your queries will confirm it, and Stripper removes the wrapper." | `docs/findings/CUAD_RECALL_GAP.md` |
| H5 | `UnderdeterminedQuery` | `query_terms.len() <= UNDERDETERMINED_MAX_TERMS`, `n_candidates >= UNDERDETERMINED_MIN_CANDIDATES`, `score_spread` is `Some(s)` with `s <= UNDERDETERMINED_MAX_SPREAD` | "A {m}-term query produced a nearly flat ranking across {n} candidates (spread {s}). Short queries can match several sections equally well. One added disambiguating word was the fix in every measured polysemy case." | `docs/CHOOSING_A_CONFIG.md` |

Ordering: hints are pushed in registry order (H1..H5). H2 suppresses
H3 (H2 is the explanation, H3 the symptom). No other suppression.

Hint count is intentionally 5. New hints require: a documented failure
shape, a citation, and a registry-table row in this doc.

## 8. Computation flow (where the code goes)

1. **`diagnosis.rs`**: types (§5), constants (§7), and three functions:
   - `pub(crate) fn compute(query: &Query, retrieved: &[RetrievalResult], selected_empty: bool, low_confidence: bool, low_confidence_threshold: f32, analyzer: &dyn Analyzer) -> Diagnosis`
     — Layer 1: F1, F3, F5, F6, then `evaluate_hints`.
   - `pub(crate) fn enrich(d: &mut Diagnosis, vocab: &HashMap<String, u32>, n_corpus_chunks: usize, low_confidence: bool, low_confidence_threshold: f32)`
     — Layer 2: F2/F4, set `corpus_stats_available = true`, clear and
     re-run `evaluate_hints`.
   - `fn evaluate_hints(...)` — pure function of the facts, applies §7.
2. **`make_report` (`context/mod.rs:943-1001`)**: call `compute(...)`
   (it already has `selected`, the low-confidence values, and
   `cfg.analyzer`), assign to the new `ContextReport.diagnosis` field.
   `build_context_expanded` (`context/mod.rs:880`) flows through the
   same `make_report`, so it needs no separate handling. Verify.
3. **`Document::context_inner` (`document/mod.rs:682-728`)**: after the
   `build_context` call, ensure `self.corpus_vocab` is built (one pass
   over `self.chunks`, analyzed term set per chunk, increment df per
   distinct term per chunk), then call a new
   `pub fn enrich_diagnosis(ctx: &mut BuiltContext, vocab: &..., n_chunks: usize, ...)`
   exported from `context` (same pattern as `attach_rewrite_trail`,
   `context/mod.rs:674`). Call it before `attach_rewrite_trail` or
   after, order is immaterial.
4. **`ContextReport::render` (`context/mod.rs:441-549`)**: append a
   `Query diagnosis` section using the existing conventions (two-line
   `──` underlined header, two-space-indented `- ` bullets). Render the
   section only when `hints` is non-empty OR `zero_match_terms` is
   non-empty. Each hint renders as its message bullet plus an indented
   `evidence: {path}` line. The existing low-confidence Warning line
   stays untouched (rendered string is unstable, but no reason to churn
   it).

## 9. Bindings

**Python (`python/src/lib.rs`)** — follow the `economics` PyDict getter
pattern (`python/src/lib.rs:429-432`). On the `ContextReport` pyclass
(`python/src/lib.rs:346-351`) add:

```python
ctx.report.diagnosis  # dict:
# {
#   "query_terms": ["refund", "window"],
#   "corpus_stats_available": True,
#   "zero_match_terms": ["cancel", "money"],
#   "term_stats": [{"term": "refund", "df": 3, "df_ratio": 0.02}, ...],
#   "terms_unmatched_in_candidates": ["cancel", "money"],
#   "n_candidates": 20,
#   "score_spread": 0.41,          # or None
#   "empty_context": False,
#   "hints": [
#     {"code": "vocab_mismatch", "message": "...", "evidence": "docs/findings/MULTIHOP_HYBRID.md"},
#   ],
# }
```

`HintCode` serializes to snake_case strings. Update the `.pyi` stub if
the package ships one (verify under `python/`).

**Node (`nodejs/src/lib.rs`)** — add `#[napi(object)] pub struct Diagnosis`
and `#[napi(object)] pub struct DiagnosisHint` (camelCase fields:
`queryTerms`, `corpusStatsAvailable`, `zeroMatchTerms`, `termStats`,
`termsUnmatchedInCandidates`, `nCandidates`, `scoreSpread:
Option<f64>`, `emptyContext`, `hints`), plus
`pub diagnosis: Diagnosis` on `Report` (`nodejs/src/lib.rs:169-237`).
Regenerate `nodejs/index.d.ts`. `code` is a string.

**Rust** — the structs are public on `redhop::context`. Document them
(`#![warn(missing_docs)]` is on in `document/mod.rs`; check the context
module's lint posture and write doc comments regardless).

## 10. Testing plan

Rust (tests module in `context/mod.rs` or `diagnosis.rs`, matching the
existing report-field test style, e.g.
`distractor_filtered_drops_low_grounding`, `context/mod.rs:1379-1405`):

1. `vocab_mismatch_hint_fires_on_paraphrase_query` — reproduce the
   docs' canonical case verbatim: corpus with "refund" and "termination
   for convenience", query "How long do I have to cancel and get my
   money back?". Assert H2 fires through `Document::context`,
   `zero_match_terms` contains "cancel", `evidence` ends in
   `MULTIHOP_HYBRID.md`.
2. `diagnosis_without_document_has_no_corpus_stats` — direct
   `build_context` call: `corpus_stats_available == false`,
   `zero_match_terms` empty, H2 cannot fire, candidate-level facts
   present.
3. `low_discrimination_hint_fires_on_templated_query` — small corpus,
   a query with ≥8 terms where most appear in >25% of chunks. Assert H4
   and the boilerplate terms appear in `term_stats` with high
   `df_ratio`.
4. `underdetermined_hint_fires_on_flat_short_query` — corpus of near-
   parallel sections, 1-term query, assert `score_spread` flat and H5.
5. `empty_context_hint` — assert H1 plus H2 co-firing and H3
   suppression by H2 (one test each).
6. `healthy_query_produces_no_hints` — a well-matched query: facts
   populated, `hints` empty. Guards against hint spam, which is this
   feature's main failure mode.
7. `diagnosis_uses_document_analyzer` — German-analyzer corpus
   containing "Bücher", query "Buch": assert "buch" is NOT in
   `zero_match_terms` (mirrors `test_german_analyzer_unifies_morphology`
   in `python/tests/test_analyzer.py`). This is the D3 anti-drift test.
8. `rewritten_query_is_what_gets_diagnosed` — `context_with_rewrites`
   with a `Stripper`: `query_terms` reflect the post-rewrite query.
9. `render_includes_diagnosis_section_only_when_warranted` — render
   with and without fired hints.

Python (`python/tests/test_diagnosis.py`): the vocab-mismatch scenario
end to end, asserting dict shape, hint code string, and that a healthy
query yields `hints == []`. Node: one smoke test in the existing test
location asserting `report.diagnosis.hints` exists and is empty on a
healthy query.

Also run `python3 scripts/check_readme_numbers.py` (no pinned strings
should change) and the full existing suite.

## 11. Docs, examples, changelog (same PR)

- **`docs/CHOOSING_A_CONFIG.md`** ("Query writing"): one short
  paragraph per failure shape noting the report now surfaces it, e.g.
  "Since 0.3.4 the report diagnoses this: `ctx.report.diagnosis` lists
  the query terms that appear nowhere in the corpus." Mirror on the
  website (`../redhop-website/src/content/docs/docs/choosing-a-config.mdx`).
  Style rule: no em dashes, no semicolons in any user-facing prose,
  including every hint message string in the code.
- **`docs/DEFAULT_PROVENANCE.md`**: new "Diagnosis hint thresholds"
  table listing the §7 constants as 🟡 convention, plus an entry in
  "Defaults flagged for re-validation" (a grid sweep over the
  thresholds against the existing CUAD/HotpotQA corpora would make them
  🟢).
- **`examples/`**: new example #12 in each language
  (`examples/python/12_diagnosis.py`, plus the Node and Rust mirrors
  using each directory's existing naming): load a small doc, run one
  healthy query and one vocabulary-mismatch query, print
  `ctx.report.render()` and walk the diagnosis fields. README feature
  mention links this file (repo rule: feature sections link runnable
  examples, not findings).
- **`README.md`**: one sentence under the Decision Report feature
  bullet + link to the example. Do not add numeric claims (keeps the
  drift-check registry untouched).
- **`CHANGELOG.md`**: `### Added` entry under 0.3.4 following the
  existing format (bold lead, mechanism summary, finding citations).
- **Website**: besides choosing-a-config, check the Decision Report
  section of `docs/overview.mdx` for a natural one-line mention.

## 12. Future extensions (explicitly deferred)

- Per-term score attribution via Tantivy explain (would upgrade H5 from
  spread-based inference to direct evidence).
- "Present but outranked" drill-down: for terms in
  `terms_unmatched_in_candidates` with corpus df > 0, report which
  chunks contain them. Cheap with the vocab map extended to postings,
  but postings cost memory; needs a use case first.
- Threshold sweep to turn the 🟡 constants 🟢 (see DEFAULT_PROVENANCE
  re-validation entry).
- Cross-source `by_source` reporting (separate feature, separate spec).

## 13. Acceptance criteria

- [ ] `ctx.report.diagnosis` populated on every `context()` and
      `build_context()` call in all three languages.
- [ ] The refund/cancel docs scenario produces H2 with the exact
      zero-match terms and a findings citation, verified by test.
- [ ] A healthy query produces zero hints (test 6).
- [ ] Direct `build_context` callers get candidate-level facts and
      `corpus_stats_available == false` without panics.
- [ ] German-analyzer anti-drift test passes (test 7).
- [ ] No retrieval or assembly behavior change: full existing test
      suite green, no benchmark deltas expected or claimed.
- [ ] No hint message contains an em dash, a semicolon, or a promised
      improvement. Every hint carries an `evidence` path that exists in
      the repo (add a unit test iterating the registry and checking the
      paths against the filesystem).
- [ ] `scripts/check_readme_numbers.py` passes unchanged.
- [ ] DEFAULT_PROVENANCE, CHANGELOG, CHOOSING_A_CONFIG (both repos),
      README mention + linked example #12 in three languages, all in
      the same PR.
