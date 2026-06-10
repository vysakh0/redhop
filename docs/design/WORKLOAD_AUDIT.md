# Design: workload audit + bring-your-own-pipeline diagnostics + observability export

**Status**: proposed, approved for implementation. Target release: 0.3.4
(all additive, non-breaking). Builds directly on
[REPORT_DIAGNOSIS.md](REPORT_DIAGNOSIS.md) (shipped on
`feat/report-diagnosis`), which this spec assumes is merged.

**Audience**: written to be picked up cold by an implementing agent.
File/line references are against `feat/report-diagnosis` at commit
`2d12a85`. Verify line numbers before editing; structures are the
contract, line numbers are hints.

---

## 1. Problem and strategic intent

Per-query diagnosis shipped in REPORT_DIAGNOSIS. Two gaps remain
between that and the thing users actually need:

1. **One query tells you about one query.** The real question is
   workload-shaped: "across my last 500 production queries, what is
   the dominant failure mode, and which knob does the evidence say to
   reach for?" Today a user must hand-roll the histogram over
   `report.diagnosis.hints`. The decision tree in
   [CHOOSING_A_CONFIG.md](../CHOOSING_A_CONFIG.md) asks users to
   introspect their corpus manually; aggregated diagnosis can answer it
   from data.

2. **Experiencing the Decision Report currently requires migrating
   retrieval.** A LangChain/LlamaIndex user has to replace their stack
   before they see what RedHop's report says about their pipeline. But
   the architecture already supports bring-your-own-retrieval:
   `analyze_context` accepts caller-supplied chunks
   (`crates/redhop/src/context/mod.rs`, `pub fn analyze_context`, and
   the Python pyfunction at `python/src/lib.rs:696` /
   Node `analyzeContext` at `nodejs/src/lib.rs:1883`), and
   `Document::from_chunks` upgrades a BYO user to full Layer-2
   corpus diagnosis. What's missing is a *documented, exampled,
   summarizable path* — the funnel door, not new machinery.

Strategic frame (agreed 2026-06-10): the library remains the product
(its measured wins come from owning chunking + retrieval + assembly).
This bundle is the acquisition channel: let users point RedHop's
diagnostics at their *existing* pipeline with ~10 lines and no behavior
change, and let every hint cite the measured finding whose fix lives in
RedHop. Observability export rides existing rails (OpenTelemetry /
Langfuse) instead of competing with them.

## 2. Goals

- `summarize_diagnoses(...)`: one call that turns N `ContextReport`s
  into a workload-level summary — hint histogram, failure rates, top
  vocabulary gaps, and at most ONE findings-cited focus recommendation.
  Rust + Python + Node.
- A flat-attribute export so a report can be attached to an OTel span
  or Langfuse trace without RedHop taking any new dependency.
- A docs page ("Diagnose your existing RAG pipeline") + example #13 in
  three languages showing the BYO loop end to end.
- Zero new required dependencies in any crate or package. Zero
  behavior change to retrieval/assembly.

## 3. Non-goals (read first)

- **No `opentelemetry`, `langfuse`, or `langchain` dependency**
  anywhere — not in crates, not in the Python package, not in
  examples, not in tests. The export helper emits a plain flat dict /
  map; the user's own SDK attaches it. LangChain glue appears only as
  *commented* snippets in docs and examples.
- **No auto-instrumentation or monkeypatching** of other frameworks.
- **No auto-tuning.** The summary recommends, with citations; it never
  changes configuration. (Bounded-architecture rule.)
- **At most one focus recommendation per summary.** A summary that
  recommends three things recommends nothing. Anti-spam is a hard
  requirement, mirroring "healthy query produces zero hints".
- **No promised lifts.** Same wording discipline as hint messages:
  observations plus "measured on <finding>" only with the citation.
  No em dashes, no semicolons in user-facing strings.
- **No hosted/cloud anything.** This spec is library + docs only.
- **No retention/recall claims in BYO mode.** Without gold labels the
  summary reports failure-shape signals, not quality scores. The docs
  page must say so explicitly (see §8, honesty section).

## 4. Design decisions, with rationale

**D1 — Mirror the existing `eval::summarize` precedent.** The repo
already has exactly this pattern for eval:
`pub fn summarize(reports: &[EvalReport]) -> EvalSummary`
(`crates/redhop/src/context/eval.rs:1135`, struct at `eval.rs:1085`).
`summarize_diagnoses` copies its shape: slice in, plain summary struct
out, mean-over-the-subset convention for optional fields (see
`EvalSummary`'s "None if zero present" convention at
`eval.rs:1101-1105`). Input is `&[ContextReport]`, not `&[Diagnosis]`,
because reports are what callers hold (`ctx.report`) and the summary
also needs report-level fields (`low_confidence_retrieval`).

**D2 — One focus, ranked by a fixed priority, gated by sample size.**
Multiple hints can dominate simultaneously (a templated workload also
trips low-confidence). The summary resolves to a single `focus` via a
priority order (§5.3) with 🟡 threshold constants registered in
DEFAULT_PROVENANCE, exactly like the hint registry. Below
`SUMMARY_MIN_QUERIES` the focus is `SampleTooSmall` and no
recommendation is made. Precedent for a recommendation string on an
analyzer output: `QuerySetReport::suggested_action`
(`crates/redhop/src/analyzer.rs:501-525`).

**D3 — Flat attributes, no SDK coupling.** The observability export is
a pure function: report in, flat `key -> scalar-or-string-list` map
out, keys under a `redhop.` namespace. Rationale: OTel attribute values
must be primitives or homogeneous arrays, and Langfuse metadata accepts
arbitrary JSON — a flat dict satisfies both. RedHop never imports an
SDK, so the zero-dependency story survives. Hint *messages* are
excluded from attributes (cardinality and size discipline; codes are
enough to alert/aggregate on). The full report is available via the
existing `report.json()` (`python/src/lib.rs`, `fn json`) for span
events or log bodies.

**D4 — Python gets a helper module; Node gets a documented snippet;
Rust gets the conventions table.** Python has a pure-Python wrapper
layer (`python/redhop/__init__.py` over `_redhop.abi3.so`) where a
dependency-free `redhop/otel.py` fits naturally. Node's `index.js` is
napi-generated (hand edits are overwritten), so Node ships the mapping
as a copy-paste snippet in the docs page instead of a helper —
acceptable asymmetry, documented in the parity test allowlist (§7).
Rust users read the conventions table and use `serde_json` on the
report. Revisit if users ask.

**D5 — Example #13 is self-contained; framework glue lives in docs.**
Examples 01-12 run with only `redhop` installed and CI-able. #13
simulates "an external retriever" with a local function returning
plain strings, and carries the real LangChain/LlamaIndex glue as a
clearly marked comment block. The docs page shows the full framework
code (it isn't executed in CI).

## 5. Component 1: `summarize_diagnoses`

### 5.1 Data structures

New code lives in `crates/redhop/src/context/diagnosis.rs` (same
module as the per-query types). All types `Debug + Clone + Serialize +
Deserialize` like `Diagnosis`.

```rust
/// Workload-level aggregation of per-query diagnoses. Observation
/// only: reports the shape of the workload's failures and at most one
/// findings-cited focus.
pub struct DiagnosisSummary {
    /// Number of reports aggregated.
    pub n: usize,
    /// Count + share per hint code, all five codes always present
    /// (count 0 included) so consumers can chart without key checks.
    pub hint_counts: Vec<HintCount>,
    /// Fraction of reports where assembly selected zero chunks.
    pub empty_context_rate: f32,
    /// Fraction of reports with `low_confidence_retrieval`.
    pub low_confidence_rate: f32,
    /// Fraction of reports with corpus stats (Layer 2). Below 1.0
    /// means part of the workload ran through direct `build_context`
    /// or `analyze_context` and got candidate-level facts only.
    pub corpus_stats_coverage: f32,
    /// Terms that zero-matched the corpus, ranked by how many queries
    /// they appeared in. Capped at TOP_TERMS_CAP. The workload's
    /// vocabulary gap, directly actionable as a `Vocabulary` dict or
    /// doc-glossary fix.
    pub top_zero_match_terms: Vec<TermCount>,
    /// Mean `score_spread` over reports where it was Some(_).
    /// None when no report carried one (mirrors EvalSummary's
    /// "None if zero present" convention).
    pub mean_score_spread: Option<f32>,
    /// How many reports carried a score_spread.
    pub n_with_score_spread: usize,
    /// The single focus recommendation (or Healthy / SampleTooSmall).
    pub focus: WorkloadFocus,
}

pub struct HintCount {
    pub code: HintCode,
    pub count: usize,
    /// count / n, in [0, 1]. 0.0 when n == 0.
    pub share: f32,
}

pub struct TermCount {
    pub term: String,
    /// Number of queries whose diagnosis listed this term.
    pub count: usize,
}

pub struct WorkloadFocus {
    pub code: FocusCode,
    /// One or two sentences. Same style rules as hint messages.
    pub message: String,
    /// Repo-relative evidence path. Empty string for Healthy and
    /// SampleTooSmall (nothing to cite).
    pub evidence: String,
}

#[non_exhaustive]
pub enum FocusCode {
    /// Fewer than SUMMARY_MIN_QUERIES reports; no recommendation.
    SampleTooSmall,
    /// No failure shape reached DOMINANT_HINT_SHARE.
    Healthy,
    /// vocab_mismatch dominates.
    VocabMismatch,
    /// low_discrimination_query dominates.
    TemplatedQueries,
    /// underdetermined_query dominates.
    UnderdeterminedQueries,
    /// empty/low-confidence rates are high but no specific hint
    /// dominates: the corpus may simply not cover the questions.
    WeakRetrieval,
}
```

### 5.2 Function

```rust
/// Aggregate per-query diagnoses into a workload summary. Mirrors
/// [`crate::context::eval::summarize`]'s slice-in / summary-out shape.
pub fn summarize_diagnoses(reports: &[ContextReport]) -> DiagnosisSummary
```

Mechanics: single pass over reports. `top_zero_match_terms` counts a
term once per query (the per-query list is already deduped), sorted by
count desc then term asc (deterministic), truncated to `TOP_TERMS_CAP`.
`n == 0` returns zeroed summary with `FocusCode::SampleTooSmall` and no
panics (mirror `EvalSummary`'s `0.0 if n == 0` convention).

Add `impl DiagnosisSummary { pub fn render(&self) -> String }`
following the report render conventions (two-line `──` headers,
two-space `- ` bullets; see `ContextReport::render`,
`crates/redhop/src/context/mod.rs:450`). Sections: header with n,
hint histogram (one line per non-zero code, count + share), rates
line, top zero-match terms line (display-ordered content-words-first
via the existing `diagnosis::display_order`), then the focus with its
evidence path. Rendered string unstable, as with the report.

### 5.3 Focus resolution (the closed registry, workload edition)

Constants in `diagnosis.rs`, all 🟡, registered in DEFAULT_PROVENANCE
(§9):

```rust
const SUMMARY_MIN_QUERIES: usize = 20;
const DOMINANT_HINT_SHARE: f32 = 0.20;
const WEAK_RETRIEVAL_MIN_RATE: f32 = 0.30;
const TOP_TERMS_CAP: usize = 20;
```

Resolution order (first match wins, exactly one focus):

| order | condition | focus | message template | evidence |
|---|---|---|---|---|
| 1 | `n < SUMMARY_MIN_QUERIES` | `SampleTooSmall` | "Only {n} queries aggregated. {min} or more are needed before the failure-shape shares are meaningful." | (empty) |
| 2 | share(`VocabMismatch`) >= `DOMINANT_HINT_SHARE` | `VocabMismatch` | "{pct}% of queries had most terms missing from the corpus. Top gap terms: {top_terms}. Rephrasing toward the documents' vocabulary is the measured first fix, and dense retrieval (retrieval=\"hybrid\") was measured to lift retention on exactly this shape." | `docs/findings/MULTIHOP_HYBRID.md` |
| 3 | share(`LowDiscriminationQuery`) >= `DOMINANT_HINT_SHARE` | `TemplatedQueries` | "{pct}% of queries are boilerplate-shaped. Run analyze_query_set on a sample to extract the template, then compile a Stripper. Template stripping was measured to lift retention on exactly this shape (CUAD three-arm run)." | `docs/findings/CUAD_CLAUSE_EXPANSION.md` |
| 4 | share(`UnderdeterminedQuery`) >= `DOMINANT_HINT_SHARE` | `UnderdeterminedQueries` | "{pct}% of queries were too short to discriminate between candidates. One added disambiguating word was the fix in every measured polysemy case. If queries come from a UI, consider prompting for one more keyword." | `docs/CHOOSING_A_CONFIG.md` |
| 5 | `empty_context_rate.max(low_confidence_rate)` >= `WEAK_RETRIEVAL_MIN_RATE` | `WeakRetrieval` | "{pct}% of queries retrieved nothing usable but no single failure shape dominates. The corpus may not cover these questions. Inspect top_zero_match_terms for what users ask about that the documents never mention." | `docs/CHOOSING_A_CONFIG.md` |
| 6 | otherwise | `Healthy` | "No failure shape exceeded {dominant_pct}% of queries. No intervention indicated." | (empty) |

Note rule 2 outranks 3: a vocab gap usually *causes* downstream
low-confidence, so it is the root-cause recommendation (same
suppression logic as H2-over-H3 per query).

### 5.4 Exports and bindings

- `crates/redhop/src/context/mod.rs`: re-export
  `summarize_diagnoses, DiagnosisSummary, HintCount, TermCount,
  WorkloadFocus, FocusCode` alongside the existing diagnosis types
  (the `pub use diagnosis::{...}` near the top of the module).
- `crates/redhop/src/lib.rs:92-95`: add them to the existing
  `pub use crate::context::{...}` block.
- **Python** (`python/src/lib.rs`): `#[pyfunction] summarize_diagnoses(
  reports: Vec<PyRef<ContextReport>>) -> PyResult<Bound<PyDict>>`
  (accept a list of the existing `ContextReport` pyclass; read
  `.inner`). Return a dict mirroring the struct: `hint_counts` as list
  of `{code, count, share}` dicts (codes via the existing
  `hint_code_to_str`), `focus` as `{code, message, evidence}` with
  snake_case focus codes, plus a `rendered` key carrying
  `summary.render()`. Check first how the existing `summarize` (eval)
  pyfunction is registered (`m.add_function` block,
  `python/src/lib.rs:2737` area) and mirror its registration and
  return style — if eval's summarize returns a pyclass instead of a
  dict, match whichever it does for consistency.
- **Node** (`nodejs/src/lib.rs`): `#[napi] pub fn summarize_diagnoses(
  reports: Vec<Report>) -> DiagnosisSummary` with `#[napi(object)]`
  structs `DiagnosisSummary`, `HintCount`, `TermCount`,
  `WorkloadFocus` (camelCase fields, codes as strings, plus a
  `rendered: String` field). The napi `Report` already carries
  `diagnosis` and `low_confidence_retrieval`, so the napi layer can map
  napi objects back to counts directly without the Rust core types.
  Regenerate `index.d.ts`.

## 6. Component 2: observability export (flat attributes)

### 6.1 Conventions (documented in the docs page, §8)

Namespace `redhop.`, types limited to OTel-legal attribute values
(bool / i64 / f64 / string / homogeneous string array):

| attribute | type | source |
|---|---|---|
| `redhop.strategy` | string | `report.strategy` (snake_case) |
| `redhop.requested_strategy` | string | `report.requested_strategy` |
| `redhop.auto_decision` | string | `report.auto_decision()` |
| `redhop.input_tokens` | int | `report.input_tokens` |
| `redhop.total_tokens` | int | `report.total_tokens` |
| `redhop.token_budget` | int | `report.token_budget` |
| `redhop.n_input_chunks` | int | `report.n_input_chunks` |
| `redhop.n_selected` | int | `report.n_selected` |
| `redhop.retained_evidence_ratio` | float | `report.retained_evidence_ratio` |
| `redhop.evidence_density` | float | `report.economics.evidence_density` |
| `redhop.estimated_waste_tokens` | int | `report.economics.estimated_waste_tokens` |
| `redhop.second_hop_rescues` | int | `report.second_hop_rescue_count` |
| `redhop.low_confidence` | bool | `report.low_confidence_retrieval` |
| `redhop.diagnosis.empty_context` | bool | `diagnosis.empty_context` |
| `redhop.diagnosis.hints` | string[] | hint codes only, in fire order |
| `redhop.diagnosis.zero_match_terms` | string[] | display-ordered, capped at 16 |
| `redhop.diagnosis.score_spread` | float | omitted when `None` |
| `redhop.diagnosis.n_candidates` | int | `diagnosis.n_candidates` |

Excluded on purpose: hint `message` strings (size/cardinality; the
`evidence` path is recoverable from the code), `term_stats` (per-term
rows belong in the JSON body, not attributes), rendered text.

### 6.2 Python helper: `python/redhop/otel.py`

New pure-Python module (no imports beyond stdlib), exported as
`redhop.otel`:

```python
def report_to_attributes(report, prefix: str = "redhop.") -> dict:
    """Flatten a ContextReport into OTel-legal span attributes.

    Works with any telemetry SDK: pass the returned dict to
    `span.set_attributes(...)` (OpenTelemetry) or as `metadata=`
    (Langfuse). RedHop imports no SDK. Full report: `report.json()`.
    """
```

Implementation notes: read the existing getters on the report pyclass
(they're already exposed: `python/src/lib.rs:355-` onward, and the
`diagnosis` dict getter added in REPORT_DIAGNOSIS). Apply the table
above, skip `score_spread` when `None`, cap `zero_match_terms` at 16
using the order the diagnosis dict provides. Add the module to the
package's public surface the same way other Python-side modules are
wired in `python/redhop/__init__.py` (verify whether `__init__.py`
re-exports submodules or users import `redhop.otel` directly — match
existing style; if nothing similar exists, plain `from redhop import
otel` via the submodule file is enough).

### 6.3 Node + Rust

No helper (D4). The docs page carries a ~15-line JS snippet
implementing the same table from `ctx.report` (all fields are plain
napi-object properties), and notes Rust users can build the map from
the public struct or `serde_json::to_value(&ctx.report)`.

## 7. Testing plan

Rust (`diagnosis.rs` tests module):

1. `summarize_empty_input_is_sample_too_small` — n=0: zeroed rates, no
   panic, focus `SampleTooSmall`.
2. `summarize_below_min_queries_makes_no_recommendation` — n=5 of
   anything → `SampleTooSmall`.
3. `summarize_vocab_dominant_workload` — synthesize ≥20 reports where
   ≥20% carry a `VocabMismatch` hint (build `Diagnosis` values
   directly; `ContextReport` has all-pub fields so construct via
   `..Default::default()` if a Default exists, else a small helper):
   focus `VocabMismatch`, evidence ends `MULTIHOP_HYBRID.md`,
   `top_zero_match_terms` ranked correctly with ties broken by term.
4. `summarize_vocab_outranks_templated` — both shapes above threshold →
   focus is `VocabMismatch` (priority rule).
5. `summarize_weak_retrieval_without_dominant_hint` — high
   empty-context rate, no hint above threshold → `WeakRetrieval`.
6. `summarize_healthy_workload_recommends_nothing` — n=30 hint-free
   reports → `Healthy`, empty evidence. The anti-spam test.
7. `summary_render_sections_and_focus` — render contains the
   histogram, the focus message, and its evidence path.
8. Extend `evidence_paths_all_exist_in_repo` to cover the focus
   registry's evidence constants.
9. Style test: extend the em-dash source check to the new message
   templates (the existing `no_em_dash_or_prose_semicolon_in_hint_strings`
   scans the file; just keep the new templates above the test module).

Python (`python/tests/test_workload_audit.py`):

- End-to-end BYO loop: build a small corpus, run ~25 queries (mix of
  healthy and the canonical paraphrase failures) through
  `Document.from_chunks(...).context(q)`, collect `ctx.report`, call
  `redhop.summarize_diagnoses(reports)`, assert focus code and that
  `rendered` is non-empty.
- `analyze_context` Layer-1 loop: same queries through
  `redhop.analyze_context(q, chunks)` (strings in, per
  `python/src/lib.rs:696`), assert `corpus_stats_coverage == 0.0` and
  the summary still resolves without error.
- `redhop.otel.report_to_attributes`: every value is
  bool/int/float/str/list-of-str (assert recursively), keys all
  prefixed, `score_spread` key absent when the report's is None, term
  list capped at 16.

Node (`nodejs/test/smoke.cjs` extension): run 25 queries, call
`summarizeDiagnoses(reports)`, assert `focus.code` is a string and
`hintCounts` has 5 entries.

Parity (`python/tests/test_parity_node.py`): check whether the
function-surface parity covers module-level functions; if it does, add
`summarize_diagnoses` to both sides and add the Python-only `otel`
helper to the PY_ONLY allowlist with a comment citing D4 of this spec.

Plus: `python3 scripts/check_readme_numbers.py` unchanged, and the
full existing suites green. **No benchmark numbers in code strings**:
the drift checker scans docs files, not Rust source, so a number baked
into a message template would rot silently when findings are re-run.
Focus messages say "measured" and cite; the numbers live in the cited
finding. (Same discipline the per-query hints follow.)

## 8. Component 3: docs page + example #13

**`docs/DIAGNOSE_YOUR_PIPELINE.md`** (new), mirrored to the website as
`../redhop-website/src/content/docs/docs/diagnose-your-pipeline.mdx`
(add to the site nav where the other docs/ pages are registered —
check `astro.config.mjs` or the sidebar config for how
choosing-a-config is listed). Structure:

1. *Who this is for*: you already run retrieval (LangChain,
   LlamaIndex, pgvector, hand-rolled) and want to know why it
   sometimes fails, without migrating anything.
2. *Step 1 — one query, zero behavior change*: retrieved texts →
   `redhop.analyze_context(query, texts)` → read
   `report.diagnosis` (Layer 1: candidate-level facts). Real
   LangChain glue shown here (`retriever.invoke(q)` →
   `[d.page_content for d in docs]`).
3. *Step 2 — corpus-level diagnosis*: load the same chunks once via
   `Document.from_chunks` to get `zero_match_terms` / `term_stats`
   (Layer 2). Note what this does and doesn't change (nothing about
   their pipeline; RedHop indexes a copy in memory).
4. *Step 3 — audit the workload*: loop a few hundred real queries,
   collect reports, `summarize_diagnoses`. Show the rendered summary
   and the hint→fix→finding mapping table (reuse the table content
   from CHOOSING_A_CONFIG's query-writing section).
5. *Step 4 — ship it to your telemetry*: `redhop.otel`
   helper + OTel snippet + Langfuse `metadata=` snippet + the Node
   snippet (per §6.3).
6. **Honesty section (required)**: without gold labels this measures
   failure *shapes*, not retention or answer quality. `analyze_context`
   reports waste; only `build_context` removes it. If the summary says
   `Healthy`, RedHop has nothing to sell you and that's the correct
   outcome. No em dashes, no semicolons (user-facing prose rule).

**Example #13** (three languages, self-contained per D5):
`examples/python/13_workload_audit.py`,
`examples/nodejs/13_workload_audit.cjs`,
`examples/rust/examples/13_workload_audit.rs`. Shape: a local
`external_search(query) -> Vec<String>` stand-in for "your existing
retriever" + a commented LangChain block (Python only); run ~25 mixed
queries through the Layer-1 path; print `summary.render()`; then the
two-line upgrade to `Document.from_chunks` for Layer 2 and print the
summary again showing `top_zero_match_terms` appear. Update
`examples/README.md` (it indexes the examples — verify format) and the
per-language example lists if they exist.

## 9. Docs / provenance / changelog (same PR)

- **DEFAULT_PROVENANCE.md**: add the four §5.3 constants to the
  diagnosis-thresholds table (🟡, with one-line rationales as in the
  existing rows) and extend re-validation entry #5 to cover them.
- **CHANGELOG.md**: extend the existing `## [Unreleased]` "Added"
  section (added in REPORT_DIAGNOSIS) with `summarize_diagnoses`, the
  `redhop.otel` helper, and the docs page; same format.
- **README.md**: one sentence + link, in the Decision Report section
  next to the diagnosis sentence: the workload-audit loop and
  "works on your existing pipeline" framing, linking
  `examples/python/13_workload_audit.py` and the docs page. No new
  numeric claims (drift registry untouched).
- **CHOOSING_A_CONFIG.md** (+ website mirror): in the query-writing
  intro callout, one added line: aggregate over a workload with
  `summarize_diagnoses`, link the new docs page.

## 10. Acceptance criteria

- [ ] `summarize_diagnoses` available and consistent in Rust, Python,
      Node; n=0 and n<20 inputs handled without panics.
- [ ] Exactly one focus per summary; healthy workload yields `Healthy`
      with no recommendation (test 6).
- [ ] Focus messages: every evidence path exists on disk (test 8); no
      em dashes or semicolons; no benchmark numbers in any code
      string (cite the finding, keep numbers in the finding).
- [ ] `redhop.otel.report_to_attributes` emits only OTel-legal value
      types, imports no third-party SDK.
- [ ] Example #13 runs with only `redhop` installed in all three
      languages; LangChain appears only in comments/docs.
- [ ] BYO Layer-1 path verified by test: plain strings through
      `analyze_context` → summarize, `corpus_stats_coverage == 0.0`.
- [ ] Docs page + website mirror live, including the honesty section
      and the Node/Rust attribute snippet.
- [ ] DEFAULT_PROVENANCE, CHANGELOG, README, CHOOSING_A_CONFIG updated;
      parity allowlists updated if the function surface is checked.
- [ ] Full existing suites + `scripts/check_readme_numbers.py` green.

## 11. Future (explicitly deferred)

- A `Vocabulary` scaffold generated from `top_zero_match_terms` (one
  step from observation to fix — needs a curation UX decision first).
- Langfuse/OTel *semantic convention* registration upstream if the
  attribute names stabilize.
- Threshold sweep to turn the §5.3 constants 🟢 (fold into the
  REPORT_DIAGNOSIS sweep; same corpora, same method).
- Hosted aggregation/dashboards: parked until the integration door
  shows pull (strategic decision 2026-06-10).
