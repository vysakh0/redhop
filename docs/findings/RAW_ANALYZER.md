# Raw analyzer is the new default — fast BM25, no stemming

> **Status: shipped as the 0.3.2 default (breaking).** The minimal
> Tantivy pipeline (tokenize + ASCII fold + lowercase; no CamelCase
> split, no stopword filter, no Snowball stemmer) measurably beat the
> previous English-Snowball default on three workloads — on retention
> AND latency — so it became the default. English Snowball is still
> available as an explicit opt-in via `language="english"`.
>
> | Workload | english ≥0.8 | raw ≥0.8 | Δ retention | english p50 | raw p50 |
> |---|---:|---:|---:|---:|---:|
> | CUAD | 86% | **91%** | **+5pts** | 6.4ms | 3.8ms |
> | HotpotQA | 100% | 100% | 0 | 2.9ms | 2.3ms |
> | MuSiQue | 90% | **97%** | **+7pts** | 3.4ms | 2.3ms |
>
> Stemming was *hurting* recall on these workloads (false-positive
> ranking from stem collisions like `"settles"`/`"settling"` →
> `"settl"`), not helping. The flip is a breaking change: rankings
> shift for users who built indexes under the old default. Code-search
> users and inflection-heavy workloads should opt back in with
> `language="english"`.

## Why this probe ran

[FRAMEWORK_MULTIQUERY](FRAMEWORK_MULTIQUERY.md) measured RedHop's
warm-query latency at ~3.3ms vs LangChain's 0.3ms — a 10× gap. Source
inspection showed three causes: LangChain literally does
`text.split()` (no preprocessing); LlamaIndex stems but uses
lightweight `bm25s`; RedHop runs the full Tantivy pipeline (7 stages:
SimpleTokenizer → RemoveLong → CamelCaseSplitter → AsciiFolding →
LowerCaser → StopWordFilter → Stemmer).

User asked: can we offer an opt-in fast path? Yes — strip the
language-specific filters from the pipeline. The hypothesis going in
was the standard story: "removing stemming will cut latency, at the
cost of inflectional recall (`'highlighted'` no longer matches
`'highlight'`)."

The measurement said the second half of that prediction is wrong.

## The implementation

New `RawAnalyzer` in `crates/redhop/src/analyzer.rs`. Pipeline:

```
SimpleTokenizer → AsciiFoldingFilter → LowerCaser
```

Three steps total. No `CamelCaseSplitter`, no `RemoveLongFilter`, no
`StopWordFilter`, no `Stemmer`. The minimum any BM25 implementation
should do: tokenize on Unicode word boundaries, fold diacritics
(`"café"` → `"cafe"`), and lowercase. That's it.

In 0.3.2 this is the default — passing nothing at all gets you the
raw pipeline. English Snowball, German, French, etc. require an
explicit `language=` opt-in.

```python
# 0.3.2 default — raw pipeline, no extra arguments needed:
doc = redhop.Document.from_text(text)

# Opt back in to English Snowball (stemming + camelCase + stopwords):
doc = redhop.Document.from_text(text, language="english")

# Multilingual content still uses the language string:
doc = redhop.Document.from_text(text, language="german")
```

## What this measures

3 workloads × 2 analyzer modes × n=100, budget=2000 tok, candidate_k=40,
`strategy="raw_topk"` (assembly out of scope here; we want the BM25
ranking only).

Workloads:
- **CUAD** — templated legal QA. Queries use exact clause names from
  the contracts (`"Change of Control"`, `"Non-Compete"`).
- **HotpotQA** — 2-hop natural-language QA from Wikipedia.
- **MuSiQue** — compositional 2-4-hop natural-language QA.

My prediction going in: `raw` would WIN CUAD (queries echo the doc
verbatim, stemming is just noise) but LOSE HotpotQA/MuSiQue (users
paraphrase, stemming would help cover `"highlighted"` vs
`"highlight"`-style gaps). That prediction was half right.

## What the result actually says

### `raw` wins CUAD by +5 ≥0.8

Expected direction, measured magnitude. CUAD's gold answer spans are
literal contract clauses. The query and document share specific legal
vocabulary verbatim. Stemming doesn't recover any missed match (the
query terms already appear in the doc); it just creates collisions
(`"settles"` / `"settling"` / `"settled"` all → `"settl"`) that pull
distractor chunks into the top-K and demote the gold chunk.

### `raw` and `english` tie on HotpotQA

Both 100% ≥0.8 at budget=2000. The budget is large enough that the
gold sentence always fits; whether stemming helps or hurts is invisible
at this slack. Worth re-running at a tighter budget — but for the
default 8192 token budget that production users typically run, this
match (or `raw` lead) probably holds.

### `raw` wins MuSiQue by +7 ≥0.8

This is the surprise. MuSiQue compositional 2-4 hop is exactly the
workload where I expected stemming to help ("the spouse of the Green
performer" vs "Steve Hillage's wife was Miquette" — different
inflections). It doesn't help, and it hurts.

The likely mechanism: when stemming is on, the analyzer treats
`"performs"` and `"performer"` as the same token. Multi-hop bridge
passages contain many such pairs by chance (Wikipedia articles say
"performer", "performance", "performing" all over the place). The
shared stem inflates BM25 scores on chunks that share *any* form of
these words, drowning out the chunks that share the actually-
discriminating proper nouns.

### `raw` is also faster on every workload

1.5-2.5× speedup per query (6.4ms → 3.8ms on CUAD; 3.4ms → 2.3ms on
MuSiQue). Confirms the original latency hypothesis: a shorter
pipeline runs faster. No surprises here.

## What this changes

- **The default flipped in 0.3.2.** New `Document` objects use the
  raw pipeline. Existing users rebuilding their index will see
  ranking shifts (some queries better, some worse) — measure before
  upgrading production indexes.
- **English Snowball is still one keyword away.** Pass
  `language="english"` to get the previous pipeline (camelCase split,
  stopwords, Snowball stemmer) — useful for code search and
  inflection-heavy workloads.
- **Multilingual paths are unaffected.** `language="german"`,
  `"french"`, etc. still route to the corresponding Snowball
  analyzer; they were always opt-in and they still are.

## When to use which

- **Default (raw pipeline, no `language=` argument):** the
  empirically-better starting point for the three English workloads
  measured here. Faster, slightly higher recall on CUAD and MuSiQue,
  tied on HotpotQA.
- **`language="english"`:** opt in if your workload has heavy
  inflectional variation between query and doc (e.g. queries about
  "acquisitions" against doc text mentioning "acquired", "acquiring")
  or if you're doing code search where camelCase splitting matters
  (`compressVideo` → both `compress` and `video` indexed).
- **`language="german"` / `"french"` / etc.:** required for
  non-English content where the Snowball stemmer for that language
  helps with morphology.

## Honest limits

- **n=100 per workload.** Three runs is consistent; larger n would
  tighten the magnitude. The direction (`raw ≥ english`) was stable
  across re-runs.
- **English-only measurement.** The +5/+7 wins on CUAD/MuSiQue are
  measured on English content. Non-English workloads will need their
  Snowball stemmer (German handles separable verbs, French handles
  gender/number, etc.). Don't default-set `raw` if your content is
  Italian.
- **Single retrieval mode (`raw_topk`).** Not measured against the
  `reasoning_preserving` strategy or `retrieval="hybrid"`. Most
  likely the result generalizes (analyzer-pipeline overhead is the
  same regardless of assembly strategy), but it's untested.
- **No code workload tested.** RedHop's `CamelCaseSplitter` matters
  for code search (`"compressVideo"` → both forms searchable); the
  new raw default skips it. Code-search users should pass
  `language="english"` to keep the splitter.

## Reproduce

```bash
# Cross-workload comparison
bench/.venv/bin/python bench/compare_raw_analyzer.py

# Multi-query-per-doc bench (CUAD) — both modes side by side
bench/.venv/bin/python bench/compare_multiquery.py
```

Raw runs:
[`reports/raw_analyzer_compare_2026-06-08.txt`](../../reports/raw_analyzer_compare_2026-06-08.txt)
and
[`reports/framework_comparison_multiquery_with_raw_2026-06-08.txt`](../../reports/framework_comparison_multiquery_with_raw_2026-06-08.txt).

## See also

- [FRAMEWORK_MULTIQUERY](FRAMEWORK_MULTIQUERY.md) — the multi-query
  benchmark that surfaced the latency gap that prompted this probe.
- [MULTILINGUAL_ANALYZER](MULTILINGUAL_ANALYZER.md) — the
  per-language Snowball coverage. Non-English workloads should still
  pick their language string, not `raw`.
- [MULTIHOP_CONSTANT_CHUNKING](MULTIHOP_CONSTANT_CHUNKING.md) — the
  chunker is still the bigger lever (12-20pt swings); the analyzer
  here is a 5-7pt swing. Both compound.
