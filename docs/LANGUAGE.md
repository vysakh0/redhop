# Language support

The honest scope of redhop's lexical tier, by language family. Empirically
verified against the same probes used to write `crates/redhop/tests/quality_suite.rs`.

## What the analyzer pipeline does

The BM25 analyzer (and the parallel grounding scorer in
`crate::context::normalize`) applies, in order:

1. **Tokenize** on Unicode whitespace + punctuation
   (`tantivy::SimpleTokenizer` / `unicode_segmentation::unicode_words`).
2. **Drop overlong tokens** (> 40 chars).
3. **Split camelCase / PascalCase / letter↔digit** boundaries
   (custom `CamelCaseSplitter`).
4. **ASCII-fold combining diacritics** (`tantivy::AsciiFoldingFilter` /
   `unicode_normalization::nfkd`). Handles `é` → `e`, `ñ` → `n`, `ß` →
   `ss`, `ø` → `o`, plus ~100 more.
5. **Lowercase** (`LowerCaser`).
6. **Drop English stopwords** (the, and, is, of, in, …).
7. **Stem** with Snowball Porter2 (English).

Steps 1-5 are language-agnostic. Steps 6 and 7 are English-only.

## What works for what

| Family | Step that breaks | Practical impact |
|---|---|---|
| **English** | — | Full power: stemming, stopwords, ASCII folding, case splits |
| **Western European Latin** (French / German / Spanish / Portuguese / Italian / Dutch / Polish) | Step 6 (their stopwords stay as tokens) + Step 7 (their morphology doesn't unify) | Exact matches work. `café` ↔ `cafe` works (Step 4). `Süßigkeit` ↔ `Sussigkeit` works. But `Bücher` ≠ `Buch`, `caminos` ≠ `camino`, `courait` ≠ `court`. |
| **Other Latin-script** (Turkish / Vietnamese / Czech / Romanian) | Steps 4 + 6 + 7 | Some Step-4 folds work (`č` → `c`), some don't (`ı` is its own letter in Turkish, treating it as `i` is locale-wrong). |
| **Cyrillic** (Russian / Ukrainian / Bulgarian) | Step 4 may strip useful info; Steps 6 + 7 don't apply | Tokens index, no morphology |
| **CJK** (Chinese / Japanese / Korean) | Step 1 doesn't word-segment without explicit spaces | **Substantially broken**: real CJK content has no inter-word whitespace. `圧縮アルゴリズム` becomes one token; a query for `圧縮` doesn't reach it. Use a CJK-aware tokenizer instead (see below). |
| **Arabic / Hebrew** | Steps 4 (irrelevant), 6, 7 don't apply; Step 1 works | Tokens index, no morphology, no diacritic normalization |
| **Indic** (Devanagari / Tamil / Bengali) | Step 1 segments on whitespace (works for many); Steps 6 + 7 don't apply | Exact matches work, no morphology |

## If you need better non-English support

The places to extend:

### Per-language stemming + stopwords

`rust-stemmers` (already in our dep tree) ships Snowball Porter2 stemmers
for: **Arabic, Danish, Dutch, English, French, German, Greek, Hungarian,
Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish,
Tamil, Turkish.**

To use a different one, replace `Stemmer::new(Language::English)` in
`crates/redhop/src/retrieval/bm25.rs::Bm25Retriever::new` with the target
language. You'll also want a matching stopword list — Snowball publishes
one per language at <https://snowballstem.org/algorithms/>.

`crate::context::normalize` would need a parallel change so the BM25 side
and the grounding scorer stay in lockstep (this is the recurring "same
tokenizer/scorer contract" — see CHANGELOG 0.1.3-0.1.4).

### CJK tokenization

For Japanese, the `lindera` or `kuromoji-rs` crates provide IPADIC-backed
morphological analysis (segmentation + lemmatization). For Chinese,
`jieba-rs`. For Korean, `lindera`'s ko-dic. Each adds ~1-10 MB of
dictionary data and a real per-language analyzer.

Wire them as a Tantivy `TextAnalyzer` alongside `STEM_ANALYZER` and
route based on a `language` field on the chunk's metadata. Out of scope
for the 0.1.x line — file an issue if you need this.

### What we won't auto-detect

We don't ship a language-detection step. A document with mixed
French/English/Japanese content would need to be split by your loader
before being handed to redhop, with each segment tagged via
`chunk.metadata["language"]`. The Tantivy multi-field analyzer setup
would then key the analyzer by language.

## Why we shipped this way

Two reasons:

1. **Validation cost.** A per-language analyzer change needs a per-language
   eval corpus to know it's actually helping. We have evidence on English
   (HotpotQA, MuSiQue, CUAD). We don't have eval data on Spanish or
   German content, so adding stemmers without evaluation is the
   "ship-a-feature-we-can't-measure" trap.

2. **Bounded scope.** The 0.1.x line ships an English-focused RAG runtime
   with explicit, observable behavior. Adding language pipelines without
   strong demand turns the library into a knobs-and-config piece of work
   we'd rather not be on the hook for. When two users ask for German,
   that's the trigger.

## Pinning tests

`crates/redhop/tests/quality_suite.rs` (T37-T40 region) locks the
current degraded-but-functional behavior in place. If a future change
accidentally regresses (e.g., breaks Spanish exact-word lookup or
German ß-folding), CI catches it.
