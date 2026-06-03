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

### Per-language stemming via the public `Analyzer` plugin (in-tree, no fork)

`crate::analyzer::SnowballAnalyzer` ships pre-baked constructors for all
18 Snowball Porter2 languages (`arabic, danish, dutch, english, finnish,
french, german, greek, hungarian, italian, norwegian, portuguese,
romanian, russian, spanish, swedish, tamil, turkish`). Swap the default
English analyzer with `Document::with_analyzer`:

```rust
use std::sync::Arc;
use redhop::analyzer::SnowballAnalyzer;

let mut doc = redhop::Document::from_text("library", "ich habe viele Bücher")?
    .with_analyzer(Arc::new(SnowballAnalyzer::german()));
let ctx = doc.context("Buch")?;   // finds the chunk via German morphology
```

ONE analyzer drives BOTH the BM25 retriever and the grounding scorer —
that's the architectural guarantee of the `Analyzer` trait. There's no
risk of the two layers disagreeing on what "the same term" is (the bug
class we fixed by hand four times through 0.1.3-0.1.4 is now structurally
impossible).

From Python:

```python
doc = redhop.Document.from_text("ich habe viele Bücher", language="german")
ctx = doc.context("Buch")     # finds it via Snowball German
```

From Node:

```javascript
const doc = Document.fromText("ich habe viele Bücher", { language: "german" });
const ctx = doc.context("Buch");
```

Unknown language strings ERROR (we don't silently fall back to English —
a typo'd `"germann"` should surface). For a CJK tokenizer or a custom
pipeline, implement the `crate::analyzer::Analyzer` trait yourself and
pass `Document::with_analyzer(Arc::new(MyAnalyzer))` directly.

### Per-language stopword lists

`SnowballAnalyzer::english()` ships with the curated list from
`crate::context::STOPWORDS`. The other 17 builtins ship with **empty**
stopword lists — we don't have curated lists per language, and shipping
uncalibrated ones violates the measure-don't-overclaim discipline.
Attach your own:

```rust
let german = SnowballAnalyzer::german()
    .with_stopwords(vec!["der".into(), "die".into(), "das".into(), /* … */]);
```

Snowball publishes per-language lists at
<https://snowballstem.org/algorithms/>; the ones from `nltk.corpus.stopwords`
are also a reasonable starting point.

### CJK tokenization

For Japanese, the `lindera` or `kuromoji-rs` crates provide IPADIC-backed
morphological analysis (segmentation + lemmatization). For Chinese,
`jieba-rs`. For Korean, `lindera`'s ko-dic. Each adds ~1-10 MB of
dictionary data and a real per-language analyzer.

Implement the `crate::analyzer::Analyzer` trait (two methods:
`build_text_analyzer()` returning a Tantivy `TextAnalyzer`, and
`tokens()` returning the term list for the grounding scorer), then
attach via `Document::with_analyzer(Arc::new(MyCjkAnalyzer))`. The
default `tokens()` impl on the trait delegates to `build_text_analyzer()`
so you only have to wire the Tantivy side. We don't ship CJK builtins
because the dictionary data is large and per-language; if you build one
worth sharing, send a PR.

### What we won't auto-detect

We don't ship a language-detection step. A document with mixed
French/English/Japanese content would need to be split by your loader
before being handed to redhop, with each segment tagged via
`chunk.metadata["language"]`. The Tantivy multi-field analyzer setup
would then key the analyzer by language.

## How the plugin guarantees parity

The `Analyzer` trait has two methods: `build_text_analyzer()` returns a
Tantivy `TextAnalyzer` (used by BM25), and `tokens(text)` returns the
list of search terms (used by the grounding scorer). The default
`tokens()` impl runs the analyzer returned by `build_text_analyzer()`
and collects its output — so the BM25 side and the grounding side go
through a **single source of truth**. There's no way to override one
without the other.

This kills, structurally, the entire class of bugs we kept finding
through 0.1.3-0.1.4 — stemming, stopwords, camelCase, ASCII-folding
mismatches between the two layers. They now follow from the architecture.

## Calibration disclaimer

Pre-baked non-English builtins (`german`, `french`, etc.) get you the
Snowball Porter2 stemmer for that language, full stop. We have **no
eval corpus** for non-English — we can't tell you whether the ranking
quality matches English Snowball's on HotpotQA / MuSiQue / CUAD.
Empirically: `Bücher` ↔ `Buch` unifies under the German analyzer in our
T41 test, but we can't promise that ranking on a real German legal
contract is as good as English Snowball on a real English one.

For demanding use cases, supply your own analyzer (implement the
`Analyzer` trait) and benchmark on your own corpus.

## Pinning tests

`crates/redhop/tests/quality_suite.rs`:

- **T37-T40** lock in the basic non-English behaviors (Spanish exact
  match, German ß-folding, French accent parity, CJK substring).
- **T41-T44** exercise the `Analyzer` plugin end-to-end: German
  morphology via `with_analyzer(German)`, French verb inflections via
  `with_analyzer(French)`, the with_analyzer call swaps both layers
  (proven by forcing the grounding scorer to run), unknown language
  strings error rather than silently falling back to English.

If a future change accidentally regresses any of these, CI catches it.
