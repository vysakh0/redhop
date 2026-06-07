# Multilingual `analyze_query_set` + `drop_template_terms` — confirmed across 5 languages

> **Status:** **Confirmed across French, German, Spanish, Chinese, and
> Japanese** with synthetic translated CUAD-shape queries (probe at
> `crates/examples/examples/multilingual_query_set_probe.rs`).
>
> **TL;DR:** [QUERY_SET_ANALYZER](QUERY_SET_ANALYZER.md) shipped with
> "English tokenization assumed" as an explicit limit. That limit
> turned out to be more conservative than necessary — the analyzer
> works on whitespace-separated non-English scripts by construction
> (Unicode `is_alphanumeric` covers Latin accented chars correctly),
> and CJK templates work too via punctuation-bounded phrase
> segmentation. `drop_template_terms` was strict-Latin-only by
> default; a small script-aware refactor (substring removal for
> Han/Hiragana/Katakana/Hangul/Thai/Lao terms, whitespace-token
> matching for everything else) makes the full detect → strip → A/B
> workflow work on CJK too without regressing the Latin behavior.

## Question

[QUERY_SET_ANALYZER.md](QUERY_SET_ANALYZER.md) explicitly listed
"English tokenization. Boilerplate detection is over lowercased
alphanumeric tokens. Non-English workloads will tokenize correctly
but the action copy mentions English findings. Multilingual probe is
a future arc."

The honest question: **how much of that is actually broken vs just
not measured?** The answer determines whether multilingual support
is a doc note or a code change.

## Setup

The probe at
[`crates/examples/examples/multilingual_query_set_probe.rs`](../../crates/examples/examples/multilingual_query_set_probe.rs)
runs the same two-failure-mode test that validated the English
heuristic (true-positive on templated + false-positive on diverse)
across five languages:

- **French / German / Spanish** — Latin script with diacritics,
  whitespace-separated, accents on alphanumeric characters.
- **Chinese (Simplified)** — Han characters, no whitespace, CJK
  punctuation marks 「」（）、。 as structural delimiters.
- **Japanese** — mixed Hiragana / Katakana / Han / Kanji, no
  whitespace, the same CJK punctuation family.

Each language gets a 6-query templated set (translation of the CUAD
template with the placeholder X varied) and an 8-query diverse set
(unrelated natural-language questions). The synthetic translations
are not real workload data; they're sufficient to characterize the
**mechanism**, not the precise share / cost magnitudes on a real
non-English workload.

## Results

| language | templated detected? | diverse stayed quiet? | template_word_share | boilerplate (top 5) |
| -------- |:--------------------:|:---------------------:| -------------------:| ------------------- |
| French   | ✓ | ✓ | 0.919 | `avocat, cas, ce, contrat, de` |
| German   | ✓ | ✓ | 0.950 | `anwalt, auf, beziehen, die, dieses` |
| Spanish  | ✓ | ✓ | 0.905 | `abogado, con, contrato, de, deberían` |
| Chinese  | ✓ | ✓ | 0.800 | `如有, 应由律师审核的部分, 相关的, 请标注本合同中与` |
| Japanese | ✓ | ✓ | 0.833 | `に関連する, もしあれば, を示してください, 弁護士の確認が必要な部分, 本契約のうち` |

All five languages register `is_templated=true` on the templated set
and `is_templated=false` on the diverse set. The diverse sets in every
language produce zero boilerplate terms — the conservative threshold
(`≥ 80% of queries AND ≥ 2 terms`) holds across scripts.

## Why CJK works (the part I expected to be broken)

The original `analyzer_tokens` is `s.to_lowercase().split(|c: char|
!c.is_alphanumeric()).filter(|w| w.len() > 1)`. My prior reading: on
Chinese/Japanese, every character is `is_alphanumeric=true` (Han,
Hiragana, Katakana all fall under Unicode "Letter" categories), so a
CJK query without whitespace collapses to one giant token. I expected
zero boilerplate detection on CJK.

The reading was wrong because **real CJK queries are full of
punctuation**: 「」「」（）、。 — quotation marks, parentheses, full-width
commas and periods. These all satisfy `!is_alphanumeric()` and act as
split points. The result is **phrase-level** segmentation:

```
请标注本合同中与「文档名称」相关的、应由律师审核的部分（如有）。
                ─┬─                 ─┬─                ─┬─
                 │                   │                  │
       segment 1 │      segment 2    │     segment 3    │
请标注本合同中与   文档名称      相关的、应由律师审核的部分   如有
```

Each segment is a "token" for the analyzer. Templated CJK queries
that all vary only inside `「…」` share the surrounding segments
verbatim, so the segments register as boilerplate and `is_templated`
fires correctly.

The catch: the boilerplate terms surface as **phrases** (e.g.,
`请标注本合同中与`), not words. That's fine for the analyzer; it
broke `drop_template_terms`.

## The `drop_template_terms` script-aware fix

Before:

```rust
// strict whitespace-token matching — CJK queries have no whitespace,
// so split_whitespace() returned the whole query as one token, which
// never matched any boilerplate phrase
query.split_whitespace().filter(|tok| {
    let key = tok.chars().filter(|c| c.is_alphanumeric())
                 .flat_map(|c| c.to_lowercase()).collect::<String>();
    !stop.contains(&key)
}).collect::<Vec<_>>().join(" ")
```

After:

```rust
// Partition boilerplate by script: terms with any Han/Hiragana/Katakana/
// Hangul/Thai/Lao/Khmer/Burmese character → phrase removal; everything
// else → existing whitespace-token matching (word-boundary safe).
let (phrase_terms, token_terms): (Vec<_>, Vec<_>) = boilerplate
    .iter().copied()
    .partition(|t| t.chars().any(is_no_space_script));

// Phase 1: substring removal, longest first, for phrase-style terms.
let mut result = query.to_string();
let mut sorted = phrase_terms.clone();
sorted.sort_by(|a, b| b.len().cmp(&a.len()));
for term in &sorted { result = result.replace(term, ""); }

// Phase 2: original whitespace-token filter for word-style terms.
// (Identical to before — no Latin behavior change.)
...
```

Why the partition matters:

- **Latin word-boundary safety preserved.** A boilerplate `"of"` must
  not erase the `"of"` inside `"office"`. The Phase 2 filter still
  applies word-boundary-style matching (via `split_whitespace`) for
  any term that contains no no-space-script character — which is every
  Latin-script term. Test:
  `drop_template_terms("the office is open", ["of", "the"])` →
  `"office is open"`, *not* `"fice is open"`.
- **CJK phrase removal works.** Boilerplate phrases coming from CJK
  templates have at least one Han/Hiragana/Katakana character, so
  they're substring-removed. Test on the Chinese query above:
  `「文档名称」、（）。` — boilerplate phrases gone, discriminator
  preserved.

`is_no_space_script` covers **Han, Hiragana, Katakana, Hangul, Thai,
Lao, Khmer, Myanmar** — the scripts that conventionally use no
whitespace between words. Arabic, Hebrew, Cyrillic, Greek, and the
Brahmic scripts that *do* use whitespace stay on the word-boundary
path.

## API contract — what's confirmed and what's still limit

**Confirmed** (probe + tests + bindings):

- `analyze_query_set` correctly detects templated workloads across
  Latin script (French / German / Spanish), Han (Chinese), and mixed
  CJK (Japanese). False-positive guarantee on diverse natural-language
  queries holds in every language.
- `drop_template_terms` strips boilerplate correctly in those same
  five languages, **with** the Latin word-boundary safety preserved
  (no regression on the English path).
- All three bindings (Rust, Python, Node) exercise CJK and
  Latin-word-boundary cases in their tests; binding parity is
  preserved.

**Still a limit** (honest scope):

- **Whitespace-required scripts that aren't in the no-space-script
  list.** Arabic / Hebrew / Cyrillic / Greek / Devanagari templates
  haven't been measured. The mechanism prediction (whitespace
  tokenization works → analyzer works → drop_template_terms uses the
  word-boundary path safely) suggests they should work, but it isn't
  measured.
- **Templated CJK queries with NO punctuation.** If someone builds a
  template like `请告诉我X的价格` (no quotation marks, no commas) and
  varies X, the surrounding phrase `请告诉我...的价格` has no internal
  split point. The analyzer would see one big token per query and miss
  the boilerplate. Real-world templated workloads almost always have
  some structural punctuation (a quoted placeholder, a `Details:` lead-
  in, a numbered prefix), so this is a corner case — but it's a real
  one for purely-prose templates.
- **`suggested_action` copy is English-only.** The string returned in
  `QuerySetReport.suggested_action` is English regardless of the query
  language. Acceptable today; could be localized in a future arc if a
  user needs it.
- **Threshold tuning.** The 0.80 boilerplate-DF threshold and 0.50
  `is_templated` floor were chosen on English workloads. The
  multilingual probe shows they hold for templated vs diverse on the
  five tested languages with shares 0.80–0.95 on templated and 0.00 on
  diverse — comfortably inside the bands — but precise tuning on a
  real non-English production workload could reveal edge cases.

## Reproduce

```bash
cargo run -p redhop-examples --example multilingual_query_set_probe --release
```

Runs in under a second. No models, no embeddings, no retrieval.

## What this changes

- New public Rust API surface: nothing new — `analyze_query_set` and
  `drop_template_terms` keep the same signatures.
- Internal change to `drop_template_terms` (script-aware partition);
  Latin behavior is byte-for-byte identical, CJK behavior now works.
- 3 new Rust unit tests pinning the CJK phrase path, the Japanese
  mixed-script path, and the Latin word-boundary regression.
- 2 new Python tests (test_query_set_analyzer.py) and 2 new Node
  assertion blocks (query_set_analyzer.cjs) mirroring the contract
  through the FFI.
- New cross-language probe harness as the reproducible evidence.
- `docs/findings/MULTILINGUAL_ANALYZER.md` (this file) + index entry.

The user-visible promise: **the detect → strip → A/B workflow
documented in `CHOOSING_A_CONFIG.md` works for any templated
workload whose queries have whitespace and/or punctuation between
the template phrases**, which is essentially all real-world
templated workloads regardless of language.
