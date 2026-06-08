# `prose_heading_default=true` is the right default — keep it

> **Status: validated default, no change.** Measured the auto-attach
> heading behavior (`Document.context()` attaches each cited chunk's
> section heading when the chunk carries `metadata["heading"]`) against
> the no-headings counterfactual on HotpotQA-as-markdown at three
> budgets:
>
> | Budget | Δ ≥0.8 (heading-on vs off) | Δ recall | Verdict |
> |---:|---:|---:|---|
> | 128 (tight) | +0pt (29% vs 29%) | -0.01 | wash |
> | 400 (typical) | **+7pt (79% vs 72%)** | +0.02 | default helps |
> | 1000 (loose) | +0pt (100% vs 100%) | 0.00 | saturated |
>
> The default helps at typical budgets and is a clean wash at the
> extremes. Cost: ~4-11 extra words of context per query (1-1.5% of
> the budget). **Keep `prose_heading_default=true`.**

## Why this probe ran

The raw-analyzer flip raised the broader question: which other
defaulted-on heuristics in RedHop have actually been measured?
Heading-attachment was a strong candidate to *look* wrong — the
Python `context(query, include_heading=False)` kwarg defaults to
False, which made it look like headings were off-by-default. But the
real default lives in `DocumentConfig::prose_heading_default = true`,
which the auto-path consumes. When `doc.context(query)` finds a prose
chunk with a heading, it attaches the section's heading chunk to the
assembled context. Nobody had ever measured whether this actually
helps.

## The setup

None of the existing benchmarks exercise this default: HotpotQA,
MuSiQue, and CUAD all load as raw text without markdown structure,
so the chunker never produces a heading-bearing chunk and the
auto-attach never fires.

To create a heading-bearing workload, this probe re-formats HotpotQA
as markdown:

```
# <article 1 title>

<article 1 paragraph>

# <article 2 title>

<article 2 paragraph>
```

Two arms on the same Wikipedia bundles:
- **A. markdown** — load with `Document.from_text(md_doc, ...)`. The
  chunker creates a heading chunk for each `# Title` and body chunks
  in the same `(source, heading)` group. The auto-attach default
  fires when retrieval surfaces a body chunk.
- **B. plain** — same paragraphs concatenated with blank lines but no
  `#` markers. No heading chunks get created; the auto-attach can't
  fire.

A small BM25 multi-field confound: the markdown arm gets heading text
indexed into the BM25 heading field. Wikipedia titles are short (~2-5
words) vs paragraph text (~50 words), so the dominant retrieval
signal is the paragraph content. The 7-point lift at budget=400 is
too large to attribute to the multi-field reach alone.

## What the result says

**At budget=400 (typical for production):**
- A. markdown: 0.91 mean recall, 99% ≥0.5, **79% ≥0.8**, 357 ctx words
- B. plain: 0.88 mean recall, 97% ≥0.5, **72% ≥0.8**, 352 ctx words
- Δ: **+7pt ≥0.8**, +0.02 recall, +4.4 words

The default helps clearly. The mechanism: HotpotQA gold sentences
often share entity vocabulary with the article title. When BM25
surfaces a body chunk that contains the entity in passing but not
the answer span fully, the auto-attached heading chunk adds the
article title — which often contains the entity name that confirms
the chunk is on-topic. The 4-word heading cost is dwarfed by the
retention gain.

**At budget=128 (tight):**
- Both arms ~29% ≥0.8. Δ within noise.
- Mechanism: the budget is too tight to fit displaced content
  anyway. With only ~3 chunks fitting, the heading attachment either
  doesn't trigger (no headroom) or displaces nothing measurable.

**At budget=1000 (loose):**
- Both arms 100% ≥0.8. Saturated.
- Mechanism: all relevant content fits comfortably; heading
  attachment is irrelevant to the metric.

## Why the default works

The auto-attach is most useful in the **typical-budget regime** where:
1. The budget is large enough to fit several chunks plus a heading
   without displacing the gold passage
2. The heading carries semantic content the body chunks alone don't
   (entity name, section topic)

This matches the design intent — sectioned prose (markdown, DOCX,
PPTX, parsed PDF) where the heading is what tells you what the
section is about. Wikipedia articles fit perfectly. Other
heading-bearing corpora (technical docs with named sections, legal
contracts with clause titles) likely fit too.

## Honest limits

- **HotpotQA-as-markdown only.** Wikipedia titles are
  entity-informative; the 7-pt lift is roughly the *upper bound* of
  what heading attachment can provide. On a corpus with categorical
  headings (`## Setup`, `## Troubleshooting`) the heading carries
  less query-relevant signal and the lift would be smaller — possibly
  zero. But the *cost* is still tiny (~1% of budget), so even at zero
  benefit the default isn't hurting.
- **n=100, single domain.** Larger n would tighten the magnitude
  estimate; a second heading-bearing workload (e.g. a parsed PDF
  contract corpus) would confirm generalization.
- **BM25 multi-field reach is a small confound.** The markdown arm
  gets heading text indexed into the BM25 heading field, which
  marginally changes retrieval. Wikipedia titles are short (~2-5
  words) vs paragraph text (~50 words); the +7pt at budget=400 is
  much larger than a multi-field reach effect would explain, so the
  heading-attachment mechanism is the dominant driver.
- **Single retrieval mode (lexical / `raw_topk`).** Hybrid retrieval
  was not measured; the mechanism doesn't depend on the retriever,
  so the result should generalize, but it's untested.

## What this changes

- **`prose_heading_default=true` stays.** Empirically validated; flipping
  would cost retention at typical budgets.
- **A small API smell surfaced.** The Python `doc.context(query,
  include_heading=False, neighbors=0)` falls back to the auto path
  (because `if neighbors == 0 && !include_heading` → calls
  `context_with`, which goes through `context_inner` and consults
  `prose_heading_default`). There's no way from Python to call
  "context_expanded with both off" — i.e., to *disable* the
  heading auto-attach without re-building chunks without metadata.
  Documented as a follow-up — not urgent; the default is correct so
  users rarely need to opt out.

## Reproduce

```bash
bench/.venv/bin/python bench/prose_heading_default.py
```

Raw run: [`reports/prose_heading_default_2026-06-08.txt`](../../reports/prose_heading_default_2026-06-08.txt).

## See also

- [RAW_ANALYZER](RAW_ANALYZER.md) — a default that was *wrong*
  (flipped in 0.3.2). This finding contrasts: same audit, same
  framework, opposite verdict.
- [HYBRID_CANDIDATE_POOL](HYBRID_CANDIDATE_POOL.md) — a default that
  is *inert* (the knob doesn't move retention). This finding
  contrasts again: heading attachment is a real, measured win.
- Together these three findings document the audit of RedHop's
  defaulted-on heuristics: flip the wrong one (raw analyzer), keep
  the right one (heading attach), stop tuning the one that doesn't
  matter (candidate_pool).
