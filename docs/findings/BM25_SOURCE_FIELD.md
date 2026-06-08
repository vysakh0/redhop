# BM25 multi-field reach is free retention — keep it

> **Status: validated default, no change.** RedHop's BM25 indexes
> three fields per chunk (`text`, `source`, `heading`) and queries all
> three with equal weight. This probe varied only the `source` value
> on identical Wikipedia content (n=100) to test whether the
> multi-field reach is a real signal, noise-vulnerable, or both:
>
> | source configuration | ≥0.8 | Δ vs neutral |
> |---|---:|---:|
> | A. signal (`<title>.md`) | **82%** | **+4** |
> | B. generic (`doc.txt`) | 78% | (control) |
> | C. noisy (random hex hash) | 78% | 0 |
>
> Signal-bearing sources are worth **+4pt ≥0.8** — and noisy sources
> cost **nothing**. The multi-field reach is one of the cleanest
> defaults in RedHop: a net win when the path carries information, a
> clean no-op when it doesn't.

## Why this probe ran

The defaulted-on heuristics audit (kicked off by
[RAW_ANALYZER](RAW_ANALYZER.md)) flagged BM25's multi-field reach as
worth checking. The design intent — a query for `"auth"` should reach
a chunk in `src/auth.rs` even when the chunk text doesn't say
"auth" — is reasonable and pinned by
`quality_suite::t08_filename_reachable_via_source_field`. But that
test only proves the field works on signal-bearing paths; it doesn't
measure whether the field *hurts* on noisy paths (random hashes,
opaque server-generated filenames, internal IDs).

The hypothesis going in: signal-bearing sources should help, but noisy
sources might displace useful ranking signal by sitting alongside the
text field with vocabulary that's never in the query.

## The setup

Three configurations of the same HotpotQA Wikipedia content, n=100.
Same text per chunk, no heading metadata, only the `source` value
varies:

- **A. signal-bearing source** — `source="<article title>.md"`. The
  path contains the article entity name. A query about the entity will
  match the source field.
- **B. generic source** — `source="doc.txt"` shared across every
  chunk. The source field has zero discriminating power; every chunk
  matches every query equally.
- **C. noisy source** — `source="<16-hex-char hash>.txt"`. The path
  carries vocabulary that's never in any query.

Each query runs the same retrieval + assembly across all three
configs. Differences come purely from how the source field
contributes to BM25 scoring.

## Result

| arm | mean recall | ≥0.5 | ≥0.8 | p50 ms |
|---|---:|---:|---:|---:|
| A. signal source | 0.93 | 99% | **82%** | 2.1 |
| B. generic source | 0.91 | 97% | 78% | 2.1 |
| C. noisy source | 0.91 | 97% | 78% | 2.0 |

**A > B = C.** Signal lifts retention by 4 points; noise does
nothing. The "best case" outcome of the four interpretive
possibilities the probe was designed to discriminate between.

## Why noise is free

Random hash strings as source values contribute terms that the BM25
analyzer indexes — but the query has no overlap with hex hashes, so
the inverse-document-frequency score for those terms against any
real query is zero. Tantivy's query parser only retrieves a
non-zero score from a field when the query term appears in that
field's posting list. Noise tokens populate the dictionary but never
contribute to scoring on real queries.

The slight latency win on arm C (-0.1ms) is within noise.

## Why signal is +4pt

HotpotQA gold sentences typically share the article subject's name
(the title). When BM25 surfaces a body chunk that mentions the
subject in passing but doesn't fully match the gold answer span, the
source field — which contains the title verbatim — adds a small
ranking boost that pulls the right article's chunk above
distractors.

This is the same mechanism as
[PROSE_HEADING_DEFAULT](PROSE_HEADING_DEFAULT.md) (entity-bearing
headings help retention), realized through a different field. They
compound: both fields carry the same title text on real
markdown-loaded files.

## What this changes

- **Multi-field reach stays.** Empirically validated as net-positive
  with no measurable downside on noisy paths.
- **No opt-out flag needed.** There's nothing to opt out of —
  worst-case performance equals the no-source-field baseline. Adding
  a flag would be all cost (config surface area) and no benefit.

## Honest limits

- **One workload (HotpotQA).** A code-search corpus with realistic
  filenames (`src/auth.rs`, `src/render.tsx`) would likely show a
  similar or larger lift since paths carry symbol-name signal. A
  CUAD-style legal corpus where each source is a contract filename
  (`AMENDMENT_NO_2_v3_final.docx`) would test the "noisy" arm more
  realistically than random hex. Both are worth checking if the
  result here is later questioned.
- **n=100, single retrieval mode** (`raw_topk` lexical). Hybrid
  retrieval routes through the same BM25 first stage, so the result
  should generalize.
- **HotpotQA titles are *short*** (2-5 words). A workload with very
  long source paths (deep directory trees, full URLs) would test
  whether long noisy sources start to *displace* signal via
  field-length-normalization effects. Untested; probability of a
  flip: low, because the unmatched terms still don't contribute
  positive score.

## Reproduce

```bash
bench/.venv/bin/python bench/bm25_source_field.py
```

Raw run: [`reports/bm25_source_field_2026-06-08.txt`](../../reports/bm25_source_field_2026-06-08.txt).

## See also

- [PROSE_HEADING_DEFAULT](PROSE_HEADING_DEFAULT.md) — companion
  finding on the heading field. Both fields are entity-carriers on
  the same kind of corpus; the +4 here and +7 there compound on
  real markdown-loaded files.
- [RAW_ANALYZER](RAW_ANALYZER.md) — the audit's positive flip.
  Contrast: that was a default that needed changing; this is one
  that's been right all along.
- [HYBRID_CANDIDATE_POOL](HYBRID_CANDIDATE_POOL.md) — the audit's
  inert knob. Three audit outcomes now documented: flip the wrong
  one, keep the right ones, ignore the inert one.
