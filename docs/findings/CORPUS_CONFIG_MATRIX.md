# Corpus × Config Matrix — when to reach for which knob

> **Hypothesis:** the lexical default is conservative; semantic + rerank +
> structural expansion materially help on real-world docs (contracts,
> handbooks, financial reports).
> **Status:** **Mostly falsified.** Lexical is the right default for 5 of 6
> document shapes we tested. Hybrid + structural expansion helps **only** on
> the legal contract (multi-section + regional overrides); cross-encoder
> rerank adds no measurable value on any of them. Structural expansion
> (`include_heading`, `neighbors`) *hurts* on the engineering handbook.
> **Setup:** 5 industry-style PDFs (a 50-page MSA, a 10-page API ref, a
> 41-page financial report, a 44-page incident runbook, a 101-page
> engineering handbook) + a `from_folder` cross-PDF routing test —
> 121 labeled queries total. Substring-or-section grader (legal PDFs use
> long-form numerals and full names — `"twelve (12) months"` rather than
> `"12 months"`, `"Singapore International Arbitration Centre"` not
> `"SIAC"` — so naive substring grading is misleading on these corpora).
> **Headline:** **lexical wins 5/6; semantic helps only on highly-structured
> contracts with near-duplicate clauses (regional overrides); rerank wins
> 0/6.**

---

## Why this finding exists

A colleague's hand-eval reported "all three tiers tied at 6/8" on the MSA
and called out polysemy ("vendor", "settle") as a failure mode. Reproducing
it surfaced two artifacts: the grader was overly literal (missed `"twelve
(12) months"` vs `"12 months"`), and the polysemy probes were intentionally
one-word queries. With both fixed, the question became: **what does each
knob actually buy on real, labeled corpora?**

## The matrix

Pass rate by config × corpus. `e<n>` = `<n>` queries returned an empty
context. Best per row in **bold**.

| Corpus | lexical (default) | hybrid+bge-small | + heading + neighbors | + cross-encoder rerank |
|---|---|---|---|---|
| **MSA** (legal contract, 50p)        | 26/29 · 89.7% · e1 | 26/29 · 89.7% · e2 | **27/29 · 93.1% · e2** | 27/29 · 93.1% · e2 |
| **API reference** (10p)              | **17/18 · 94.4%** | 17/18 · 94.4% | 17/18 · 94.4% | 17/18 · 94.4% |
| **Financial report** (41p)           | **18/18 · 100%** | 18/18 · 100% | 18/18 · 100% | 18/18 · 100% |
| **Incident runbook** (44p)           | **18/18 · 100%** | 18/18 · 100% | 18/18 · 100% | 18/18 · 100% |
| **Engineering handbook** (101p)      | **30/31 · 96.8%** | 30/31 · 96.8% | 29/31 · 93.5% | 29/31 · 93.5% |
| **Cross-PDF routing** (5-file folder) | **8/8 · 100%** | 8/8 · 100% | 8/8 · 100% | 8/8 · 100% |

## Reading the matrix

**Lexical is the right default.** On 5 of 6 corpora, the no-model
lexical tier (BM25, sub-100ms warm, zero dependencies) ties or beats every
other configuration. **There is no general accuracy win to "climbing"** to
hybrid + semantic on these corpora — for code, API refs, runbooks,
financial reports, and even a 101-page handbook, BM25 alone wins.

**Hybrid + structural expansion is a +1-query lift on the MSA only.** The
MSA has near-duplicate clauses (8 governing-law clauses: 1 main + 7
regional overrides), so heading-aware retrieval (`include_heading=True,
neighbors=1`) genuinely disambiguates "EU law" → §G.1 Ireland from "main
governing law" → §9.1 Delaware. This shape is real — it's what RedHop
ships heading and neighbors for — but it generalizes narrowly. The same
expansion *hurt* the handbook (97% → 94%) because clean, well-titled
chapters don't benefit from added neighbors; the extra context dilutes.

**Cross-encoder rerank added zero pass-rate** across all 6 corpora.
It *did* add 5–10× query latency. On corpora where the failures are
"empty context because BM25 returned nothing" (the MSA #22, #24
paraphrases), rerank can't help because it reranks an empty list. On
corpora where lexical already wins, rerank has nothing to fix. The
[employee↔staff synonym test](../../) where rerank *did* help is a real
case — just not one that shows up at the rates a default config should
target.

**Cross-PDF routing works at lexical default.** All 8 routing queries
(asking "what HTTP code…?" against a folder of 5 different PDFs) correctly
land on the right source document with lexical alone — no semantic
required for inter-document attribution either.

## The 1–2 stubborn failures

Across all configs, two MSA paraphrases stay broken:

- **"How long do I have to cancel and get my money back?"** (answer: §3.4
  30-day refund) — lexical/hybrid/rerank return **empty contexts**. The
  PDF uses *"refund"* and *"termination for convenience"*, not "cancel"
  or "money back". `retrieval="semantic"` (full dense, BM25 bypassed)
  returns *something*, but the wrong clause.
- **"How much could I sue for?"** (answer: §7.2 liability cap, twelve
  (12) months of fees) — same shape. The PDF uses *"limitation of
  liability"* and *"damages"*, not "sue".

The interesting bit: **hybrid sometimes returns *fewer* candidates than
lexical alone.** That's wrong by construction — hybrid should be a
superset. Tracked in
[issue #1 — hybrid empty-context regression](https://github.com/vysakh0/redhop/issues/1).

## Recommended config, by document shape

This is the [docs/guides/CHOOSING_A_CONFIG.md] matrix in one line per
shape — the matrix above is the data, this is the prescription:

| If your corpus is… | Use… | Why |
|---|---|---|
| Code, API refs, internal docs, runbooks, financial reports | **`Document.from_file(path)`** (lexical default) | BM25 wins; no model download |
| A handbook / well-titled chapter doc | **`Document.from_file(path)`** (lexical default) | Lexical wins; structural expansion *hurts* clean chapters |
| A contract or policy with regional overrides / near-duplicate clauses | **`Document.from_file(path, retrieval="hybrid", model="bge-small").context(q, include_heading=True, neighbors=1)`** | Heading awareness disambiguates parallel clauses |
| A synonym-heavy corpus (HR FAQs, support tickets where query and answer share no words) | **`Document.from_file(path, retrieval="hybrid", model="bge-small", rerank="cross-encoder")`** | Cross-encoder closes the synonym gap — but verify on your corpus; rerank is *not* free |
| Multiple files (`from_folder`) | **lexical default works** — `from_folder` routing is perfect at BM25 | No hybrid required for inter-document attribution |

## What this means for the default

- **Keep `retrieval="lexical"` as the default.** The data does not support
  switching the default to hybrid.
- **Promote `include_heading=True, neighbors=1` from a buried context()
  kwarg to a documented "structured-docs recipe".** The MSA result earns it.
- **De-emphasize cross-encoder rerank in the default-path docs.** It has a
  narrow use case (paraphrase-heavy corpora) that the average user doesn't
  have. Keep it documented for the cases where it helps, but don't make it
  part of the recommended path.

## Caveats

- 121 labeled queries × 6 corpora is enough to **rank configs** with
  confidence on these shapes; it's not enough to claim "lexical wins on
  every legal contract" in general. Your corpus may differ.
- The grader is substring-or-section. It misses cases where the right
  clause is in a chunk that wasn't the top-3 citation but appeared deeper
  in the assembled context. Bias is toward generous-to-RedHop.
- The corpora are LLM-generated synthetic docs in the style of real ones.
  Real-world OCR noise, legacy formatting, scanned-to-PDF artifacts, and
  multi-column layout could shift the picture.
- No latency-tier comparison here — the speed delta (lexical ~50ms warm vs
  cross-encoder 1000+ms) is large enough that even ties on accuracy are
  decisive wins for lexical in production paths.

## Reproduce

The eval scripts (and the colleague's source corpus) live outside the
public repo. The shape is:

```python
# Per-PDF eval with substring-or-section grading
for cfg_name, ff_kwargs, ctx_kwargs in CONFIGS:
    doc = redhop.Document.from_file(path, **ff_kwargs)
    for query, accept_substrings, section_regex, _label in QUERIES[corpus]:
        ctx = doc.context(query, **ctx_kwargs)
        ok = any(s in ctx.text().lower() for s in accept_substrings) or \
             any(re.search(section_regex, c.get("heading","").lower())
                 for c in ctx.citations[:3])
```

Full script in `research/colleague_eval/eval_all.py` (gitignored). The
publishable result is this matrix.
