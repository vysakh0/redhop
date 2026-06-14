# The Catalog Regime — short noisy queries on a near-duplicate corpus

> **Hypothesis:** the levers that win on RedHop's measured regime (long multi-hop
> questions over diverse prose) behave differently in the catalog regime — short,
> noisy, 2-5 token queries over a high-cardinality, near-duplicate corpus.
> Specifically: (1) a char-ngram subword retriever recovers transcription noise
> word-BM25 cannot; (2) per-field weighting lifts strict set-coverage; (3) a
> single-gold recall@k metric hides whole-variant-family misses.
> **Status:** char-ngram typo recovery **Confirmed** (large, robust). set-coverage
> hides behind recall@k **Confirmed**. Char-ngram pays a clean set-coverage cost
> at scale (the inversion *direction*) **Partially confirmed** (the word-BM25
> baseline is tie-break-confounded, see Method). Per-field-weight set-coverage
> lift **Falsified on this workload** (exact null: a boost on a field the
> near-duplicates share changes nothing). Bootstrap 95% CIs over queries.
> **Setup:** synthetic, deterministic, model-free catalog of generic public goods
> (no real brands); short queries with brand-token OCR/transcription typos; three
> corpus sizes (144 / 600 / 2500). An **external regime re-derived on a redhop
> rig** — single-domain synthetic evidence, every number a hypothesis, not a
> portable fact.
> **Headline:** on brand-typo'd queries char-ngram holds R@1 ~0.98 and AmbCov@20
> 0.83-1.0 at every scale while word-BM25 craters to R@1 0.10 / AmbCov 0.25. R@20
> reads a perfect 1.000 throughout while AmbCov@20 ranges 0.25-1.0 — the metric gap
> `set_coverage` exists to catch. Per-field boosting changed nothing.
> **Reproduce:** `cargo run -p redhop-examples --example catalog_regime_probe --release`
> (raw capture: `reports/catalog_regime_probe_2026-06-14.txt`).
> **Justifies API:** `CharNgramAnalyzer` (subword typo tier), `EvalGold::AllOf` +
> `set_coverage` (the AmbCov metric), `Bm25Retriever::with_field_weights` /
> `DocumentConfig::bm25_field_weights` (the knob — shipped zero-regression, lift
> not assumed). See `docs/CHOOSING_A_CONFIG.md` (When your corpus is a catalog).
> **Caveats:** synthetic single-domain; n=24 clarify families; the clarify queries
> produce score ties that a deterministic tie-break resolves (Method), which
> flatters word-BM25's clean AmbCov; brand-only typo noise; R@20 saturates at
> 1.000 so only R@1 / AmbCov discriminate. See §Honest limits.

---

RedHop's evidence suite (HotpotQA / MuSiQue / CUAD) is built on **long, diverse,
multi-hop questions over prose**. An external retrieval evaluation surfaced a
regime that suite never exercises: short (2-5 token), frequently
**mis-transcribed** queries mapped onto a **large, high-cardinality catalog**
where one brand has dozens of price / size / flavor variants differing by a token
or two. This finding re-derives its conclusions on a redhop rig rather than
importing the numbers.

## Method (hermetic, deterministic)

The rig (`catalog_regime_probe`) generates a deterministic catalog of generic
public goods (13 made-up brands × 10 product types × a flavor / size / price
lattice). It fixes 24 `(brand, product)` **probe families** (6 variants each)
that are always present, then pads with near-duplicate sibling SKUs to reach
144 / 600 / 2500 items. Queries are short and come in two noise modes:

- **resolve** — `"brand product flavor size"`, one gold SKU. Metric: `R@k`.
- **clarify** — `"brand product"`, the whole variant family is gold. Metric:
  **strict AmbCov@20** = the *entire* family present in the top-20 (you cannot
  offer a disambiguation option you never retrieved). This is the retrieval-layer
  twin of the shipped `EvalGold::AllOf` / `set_coverage`.

`noisy` corrupts the brand token with one OCR-style edit (`lays` → `1ays`).
Everything is hermetic and deterministic (a seeded LCG drives the typos and the
bootstrap).

**The score-tie tie-break (read this before reading AmbCov).** A clarify query
`"acme chips"` scores *every* `acme chips` variant identically under word-BM25
(they share exactly the matched tokens) — the whole family and all its siblings
tie. With more SKUs than top-k slots, *which* tied SKUs land in the top-20 is
otherwise decided by Tantivy's non-deterministic multi-segment doc ordering. The
probe makes the metric reproducible by retrieving the full matching pool and
re-breaking ties by `(quantized score desc, id asc)`. Base-family SKUs have the
lowest ids, so this tie-break **systematically favors families** for retrievers
that leave them tied (word-BM25). Read word-BM25's clean AmbCov of 1.000 with
that caveat: it reflects the tie-break, not a discrimination the retriever made.
The trustworthy signals are the ones a tie-break cannot manufacture: the **noisy**
gap (char-ngram genuinely scores the typo'd brand, word-BM25 scores it zero) and
char-ngram's **own** scale-degradation (it breaks the ties and pays for it).

## Panel A — the char-ngram typo tier (the robust win)

Brand-typo'd queries, all three corpus sizes. `R@1` is resolve early precision;
`AmbCov@20` is strict clarify set-coverage.

| arm | n=144 R@1 | AmbCov@20 | n=600 R@1 | AmbCov@20 | n=2500 R@1 | AmbCov@20 |
| --- | --- | --- | --- | --- | --- | --- |
| word-bm25  | 0.104 | 0.250 | 0.104 | 0.250 | 0.104 | 0.250 |
| char-ngram | **0.986** | **1.000** | **0.972** | **0.958** | **0.979** | **0.833** |

A single brand-token typo zeroes word-BM25's early precision (`R@1` 0.10): the
typo'd brand contributes no term, so the gold SKU ranks below same-product SKUs
from every other brand (word-BM25 still reaches it by R@10 ≈ 0.83 via the
product / flavor / size tokens, but not at rank 1). Char-ngram subword matching
recovers the typo'd token with no model (`lays` and `1ays` still share `ays`,
`ys `) and puts the gold at rank 1 ~0.98 of the time, holding most of its
clarify coverage at every scale. This is the strongest, most robust effect in the
probe and the reason `CharNgramAnalyzer` ships.

## Panel B — set-coverage hides behind recall@k (why `EvalGold::AllOf` exists)

Every word-based arm scores **R@20 = 1.000 at every size and noise level** — the
single-gold recall metric says retrieval is solved. The strict whole-family
metric over the same runs disagrees:

| run | R@20 (resolve) | AmbCov@20 (clarify) |
| --- | --- | --- |
| word-bm25, clean, n=2500 | 1.000 | 1.000 *(tie-break aided)* |
| word-bm25, noisy, n=2500 | 1.000 | **0.250** |
| char-ngram, clean, n=2500 | 1.000 | 0.833 |
| char-ngram, noisy, n=2500 | 1.000 | 0.833 |

A "perfect" R@20 of 1.000 coexists with AmbCov as low as 0.250 (18 of 24 families
un-offerable, word-BM25 + brand typo). recall@k against a single gold chunk cannot
see a half-retrieved variant family; `set_coverage` (the fraction of families
*fully* present) can. This is exactly the failure `EvalGold::AllOf` catches.

## Panel C — char-ngram's clean cost at scale (the inversion direction)

Clean queries, strict AmbCov@20 by corpus size, bootstrap 95% CIs:

| arm | n=144 | n=600 | n=2500 |
| --- | --- | --- | --- |
| word-bm25  | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| char-ngram | 1.000 [1.000, 1.000] | 0.958 [0.875, 1.000] | **0.833** [0.708, 0.958] |

Char-ngram's clean set-coverage **erodes with scale** (1.000 → 0.958 → 0.833)
while word-BM25's stays flat. The honest reading, per Method: word-BM25 leaves the
family-vs-sibling near-duplicates *tied*, so its flat 1.000 is the tie-break (low
family ids) holding, not a retrieval decision. Char-ngram's overlapping-gram
vocabulary gives the near-duplicates *distinct* scores — it actually reranks them
— and at scale those fine-grained gram distinctions increasingly rank a sibling
above a family member, scattering the family out of the top-20. So char-ngram
pays a measured clean cost (the external evidence's "inverts at scale" *direction*)
precisely because it is doing more than word-BM25, while word-BM25's apparent
robustness here is partly an artifact of leaving the hard call untaken. **The
conclusion that survives either way:** char-ngram is a recall booster for short /
noisy tokens, **not a drop-in** — pair it with word-BM25, do not replace it.

## Panel D — per-field weighting: the lift did NOT replicate (exact null)

We boosted the family-key field (`heading = brand product`) at ×3 and ×8, and a
too-broad field (`heading = brand` alone) at ×3. Every boosted arm reproduced
word-BM25 **bit for bit** on AmbCov, clean and noisy, at every size:

| arm | clean AmbCov@2500 | noisy AmbCov@2500 |
| --- | --- | --- |
| word-bm25 (baseline) | 1.000 | 0.250 |
| boost key (brand+product) ×3 | 1.000 | 0.250 |
| boost key (brand+product) ×8 | 1.000 | 0.250 |
| boost brand-only ×3 | 1.000 | 0.250 |

**No effect at all.** The mechanism is the lesson: a clarify query `"acme chips"`
matches the boosted field identically for the whole family *and* its siblings (they
all carry the same `brand product` heading), so a boost scales them all by the
same factor and reorders nothing. A field boost can only help when the boosted
field **separates** the answer from its near-duplicates; here it does not, so it
is inert. The external evidence's lift came from a brand+flavor boost whose flavor
leg discriminated *within* the family — a different field structure than this rig.

**What this means for the shipped knob.** `Bm25Retriever::with_field_weights` /
`DocumentConfig::bm25_field_weights` ship anyway, because they are
**zero-regression** (unit-tested: `FieldWeights::uniform()` reproduces the default
ranking *and* scores bit-for-bit, and a `1.0` weight is skipped before it reaches
Tantivy) and because field importance is genuinely domain-specific. But the
default stays equal-weight and the guidance is the deliverable, not the knob:
**boosting helps only when the boosted field discriminates the answer from its
near-duplicates — sweep on a held-out set with your own eval and watch
set-coverage, because a boost on a shared field is inert.** Same shape as
`CUAD_PRF_NULL` / `SUB_IDF_AUTO_DROP_NULL`: a plausible reweighting that fails
unless the signal it amplifies is actually discriminative.

## Interpretation

The catalog regime separates two failure axes the prose suite conflates:

- **Lexical noise (short tokens).** A transcription typo on a 1-2 token brand
  zeroes token-exact BM25, and dense can't rescue a 1-2 token query either
  (`SEMANTIC_ZERO_DEP`, the 0.56 ceiling). The lever is **subword lexical**
  matching, no model — Panel A, the cleanest and largest effect.
- **Cardinality (near-duplicates).** A query that maps to a *set* ties across the
  family, so single-gold recall stops meaning "the answer set is offerable"
  (Panel B), a field boost can only amplify what already ties (Panel D, inert),
  and a retriever that *does* break the ties (char-ngram) pays for the fine
  distinctions at scale (Panel C).

The throughline with the rest of the evidence layer: a lever that wins in one
regime is not free in another, and the honest output is teaching the user *which
regime they are in* (corpus size, query length, noise) rather than flipping a
default.

## Honest limits

- **Synthetic, single-domain.** One generated catalog of generic goods; treat as
  mechanism demonstration, not a portable benchmark — same discipline as
  `SPIDER_ENRICH`.
- **Score ties dominate clarify AmbCov.** The near-duplicate families tie under
  word-BM25, so clean AmbCov is decided by the deterministic tie-break, not by the
  retriever. Cross-arm clean-AmbCov comparisons (Panel C) are read with that
  confound stated. The noisy gap (Panel A) and the metric gap (Panel B) are not
  tie-break artifacts and are the load-bearing results.
- **n=24 clarify families.** set-coverage CIs are wide. The Panel-A typo gap is
  large enough to read through the noise; the Panel-C degradation has CIs that
  exclude 1.000 at n=2500 but the baseline is confounded as above.
- **R@20 saturates** at 1.000, so the resolve signal lives entirely at R@1 and the
  set signal at AmbCov@20. A harder corpus would move R@20 too.
- **Brand-only typo model** — one deterministic edit on the brand token; heavier
  noise would widen char-ngram's lead.
- **Field-weight null is workload-shaped.** It says a boost on a *shared* field is
  inert, not that field weights never help. A corpus whose hard cases are
  cross-field competition could still show a lift, which is why the guidance is
  "sweep, don't assume."

## What changed afterward

- Shipped `CharNgramAnalyzer` (subword typo / short-token tier) behind the existing
  `Analyzer` trait, with `Analyzer::union_subword_query()` so a multi-gram query
  ORs its grams instead of phrase-matching them.
- Shipped `EvalGold::AllOf` + `EvalReport::set_coverage` (+ `EvalSummary`
  aggregation) — the strict variant-family coverage metric, the AmbCov twin.
- Shipped `Bm25Retriever::with_field_weights` + `DocumentConfig::bm25_field_weights`
  (default equal-weight, zero-regression), recorded the lift as an **exact null on
  this workload**.
- Added the corpus-size / query-length axis and the typo-tier and field-weight
  procedures to `docs/CHOOSING_A_CONFIG.md`.
