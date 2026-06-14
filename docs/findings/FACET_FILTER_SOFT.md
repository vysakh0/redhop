# Facet / metadata filters at scale — the soft-filter default (guidance, no code)

> **Hypothesis:** the over-filtering lesson RedHop already measured for
> *relevance* cuts (`SECOND_HOP_TAX`, `REASONING_PRESERVATION`, `DISTRACTOR_ROBUSTNESS`)
> generalizes to *metadata / facet* cuts: a hard categorical filter can remove
> answer-bearing chunks whenever the legitimate answer set spans the filtered
> facet, and the damage grows with corpus size.
> **Status:** Open / guidance only. **RedHop ships no metadata-filter primitive
> today** (verified — see below), so there is nothing to soften and no code lands
> here. This records the design rule for if/when one is built.
> **Setup:** external catalog evaluation (an unmeasured regime) + RedHop's own
> relevance-filter findings. Not re-derived on a redhop rig because the
> infrastructure does not exist to derive it on.
> **Headline:** a hard price filter helped at small scale (it prunes wrong-price
> variants) but stopped helping — and on weak base retrievers actively hurt
> recall — at large scale, because the same product legitimately exists at many
> prices, so the answer set spans the filtered facet. The fix is the same shape as
> `ReasoningPreserving`: degrade a facet filter to a soft rerank / boost signal,
> not a hard cut, whenever answers can span the facet.
> **Caveats:** the "hard filter inverts at scale" claim is **conditional on
> retriever strength** (see the nuance below); do not overstate it.

---

## What RedHop has today

There is **no metadata / categorical / facet filtering anywhere in the context
or retrieval path**. `Chunk` carries a `metadata: HashMap<String, Value>` (used
for citations, headings, and neighbor expansion), but no `ContextStrategy`
(`RawTopK`, `DistractorFiltered`, `RedundancyPruned`, `MaxDensity`,
`ReasoningPreserving`, `Auto`) and no retriever reads it as a *filter*. Every
selection RedHop makes is **relevance / grounding based**, never categorical.

So this finding is not "fix the metadata filter" — it is "here is the rule the
metadata filter must follow the day it is built, derived from evidence we already
have." We do not ship speculative filtering infrastructure (bounded-architecture
discipline); we record the constraint so it is not re-learned the hard way.

## The evidence (external, not re-derived)

The external catalog evaluation added a metadata **price filter** to its retriever:

- At a **small** catalog (~74 items) the hard price filter **helped** — it
  collapses wrong-price variants of a product, so the right one surfaces.
- At a **large** catalog (~2500 items) it **stopped helping**, and on the weaker
  base retrievers (a char-ngram vector retriever, an RRF hybrid) it **hurt**
  recall: with hundreds of same-price distractors the hard filter removed gold
  SKUs more often than it removed noise, because the legitimate answer set for a
  query spans many prices (the same product exists at 25 / 52 / 90 / 150 / 200 g
  with different MRPs).

This is the categorical sibling of `SECOND_HOP_TAX`: there the hard cut was on
*query relevance* and it dropped the bridge chunk; here the hard cut is on a
*facet* and it drops a legitimate same-product variant. Same geometry — a hard,
context-blind cut removes evidence that the downstream task needed — extended to
metadata and to the scale axis (corpus size is not a variable in `SECOND_HOP_TAX`;
it is the whole story here).

## The honest nuance — do not overstate the inversion

The external evaluation's *first* pass concluded "hard facet filter inverts from
helpful to harmful at scale." Its *follow-up*, with a stronger multi-field base
retriever, found the filter did **not** invert: it fired on few of the ambiguous
(answer-spans-the-facet) queries, flipped zero of them, and helped the
single-answer queries by collapsing same-name siblings. The lesson is therefore
**conditional**, not universal:

- A hard facet filter is dangerous **in proportion to how much the base
  retriever's recall is already leaking** and **how often the answer set spans
  the facet**. On a strong retriever whose recall holds, and on queries whose
  answer is single-valued in the facet, a hard filter is fine and even helpful.
- The risk concentrates where (a) recall is marginal and (b) the answer
  legitimately spans the filtered field. That is exactly the regime
  `ReasoningPreserving` was built for, one axis over.

## The rule (for if / when a facet filter is built)

1. **Default to soft.** A facet constraint should down-weight / rerank
   (a multiplicative penalty on non-matching chunks), not hard-drop them, so a
   legitimately-spanning answer can still surface. Mirror the `ReasoningPreserving`
   philosophy: prefer demotion to deletion when the cut is context-blind.
2. **Make hard-cut an explicit opt-in**, documented as "use only when the answer
   is single-valued in this facet (a true equality constraint), never when it can
   span the facet."
3. **Set-aware where possible.** When the query maps to a *set* (the
   `EvalGold::AllOf` / `set_coverage` case), measure the filter against
   set-coverage, not single-gold recall — a hard facet cut is exactly the thing
   that silently drops one variant of a family (see `CATALOG_REGIME` Panel B).
4. **Re-derive before shipping a default.** None of the above changes a RedHop
   default today; if a metadata filter is added, its soft-vs-hard default needs a
   redhop-rig measurement on a faceted corpus first.

## What changed afterward

Nothing in code. This is a recorded design constraint plus a `CHOOSING_A_CONFIG`
note. It connects the catalog-regime evidence (`CATALOG_REGIME`) to RedHop's
existing over-filtering geometry (`SECOND_HOP_TAX`, `REASONING_PRESERVATION`,
`DISTRACTOR_ROBUSTNESS`) so the metadata-filter design starts from the right
prior the day someone needs it.
