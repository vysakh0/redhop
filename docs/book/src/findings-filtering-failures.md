# Filtering failures

> Full docs: [`DISTRACTOR_ROBUSTNESS.md`](https://github.com/redhop/redhop/blob/main/docs/findings/DISTRACTOR_ROBUSTNESS.md)
> and [`REASONING_PRESERVATION.md`](https://github.com/redhop/redhop/blob/main/docs/findings/REASONING_PRESERVATION.md)

**Hypothesis.** Distractors hurt generated answers, so aggressive distractor
filtering is a free quality win.

**Status: PARTIALLY FALSIFIED** on multi-hop — and the correction is the
strongest end-to-end result in the project.

**Experiment 1 (sign-flip).** End-to-end QA on HotpotQA via an LLM (`claude
haiku`): build clean / polluted / filtered contexts and score answer quality.

**Result 1.** Distractors *do* degrade answers (causal, +0.033 both runs). But
the **net benefit of filtering flipped sign** between n=20 (−0.050) and n=30
(+0.020) — within noise. At this scale, whether aggressive filtering nets
positive on multi-hop is **unresolved**. Reported as the finding, not hidden.

**Experiment 2 (n=300, CIs).** Gap-qualified multi-hop, four contexts from the
same polluted input at an aggressive threshold; paired bootstrap CIs.

**Result 2.** The surprise: the distractors barely hurt the strong generator
(polluted 0.829 ≈ gold-only 0.830), but the aggressive **filter** crashed
quality to 0.705 — *the cure was worse than the disease*. `reasoning_preserving`
recovered a CI-significant slice (+0.035, CI [+0.003, +0.067]), and the gain was
**causally localized** to the rescued evidence: +0.173 where it saved gold the
filter dropped, ~0 (CI spans zero) where gold retention was identical.

**Caveats.** One generator (haiku), lexical kw-recall proxy, rescued subset
small (n=25). Single dataset.

**Implications.** "Don't over-filter." On distractor-robust generators the
unfiltered context already beats aggressive filtering; if you must filter, make
it reasoning-safe. This is why `reasoning_preserving` is the default and
`distractor_filtered` is recommended only at a *low* threshold.
