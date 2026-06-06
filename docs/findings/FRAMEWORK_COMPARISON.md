# RedHop vs LangChain vs LlamaIndex — head-to-head (Tier 1 + Tier 3)

> **Question:** is RedHop's context assembly actually better than the big
> frameworks'? (Prompted by the hypothesis "RedHop does it better than
> LangChain/LlamaIndex ever will.")
> **Status:** Answered, honestly — **competitive, not dominant.** RedHop ties
> LlamaIndex and beats LangChain on downstream answers; its retention edge on
> multi-hop does not translate into a large answer-quality lead.
> **Setup:** same documents, **BM25 retrieval for all three**, same token budget
> (forced below doc size so selection happens), RedHop at the 128-token default.
> Tier-1: gold-evidence word-recall (free). Tier-3: gpt-4o-mini answers each
> system's context, scored SQuAD-style (F1/EM). Harnesses in `bench/`.

> **Rerun — 2026-06-06 (current main, 0.2.2).** Fresh run of `bench/compare.py`
> on the current main. **HotpotQA RedHop[topk] +3 points: 77% → 80%** ≥0.8
> retention, opening the multi-hop lead from +5 to +8 over LlamaIndex (LangChain
> and LlamaIndex unchanged). CUAD numbers are identical. The improvement is
> attributable to the BM25 silent-wildcard fix + analyzer sharpening from 0.2.1;
> it landed cleanly on multi-hop and didn't regress anything else. The CUAD
> 4-point gap to LlamaIndex was investigated separately — see
> [CUAD_RECALL_GAP.md](CUAD_RECALL_GAP.md): mechanism is BM25 template-boilerplate
> dilution, closeable with a 6-line query preprocessor that takes RedHop to
> 88% ≥0.8 (overtaking LlamaIndex by 2 points). Raw output:
> [`reports/framework_comparison_2026-06-06.txt`](../../reports/framework_comparison_2026-06-06.txt).
> The Tier-1 table below is the updated 2026-06-06 numbers; Tier 3 has not been
> rerun.

---

## Tier 1 — evidence retention (≥0.8 word-recall, no LLM, n=300)

| dataset | redhop (best) | LangChain | LlamaIndex |
| ------- | ------------- | --------- | ---------- |
| HotpotQA multi-hop | **80%** (was 77%) | 71% | 72% |
| CUAD contracts | 82% | 73% | **86%** |

## Tier 3 — downstream answer quality (gpt-4o-mini, n=150)

**CUAD (verbatim span extraction):**

| system | F1 | EM | refusal |
| ------ | -- | -- | ------- |
| redhop | 0.324 | 0.153 | 49% |
| redhop[topk] | 0.342 | 0.173 | 47% |
| langchain | 0.248 | 0.107 | 59% |
| llamaindex | **0.350** | 0.160 | 47% |

**HotpotQA (concise factoid QA):**

| system | F1 | EM | refusal |
| ------ | -- | -- | ------- |
| redhop | **0.514** | 0.413 | 30% |
| redhop[topk] | 0.515 | 0.413 | 30% |
| langchain | 0.499 | 0.387 | 33% |
| llamaindex | 0.497 | **0.420** | 33% |

## Honest reading

- **RedHop is competitive, not a blowout.** On answers it's **≈ LlamaIndex and
  ahead of LangChain** — on CUAD RedHop[topk] (0.342) ≈ LlamaIndex (0.350) ≫
  LangChain (0.248); on HotpotQA all three are within ~2 points (RedHop edges F1,
  LlamaIndex edges EM). The original "better than they'll ever be" is **not**
  supported; "holds its own with the leaders" is.
- **Retention is a loose proxy for answers.** RedHop's clear multi-hop *retention*
  lead (77% vs 71–72%) shrank to a near-tie on *answer quality* — at these budgets
  every system hands the model enough to roughly tie. Worth remembering before
  over-reading any retention number.
- **The strategy still isn't the advantage — downstream too.** `reasoning_preserving`
  vs `raw_topk` is indistinguishable on answers (CUAD 0.324 vs 0.342, Hotpot 0.514
  vs 0.515). Consistent with every prior finding: RedHop's value isn't a magic
  optimizer.
- **LangChain's deficit is mostly refusals** (CUAD 59% vs ~47%) — its
  chunking/retrieval surfaced the answer span less often, so the model bailed more.

## So where does RedHop actually stand?

Competitive on the core quality metrics with the category leaders, and
**differentiated on what they don't offer: an interpretable, conditional,
observable runtime** (the Decision Report, Auto, the evidence layer) — not on
raw retrieval/answer quality. That's the honest, defensible position.

## Caveats
- BM25 across all three (controlled); their default *vector* retrievers untested.
- gpt-4o-mini only; one budget per dataset; CUAD extraction F1 is low in absolute
  terms (hard task, high refusal) — the *relative* ranking is the signal.
- LlamaIndex's contract edge is real and unexplained (its node parsing /
  tokenization may suit legalese); worth a look.

Reproduce: `bench/.venv/bin/python bench/compare.py` (Tier 1),
`bench/.venv/bin/python bench/tier3.py --n 150` (Tier 3). Raw output in
`reports/framework_comparison.txt`, `reports/framework_tier3.txt`.
