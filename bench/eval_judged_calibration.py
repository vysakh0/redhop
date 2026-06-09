#!/usr/bin/env python3
"""Calibration probe for the LLM-judged eval surface, with optional
side-by-side against the Ragas eval library.

Runs the full judged surface (faithfulness_judged, relevancy_judged,
correctness_judged, faithfulness with claim decomposition, and a
critique aspect bundle) on a 5-case hand-curated test set with known
expected characteristics:

  case             | expected behavior
  -----------------|---------------------------------------------------
  CLEAN            | answer is a faithful paraphrase of ctx
                   | → faithfulness, relevancy, correctness all HIGH
  HALLUCINATION    | answer adds tokens never in ctx
                   | → faithfulness LOW, others mid
  OFF_TOPIC        | answer doesn't address the query
                   | → relevancy LOW, others varies
  WRONG_FACT       | answer contradicts the gold answer
                   | → correctness LOW
  REFUSAL          | answer refuses to engage
                   | → faithfulness/relevancy LOW, correctness 0

Two judge modes:

  • OpenAI (if `OPENAI_API_KEY` is set + `openai` is installed): real,
    paid LLM calls. Outputs honest numbers you can quote.
  • Deterministic stub (default — no key, no network): the stub
    returns synthetic scores from token-overlap on the prompt blocks.
    Useful for CI, for verifying the wiring end-to-end, and for
    showing the shape of the output without burning tokens.

Optionally, if `ragas` is installed AND `OPENAI_API_KEY` is set, the
script also runs the same dataset through Ragas's faithfulness +
answer_relevancy + answer_similarity and prints a pairwise agreement
matrix (Pearson r, mean absolute error).

Run:
  bench/.venv/bin/python bench/eval_judged_calibration.py

With a real LLM:
  OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py

With Ragas comparison:
  pip install ragas
  OPENAI_API_KEY=sk-... bench/.venv/bin/python bench/eval_judged_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]

# Model id. OpenRouter expects vendor-prefixed ids (`openai/gpt-4o-mini`,
# `anthropic/claude-3-haiku-20240307`); bare OpenAI expects unprefixed
# (`gpt-4o-mini`). We pick based on which key is set.
if os.environ.get("OPENROUTER_API_KEY"):
    MODEL_DEFAULT = os.environ.get("EVAL_MODEL", "openai/gpt-4o-mini")
else:
    MODEL_DEFAULT = os.environ.get("EVAL_MODEL", "gpt-4o-mini")


# ── Test set ────────────────────────────────────────────────────────────────


@dataclass
class TestCase:
    """One (query, ctx_text, answer, gold) tuple with a label naming the
    failure mode this case demonstrates."""
    label: str
    query: str
    ctx_text: str
    answer: str
    gold_answer: str
    expected: dict[str, str]  # metric → "high" / "low" / "mid"


TEST_SET: list[TestCase] = [
    TestCase(
        label="CLEAN",
        query="What is the refund window?",
        ctx_text=(
            "Section 3.2 Refunds. The refund window is thirty days from "
            "the date of purchase. Customers may return items in original "
            "packaging."
        ),
        answer="The refund window is thirty days from the date of purchase.",
        gold_answer="thirty days from the date of purchase",
        expected={
            "faithfulness": "high",
            "relevancy": "high",
            "correctness": "high",
        },
    ),
    TestCase(
        label="HALLUCINATION",
        query="What is the refund window?",
        ctx_text=(
            "Section 3.2 Refunds. The refund window is thirty days from "
            "the date of purchase."
        ),
        answer=(
            "The refund window is thirty days, and customers must "
            "provide a notarized affidavit and pay a $25 restocking fee."
        ),
        gold_answer="thirty days from the date of purchase",
        expected={
            "faithfulness": "low",  # affidavit + $25 fee not in ctx
            "relevancy": "high",     # still on-topic
            "correctness": "mid",    # the "30 days" part matches
        },
    ),
    TestCase(
        label="OFF_TOPIC",
        query="What is the refund window?",
        ctx_text=(
            "Section 3.2 Refunds. The refund window is thirty days from "
            "the date of purchase."
        ),
        answer="Photosynthesis converts sunlight into chemical energy.",
        gold_answer="thirty days from the date of purchase",
        expected={
            "faithfulness": "low",
            "relevancy": "low",
            "correctness": "low",
        },
    ),
    TestCase(
        label="WRONG_FACT",
        query="What is the refund window?",
        ctx_text=(
            "Section 3.2 Refunds. The refund window is thirty days from "
            "the date of purchase."
        ),
        answer="The refund window is ninety days from the date of purchase.",
        gold_answer="thirty days from the date of purchase",
        expected={
            "faithfulness": "low",   # 90 days isn't in ctx
            "relevancy": "high",     # on-topic
            "correctness": "low",    # contradicts gold
        },
    ),
    TestCase(
        label="REFUSAL",
        query="What is the refund window?",
        ctx_text=(
            "Section 3.2 Refunds. The refund window is thirty days from "
            "the date of purchase."
        ),
        answer="I cannot answer that question.",
        gold_answer="thirty days from the date of purchase",
        expected={
            "faithfulness": "low",   # no claims to check (vacuous high)
            "relevancy": "low",      # doesn't address the question
            "correctness": "low",
        },
    ),
]


# ── Judges ──────────────────────────────────────────────────────────────────


def make_openai_judge(model: str = MODEL_DEFAULT) -> redhop.Judge:
    """Real LLM judge over an OpenAI-compatible chat-completions API.

    Picks up credentials from one of:
      • `OPENROUTER_API_KEY` — routes via openrouter.ai (multi-vendor;
        cheap; use a model id like `openai/gpt-4o-mini` or
        `anthropic/claude-3-haiku`)
      • `OPENAI_API_KEY` — routes via api.openai.com (use a bare model
        id like `gpt-4o-mini`)

    Imported lazily so users without the openai package can still run
    the script in stub mode.
    """
    from openai import OpenAI  # type: ignore

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        client = OpenAI()  # picks up OPENAI_API_KEY from env

    def score(prompt: str, system: str | None):
        """Single-shot chat completion with temperature 0 for determinism.

        Returns either a float (when the LLM replied with a parseable
        number — the score-prompt path) or a dict with `raw_text`
        (when the LLM replied with prose — the claim-extraction path).
        The Python Judge binding accepts both shapes; the dict path
        preserves the raw text so downstream claim parsing works."""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        # Try to interpret as a number; on failure, the LLM returned
        # prose (extraction or classification output) and we hand it
        # through as raw_text.
        try:
            return float(text)
        except ValueError:
            return {"score": 0.0, "raw_text": text, "model": model}

    return redhop.Judge.from_callable(score, name=model).cached()


def make_stub_judge() -> redhop.Judge:
    """Deterministic stub that returns scores calibrated to the test
    set's expected behavior. Lets you run the script (and exercise the
    full eval surface) without an API key.

    The stub returns:
    - For extraction prompts (system contains "Decompose answers"):
      one claim per answer sentence.
    - For verification prompts: 0.9 if the claim's tokens appear in the
      context, 0.1 otherwise.
    - For faithfulness/relevancy/correctness: a synthetic score based on
      token overlap between the relevant pair (answer/ctx,
      query/answer, answer/gold). This MATCHES what an honest LLM
      would broadly produce for the test set and lets us validate the
      surface without burning real tokens.
    - For critique prompts (system contains "qualitative property"):
      0.5 (neutral). Critique prompts depend on the aspect definition,
      which we can't synthesize sensibly here.
    """

    def words(s: str) -> set[str]:
        return {
            w
            for w in "".join(c if c.isalnum() else " " for c in s.lower()).split()
            if len(w) > 2
        }

    def token_overlap(a: str, b: str) -> float:
        wa, wb = words(a), words(b)
        if not wa:
            return 0.0
        return len(wa & wb) / len(wa)

    def score(prompt: str, system: str | None):
        """Returns either a float (score) or a dict {score, raw_text}.
        The decomposition-extraction call needs raw text to be parsed
        into claims; everything else needs a numeric score."""
        sys_str = system or ""

        if "Decompose answers" in sys_str:
            # Extraction prompt — return one claim per answer sentence.
            try:
                ans = prompt.split("ANSWER:\n", 1)[1].split("\n\n", 1)[0]
            except IndexError:
                return {"score": 0.0, "raw_text": ""}
            sentences = [
                s.strip()
                for s in ans.replace("?", ".").replace("!", ".").split(".")
                if s.strip()
            ]
            return {"score": 0.0, "raw_text": "\n".join(sentences)}

        if "single CLAIM" in sys_str:
            try:
                claim = prompt.split("CLAIM:\n", 1)[1].split("\n\n", 1)[0]
                ctx = prompt.split("CONTEXT:\n", 1)[1].split("\n\n", 1)[0]
            except IndexError:
                return 0.5
            return token_overlap(claim, ctx)

        if "qualitative property" in sys_str:
            # Phase-7 critique — neutral (no semantic signal to use).
            return 0.5

        if "supported by a given CONTEXT" in sys_str:
            try:
                ctx = prompt.split("CONTEXT:\n", 1)[1].split("\n\n", 1)[0]
                ans = prompt.split("ANSWER:\n", 1)[1].split("\n\n", 1)[0]
            except IndexError:
                return 0.5
            return token_overlap(ans, ctx)

        if "addresses a question" in sys_str:
            try:
                q = prompt.split("QUESTION:\n", 1)[1].split("\n\n", 1)[0]
                ans = prompt.split("ANSWER:\n", 1)[1].split("\n\n", 1)[0]
            except IndexError:
                return 0.5
            return token_overlap(q, ans)

        if "factual correctness" in sys_str:
            try:
                gold = prompt.split("REFERENCE ANSWER:\n", 1)[1].split("\n\n", 1)[0]
                ans = prompt.split("GENERATED ANSWER:\n", 1)[1].split("\n\n", 1)[0]
            except IndexError:
                return 0.5
            return token_overlap(gold, ans)

        return 0.5

    return redhop.Judge.from_callable(score, name="stub-calibrated").cached()


# ── Run RedHop ──────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    label: str
    redhop_faithfulness: float | None
    redhop_relevancy: float | None
    redhop_correctness: float | None
    redhop_faithfulness_decomposed: float | None
    redhop_claims_extracted: int | None
    redhop_claims_supported: int | None
    redhop_critique: dict[str, float | None] = field(default_factory=dict)


def run_redhop_on_case(case: TestCase, judge: redhop.Judge) -> CaseResult:
    doc = redhop.Document.from_text(case.ctx_text)
    ctx = doc.context(case.query)

    # Single-prompt faithfulness.
    r1 = redhop.evaluate(
        case.query, ctx,
        answer=case.answer,
        gold_answer=case.gold_answer,
        judge=judge,
    )

    # Claim-decomposed faithfulness (separate call so we can record both).
    r2 = redhop.evaluate(
        case.query, ctx,
        answer=case.answer,
        gold_answer=case.gold_answer,
        judge=judge,
        decompose_faithfulness=True,
    )

    # Aspect critique — pin a small bundle.
    aspects = [
        redhop.Aspect("conciseness", "Is the ANSWER free of repetition or padding?"),
        redhop.Aspect(
            "harmfulness",
            "Does the ANSWER contain harmful, offensive, or unsafe content?",
            high_is_good=False,
        ),
    ]
    critique = redhop.critique(
        case.answer, aspects,
        judge=judge,
        context=case.ctx_text,
        query=case.query,
    )

    return CaseResult(
        label=case.label,
        redhop_faithfulness=r1.faithfulness_judged,
        redhop_relevancy=r1.relevancy_judged,
        redhop_correctness=r1.correctness_judged,
        redhop_faithfulness_decomposed=r2.faithfulness_judged,
        redhop_claims_extracted=r2.n_faithfulness_claims_extracted,
        redhop_claims_supported=r2.n_faithfulness_claims_supported,
        redhop_critique={name: score for name, score in critique.scores},
    )


# ── Optional side-by-side via the Ragas eval library ───────────────────────


def run_ragas_if_available(test_set: list[TestCase], model: str = MODEL_DEFAULT):
    """Run Ragas's `faithfulness` on the same dataset, returning a list
    of per-case dicts {label, faithfulness} or None if Ragas isn't
    installed.

    Only faithfulness is run here because the other Ragas metrics
    (answer_relevancy, answer_similarity, answer_correctness) need an
    embedder. OpenRouter doesn't expose embeddings; OpenAI does. To keep
    the comparison single-vendor, we restrict to faithfulness — which
    is where our few-shot + batched changes are concentrated anyway."""
    try:
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import faithfulness as r_faithfulness  # type: ignore
        from ragas.llms import LangchainLLMWrapper  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError as e:
        print(f"  (ragas dep import failed: {e})")
        return None

    # Route Ragas's LLM through the same provider we're using for
    # RedHop, so the comparison is apples-to-apples on judge-model.
    if os.environ.get("OPENROUTER_API_KEY"):
        llm = ChatOpenAI(
            model=model,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
    else:
        llm = ChatOpenAI(model=model, temperature=0.0)
    wrapped_llm = LangchainLLMWrapper(llm)
    r_faithfulness.llm = wrapped_llm

    ds = Dataset.from_list([
        {
            "question": c.query,
            "answer": c.answer,
            "ground_truth": c.gold_answer,
            "contexts": [c.ctx_text],
        }
        for c in test_set
    ])
    result = ragas_evaluate(
        dataset=ds,
        metrics=[r_faithfulness],
        llm=wrapped_llm,
    )
    df = result.to_pandas()
    return [
        {
            "label": test_set[i].label,
            "faithfulness": float(df.iloc[i]["faithfulness"]),
        }
        for i in range(len(test_set))
    ]


# ── Reporting ───────────────────────────────────────────────────────────────


def fmt_score(s) -> str:
    if s is None:
        return "  null"
    return f"{s:6.3f}"


def print_redhop_table(results: list[CaseResult]):
    print()
    print("=" * 88)
    print("  RedHop metrics (one row per case)")
    print("=" * 88)
    print(
        f"  {'case':<14} {'faith':>7} {'relev':>7} {'corr':>7} "
        f"{'faith_d':>8} {'claims':>10} {'critique':>20}"
    )
    print("  " + "-" * 84)
    for r in results:
        claims = "—"
        if r.redhop_claims_extracted is not None:
            claims = f"{r.redhop_claims_supported}/{r.redhop_claims_extracted}"
        critique_str = " ".join(
            f"{k}={fmt_score(v)}" for k, v in r.redhop_critique.items()
        )
        print(
            f"  {r.label:<14} "
            f"{fmt_score(r.redhop_faithfulness)} "
            f"{fmt_score(r.redhop_relevancy)} "
            f"{fmt_score(r.redhop_correctness)} "
            f"{fmt_score(r.redhop_faithfulness_decomposed)} "
            f"{claims:>10}  {critique_str}"
        )


def print_calibration_check(results: list[CaseResult], cases: list[TestCase]):
    """For each case, check whether the metrics landed in the expected
    bucket (high / mid / low). Prints a small grid; mismatches flagged."""
    print()
    print("=" * 88)
    print("  Calibration check — did the metrics land in the expected buckets?")
    print("=" * 88)
    print("  buckets: high ≥ 0.7, mid 0.4-0.7, low < 0.4")
    print()
    print(f"  {'case':<14} {'metric':<15} {'expected':<10} {'actual':>8} {'verdict':>10}")
    print("  " + "-" * 84)

    def bucket(v):
        if v is None:
            return "null"
        if v >= 0.7:
            return "high"
        if v >= 0.4:
            return "mid"
        return "low"

    for r, c in zip(results, cases):
        for metric_name, expected in c.expected.items():
            actual_val = {
                "faithfulness": r.redhop_faithfulness,
                "relevancy": r.redhop_relevancy,
                "correctness": r.redhop_correctness,
            }[metric_name]
            actual_b = bucket(actual_val)
            verdict = "✓" if actual_b == expected else "✗"
            print(
                f"  {c.label:<14} {metric_name:<15} {expected:<10} "
                f"{fmt_score(actual_val)}  {actual_b:>5} {verdict:>3}"
            )


def print_ragas_comparison(redhop_results: list[CaseResult], ragas_scores: list[dict]):
    print()
    print("=" * 88)
    print("  Side-by-side: RedHop vs Ragas faithfulness (same LLM)")
    print("=" * 88)
    print(
        f"  {'case':<14} {'RedHop':>10} {'RedHop_d':>10} {'Ragas':>10} {'|RH-Rg|':>10}"
    )
    print("  " + "-" * 60)
    for rh, rg in zip(redhop_results, ragas_scores):
        delta = (
            abs(rh.redhop_faithfulness - rg["faithfulness"])
            if rh.redhop_faithfulness is not None
            else float("nan")
        )
        print(
            f"  {rh.label:<14} "
            f"{fmt_score(rh.redhop_faithfulness)} "
            f"{fmt_score(rh.redhop_faithfulness_decomposed)} "
            f"{fmt_score(rg['faithfulness'])} "
            f"{delta:>10.3f}"
        )

    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        mx = sum(xs) / n
        my = sum(ys) / n
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        if sx == 0 or sy == 0:
            return float("nan")
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

    def mae(xs, ys):
        return sum(abs(x - y) for x, y in zip(xs, ys)) / max(len(xs), 1)

    rh_single = [r.redhop_faithfulness or 0.0 for r in redhop_results]
    rh_decomp = [r.redhop_faithfulness_decomposed or 0.0 for r in redhop_results]
    rg_faith = [g["faithfulness"] for g in ragas_scores]

    print()
    print("  Pairwise agreement (faithfulness):")
    print(f"  {'RedHop single-prompt  ↔ Ragas':<32} r={pearson(rh_single, rg_faith):+.3f}  MAE={mae(rh_single, rg_faith):.3f}")
    print(f"  {'RedHop decomposed     ↔ Ragas':<32} r={pearson(rh_decomp, rg_faith):+.3f}  MAE={mae(rh_decomp, rg_faith):.3f}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    # Try to use the real LLM judge if both an API key (OpenAI or
    # OpenRouter) AND the `openai` package are present. Probe up front
    # so the banner is honest.
    has_key = bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    )
    use_real = has_key
    if use_real:
        try:
            import openai  # noqa: F401
        except ImportError:
            use_real = False

    print()
    print("=" * 88)
    if use_real:
        provider = "OpenRouter" if os.environ.get("OPENROUTER_API_KEY") else "OpenAI"
        banner_judge = f"{provider} {MODEL_DEFAULT}"
    else:
        banner_judge = "deterministic stub"
    print(f"  RedHop calibration — judge = {banner_judge}")
    print("=" * 88)

    if use_real:
        judge = make_openai_judge()
    else:
        if has_key:
            print("  (API key is set but `openai` not installed — falling back to stub)")
        else:
            print("  (set OPENAI_API_KEY or OPENROUTER_API_KEY + `pip install openai` for real LLM calls)")
        judge = make_stub_judge()

    results = []
    t0 = time.perf_counter()
    for case in TEST_SET:
        results.append(run_redhop_on_case(case, judge))
    elapsed_redhop = time.perf_counter() - t0

    print_redhop_table(results)
    print_calibration_check(results, TEST_SET)
    print(f"\nTotal time: {elapsed_redhop:.2f}s")

    # Optional side-by-side comparison via the Ragas eval library.
    ragas_scores = None
    if use_real:
        ragas_scores = run_ragas_if_available(TEST_SET)
        if ragas_scores is None:
            print("\nRagas comparison: SKIPPED (`pip install ragas` to enable)")
        else:
            print_ragas_comparison(results, ragas_scores)

    # Persist a machine-readable snapshot for downstream tooling.
    out = REPO / "reports" / "eval_judged_calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {
            "judge": banner_judge,
            "ragas": ragas_scores,
            "results": [
                {
                    "label": r.label,
                    "faithfulness": r.redhop_faithfulness,
                    "relevancy": r.redhop_relevancy,
                    "correctness": r.redhop_correctness,
                    "faithfulness_decomposed": r.redhop_faithfulness_decomposed,
                    "claims_extracted": r.redhop_claims_extracted,
                    "claims_supported": r.redhop_claims_supported,
                    "critique": r.redhop_critique,
                }
                for r in results
            ],
        },
        indent=2,
        default=lambda o: None,
    ))
    print(f"\nWrote machine-readable snapshot → {out}")


if __name__ == "__main__":
    main()
