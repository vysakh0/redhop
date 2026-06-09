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
MODEL_DEFAULT = "gpt-4o-mini"


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
    """Real OpenAI judge. Requires `OPENAI_API_KEY` env var.

    Imported lazily so users without the openai package can still run
    the script in stub mode."""
    from openai import OpenAI  # type: ignore

    client = OpenAI()

    def score(prompt: str, system: str | None) -> str:
        # Single-shot chat completion with temperature 0 for max
        # determinism. Return the raw text so RedHop's parse_score can
        # handle "0.85" AND so the claim-extraction path (which expects
        # raw text, not a number) works without special-casing.
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

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
    """Returns a list of per-case dicts {label, faithfulness, relevancy,
    similarity} from Ragas, or None if Ragas isn't installed. Real LLM
    calls — needs `OPENAI_API_KEY`."""
    try:
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness as r_faithfulness,
            answer_relevancy as r_relevancy,
            answer_similarity as r_similarity,
        )
        from datasets import Dataset  # type: ignore
    except ImportError:
        return None

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
        metrics=[r_faithfulness, r_relevancy, r_similarity],
    )
    df = result.to_pandas()
    return [
        {
            "label": test_set[i].label,
            "faithfulness": float(df.iloc[i]["faithfulness"]),
            "relevancy": float(df.iloc[i]["answer_relevancy"]),
            "similarity": float(df.iloc[i]["answer_similarity"]),
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
    print("  Side-by-side: RedHop vs Ragas (per-case scores)")
    print("=" * 88)
    print(
        f"  {'case':<14} "
        f"{'faith RH':>10} {'faith Rg':>10} "
        f"{'relev RH':>10} {'relev Rg':>10} "
        f"{'corr RH':>10} {'sim Rg':>10}"
    )
    print("  " + "-" * 84)
    for rh, rg in zip(redhop_results, ragas_scores):
        print(
            f"  {rh.label:<14} "
            f"{fmt_score(rh.redhop_faithfulness)} {fmt_score(rg['faithfulness'])} "
            f"{fmt_score(rh.redhop_relevancy)} {fmt_score(rg['relevancy'])} "
            f"{fmt_score(rh.redhop_correctness)} {fmt_score(rg['similarity'])}"
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

    print()
    print("  Pairwise agreement:")
    for label, getter_rh, getter_rg in [
        ("faithfulness", lambda r: r.redhop_faithfulness, lambda g: g["faithfulness"]),
        ("relevancy", lambda r: r.redhop_relevancy, lambda g: g["relevancy"]),
        ("correctness↔similarity", lambda r: r.redhop_correctness, lambda g: g["similarity"]),
    ]:
        rhv = [getter_rh(r) or 0.0 for r in redhop_results]
        rgv = [getter_rg(g) for g in ragas_scores]
        print(f"  {label:<25} r={pearson(rhv, rgv):+.3f}  MAE={mae(rhv, rgv):.3f}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    # Try to use the real LLM judge if both the env var AND the openai
    # package are present. The "ImportError" check inside
    # make_openai_judge would only fire AFTER the banner if we did the
    # check naively; probe up front so the banner is honest.
    use_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if use_openai:
        try:
            import openai  # noqa: F401
        except ImportError:
            use_openai = False

    print()
    print("=" * 88)
    banner_judge = f"OpenAI {MODEL_DEFAULT}" if use_openai else "deterministic stub"
    print(f"  RedHop calibration — judge = {banner_judge}")
    print("=" * 88)

    if use_openai:
        judge = make_openai_judge()
    else:
        if os.environ.get("OPENAI_API_KEY"):
            print("  (OPENAI_API_KEY is set but `openai` not installed — falling back to stub)")
        else:
            print("  (set OPENAI_API_KEY + `pip install openai` for real LLM calls)")
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
    if use_openai:
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
            "judge": "openai" if use_openai else "stub",
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
