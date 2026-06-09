#!/usr/bin/env python3
"""Real-workload faithfulness correlation: RedHop vs Ragas on HotpotQA.

Loads N HotpotQA dev examples, generates an answer for each via the
same LLM both libraries use as judge, then scores faithfulness with
both libraries and reports correlation.

Why this exists: the 5-case calibration probe in
`eval_judged_calibration.py` is a wiring smoke + edge-case bucket
check. This is the actual "does our metric track Ragas's on real RAG
outputs" measurement. Strong correlation here is strong evidence that
the few-shot + batched-verification path produces equivalent numbers
to Ragas's per-statement verification.

Run:
  OPENROUTER_API_KEY=sk-or-... bench/.venv/bin/python bench/eval_correlation_hotpot.py
  OPENROUTER_API_KEY=sk-or-... bench/.venv/bin/python bench/eval_correlation_hotpot.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="number of HotpotQA examples")
    p.add_argument(
        "--model",
        default=os.environ.get("EVAL_MODEL", "openai/gpt-4o-mini"),
        help="LLM model id (OpenRouter: vendor-prefixed; OpenAI: bare)",
    )
    p.add_argument(
        "--context",
        choices=("supporting", "distractor_only", "all"),
        default="distractor_only",
        help=(
            "supporting=gold ctx (zero variance, uninformative); "
            "distractor_only=non-supporting paras (LLM hallucinates/refuses → variance); "
            "all=full HotpotQA distractor setting"
        ),
    )
    return p.parse_args()


# ── Workload ────────────────────────────────────────────────────────────────


@dataclass
class Case:
    qid: str
    query: str
    ctx_text: str
    gold_answer: str


def load_hotpot(limit: int, context_mode: str = "distractor_only") -> list[Case]:
    """Load N HotpotQA dev examples.

    `context_mode` controls what gets put into the ctx_text:
      • "supporting"      — only the supporting paragraphs (gold ctx).
                            LLM will answer faithfully → all faithfulness
                            ≈ 1.0 → zero variance, useless for correlation.
      • "distractor_only" — only the NON-supporting paragraphs. The LLM
                            either refuses, partially guesses, or
                            hallucinates → real faithfulness variance →
                            informative correlation signal.
      • "all"             — supporting + distractor (the default
                            HotpotQA distractor setting, ~10 paragraphs).
                            Mixed signal.
    """
    path = REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json"
    data = json.loads(path.read_text())
    out: list[Case] = []
    for ex in data:
        if len(out) >= limit:
            break
        gold = ex.get("answer", "").strip()
        if not gold or gold.lower() in {"yes", "no"}:
            continue
        paras = {title: sents for title, sents in ex["context"]}
        supporting_titles = set(title for title, _idx in ex["supporting_facts"])

        if context_mode == "supporting":
            keep = [t for t in paras if t in supporting_titles]
        elif context_mode == "distractor_only":
            keep = [t for t in paras if t not in supporting_titles]
        elif context_mode == "all":
            keep = list(paras.keys())
        else:
            raise ValueError(f"unknown context_mode: {context_mode}")

        ctx_text = "\n\n".join(" ".join(paras[t]) for t in keep)
        if not ctx_text.strip():
            continue
        out.append(
            Case(
                qid=ex["_id"],
                query=ex["question"],
                ctx_text=ctx_text,
                gold_answer=gold,
            )
        )
    return out


# ── LLM client (OpenRouter or OpenAI) ───────────────────────────────────────


def make_openai_client():
    from openai import OpenAI  # type: ignore

    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    return OpenAI()


def generate_answer(client, model: str, query: str, ctx_text: str) -> str:
    """Use the LLM to generate a candidate answer given the gold context.
    Temperature 0 for reproducibility."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the QUESTION using ONLY the CONTEXT. Be concise: one or "
                    "two short sentences. If the CONTEXT does not contain the answer, "
                    "say 'I don't know.'"
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{ctx_text}\n\nQUESTION: {query}",
            },
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ── RedHop side ─────────────────────────────────────────────────────────────


def make_redhop_judge(client, model: str) -> redhop.Judge:
    def score(prompt, system):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        try:
            return float(text)
        except ValueError:
            return {"score": 0.0, "raw_text": text, "model": model}

    return redhop.Judge.from_callable(score, name=model).cached()


def run_redhop(cases: list[Case], answers: list[str], judge: redhop.Judge):
    """For each case, build a single-chunk Document and score faithfulness
    via BOTH single-prompt and decomposed paths."""
    out = []
    for c, ans in zip(cases, answers):
        doc = redhop.Document.from_text(c.ctx_text)
        ctx = doc.context(c.query)
        r_single = redhop.evaluate(c.query, ctx, answer=ans, judge=judge)
        r_decomp = redhop.evaluate(
            c.query, ctx, answer=ans, judge=judge, decompose_faithfulness=True,
        )
        out.append(
            {
                "qid": c.qid,
                "redhop_single": r_single.faithfulness_judged,
                "redhop_decomposed": r_decomp.faithfulness_judged,
                "n_claims_extracted": r_decomp.n_faithfulness_claims_extracted,
                "n_claims_supported": r_decomp.n_faithfulness_claims_supported,
            }
        )
    return out


# ── Ragas side ──────────────────────────────────────────────────────────────


def run_ragas(cases: list[Case], answers: list[str], model: str):
    from ragas import evaluate as ragas_evaluate  # type: ignore
    from ragas.metrics import faithfulness as r_faithfulness  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore
    from datasets import Dataset  # type: ignore

    if os.environ.get("OPENROUTER_API_KEY"):
        llm = ChatOpenAI(
            model=model,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
    else:
        llm = ChatOpenAI(model=model, temperature=0.0)
    wrapped = LangchainLLMWrapper(llm)
    r_faithfulness.llm = wrapped

    ds = Dataset.from_list(
        [
            {
                "question": c.query,
                "answer": a,
                "ground_truth": c.gold_answer,
                "contexts": [c.ctx_text],
            }
            for c, a in zip(cases, answers)
        ]
    )
    result = ragas_evaluate(dataset=ds, metrics=[r_faithfulness], llm=wrapped)
    df = result.to_pandas()
    return [
        {"qid": cases[i].qid, "ragas_faithfulness": float(df.iloc[i]["faithfulness"])}
        for i in range(len(cases))
    ]


# ── Stats ───────────────────────────────────────────────────────────────────


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
    if not xs:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("ERROR: set OPENROUTER_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    provider = "OpenRouter" if os.environ.get("OPENROUTER_API_KEY") else "OpenAI"
    print()
    print("=" * 88)
    print(f"  HotpotQA faithfulness correlation — {provider} {args.model}, n={args.n}")
    print("=" * 88)

    print(f"\nLoading HotpotQA-{args.n} (context_mode={args.context})…")
    cases = load_hotpot(args.n, context_mode=args.context)
    print(f"  loaded {len(cases)} cases")

    client = make_openai_client()

    print("\nGenerating answers (one LLM call per case)…")
    t0 = time.perf_counter()
    answers = [generate_answer(client, args.model, c.query, c.ctx_text) for c in cases]
    print(f"  generated {len(answers)} answers in {time.perf_counter() - t0:.1f}s")

    print("\nRedHop scoring (single-prompt + decomposed)…")
    t0 = time.perf_counter()
    judge = make_redhop_judge(client, args.model)
    redhop_out = run_redhop(cases, answers, judge)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("\nRagas scoring (faithfulness)…")
    t0 = time.perf_counter()
    ragas_out = run_ragas(cases, answers, args.model)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print()
    print("=" * 88)
    print("  Per-case faithfulness scores")
    print("=" * 88)
    print(f"  {'qid':<24} {'RedHop S':>10} {'RedHop D':>10} {'Ragas':>10} {'|D-Rg|':>10}")
    print("  " + "-" * 70)
    for rh, rg in zip(redhop_out, ragas_out):
        s = rh["redhop_single"]
        d = rh["redhop_decomposed"]
        r = rg["ragas_faithfulness"]
        delta_d = abs(d - r) if d is not None else float("nan")
        s_str = f"{s:10.3f}" if s is not None else "      null"
        d_str = f"{d:10.3f}" if d is not None else "      null"
        r_str = f"{r:10.3f}"
        print(f"  {rh['qid']:<24} {s_str} {d_str} {r_str} {delta_d:10.3f}")

    print()
    rh_single = [r["redhop_single"] or 0.0 for r in redhop_out]
    rh_decomp = [r["redhop_decomposed"] or 0.0 for r in redhop_out]
    rg_faith = [r["ragas_faithfulness"] for r in ragas_out]
    print(f"  RedHop single-prompt  ↔ Ragas:  r={pearson(rh_single, rg_faith):+.3f}  MAE={mae(rh_single, rg_faith):.3f}")
    print(f"  RedHop decomposed     ↔ Ragas:  r={pearson(rh_decomp, rg_faith):+.3f}  MAE={mae(rh_decomp, rg_faith):.3f}")

    out_dir = REPO / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval_correlation_hotpot_n{len(cases)}.json"
    out.write_text(
        json.dumps(
            {
                "provider": provider,
                "model": args.model,
                "n": len(cases),
                "cases": [
                    {
                        **rh,
                        **rg,
                        "answer": ans,
                        "gold_answer": c.gold_answer,
                    }
                    for rh, rg, ans, c in zip(redhop_out, ragas_out, answers, cases)
                ],
                "pearson": {
                    "redhop_single_vs_ragas": pearson(rh_single, rg_faith),
                    "redhop_decomposed_vs_ragas": pearson(rh_decomp, rg_faith),
                },
                "mae": {
                    "redhop_single_vs_ragas": mae(rh_single, rg_faith),
                    "redhop_decomposed_vs_ragas": mae(rh_decomp, rg_faith),
                },
            },
            indent=2,
        )
    )
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
