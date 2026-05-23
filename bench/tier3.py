#!/usr/bin/env python3
"""Tier-3: downstream answer quality — RedHop vs LangChain vs LlamaIndex.

The decisive test. Reuses the *same* Tier-1 contexts (same docs, BM25, same
budget) but now feeds each system's assembled context to an LLM and scores the
answer against gold (SQuAD-style token F1 + exact match). Retention is a proxy;
this is whether the context actually answers the question.

Systems: redhop (reasoning_preserving, the shipped strategy), redhop[topk]
(does the strategy matter downstream?), langchain, llamaindex — all BM25, same
budget, same chunk defaults as Tier 1.

Calls gpt-4o-mini via OpenRouter (parallel, cached to disk — resumable, no
double-spend). Run:  bench/.venv/bin/python bench/tier3.py [--n 150]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import redhop

from compare import CANDIDATE_K, ctx_langchain, ctx_llamaindex

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "bench" / "tier3_cache.json"
MODEL = "openai/gpt-4o-mini"


# Tier-3 golds are the actual ANSWERS (not the evidence used for retention).
def cuad_qa(limit: int):
    data = json.loads((REPO / "data/cuad/cuad_sample.json").read_text())["data"]
    n = 0
    for c in data:
        for p in c["paragraphs"]:
            for qa in p["qas"]:
                if n >= limit:
                    return
                if qa["answers"]:
                    yield p["context"], qa["question"], qa["answers"][0]["text"]
                    n += 1


def hotpot_qa(limit: int):
    data = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    for ex in data[:limit]:
        doc = "\n\n".join(" ".join(s) for _, s in ex["context"])
        yield doc, ex["question"], ex["answer"]  # the real short answer


def ctx_redhop(doc, q, budget, strategy="reasoning_preserving"):
    d = redhop.Document.from_text(doc, strategy=strategy, candidate_k=CANDIDATE_K)
    return d.context(q, budget=budget).text()


SYSTEMS = {
    "redhop": lambda d, q, b: ctx_redhop(d, q, b, "reasoning_preserving"),
    "redhop[topk]": lambda d, q, b: ctx_redhop(d, q, b, "raw_topk"),
    "langchain": ctx_langchain,
    "llamaindex": ctx_llamaindex,
}

REFUSALS = ("insufficient", "cannot", "not enough", "no information", "does not contain")


# ---- SQuAD-style scoring ----

def norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred: str, gold: str) -> float:
    return float(norm(pred) == norm(gold))


def ask(prompt: str) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    body = json.dumps(
        {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR {e}]"


def prompt_for(ctx: str, q: str, task: str) -> str:
    if task == "extract":  # CUAD: span extraction
        instr = (
            "Extract the verbatim span from the contract below that answers the question. "
            "Output only that span. If the contract does not contain it, reply exactly INSUFFICIENT."
        )
    else:  # HotpotQA: concise factoid QA
        instr = (
            "Answer the question using ONLY the context below. Be concise (a few words). "
            "If the context does not contain the answer, reply exactly INSUFFICIENT."
        )
    return f"{instr}\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


def run(items, budget: int, label: str, task: str, cache: dict):
    items = list(items)
    # Build every system's context for every item, collect LLM jobs.
    jobs = {}  # key -> prompt
    meta = []  # (idx, gold)
    for idx, (doc, q, gold) in enumerate(items):
        meta.append((idx, gold))
        for sysname, fn in SYSTEMS.items():
            try:
                ctx = fn(doc, q, budget)
            except Exception:  # noqa: BLE001
                ctx = ""
            jobs[f"{label}|{sysname}|{idx}"] = prompt_for(ctx, q, task)

    pending = {k: v for k, v in jobs.items() if k not in cache}
    print(f"[{label}] {len(items)} items × {len(SYSTEMS)} systems; {len(pending)} LLM calls")
    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(ask, p): k for k, p in pending.items()}
            for fut in as_completed(futs):
                res = fut.result()
                if not res.startswith("[ERROR"):
                    cache[futs[fut]] = res
                done += 1
                if done % 100 == 0:
                    CACHE.write_text(json.dumps(cache))
                    print(f"  ...{done}/{len(pending)}")
        CACHE.write_text(json.dumps(cache))

    agg = {s: {"f1": 0.0, "em": 0.0, "ref": 0, "n": 0} for s in SYSTEMS}
    for idx, gold in meta:
        for s in SYSTEMS:
            ans = cache.get(f"{label}|{s}|{idx}", "")
            a = agg[s]
            a["f1"] += f1(ans, gold)
            a["em"] += em(ans, gold)
            a["ref"] += int(any(m in ans.lower() for m in REFUSALS))
            a["n"] += 1

    print(f"\n==== {label}  (gpt-4o-mini, budget {budget}, n={len(items)}) ====")
    print(f"  {'system':<14} {'F1':>6} {'EM':>6} {'refusal%':>9}")
    print("  " + "-" * 38)
    for s in SYSTEMS:
        a = agg[s]
        n = max(a["n"], 1)
        print(f"  {s:<14} {a['f1'] / n:>6.3f} {a['em'] / n:>6.3f} {100 * a['ref'] / n:>8.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    run(cuad_qa(args.n), budget=2000, label="CUAD", task="extract", cache=cache)
    run(hotpot_qa(args.n), budget=400, label="HotpotQA", task="qa", cache=cache)


if __name__ == "__main__":
    main()
