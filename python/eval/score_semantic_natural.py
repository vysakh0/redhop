#!/usr/bin/env python3
"""Tier-3 for the natural semantic-mismatch split: does retrieval mode change
ANSWERS, and does the lexical/semantic split predict where dense helps?

Reads exports/semantic_natural_contexts.jsonl (emitted by the Rust
`semantic_natural` example): one row per (item, mode) with the assembled
context + question + gold answer + subset (lexical / semantic). Answers each with
gpt-4o-mini and scores SQuAD-style F1/EM, aggregated by subset × mode.

Parallel, cached to disk (resumable). Needs OPENROUTER_API_KEY.
Run:  python python/eval/score_semantic_natural.py [--n 0=all]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTEXTS = REPO / "exports" / "semantic_natural_contexts.jsonl"
SIGNALS = REPO / "exports" / "semantic_natural_signals.jsonl"
CACHE = REPO / "exports" / "semantic_natural_cache.json"
MODEL = "openai/gpt-4o-mini"


def norm(s: str) -> str:
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
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


def prompt_for(ctx: str, q: str) -> str:
    return (
        "Answer the question using ONLY the context below. Be concise (a few words). "
        "If the context does not contain the answer, reply exactly INSUFFICIENT.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap rows (0 = all)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in CONTEXTS.read_text().splitlines() if l.strip()]
    if args.n:
        rows = rows[: args.n]
    cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def key(r):
        return f"{r['id']}|{r['mode']}"

    pending = [
        (key(r), prompt_for(r["context"], r["question"])) for r in rows if key(r) not in cache
    ]
    print(f"{len(rows)} rows; {len(pending)} LLM calls (model={MODEL})")
    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(ask, p): k for k, p in pending}
            for fut in as_completed(futs):
                res = fut.result()
                if not res.startswith("[ERROR"):
                    cache[futs[fut]] = res
                done += 1
                if done % 100 == 0:
                    CACHE.write_text(json.dumps(cache))
                    print(f"  ...{done}/{len(pending)}")
        CACHE.write_text(json.dumps(cache))

    agg = defaultdict(lambda: {"f1": 0.0, "em": 0.0, "n": 0})
    for r in rows:
        ans = cache.get(key(r), "")
        for bucket in ((r["subset"], r["mode"]), ("ALL", r["mode"])):
            a = agg[bucket]
            a["f1"] += f1(ans, r["gold_answer"])
            a["em"] += em(ans, r["gold_answer"])
            a["n"] += 1

    print("\nDownstream answer quality (gpt-4o-mini) — F1 / EM by subset × mode")
    print(f"  {'subset':<12} {'bm25':>13} {'dense':>13} {'hybrid':>13}")
    print("  " + "-" * 54)
    for subset in ("lexical", "semantic", "ALL"):
        cells = []
        for mode in ("bm25", "dense", "hybrid"):
            a = agg[(subset, mode)]
            n = max(a["n"], 1)
            cells.append(f"{a['f1'] / n:.2f}/{a['em'] / n:.2f}")
        n = agg[(subset, "bm25")]["n"]
        print(f"  {subset:<8} (n={n:>4}) {cells[0]:>13} {cells[1]:>13} {cells[2]:>13}")

    # Conditional escalation downstream (reuses cached answers — no new calls).
    if not SIGNALS.exists():
        return
    margin = {
        r["id"]: r["margin"]
        for r in (json.loads(l) for l in SIGNALS.read_text().splitlines() if l.strip())
    }
    gold = {r["id"]: r["gold_answer"] for r in rows}
    ids = sorted(gold)

    def policy_f1(pick):  # pick(id) -> mode
        f1s = ems = esc = 0.0
        for i in ids:
            mode = pick(i)
            ans = cache.get(f"{i}|{mode}", "")
            f1s += f1(ans, gold[i])
            ems += em(ans, gold[i])
            esc += int(mode == "dense")
        m = len(ids)
        return f1s / m, ems / m, 100 * esc / m

    print("\nConditional escalation downstream (margin < τ → dense)  [cached, free]")
    print(f"  {'policy':<22} {'F1':>6} {'EM':>6} {'escalated%':>11}")
    for label, pick in [
        ("always bm25", lambda i: "bm25"),
        ("always dense", lambda i: "dense"),
    ]:
        a, b, e = policy_f1(pick)
        print(f"  {label:<22} {a:>6.2f} {b:>6.2f} {e:>10.0f}%")
    for tau in (0.20, 0.30, 0.50):
        a, b, e = policy_f1(lambda i, t=tau: "dense" if margin.get(i, 1.0) < t else "bm25")
        print(f"  {'margin<' + format(tau, '.2f'):<22} {a:>6.2f} {b:>6.2f} {e:>10.0f}%")


if __name__ == "__main__":
    main()
