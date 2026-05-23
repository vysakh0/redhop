#!/usr/bin/env python3
"""DILUTION test: does pruning a bloated-but-fitting context RECOVER accuracy?

The reasoning-QA test ran at a budget where nothing had to be cut and found
pruning ≈ no-op. That regime is rigged — when everything fits, any filter only
loses info. This script tests the regime that actually matters for large
windows: a context that FITS but is mostly junk (lost-in-the-middle).

Per gap-qualified multi-hop HotpotQA query (built by the Rust emit_dilution),
four contexts from the SAME large polluted pool:

  ctx_gold_only  gold only (clean ceiling)
  ctx_polluted   gold + ~1000 distractors, ALL of it (~30k tok; stuff-it-all)
  ctx_pruned     polluted → ReasoningPreserving, pruned to budget (~2k tok)
  ctx_topk       polluted → MaxDensity, truncated to budget (naive relevance)

Decisive comparisons (paired bootstrap 95% CI):
  pruned − polluted : does pruning the bloat RECOVER accuracy? (>0 ⇒ dilution
                      real ⇒ the optimizer earns its keep at big windows)
  pruned − topk     : does bridge-aware pruning beat naive truncation at the
                      same budget? (the second-hop tax under real cuts)

Boundary: RedHop builds contexts; this lab script judges answers.
Usage: python python/eval/score_dilution.py [--n 200] [--model <id>]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTEXTS = REPO / "exports" / "dilution_contexts.jsonl"

CONDITIONS = ["ctx_gold_only", "ctx_polluted", "ctx_pruned", "ctx_topk"]

STOP = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "was", "were", "are", "be"}
REFUSAL_MARKERS = [
    "insufficient", "cannot determine", "not enough information", "no information",
    "does not contain", "doesn't contain", "unable to", "i don't know", "cannot answer",
    "not provide", "not mention", "not specify",
]


def keywords(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return {t for t in toks if len(t) > 1 and t not in STOP}


def kw_recall(answer: str, gold: str) -> float:
    g = keywords(gold)
    if not g:
        return 1.0
    return len(g & keywords(answer)) / len(g)


def is_refusal(answer: str) -> bool:
    al = answer.lower()
    return any(m in al for m in REFUSAL_MARKERS)


def ask_llm(question: str, context: str, model: str) -> str:
    prompt = (
        "Answer the question using ONLY the context below. Be concise (a few "
        "words). If the context does not contain the answer, reply exactly "
        "INSUFFICIENT.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    if "/" in model:
        return _ask_openrouter(prompt, model)
    import subprocess
    try:
        out = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True, text=True, timeout=120,
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR {e}]"


def _ask_openrouter(prompt: str, model: str) -> str:
    import json as _json
    import os
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return "[ERROR no OPENROUTER_API_KEY]"
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = _json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR {e}]"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def boot_ci(deltas: list[float], iters: int = 2000) -> tuple[float, float, float]:
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(0x5EED)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return mean(deltas), means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", type=str, default="haiku")
    ap.add_argument("--input", type=str, default=str(CONTEXTS))
    args = ap.parse_args()

    contexts = Path(args.input)
    cache_file = contexts.with_name(contexts.stem + f"_cache_{args.model.replace('/', '_')}.json")

    rows = [json.loads(l) for l in contexts.read_text().splitlines() if l.strip()][: args.n]
    cache: dict[str, str] = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    pending = [
        (f"{r['id']}|{cond}", r["question"], r[cond])
        for r in rows for cond in CONDITIONS
        if f"{r['id']}|{cond}" not in cache
    ]
    print(f"scoring {len(rows)} queries × {len(CONDITIONS)} conds (model={args.model}); "
          f"{len(pending)} uncached")
    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(ask_llm, q, ctx, args.model): key for key, q, ctx in pending}
            for fut in as_completed(futs):
                res = fut.result()
                if res.startswith("[ERROR"):
                    continue  # don't cache failures — let a re-run retry them
                cache[futs[fut]] = res
                done += 1
                if done % 50 == 0:
                    cache_file.write_text(json.dumps(cache))
                    print(f"  ...{done}/{len(pending)}")
        cache_file.write_text(json.dumps(cache))

    kw = {c: [] for c in CONDITIONS}
    ref = {c: 0 for c in CONDITIONS}
    d_prune_poll: list[float] = []   # pruned − polluted (dilution recovery)
    d_prune_topk: list[float] = []   # pruned − topk (bridge-aware vs naive)
    d_gold_poll: list[float] = []    # gold − polluted (dilution ceiling gap)
    # Subset where pruned kept the second hop but topk dropped it.
    rescued_delta: list[float] = []
    for r in rows:
        per = {}
        for cond in CONDITIONS:
            ans = cache.get(f"{r['id']}|{cond}", "")
            rec = kw_recall(ans, r["gold_answer"])
            per[cond] = rec
            kw[cond].append(rec)
            ref[cond] += int(is_refusal(ans))
        d_prune_poll.append(per["ctx_pruned"] - per["ctx_polluted"])
        d_prune_topk.append(per["ctx_pruned"] - per["ctx_topk"])
        d_gold_poll.append(per["ctx_gold_only"] - per["ctx_polluted"])
        if r["second_hop_in_pruned"] and not r["second_hop_in_topk"]:
            rescued_delta.append(per["ctx_pruned"] - per["ctx_topk"])

    print("\n──── downstream QA by condition (n={}, {}) ────".format(len(rows), args.model))
    print(f"  {'condition':<16} {'kw_recall':>10} {'refusal%':>10}")
    print("  " + "─" * 40)
    for c in CONDITIONS:
        print(f"  {c:<16} {mean(kw[c]):>10.3f} {ref[c]/max(len(rows),1)*100:>9.0f}%")

    print("\n──── decisive paired comparisons (95% bootstrap CI) ────")
    for label, d in [
        ("DILUTION: pruned − polluted (recovery)", d_prune_poll),
        ("bridge-aware: pruned − topk          ", d_prune_topk),
        ("ceiling gap: gold − polluted         ", d_gold_poll),
    ]:
        m, lo, hi = boot_ci(d)
        sig = "✓ sig" if (lo > 0 or hi < 0) else "~ ns"
        print(f"  {label}:  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")

    print(f"\n  second-hop-rescued subset (pruned kept it, topk dropped): n={len(rescued_delta)}")
    if rescued_delta:
        m, lo, hi = boot_ci(rescued_delta)
        print(f"     pruned − topk on rescued:  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print("\n  Read: pruned−polluted > 0 ⇒ dilution is real and pruning recovers")
    print("  accuracy (the library earns its keep at large windows). ≈ 0 ⇒ the")
    print("  model handles the bloat fine and the optimizer adds no accuracy.")


if __name__ == "__main__":
    main()
