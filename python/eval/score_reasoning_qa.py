#!/usr/bin/env python3
"""Large-n end-to-end test of the ReasoningPreserving context strategy.

The retention experiment (Rust, hermetic, n=1327, CIs) proved
ReasoningPreserving rescues second hops an aggressive relevance filter
drops. That is *reachability*. This script tests *reasoning success*:
does keeping the second hop produce better answers?

RedHop (Rust, emit_reasoning_qa) built, per gap-qualified multi-hop
HotpotQA query, four contexts from the SAME polluted input at an
aggressive filter threshold:

  ctx_gold_only  gold only (clean ceiling)
  ctx_polluted   gold + 8 off-document distractors (noisy retrieval)
  ctx_filtered   polluted → DistractorFiltered (taxes the second hop)
  ctx_reasoning  polluted → ReasoningPreserving (rescues the second hop)

Decisive comparison: ctx_reasoning − ctx_filtered, with a paired
bootstrap 95% CI. Same junk-suppression regime; the systematic
difference is the rescued second hop.

Mechanism check (separating reachability from reasoning success): the
Rust emitter labels, per query, whether each strategy kept the labeled
second hop. We isolate the RESCUED subset — queries where reasoning kept
the second hop but filtered dropped it — and test whether the answer
improvement concentrates there, as the second-hop-tax theory predicts.

Boundary: RedHop builds contexts; this lab script judges answers.
LLM access: the `claude` CLI. Responses are cached to disk (resumable).

Usage:
    python scripts/score_reasoning_qa.py [--n 300] [--model haiku]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTEXTS = REPO / "exports" / "reasoning_qa_contexts.jsonl"
CACHE = REPO / "exports" / "reasoning_qa_cache.json"

CONDITIONS = ["ctx_gold_only", "ctx_polluted", "ctx_filtered", "ctx_reasoning"]

STOP = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "was", "were", "are", "be"}
REFUSAL_MARKERS = [
    "insufficient",
    "cannot determine",
    "not enough information",
    "no information",
    "does not contain",
    "doesn't contain",
    "unable to",
    "i don't know",
    "cannot answer",
    "not provide",
    "not mention",
    "not specify",
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
    # OpenRouter for model ids of the form "vendor/model" (e.g. openai/gpt-4o-mini);
    # the local `claude` CLI otherwise (haiku/sonnet).
    if "/" in model:
        return _ask_openrouter(prompt, model)
    try:
        out = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True,
            text=True,
            timeout=60,
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
    body = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = _json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR {e}]"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def boot_ci(deltas: list[float], iters: int = 2000) -> tuple[float, float, float]:
    """Mean + 95% bootstrap CI of a paired per-query delta vector."""
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(0x5EED)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return mean(deltas), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", type=str, default="haiku")
    ap.add_argument(
        "--input",
        type=str,
        default=str(CONTEXTS),
        help="contexts JSONL (default: exports/reasoning_qa_contexts.jsonl)",
    )
    args = ap.parse_args()

    contexts = Path(args.input)
    # Cache is per-context-set: different contexts (e.g. different τ) must not
    # collide on the (id|cond|model) key.
    cache_file = contexts.with_name(contexts.stem + "_cache.json")

    rows = [json.loads(l) for l in contexts.read_text().splitlines() if l.strip()]
    rows = rows[: args.n]
    cache: dict[str, str] = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    total = len(rows) * len(CONDITIONS)
    cached_n = sum(1 for r in rows for c in CONDITIONS if f"{r['id']}|{c}|{args.model}" in cache)
    print(
        f"scoring {len(rows)} queries × {len(CONDITIONS)} conditions = {total} LLM calls "
        f"(model={args.model}, already cached={cached_n})\n"
    )

    # Per-query, per-condition kw_recall + refusal.
    kw = {c: [] for c in CONDITIONS}
    ref = dict.fromkeys(CONDITIONS, 0)
    # Per-query deltas for the decisive paired comparisons.
    d_reason_filt: list[float] = []  # reasoning − filtered
    d_gold_poll: list[float] = []  # gold_only − polluted
    # Rescued subset: reasoning kept the second hop, filtered dropped it.
    rescued_delta: list[float] = []
    both_kept_delta: list[float] = []

    # Phase 1: fill the cache for all uncached (query, condition) pairs in
    # parallel. The calls are pure I/O waits (network + model latency), so a
    # thread pool turns a ~1h sequential run into a few minutes. Resumable:
    # anything already cached is skipped.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending = [
        (f"{r['id']}|{cond}|{args.model}", r["question"], r[cond])
        for r in rows
        for cond in CONDITIONS
        if f"{r['id']}|{cond}|{args.model}" not in cache
    ]
    if pending:
        print(f"  dispatching {len(pending)} uncached calls across 16 workers...")
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
                    print(f"  ...{done}/{len(pending)} new calls")
        cache_file.write_text(json.dumps(cache))

    # Phase 2: analysis reads purely from the now-complete cache.
    for r in rows:
        per = {}
        for cond in CONDITIONS:
            key = f"{r['id']}|{cond}|{args.model}"
            ans = cache.get(key, "")
            rec = kw_recall(ans, r["gold_answer"])
            per[cond] = rec
            kw[cond].append(rec)
            ref[cond] += int(is_refusal(ans))
        d_reason_filt.append(per["ctx_reasoning"] - per["ctx_filtered"])
        d_gold_poll.append(per["ctx_gold_only"] - per["ctx_polluted"])
        # Full-gold-retention split (refined): RESCUED = reasoning kept MORE
        # gold than filtered; CONTROL = identical gold retention. (reasoning's
        # kept-gold is always a superset of filtered's, verified 0 violations.)
        if r["gold_kept_reasoning"] > r["gold_kept_filtered"]:
            rescued_delta.append(per["ctx_reasoning"] - per["ctx_filtered"])
        else:
            both_kept_delta.append(per["ctx_reasoning"] - per["ctx_filtered"])

    cache_file.write_text(json.dumps(cache))

    print(f"\n──── downstream QA by condition (n={len(rows)}, {args.model}) ────")
    print(f"  {'condition':<16} {'kw_recall':>10} {'refusal%':>10}")
    print("  " + "─" * 40)
    for c in CONDITIONS:
        print(f"  {c:<16} {mean(kw[c]):>10.3f} {ref[c] / max(len(rows), 1) * 100:>9.0f}%")

    print("\n──── decisive paired comparisons (95% bootstrap CI) ────")
    m, lo, hi = boot_ci(d_gold_poll)
    print(f"  distractors hurt   (gold_only − polluted):  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    m, lo, hi = boot_ci(d_reason_filt)
    print(f"  REASONING vs FILTER (reasoning − filtered):  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    sig = lo > 0
    verdict = (
        "✓ ReasoningPreserving SIGNIFICANTLY beats aggressive filtering (CI excludes 0)"
        if sig
        else "~ reasoning vs filter within noise at this n (CI includes 0) — report honestly"
    )
    print(f"      → {verdict}")

    print("\n──── mechanism: reachability → reasoning success (full-gold split) ────")
    print(f"  RESCUED subset (reasoning kept MORE gold than filtered):  n={len(rescued_delta)}")
    if rescued_delta:
        m, lo, hi = boot_ci(rescued_delta)
        print(f"     reasoning − filtered on rescued:  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"  CONTROL subset (IDENTICAL gold retention):                n={len(both_kept_delta)}")
    if both_kept_delta:
        m, lo, hi = boot_ci(both_kept_delta)
        print(f"     reasoning − filtered on control:  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("  Theory: the gain should concentrate in RESCUED (gold reachability")
    print("  differs) and CONTROL should be ~0 (same gold). A non-zero CONTROL now")
    print("  means the residual is NOT gold loss but junk readmission / reordering.")


if __name__ == "__main__":
    main()
