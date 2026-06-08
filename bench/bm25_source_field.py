#!/usr/bin/env python3
"""Does the BM25 multi-field reach (text + source + heading) help, or
hurt on workloads with noisy file paths?

RedHop's BM25 indexes three fields per chunk — text, source (file path
or logical handle), heading (metadata field). The query parser searches
all three with equal weight; the design intent is that a query like
`"auth"` should reach a chunk in `src/auth.rs` even when the chunk's
own text doesn't contain the word. Validated by
`quality_suite::t08_filename_reachable_via_source_field`.

But there's no measurement of what happens when source is *noise*
rather than signal — random hash paths (`/var/data/page_12345.html`),
auto-generated filenames, internal IDs. In that case the source field
is fighting BM25 with random vocabulary; it could be displacing real
signal in the ranking.

This probe tests three configurations of the same Wikipedia content:

- **A. signal-bearing source** — `source="<article title>.md"`.
  The path contains the entity name, so multi-field reach helps when
  the query is about the entity.
- **B. empty/generic source** — `source="doc.txt"` shared across all
  chunks. Effectively neutralizes the source field (every chunk's
  source matches every query equally, contributing no ranking signal).
- **C. noisy source** — `source="<random 16-char hash>.txt"`. The
  source field carries vocabulary that's never in the query, but it
  still occupies a BM25 field and (via length normalization) affects
  ranking slightly.

Heading is empty in all arms so we isolate the source-field effect.
Same paragraphs, same query, same gold across all three.

Run:  bench/.venv/bin/python bench/bm25_source_field.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import redhop

REPO = Path(__file__).resolve().parents[1]
BUDGET = 400


def words(s: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 1}


def span_recall(gold: str, ctx: str) -> float:
    g = words(gold)
    if not g:
        return 1.0
    cw = words(ctx)
    return len(g & cw) / len(g)


def fake_hash(seed: str) -> str:
    """Stable 16-char hex prefix so re-runs are deterministic."""
    return hashlib.sha1(seed.encode()).hexdigest()[:16]


# ── HotpotQA with three source configurations ─────────────────────────────


def hotpot_items(limit: int):
    """Yield (signal_chunks, generic_chunks, noisy_chunks, query, gold) per ex."""
    data = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    for ex in data[:limit]:
        signal: list = []
        generic: list = []
        noisy: list = []
        for title, sents in ex["context"]:
            text = " ".join(sents)
            if text.strip():
                signal.append(redhop.Chunk(text, source=f"{title}.md"))
                generic.append(redhop.Chunk(text, source="doc.txt"))
                noisy.append(redhop.Chunk(text, source=f"{fake_hash(title)}.txt"))
        paras = {title: sents for title, sents in ex["context"]}
        gold_sents = []
        for title, idx in ex["supporting_facts"]:
            if title in paras and idx < len(paras[title]):
                gold_sents.append(paras[title][idx])
        gold = " ".join(gold_sents)
        if gold.strip() and signal:
            yield signal, generic, noisy, ex["question"], gold


# ── Arms ───────────────────────────────────────────────────────────────────


def arm(chunks, query: str, budget: int) -> str:
    doc = redhop.Document.from_chunks(chunks, strategy="raw_topk", token_budget=budget)
    return doc.context(query).text()


# ── Runner ────────────────────────────────────────────────────────────────


def eval_arm(items_list, chunk_picker, label: str):
    rec_sum = 0.0
    r50 = 0
    r80 = 0
    latencies = []
    ctx_words = 0
    for example in items_list:
        chunks = chunk_picker(example)
        query = example[3]
        gold = example[4]
        t0 = time.perf_counter()
        try:
            ctx = arm(chunks, query, BUDGET)
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] error: {e}", file=sys.stderr)
            ctx = ""
        latencies.append((time.perf_counter() - t0) * 1000)
        r = span_recall(gold, ctx)
        rec_sum += r
        r50 += int(r >= 0.5)
        r80 += int(r >= 0.8)
        ctx_words += len(ctx.split())
    n = max(len(items_list), 1)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    return rec_sum / n, r50 * 100 / n, r80 * 100 / n, p50, ctx_words / n


def main() -> None:
    print()
    print("=" * 78)
    print("  BM25 source-field probe")
    print("  Does multi-field reach help when source is signal, hurt when noise?")
    print("=" * 78)

    items_list = list(hotpot_items(100))
    n = len(items_list)
    print()
    print(f"  HotpotQA (n={n}, budget={BUDGET} tok)")
    print(f"  All arms: same paragraphs, no heading metadata. Source varies.")
    print()
    print(f"  {'arm':<40} {'recall':>8} {'≥0.5':>6} {'≥0.8':>6} {'p50 ms':>8} {'avg ctx words':>14}")
    print("  " + "-" * 84)

    a = eval_arm(items_list, lambda ex: ex[0], "signal (article-title path)")
    b = eval_arm(items_list, lambda ex: ex[1], "generic ('doc.txt' shared)")
    c = eval_arm(items_list, lambda ex: ex[2], "noisy (random hex hash)")

    def row(label, r):
        print(f"  {label:<40} {r[0]:>8.2f} {r[1]:>5.0f}% {r[2]:>5.0f}% {r[3]:>7.1f} {r[4]:>14.1f}")

    row("A. signal-bearing source (current ideal)", a)
    row("B. generic source (neutral control)", b)
    row("C. noisy source (random hash path)", c)
    print()
    print(f"  Δ A − B (signal lift vs neutral): "
          f"recall {a[0]-b[0]:+.2f}  ≥0.5 {a[1]-b[1]:+.0f}  ≥0.8 {a[2]-b[2]:+.0f}")
    print(f"  Δ C − B (noise penalty vs neutral): "
          f"recall {c[0]-b[0]:+.2f}  ≥0.5 {c[1]-b[1]:+.0f}  ≥0.8 {c[2]-b[2]:+.0f}")
    print()
    print("Reading the result:")
    print("  • A > B > C  → multi-field reach is a real signal, hurt by noise")
    print("  • A ≈ B ≈ C  → source field doesn't move retention; safe in all regimes")
    print("  • A ≈ B, C < B → safe when meaningful, hurts on noise (warning case)")
    print("  • A > B, C ≈ B → signal helps, noise is harmless (best case)")
    print()


if __name__ == "__main__":
    main()
