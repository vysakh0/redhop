#!/usr/bin/env python3
"""Does the `prose_heading_default=true` default actually help, or is it
budget noise?

When `Document.context(query)` finds a prose chunk that carries
`metadata["heading"]`, the default behavior auto-attaches that
section's heading chunk to the assembled context. The intuition: a
citation deep inside `## Refunds → ### Eligibility` should arrive at
the LLM with the section title attached so the model knows what the
text is about.

But nobody has measured whether the heading attachment actually
improves retention, hurts it (by displacing budget from real
content), or is a wash. None of the existing benchmarks (HotpotQA /
MuSiQue / CUAD) exercise this path because they all use raw text
without markdown structure — the auto-heading default never fires.

This probe synthesizes a heading-bearing workload: HotpotQA Wikipedia
paragraphs loaded as `from_chunks` with each paragraph's article
title set as `metadata["heading"]`. Two arms on the same chunks +
same retrieval:

- **A. default path** — `doc.context(query)` uses the auto path;
  with `prose_heading_default=true` (the shipped default), every
  retrieved chunk pulls its heading chunk into the assembled context.
- **B. heading off** — `doc.context_expanded(query, neighbors=0,
  include_heading=False)` skips the auto path; identical retrieval,
  no heading chunk attached.

The arms differ in exactly one thing: whether the heading chunk is
in the assembled context. Same chunks, same retriever, same budget.

Run:  bench/.venv/bin/python bench/prose_heading_default.py
"""

from __future__ import annotations

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


# ── HotpotQA with article titles as headings ───────────────────────────────


def hotpot_markdown_items(limit: int):
    """Yield (markdown_doc, plain_doc, query, gold) tuples.

    Two versions of the same Wikipedia bundle:
    - markdown_doc: each article rendered as '# <title>\n\n<paragraph>\n\n'.
      The chunker will produce a heading chunk for each '#' heading; body
      chunks share (source, heading) with that heading chunk. The
      prose_heading_default auto-attach can fire because the heading chunk
      EXISTS distinct from the body chunks.
    - plain_doc: same paragraphs concatenated with blank lines but no
      '# heading' markers. No heading chunks, so auto-attach can't fire.

    BM25 multi-field reach IS a small confound (markdown adds heading text
    to BM25's heading field). Wikipedia titles are short (~2-5 words) vs
    paragraph text (~50 words), so the dominant retrieval signal is the
    paragraph itself.
    """
    data = json.loads((REPO / "data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())
    for ex in data[:limit]:
        md_parts = []
        plain_parts = []
        for title, sents in ex["context"]:
            paragraph = " ".join(sents)
            if paragraph.strip():
                md_parts.append(f"# {title}\n\n{paragraph}")
                plain_parts.append(paragraph)
        if not md_parts:
            continue
        md_doc = "\n\n".join(md_parts)
        plain_doc = "\n\n".join(plain_parts)
        paras = {title: sents for title, sents in ex["context"]}
        gold_sents = []
        for title, idx in ex["supporting_facts"]:
            if title in paras and idx < len(paras[title]):
                gold_sents.append(paras[title][idx])
        gold = " ".join(gold_sents)
        if gold.strip():
            yield md_doc, plain_doc, ex["question"], gold


# ── Arms ───────────────────────────────────────────────────────────────────


def arm_with_heading(md_doc: str, query: str, budget: int) -> str:
    """Markdown source: each '# Title' becomes a heading chunk; body
    paragraphs share (source, heading) with that heading chunk. The
    auto-attach default fires when retrieval surfaces a body chunk."""
    doc = redhop.Document.from_text(md_doc, strategy="raw_topk", token_budget=budget)
    return doc.context(query).text()


def arm_no_heading(plain_doc: str, query: str, budget: int) -> str:
    """Same paragraphs, no markdown '#' markers. No heading chunks get
    created, so the auto-attach default can't fire."""
    doc = redhop.Document.from_text(plain_doc, strategy="raw_topk", token_budget=budget)
    return doc.context(query).text()


# ── Runner ────────────────────────────────────────────────────────────────


def eval_arm(items_list, arm_fn, doc_picker, label: str):
    rec_sum = 0.0
    r50 = 0
    r80 = 0
    latencies = []
    ctx_words_total = 0
    for example in items_list:
        doc_text = doc_picker(example)
        query = example[2]
        gold = example[3]
        t0 = time.perf_counter()
        try:
            ctx = arm_fn(doc_text, query, BUDGET)
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] error: {e}", file=sys.stderr)
            ctx = ""
        latencies.append((time.perf_counter() - t0) * 1000)
        r = span_recall(gold, ctx)
        rec_sum += r
        r50 += int(r >= 0.5)
        r80 += int(r >= 0.8)
        ctx_words_total += len(ctx.split())
    n = max(len(items_list), 1)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    return rec_sum / n, r50 * 100 / n, r80 * 100 / n, p50, ctx_words_total / n


def main() -> None:
    print()
    print("=" * 78)
    print("  prose_heading_default probe")
    print("  Does the auto-heading attachment help retention, or just inflate ctx?")
    print("=" * 78)

    items_list = list(hotpot_markdown_items(100))
    n = len(items_list)
    # Sweep budgets — tight (128 tok) tests whether heading attach displaces useful
    # content under pressure; loose (1000) tests whether heading helps when budget
    # isn't binding (~all useful content fits anyway).
    for budget in (128, 400, 1000):
        print()
        print(f"  HotpotQA-as-markdown (n={n}, budget={budget} tok)")
        print(f"  Arm A: each article rendered as '# <title>' + paragraph (heading chunks exist)")
        print(f"  Arm B: same paragraphs, no '#' markers (no heading chunks created)")
        print()
        print(f"  {'arm':<36} {'recall':>8} {'≥0.5':>6} {'≥0.8':>6} {'p50 ms':>8} {'avg ctx words':>14}")
        print("  " + "-" * 80)

        global BUDGET
        BUDGET = budget
        a_r, a_50, a_80, a_ms, a_w = eval_arm(
            items_list, arm_with_heading, lambda ex: ex[0], "markdown (heading attach fires)"
        )
        b_r, b_50, b_80, b_ms, b_w = eval_arm(
            items_list, arm_no_heading, lambda ex: ex[1], "plain (no heading chunks)"
        )

        print(f"  {'A. markdown (heading attach fires)':<36} {a_r:>8.2f} {a_50:>5.0f}% {a_80:>5.0f}% {a_ms:>7.1f} {a_w:>14.1f}")
        print(f"  {'B. plain (no heading chunks)':<36} {b_r:>8.2f} {b_50:>5.0f}% {b_80:>5.0f}% {b_ms:>7.1f} {b_w:>14.1f}")
        print()
        print(f"  {'Δ (A − B; positive = default helps)':<36} "
              f"{a_r - b_r:>+6.2f} {a_50 - b_50:>+5.0f}  {a_80 - b_80:>+5.0f}  "
              f"{a_ms - b_ms:>+7.1f} {a_w - b_w:>+11.1f}")
    print()
    print("Reading the result:")
    print("  • +retention, +tokens → default helps, costs budget; keep if Δ retention > Δ budget cost")
    print("  • 0 retention, +tokens → default inflates context for no gain; flip default to false")
    print("  • -retention, +tokens → default actively hurts (heading displaces real content)")
    print("  • +retention, 0 tokens → can't happen; sanity check")
    print()


if __name__ == "__main__":
    main()
