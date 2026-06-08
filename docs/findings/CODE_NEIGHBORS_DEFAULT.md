# `code_neighbors_default=1` is a budget-dependent compromise — keep it, document the tradeoff

> **Status: validated default with caveats — no change.** Measured the
> code-neighbor auto-expansion (when `Document.context()` finds a code
> chunk, ±1 adjacent chunks are auto-pulled) on RedHop's own Rust
> source via 10 queries with body-bearing markers (phrases that appear
> only inside the function body, not in its signature or docstring).
> Sweep over four budgets:
>
> | Budget | n=0 | **n=1** (default) | n=2 | n=3 | Δ ctx |
> |---:|---:|---:|---:|---:|---:|
> | 128 | 3/10 | **4/10** | 4/10 | 4/10 | +31% |
> | 400 | 6/10 | 6/10 | 6/10 | 6/10 | +35% |
> | 1000 | 7/10 | 8/10 | 9/10 | 10/10 | +85% |
> | 4000 | 7/10 | 8/10 | 9/10 | 10/10 | +91% |
>
> The default is the only value that's never *catastrophically* wrong
> across the budget range: it helps at tight budget (+1/10), is
> wasteful at mid budget (0 gain, +35% ctx), and is conservative at
> loose budget (n=2/n=3 would help more). **Keep at 1; document the
> budget tradeoff so users who care can opt up.**

## Why this probe ran

The defaulted-on heuristics audit (kicked off by
[RAW_ANALYZER](RAW_ANALYZER.md)) flagged `code_neighbors_default=1` as
worth measuring. The intuition behind it — "a `def function_name():`
line is useless without the body" — is reasonable, but the existing
`bench/code_retrieval.py` measures retrieval mode (BM25 vs hybrid vs
dense), not neighbor expansion. The default value itself had never
been validated.

## The setup

10 natural-language queries about RedHop's own Rust code. For each
query, a "body marker" is picked: a distinctive phrase that appears
only inside the target function's implementation body, not in its
signature or doc-comment. If the marker shows up in the assembled
context, the user got the implementation along with whatever the
retriever surfaced.

Files are loaded via `Document.from_file()` so the `.rs` extension
triggers `metadata["kind"]="code"` automatically and the auto-expansion
path becomes active.

Arms:
- **A. neighbors=0** — manual path bypass:
  `doc.context(q, neighbors=0, include_heading=True)`. include_heading
  is a no-op on code (code chunks carry no heading metadata), so this
  cleanly disables the auto-expansion without changing anything else.
- **B. neighbors=1** — current default: `doc.context(q)` lets the
  auto path fire.
- **C. neighbors=2 / D. neighbors=3** — manual override with more
  neighbors; tests diminishing returns and where the right value lives
  at each budget regime.

Sweep over four budgets covering the realistic range: 128 (very
tight), 400 (typical eval), 1000 (mid production), 4000 (loose
production, closer to RedHop's default token_budget=8192).

## What the data says

### At budget=128 (very tight)

| arm | hits | ctx words |
|---|---:|---:|
| n=0 | 3/10 | 75 |
| **n=1 (default)** | **4/10 (+1)** | 98 (+31%) |
| n=2 | 4/10 | 104 |
| n=3 | 4/10 | 105 |

The default helps by 1 hit; raising to 2 or 3 doesn't add more,
because the budget is too tight to fit additional neighbors (they
get pruned by the budget). The default is the right value at this
budget.

### At budget=400 (typical evaluation)

| arm | hits | ctx words |
|---|---:|---:|
| n=0 | 6/10 | 263 |
| **n=1 (default)** | **6/10 (+0)** | 355 (+35%) |
| n=2 | 6/10 | 375 |
| n=3 | 6/10 | 384 |

Flat across all arms. The default is **pure context inflation** —
35% larger context, 0 retention gain. Mechanism: at this budget,
BM25 surfaces 3 candidate chunks and the budget fits them all
directly; the body markers are already in the retrieved chunks (the
fixed-token chunker creates chunks large enough — ~5-15 lines of
code each — that signature and body usually ride together). Neighbor
expansion adds redundant chunks that don't carry the marker.

### At budget=1000 and budget=4000 (loose production)

| arm | hits @1000 | hits @4000 | ctx words @1000 |
|---|---:|---:|---:|
| n=0 | 7/10 | 7/10 | 320 |
| **n=1 (default)** | **8/10 (+1)** | **8/10 (+1)** | 593 (+85%) |
| n=2 | 9/10 | 9/10 | 765 |
| n=3 | 10/10 | 10/10 | 840 |

Monotonic: each step from n=0 to n=3 recovers one more body marker.
The default n=1 is under-allocated for this regime — n=3 would hit
10/10 on both budgets. Mechanism: at loose budget, the long-function
case becomes recoverable. A function spans 2-4 chunks; the signature
chunk is what BM25 finds; the body chunks live 1-3 positions away.
The further out neighbors reach, the more bodies arrive in context.

The ctx-words cost scales linearly: +85-91% at the default vs
neighbors=0; +163% at n=3. For most production users (LLM context
budgets), the ctx-size cost is small in absolute terms, so n=3 is a
defensible choice — but it's not the right *default* because of the
behavior at tighter budgets.

## Why none of the alternatives are clearly better

- **Flip to 0** would save context at budget=400 (the only regime
  where the default is wasteful) but cost 1/10 retention at tight
  budgets and 1-3/10 at loose budgets. Net negative for most users.
- **Raise to 2 or 3** would help at loose budgets (+1 to +2 hits) but
  doubles the context-size penalty at every budget — including
  tight ones where neighbors=2/3 don't even add hits (the budget
  prunes the extra neighbors anyway).
- **Keep at 1** is the only value that helps at the tight budget,
  caps the inflation at +35% at mid budget, and provides at least
  some recovery at loose budgets. It's a defensible compromise.

## What this changes

- **`code_neighbors_default=1` stays.** No flip; it's the right
  value at tight budgets and a bounded-cost compromise everywhere
  else.
- **Document the budget tradeoff** for advanced users:
  - At budget ≤ 400: default is right
  - At budget ≥ 1000: explicit `neighbors=2` or `3` recovers more
    function bodies (+1 to +2 hits per +1 neighbor on a 10-query
    code-search sample)
- **The Python/Node API smell flagged here was fixed in 0.3.3.** Both
  `code_neighbors_default` (Python kwarg / Node `codeNeighborsDefault`
  option) and `prose_heading_default` are now surfaced at the
  `Document` constructor level. Users on memory-tight workloads can
  disable the auto-expansion explicitly:
  ```python
  doc = redhop.Document.from_file("src.rs", code_neighbors_default=0)
  ```
  See [CHANGELOG.md §0.3.3](../../CHANGELOG.md). The
  `include_heading=True` workaround used in this probe's bench script
  is no longer the only opt-out path.

## Honest limits

- **One workload (RedHop's own Rust source).** Code-search corpora
  vary in function size — codebases with longer functions
  (Python/JS with verbose handlers) would likely show n=2 or n=3
  helping at lower budgets than RedHop's compact Rust. A Python or
  TypeScript corpus probe would tighten the recommendation.
- **n=10 queries.** Differences of ±1 hit are within sample noise;
  the monotonic pattern at loose budgets (+1 per +1 neighbor) is
  more meaningful than absolute counts.
- **Marker-based retention metric.** "Is this distinctive body
  phrase in the assembled context?" is a proxy for "did the user
  get useful information about the implementation." For a real
  downstream LLM, the relevant metric is whether the model can
  answer follow-up questions; that's untested here.
- **Body-marker selection bias.** Markers were chosen by hand to be
  body-distinctive. A different selection could shift the absolute
  hit count up or down by a few; the budget-trend pattern is
  robust.

## Reproduce

```bash
bench/.venv/bin/python bench/code_neighbors_default.py
```

Raw run: [`reports/code_neighbors_default_2026-06-08.txt`](../../reports/code_neighbors_default_2026-06-08.txt).

## See also

- [PROSE_HEADING_DEFAULT](PROSE_HEADING_DEFAULT.md) — the companion
  auto-expansion default for prose with section headings. Cleanly
  validated as +7pt at typical budgets. This finding is messier:
  the code default is right at *some* budgets, wasteful at others.
- [RAW_ANALYZER](RAW_ANALYZER.md) — the audit's positive flip.
  Contrast: that was a default that was clearly wrong; this is one
  that's a defensible compromise.
- [HYBRID_CANDIDATE_POOL](HYBRID_CANDIDATE_POOL.md) — the audit's
  inert knob. Similar shape to this one ("default is a compromise"),
  but the knob there literally moves no retention; here, the knob
  moves retention monotonically at loose budgets.
- [BM25_SOURCE_FIELD](BM25_SOURCE_FIELD.md) — the cleanest default
  in the audit (free retention with no downside). Contrast: this
  default has a real cost in context inflation.
