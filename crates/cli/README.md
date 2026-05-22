# neorag — eval & observability CLI

A thin, Unix-like shell over the NeoRAG context API (`build_context`,
`analyze_context`, `context_economics`). It exists for **evaluation,
observability, benchmarking, reproducibility, and context inspection** — not
as a serving runtime, workflow engine, or orchestration layer. NeoRAG remains
a reasoning-preserving context optimization library; this just exposes it.

```bash
cargo build -p neorag-cli --release      # produces target/release/neorag
```

Input is the JSON your retriever already produces:
`{"query": "...", "chunks": [{"id","text","source"?,"token_count"?,"embedding"?}]}`
(only `text` is required; pass `-` to read from stdin).

## Commands

### `compare` — strategies side-by-side (the strongest demo surface)
```bash
neorag compare --query "Who was the British PM during WWII?" \
  --input retrieval.json \
  --strategies raw_topk,distractor_filtered,reasoning_preserving \
  --gold-ids c3,c7 --second-hop-id c7        # optional → retention columns
```
Prints chunks in→out, tokens, removed, rescues, distractor ratio, evidence
density, optional gold/second-hop retention, and context previews. `--json out.json`
writes a structured artifact.

### `analyze-context` — non-destructive observability
```bash
neorag analyze-context context.json --query "..."
```
Renders the `Context Optimization Report` (density, distractors, rescues,
estimated waste, warnings). `--json` for the raw report.

### `benchmark` — reproducible strategy sweep
```bash
neorag benchmark --input labeled.json \
  --strategies raw_topk,distractor_filtered,reasoning_preserving \
  --budgets 250,800,12000 --out-dir out/
```
Labeled input adds `gold_ids` / `second_hop_id` per query. Emits
`results.json` + `SUMMARY.md` with 95% bootstrap CIs. No fabricated metrics —
every number is computed from the provided labels. (The canonical hermetic
HotpotQA run lives in [`benchmarks/context/`](../../benchmarks/context/).)

### `report` — render an artifact to markdown / HTML
```bash
neorag report results.json --markdown report.md --html report.html
```
Works on `compare --json` output and on `benchmarks/context/results.json`.

## Evidence layer

Outputs link back to the measured findings that justify the behavior:
[`docs/findings/`](../../docs/findings/README.md),
[`benchmarks/`](../../benchmarks/README.md), [`reports/`](../../reports/README.md).
