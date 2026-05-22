# CLI usage

A thin, Unix-like eval/observability CLI. Build it:

```bash
cargo build -p redhop-cli --release   # → target/release/redhop
```

Input is the JSON your retriever already produces:
`{"query": "...", "chunks": [{"id","text","source"?,"token_count"?,"embedding"?}]}`
(only `text` required; `-` reads stdin).

## `compare` — strategies side by side

```bash
redhop compare --input retrieval.json \
  --strategies raw_topk,distractor_filtered,reasoning_preserving \
  --gold-ids c3,c7 --second-hop-id c7      # optional → retention columns
```

```text
strategy                chunks   tokens   removed  rescued  distr  density  gold_ret  2nd_hop
raw_topk                8→8      100      0        0        0.88   0.10     1.00      ✓
distractor_filtered     8→1      17       7        0        0.00   0.29     0.50      ✗
max_density             8→8      100      0        0        0.88   0.10     1.00      ✓
reasoning_preserving    8→2      30       6        1        0.50   0.20     1.00      ✓
```

`distractor_filtered` drops the second hop (`gold_ret` 0.50, `2nd_hop` ✗);
`reasoning_preserving` keeps it while pruning 6 distractors. `--json out.json`
writes a structured artifact.

## `analyze-context` — observability

```bash
redhop analyze-context context.json --query "..."
```

Renders the Context Optimization Report (`--json` for the raw report).

## `benchmark` — reproducible sweep

```bash
redhop benchmark --input labeled.json --budgets 250,800,12000 --out-dir out/
```

Labeled input adds `gold_ids` / `second_hop_id` per query. Emits `results.json`
+ `SUMMARY.md` with 95% bootstrap CIs — every number computed from the labels.

## `report` — render artifacts

```bash
redhop report results.json --markdown report.md --html report.html
```

Works on `compare --json` output and on `benchmarks/context/results.json`.
