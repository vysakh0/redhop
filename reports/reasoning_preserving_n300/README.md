# Report: reasoning_preserving_n300

Captured raw output of the end-to-end reasoning-preservation experiment.
Full interpretation: [docs/findings/REASONING_PRESERVATION.md](../../docs/findings/REASONING_PRESERVATION.md).

- **Date:** 2026-05-21
- **n:** 300 gap-qualified multi-hop HotpotQA queries × 4 conditions = 1200 `claude haiku` calls
- **Generator:** `claude haiku` (via the `claude` CLI)
- **Filter threshold:** 0.20 (aggressive — where the second-hop tax bites)
- **Contexts built by:** `cargo run -p neorag-examples --example emit_reasoning_qa --release`
- **Scored by:** `python ../neorag/scripts/score_reasoning_qa.py --n 300 --model haiku`

## Files
- `result.txt` — the final result tables (current, full-gold mechanism split)
- `raw_output_initial.txt` — the first run's stdout (single-chunk proxy split; kept for the documented correction)

## Headline
- reasoning − filtered: **+0.035** [+0.003, +0.067] (CI excludes 0)
- mechanism: **+0.173** on the rescued subset (gold reachability differs) vs **+0.022 [−0.007, +0.054]** on the identical-gold control → the gain is *caused* by preserving low-relevance gold, not by reordering/denoising.
- surprise: the aggressive **filter** crashed quality (0.829→0.705); the **distractors did not** (0.829 ≈ 0.830 ceiling) on this distractor-robust generator.

Reproducible from cache: the scorer caches every LLM response
(`exports/reasoning_qa_cache.json` in the Python lab), so re-running is
free and deterministic.
