# RedHop · examples

Real-world usage demos, organized by language. These are **not**
measurement probes (those live in
[`crates/examples/examples/`](../crates/examples/examples/) alongside
the evidence layer in [`docs/findings/`](../docs/findings/)). These
are runnable code that shows how to use the 0.3.0 API for common
scenarios.

## What's here

| Language | Folder | Status |
| -------- | ------ | ------ |
| Python | [`python/`](python/) | ✓ 15 examples covering the 3-call surface, typed Chunk, rewrite chain + audit, chunk enrich, A/B eval, chat RAG, retrieval tiers (lexical/hybrid/semantic), structural expansion, multilingual analyzer, assembly strategies, folder indexing, query diagnosis, workload audit (BYO pipeline), catalog search (char-ngram + field weights + set-coverage), safe auto-answers (confidence routing) |
| Node.js | [`nodejs/`](nodejs/) | ✓ 15 examples: same scenarios as Python, `.cjs` mirrors using the Node.js camelCase API surface |
| Rust | [`rust/`](rust/) | ✓ 15 examples: same scenarios mirrored against the Rust core API. `crates/examples/examples/` continues to house the measurement probes (evidence layer) |

Each language folder has its own `README.md` describing the demos and
how to run them.

## What's NOT here

Measurement probes (CUAD harnesses, multilingual probes, dilution
sweeps, the four-corner-observation falsifications) stay in
[`crates/examples/examples/`](../crates/examples/examples/) because
they are evidence-layer infrastructure, not API showcases. The split:

- `examples/`: *how* to use the API. Real-world scenarios,
  inline data, demo-shaped output. All three languages.
- `crates/examples/`: *what is true* on a measured workload.
  Reproducible benchmarks behind every claim in
  [`docs/findings/`](../docs/findings/). Rust-only because the probes
  need fast iteration and the existing CUAD/HotpotQA harness shape.

If you want to know whether a feature works, look at the finding doc
+ its harness. If you want to know how to call the feature, look
here.

## Suggested entry points

- **New to RedHop?** Start with
  [`python/01_quickstart.py`](python/01_quickstart.py): the
  3-call surface in 80 lines.
- **Building a RAG system over your own pre-chunked content?** Read
  [`python/02_structured_corpus.py`](python/02_structured_corpus.py)
  for the typed-`Chunk` pattern.
- **Templated queries** (legal / support-ticket / form-driven)? See
  [`python/03_templated_workload.py`](python/03_templated_workload.py).
- **Want to A/B before adopting a knob?**
  [`python/05_evaluate_ab.py`](python/05_evaluate_ab.py) shows the
  deterministic eval surface with no LLM judge.
- **Already running retrieval somewhere else** (LangChain,
  LlamaIndex, pgvector, hand-rolled)? See
  [`python/13_workload_audit.py`](python/13_workload_audit.py) for the
  BYO loop: point RedHop's diagnostics at your existing pipeline,
  aggregate across a workload, get one focus recommendation. The full
  walk-through lives in
  [`docs/DIAGNOSE_YOUR_PIPELINE.md`](../docs/DIAGNOSE_YOUR_PIPELINE.md).
