<p align="center">
  <img src=".github/icon.svg" width="96" height="96" alt="RedHop">
</p>

<h1 align="center">RedHop</h1>

<p align="center"><b>The context layer that shows its work.</b></p>

<p align="center">
  <a href="https://pypi.org/project/redhop/"><img alt="PyPI" src="https://img.shields.io/pypi/v/redhop?label=pypi&color=e11d48"></a>
  <a href="https://crates.io/crates/redhop"><img alt="crates.io" src="https://img.shields.io/crates/v/redhop?label=crates.io&color=e11d48"></a>
  <a href="https://www.npmjs.com/package/redhop"><img alt="npm" src="https://img.shields.io/npm/v/redhop?label=npm&color=e11d48"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="docs/findings/README.md"><img alt="evidence layer" src="https://img.shields.io/badge/evidence-layer-blue"></a>
</p>

<p align="center">
Hand it a document and a question. It chunks, retrieves, and allocates the
context your model should actually see. Then it tells you <b>what it kept,
what it dropped, and why</b>, with citations back to the source. No vector
database, no LLM, all in-process. Measured on real contracts: <b>−80% prompt
tokens with the gold evidence retained, at ~1.7ms per query</b>
(<a href="docs/findings/DOCUMENT_EVAL_CUAD.md">DOCUMENT_EVAL_CUAD</a>).
</p>

---

## Get started in 60 seconds

```bash
pip install redhop                            # Python  (PyPI)
# OR
cargo add redhop --features files,semantic    # Rust    (crates.io)
# OR
npm install redhop                            # Node.js (npm)
```

```python
import redhop

doc = redhop.Document.from_file("contract.pdf")    # parses + chunks + indexes
ctx = doc.context("What is the governing law?")    # retrieves + assembles
answer = llm.generate(ctx.text())                  # any LLM, no lock-in
```

Same three-line shape in Node and Rust:

```js
const doc = Document.fromFile("contract.pdf");
const ctx = doc.context("What is the governing law?");
```

```rust
let mut doc = redhop::read_file("contract.pdf")?;
let ctx = doc.context("What is the governing law?")?;
```

Already chunked your own content? `Document.from_chunks([redhop.Chunk(text, source=...), ...])`.

Eleven runnable walkthroughs, in all three languages, live in
[examples/](examples/README.md). Start with
[01_quickstart.py](examples/python/01_quickstart.py).

---

RedHop is the layer between your documents and the LLM. It is **not** a vector
database, an agent framework, or a workflow engine. It does one thing: turn a
document and a query into the right prompt context, **and explain the decision**.

The core idea it's built on: **retrieval quality is not the same as reasoning
quality.** Transformers tolerate irrelevant context far better than they
tolerate *missing reasoning links*. The chunk a multi-hop answer depends on is
often low-relevance to the query and gets silently pruned, so RedHop's default
keeps it and makes the trade-off visible. Every default traces to a measured
finding, including the hypotheses that failed, in the
[evidence layer](docs/findings/README.md).

## It explains every decision

Every call returns a **Decision Report**: what it kept, what it dropped, and
*why*, including when it deliberately leaves a small context untouched.

<p align="center">
  <img src=".github/decision_report.svg" alt="Sample Decision Report" width="100%">
</p>

The same fields are available programmatically: `ctx.report.auto_decision`,
`ctx.report.total_tokens`, `ctx.report.retained_evidence_ratio`. When retrieval
looks weak, `ctx.report.diagnosis` lists the query terms that appear nowhere
in the corpus and fires bounded hints (e.g. vocabulary mismatch, templated
boilerplate, polysemy) with a link to the finding behind each one. See
[`examples/python/12_diagnosis.py`](examples/python/12_diagnosis.py). Call
`doc.analyze(query)` to get the report **without** assembling a context. Query
rewrites (boilerplate stripping, synonym expansion) land on the same report as
a per-stage audit trail via `ctx.report.query_rewrites`.

## Cite the evidence

Every selected chunk remembers where it came from:

```python
for c in ctx.citations:
    print(c["source"], c["page"], c["heading"])
    # contract.pdf  3     None      →  "contract.pdf, p.3"
    # notes.md      None  "Refunds" →  "notes.md → Refunds"
```

`source` plus whichever of `page` / `heading` / `line` the format provides.
No separate store, no second lookup.

## How it compares

Same documents, same budgets, BM25 for all three. Evidence retention
(share of queries with ≥0.8 gold-evidence recall, n=300):

| dataset | RedHop | LangChain | LlamaIndex |
| --- | ---:| ---:| ---:|
| HotpotQA (multi-hop) | **80%** | 71% | 72% |
| MuSiQue (compositional multi-hop) | **22%** | 19% | 17% |
| CUAD (contracts, raw template query) | 82% | 73% | **86%** |

Read it honestly:

- **Multi-hop retention is RedHop's durable lead**, replicated on two datasets.
  It comes from the chunking and retrieval defaults, not a magic assembly
  strategy. We say so because it's true.
- **LlamaIndex leads on raw contract queries.** The gap is mechanism-known
  (BM25 boilerplate dilution) and closeable with RedHop's query-rewrite
  workflow (82% → 90.7%), but the same preprocessing also lifts LlamaIndex.
  The retrieval engines are roughly comparable on contracts.
- Need more multi-hop? `retrieval="hybrid"` lifts HotpotQA ≥0.8 retention
  71% → 81% (n=100) at roughly 100× per-query latency.

Full numbers, fair-preprocessing results, the hybrid head-to-head, and the
caveats: [docs/COMPARISON.md](docs/COMPARISON.md).

<p align="center">
  <img src=".github/retention_vs_frameworks.svg" alt="Evidence retention vs LangChain vs LlamaIndex" width="100%">
</p>

## How it works

<p align="center">
  <img src=".github/architecture.svg" alt="RedHop pipeline" width="100%">
</p>

Five stages: **you bring documents and a query**, RedHop owns parsing,
chunking, retrieval, and context allocation, and **you get a `BuiltContext`**
with the assembled prompt, citations, and a Decision Report.

**Loaders.** `from_text`, `from_chunks`, `from_file` (PDF, DOCX, PPTX, XLSX,
Markdown, text/code), `from_bytes`, and `from_folder`. Code is chunked
verbatim and labeled with its nearest definition, prose is sentence-packed,
and every format carries its structural location (page, heading, line) for
citations. Folder indexing with an incremental cache:
[11_folder_indexing.py](examples/python/11_folder_indexing.py).

**Retrieval tiers.** All tiers run in-process, with no ANN and no index server:

| `retrieval=` | What it does | Reach for it when |
| --- | --- | --- |
| `"lexical"` *(default)* | BM25. Zero dependencies, fully offline | most document QA, where the words in the question are usually the words in the answer |
| `"hybrid"` | BM25 prunes to a pool, a dense model reorders it | multi-hop bridge passages, or parallel near-duplicate clauses |
| `"semantic"` | dense over every chunk, exact cosine | queries and answers share no vocabulary at all |

All three side by side on the same corpus:
[07_retrieval_tiers.py](examples/python/07_retrieval_tiers.py). Non-English?
`language="german"` (18 Snowball languages) or a custom analyzer for CJK. See
[docs/LANGUAGE.md](docs/LANGUAGE.md) and
[09_multilingual.py](examples/python/09_multilingual.py).

**Assembly strategies.** `reasoning_preserving` *(default)*,
`distractor_filtered`, `max_density`, `raw_topk`, and `auto`, which is
size-gated: small contexts pass through untouched, large or diluted ones get
pruned. Worked comparison:
[10_strategy_choice.py](examples/python/10_strategy_choice.py).

→ Picking a config in 60 seconds: [docs/CHOOSING_A_CONFIG.md](docs/CHOOSING_A_CONFIG.md).

## Templated queries? There's a measured recipe

If every query in your workload shares fixed boilerplate (legal QA templates,
support-ticket forms), the boilerplate dilutes BM25's signal terms. RedHop
ships a detect, strip, expand workflow that lifted CUAD retention
**81.3% → 87.7% → 90.7%**, with each rewrite stage recorded on the Decision
Report:

```python
report = redhop.analyze_query_set(my_queries)        # 1. detect the template
if report.is_templated:
    stripper = redhop.Stripper(report.boilerplate_terms)          # 2. strip
    vocab = redhop.Vocabulary({"change of control": ["merger"]})  # 3. expand
    ctx = doc.context_with_rewrites(user_query, [stripper, vocab])
    for rec in ctx.report.query_rewrites:            # 4. the audit trail
        print(rec.stage, rec.matched, rec.added, rec.removed)
```

Full runnable walkthrough:
[03_templated_workload.py](examples/python/03_templated_workload.py). The
decision rule lives in [docs/CHOOSING_A_CONFIG.md](docs/CHOOSING_A_CONFIG.md).

## Evaluate in CI, no LLM required

`redhop.evaluate(...)` returns deterministic lexical metrics
(`context_recall`, `context_precision`, `faithfulness_lexical`, and more) in
about a millisecond. It is built from the same primitives as the Decision
Report, so eval and runtime can't disagree. Run it on every PR:

```python
eval_a = redhop.evaluate(query, doc.context(query), gold_chunks=gold_ids)
eval_b = redhop.evaluate(query, ctx_with_rewrites, gold_chunks=gold_ids)
print("lift:", eval_b.overall - eval_a.overall)    # deterministic A/B
```

When you need judged metrics, opt in with your own LLM caller.
`judge=Judge.from_callable(my_llm).cached()` adds faithfulness, relevancy, and
correctness. `decompose_faithfulness=True` is **Ragas-calibrated** (r=+0.664,
n=200 HotpotQA, see [COMPARISON_RAGAS](docs/COMPARISON_RAGAS.md)).
`critique(answer, aspects=[...])` scores open-ended dimensions like
conciseness or brand voice.

Runnable A/B walkthrough:
[05_evaluate_ab.py](examples/python/05_evaluate_ab.py). Design and the full
field list: [ANSWER_QUALITY_EVAL](docs/findings/ANSWER_QUALITY_EVAL.md).

## Documentation

- **Choosing a configuration**: [docs/CHOOSING_A_CONFIG.md](docs/CHOOSING_A_CONFIG.md)
- **Runnable examples** (Python, Node, Rust): [examples/](examples/README.md)
- **Comparison vs LangChain / LlamaIndex**: [docs/COMPARISON.md](docs/COMPARISON.md)
- **Comparison vs Ragas**: [docs/COMPARISON_RAGAS.md](docs/COMPARISON_RAGAS.md)
- **Evidence layer** (every finding, including the falsified ones, plus the literature it leans on): [docs/findings/](docs/findings/README.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Retrieval tips**: [docs/retrievaltips.md](docs/retrievaltips.md)
- **Python**: [python/README.md](python/README.md) and **Node**: [nodejs/README.md](nodejs/README.md)
- **API stability**: [docs/API_STABILITY.md](docs/API_STABILITY.md)
- **FAQ**: [FAQ.md](FAQ.md) and **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
