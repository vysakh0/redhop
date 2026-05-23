# bench/ — head-to-head context-assembly benchmark

Compares **RedHop vs LangChain vs LlamaIndex** on *context assembly*: given a
document and a query, what context does each hand the LLM, and how good is it?
Isolated from the library (its own venv, heavyweight deps, never in CI).

This is a deliberately **fair** benchmark — it can show "comparable" or "RedHop
behind" as readily as "RedHop ahead". Treat the numbers as findings, not
marketing.

## Setup

```bash
python3.13 -m venv bench/.venv
bench/.venv/bin/pip install rank-bm25 langchain-community llama-index-core llama-index-retrievers-bm25
bench/.venv/bin/pip install ./python          # builds the redhop wheel (needs Rust)
```

## Run

```bash
bench/.venv/bin/python bench/compare.py         # Tier 1: tokens + evidence retention (free, no LLM)
```

## Method (Tier 1)

- All three retrieve with **BM25** (LangChain/LlamaIndex `BM25Retriever`,
  RedHop's internal Tantivy BM25) — isolates assembly from retrieval-engine.
- Same **token budget**, set *below* document size so selection is forced.
- RedHop runs `reasoning_preserving` (the strategy under test); the default Auto
  policy would pass small docs through unpruned.
- Metric: gold-evidence **word-recall** in the assembled context (CUAD: answer
  span; HotpotQA: gold supporting sentences).

**Retention is a proxy, not the verdict** — RedHop deliberately drops low-relevance
chunks, so lower retention may or may not hurt the actual answer. Downstream
answer quality (Tier 3, LLM) is the decisive test.

Versions pinned at first run: langchain-community 0.4.x (note: officially
sunsetting; its BM25Retriever still works), llama-index-core + bm25 retriever.
Raw results: `reports/framework_comparison*.txt`.
