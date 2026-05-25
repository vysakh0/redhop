#!/usr/bin/env python
"""Code-retrieval probe: does type-aware indexing help on a real codebase?

Indexes RedHop's own Rust source and asks natural-language questions whose answer
is a specific function/file. Reports recall@3 per retrieval mode. Use it to A/B
the typed-hybrid change (code → BM25, prose → dense, RRF) against the prior
behavior by rebuilding before/after.

Run:  bench/.venv/bin/python bench/code_retrieval.py
"""
import os
import sys
import time

import redhop

ROOT = os.path.join(os.path.dirname(__file__), "..", "crates")

# (natural-language query, target file substr, marker that must be in the chunk)
QUERIES = [
    ("how are two ranked result lists combined by reciprocal rank fusion",
     "fusion.rs", "reciprocal_rank_fusion"),
    ("split text into chunks keeping the original formatting verbatim",
     "document/src/lib.rs", "verbatim_chunks"),
    ("decide a chunk is code so it is retrieved lexically not embedded",
     "local_rerank.rs", "is_code"),
    ("extract text from a PDF page by page from bytes in memory",
     "pdf.rs", "extract_text_from_mem_by_pages"),
    ("split a markdown document into sections by heading",
     "text.rs", "markdown_sections"),
    ("build a BM25 lexical index using tantivy in memory",
     "bm25.rs", "Bm25Retriever"),
    ("set the number of intra-op threads for the ONNX session",
     "onnx.rs", "intra_threads"),
    ("cosine similarity between two embedding vectors",
     "local_rerank.rs", "fn cosine"),
    ("classify a source file by extension as code data or prose",
     "document/src/lib.rs", "chunk_kind"),
    ("attach adjacent neighbor chunks for structural context expansion",
     "context/src/lib.rs", "ExpansionPlan"),
    ("read cells from an XLSX spreadsheet sheet by sheet",
     "xlsx.rs", "worksheet_range"),
    ("renumber chunk ids so a merged set is unique",
     "document/src/lib.rs", "reassign_ids"),
]
TOPK = 3


def main():
    kw_by_mode = {
        "lexical": dict(retrieval="lexical"),
        "hybrid": dict(retrieval="hybrid", model="bge-small"),
        "semantic": dict(retrieval="semantic", model="bge-small"),
    }
    modes = sys.argv[1:] or ["lexical", "hybrid", "semantic"]
    for mode in modes:
        t = time.time()
        doc = redhop.Document.from_folder(ROOT, **kw_by_mode[mode])
        setup = time.time() - t
        hits = 0
        for q, fpath, marker in QUERIES:
            cites = doc.context(q, budget=4000).citations[:TOPK]
            ok = any(fpath in c["source"] and marker in c["text"] for c in cites)
            hits += 1 if ok else 0
        print(f"{mode:>9}: recall@{TOPK} = {hits}/{len(QUERIES)} ({100*hits//len(QUERIES)}%)  "
              f"| {len(doc)} chunks | setup {setup:.1f}s")


if __name__ == "__main__":
    main()
