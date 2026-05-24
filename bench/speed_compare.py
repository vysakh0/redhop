#!/usr/bin/env python
"""Speed comparison: RedHop (BM25 + dense rerank) vs LangChain vs LlamaIndex.

Measures the *indexing/setup* cost (raw text -> queryable) and *per-query* cost on
real CUAD contracts, at growing document sizes. The point is the architectural
difference, not micro-optimisation:

  - RedHop (BM25 default) and RedHop (dense rerank) embed **0 chunks at index time**.
    Dense rerank embeds only a small BM25 candidate pool, at query time.
  - LangChain / LlamaIndex must embed **every chunk** to build a vector index.

Fairness:
  - Each framework uses its OWN splitter and OWN vector store (LangChain
    InMemoryVectorStore, LlamaIndex VectorStoreIndex, RedHop internal BM25).
  - Embeddings are the same model family (intfloat/e5-small-v2) on every side:
    ONNX for RedHop's rerank path, PyTorch (sentence-transformers) for LC/LI —
    i.e. each as it is actually run. CPU only.
  - PDF parsing is excluded (identical for all); we feed plain text.

Run (offline):
    HF_HUB_OFFLINE=1 bench/.venv/bin/python bench/speed_compare.py
Prereqs: bench/models/e5-small-onnx (optimum-cli export onnx --model
intfloat/e5-small-v2 --task feature-extraction bench/models/e5-small-onnx).
"""
import json, time, statistics, os, sys

CUAD = "data/cuad/cuad_sample.json"
ONNX_DIR = "bench/models/e5-small-onnx"
SIZES = [1, 5, 15]          # number of CUAD contracts concatenated
QUERIES = [
    "What is the governing law of this agreement?",
    "How can either party terminate the contract?",
    "What are the confidentiality obligations?",
    "Is there an exclusivity or non-compete provision?",
    "What are the payment terms?",
]
TOPK = 4

def load_contracts(n):
    data = json.load(open(CUAD))["data"]
    texts = [d["paragraphs"][0]["context"] for d in data[:n]]
    return "\n\n".join(texts)

def approx_tokens(text):
    return len(text) // 4

def timed(fn, reps=1):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

# ---- shared embedding model (e5-small, PyTorch) for LC/LI -------------------
from sentence_transformers import SentenceTransformer
ST = SentenceTransformer("intfloat/e5-small-v2")
def st_passages(texts): return ST.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=64)
def st_query(q): return ST.encode("query: " + q, normalize_embeddings=True)

# ---- RedHop -----------------------------------------------------------------
import redhop
def rh_bm25_index(text): return redhop.Document.from_text(text)
def rh_rerank_index(text):
    return redhop.Document.from_text(
        text, retrieval="rerank",
        embedder_model=f"{ONNX_DIR}/model.onnx",
        embedder_tokenizer=f"{ONNX_DIR}/tokenizer.json",
        embedder_dim=384, embedder_pooling="mean",
        embedder_query_prefix="query: ", embedder_passage_prefix="passage: ",
    )
def rh_query(doc, q): doc.context(q, budget=2000)

# ---- LangChain --------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
class STEmb(Embeddings):
    def embed_documents(self, texts): return [v.tolist() for v in st_passages(texts)]
    def embed_query(self, text): return st_query(text).tolist()
LC_SPLIT = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
def lc_index(text):
    chunks = LC_SPLIT.split_text(text)
    vs = InMemoryVectorStore(STEmb())
    vs.add_texts(chunks)                  # embeds every chunk
    return vs, len(chunks)
def lc_query(vs, q): vs.similarity_search(q, k=TOPK)

# ---- LlamaIndex -------------------------------------------------------------
from llama_index.core import VectorStoreIndex, Document as LIDoc, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
Settings.llm = None
class STEmbLI(BaseEmbedding):
    def _get_text_embedding(self, text): return st_passages([text])[0].tolist()
    def _get_text_embeddings(self, texts): return [v.tolist() for v in st_passages(texts)]
    def _get_query_embedding(self, q): return st_query(q).tolist()
    async def _aget_query_embedding(self, q): return self._get_query_embedding(q)
LI_SPLIT = SentenceSplitter(chunk_size=128, chunk_overlap=16)
LI_EMB = STEmbLI(embed_batch_size=64)
def li_index(text):
    nodes = LI_SPLIT.get_nodes_from_documents([LIDoc(text=text)])
    idx = VectorStoreIndex(nodes, embed_model=LI_EMB)   # embeds every node
    return idx, len(nodes)
def li_query(idx, q): idx.as_retriever(similarity_top_k=TOPK).retrieve(q)

# ---- BM25 retrievers for the lexical (no-embed) scenario --------------------
# Like-for-like vs RedHop (BM25): each framework's own BM25 retriever, no embeddings.
from langchain_community.retrievers import BM25Retriever as LCBM25
def lc_bm25_index(text):
    chunks = LC_SPLIT.split_text(text)
    r = LCBM25.from_texts(chunks); r.k = TOPK
    return r, len(chunks)
def lc_bm25_query(r, q): r.invoke(q)

from llama_index.retrievers.bm25 import BM25Retriever as LIBM25
def li_bm25_index(text):
    nodes = LI_SPLIT.get_nodes_from_documents([LIDoc(text=text)])
    r = LIBM25.from_defaults(nodes=nodes, similarity_top_k=TOPK)
    return r, len(nodes)
def li_bm25_query(r, q): r.retrieve(q)

# ---- warmup (exclude model load from timings) -------------------------------
print("warming up models…", file=sys.stderr)
_warm = "Governing law is New York. " * 30
st_passages(["a", "b"]); st_query("a")
_d = rh_rerank_index(_warm); rh_query(_d, "law?")
_d2 = rh_bm25_index(_warm); rh_query(_d2, "law?")

def measure(idx_fn, q_fn, returns_count, text):
    t = time.perf_counter(); handle = idx_fn(text); build_s = time.perf_counter() - t
    nchunks = "—"
    if returns_count:
        handle, nchunks = handle
    t = time.perf_counter(); q_fn(handle, QUERIES[0]); first_s = time.perf_counter() - t  # cold
    warm_ms = timed(lambda: [q_fn(handle, q) for q in QUERIES[1:]], reps=2) / max(1, len(QUERIES)-1) * 1000
    return nchunks, build_s, first_s, build_s + first_s, warm_ms

def run():
    # Two scenarios, like-for-like within each:
    #   LEXICAL  — nobody embeds (BM25 all round): RedHop vs LangChain vs LlamaIndex BM25.
    #   SEMANTIC — everybody embeds: RedHop dense rerank vs LangChain/LlamaIndex vector.
    SCENARIOS = [
        ("LEXICAL  (BM25, no embeddings)", [
            ("RedHop (BM25)",       rh_bm25_index,  rh_query,      False),
            ("LangChain (BM25)",    lc_bm25_index,  lc_bm25_query, True),
            ("LlamaIndex (BM25)",   li_bm25_index,  li_bm25_query, True),
        ]),
        ("SEMANTIC (embeddings, e5-small)", [
            ("RedHop (dense rerank)", rh_rerank_index, rh_query, False),
            ("LangChain (vector)",    lc_index,        lc_query, True),
            ("LlamaIndex (vector)",   li_index,        li_query, True),
        ]),
    ]
    for n in SIZES:
        text = load_contracts(n)
        toks = approx_tokens(text)
        print(f"\n{'='*84}\n{n} contract(s) — ~{toks:,} tokens (~{len(text):,} chars)\n{'='*84}")
        for label, systems in SCENARIOS:
            print(f"\n  {label}")
            print(f"  {'system':<24}{'chunks':>8}{'build_s':>9}{'1st-q_s':>9}{'TTFA_s':>9}{'warm_ms':>10}")
            print("  " + "-"*68)
            for name, idx_fn, q_fn, rc in systems:
                nchunks, build_s, first_s, ttfa, warm_ms = measure(idx_fn, q_fn, rc, text)
                print(f"  {name:<24}{str(nchunks):>8}{build_s:>8.2f}s{first_s:>8.2f}s{ttfa:>8.2f}s{warm_ms:>8.1f}ms")

if __name__ == "__main__":
    run()
