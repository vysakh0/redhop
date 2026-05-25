#!/usr/bin/env python
"""Large-document scaling: how fast is a thousands-of-page PDF query-ready?

Generates synthetic N-page PDFs and times the from_file path end to end —
parse + chunk + index (lazy, on first query) + retrieval. This is a *latency*
measurement (parse/index/query throughput), not an answer-quality one: the text
is random tokens, so the only correctness check is that a uniquely-planted
phrase is retrieved with the right page citation.

Run:
    bench/.venv/bin/pip install fpdf2          # one-time, to generate PDFs
    bench/.venv/bin/python bench/large_pdf.py
"""
import os
import random
import sys
import tempfile
import time

import redhop

SIZES = [1000, 2000, 4000]
WORDS = (
    "agreement party shall terminate notice governing law delaware refund policy "
    "customer data breach liability indemnify warranty payment invoice confidential "
    "information disclose obligations representations covenants amendment waiver "
    "severability arbitration jurisdiction clause section exhibit schedule"
).split()


def make_pdf(pages: int, path: str) -> int:
    """Write an `pages`-page PDF; plant a unique phrase ~70% through. Returns the
    1-based planted page."""
    from fpdf import FPDF

    random.seed(1)
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_font("Helvetica", size=11)
    plant = pages * 7 // 10
    for i in range(pages):
        pdf.add_page()
        body = ". ".join(
            " ".join(random.choice(WORDS) for _ in range(random.randint(8, 16)))
            for _ in range(6)
        )
        text = f"Page {i + 1}. {body}."
        if i == plant:
            text += " XYZZY special clause: the secret governing jurisdiction is Atlantis."
        pdf.multi_cell(w=pdf.epw, h=6, text=text)
    pdf.output(path)
    return plant + 1


def main(semantic: bool = False):
    mode = "semantic" if semantic else "lexical"
    print(f"== {mode} (from_file) ==")
    with tempfile.TemporaryDirectory() as d:
        for pages in SIZES:
            path = os.path.join(d, f"big_{pages}.pdf")
            planted = make_pdf(pages, path)

            t = time.time()
            kw = dict(retrieval="semantic", model="bge-small") if semantic else {}
            doc = redhop.Document.from_file(path, **kw)
            ctx = doc.context("XYZZY secret governing jurisdiction Atlantis")
            first = time.time() - t  # parse + chunk + index + (embed-all if semantic)

            t = time.time()
            doc.context("refund policy warranty payment")
            warm = (time.time() - t) * 1000

            top = ctx.citations[0] if ctx.citations else None
            ok = "OK" if top and "Atlantis" in top["text"] else "miss"
            print(
                f"{pages:>5}p | {len(doc):>5} chunks | first answer {first:6.2f}s | "
                f"warm {warm:5.1f}ms | planted→page {top['page'] if top else '-'} ({ok})"
            )


if __name__ == "__main__":
    main(semantic="--semantic" in sys.argv)
