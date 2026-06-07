"""11 · Folder indexing — `Document.from_folder(path, ...)` with .gitignore
   support, ignore globs, and the incremental persistent on-disk index.

Real-world scenario:
    An engineering team has a `docs/` directory with mixed Markdown,
    code samples, and the occasional vendored upstream file they don't
    want indexed. They want:
      - One combined index over all readable files (so a query
        anywhere returns hits from any file with a per-file source
        citation).
      - `.gitignore` honored automatically.
      - Custom `ignore` globs for the vendored-but-not-gitignored files.
      - `persist=True` so the second invocation skips re-indexing
        unchanged files (incremental on-disk cache, default location
        `<folder>/.redhop/`).

What this demonstrates:
    - `Document.from_folder(path)` — recursive indexing over a directory.
    - `recursive=False` to flat-index just one level.
    - `gitignore=True` (default) — `.gitignore` filtering.
    - `ignore=[glob1, glob2, ...]` — extra gitignore-style globs.
    - `persist=True` — incremental on-disk index. Second run reads
      the cache; only changed files re-parse.
    - `doc.n_files` / `doc.skipped_files` — observability over what
      was actually indexed vs why something was skipped.
    - `Document.from_bytes(data, source, ...)` — for blobs you fetched
      from S3 / GCS / a DB rather than the local FS.

Requires `redhop[files]`:
    pip install "redhop[files]"   # adds PDF/DOCX/PPTX/XLSX parsers

Run:
    python examples/python/11_folder_indexing.py
"""

import shutil
import tempfile
from pathlib import Path

import redhop


def setup_demo_docs(root: Path) -> None:
    """Build a small synthetic docs/ tree to index."""
    # Top-level files.
    (root / "README.md").write_text(
        "# Acme Inc Engineering Handbook\n\n"
        "Welcome. Start with onboarding.md for new hires.\n"
    )
    (root / "onboarding.md").write_text(
        "# Onboarding\n\n"
        "New hires get a laptop on day 1 and access provisioned in 24 hours.\n"
        "Talk to it@acme.com if something is missing.\n"
    )

    # A subdirectory.
    (root / "policies").mkdir()
    (root / "policies" / "refunds.md").write_text(
        "# Refund Policy\n\n"
        "Customers get a full refund within 30 days of delivery.\n"
    )
    (root / "policies" / "shipping.md").write_text(
        "# Shipping Policy\n\n"
        "Standard ships in 3-5 business days. Express in 1-2.\n"
    )

    # A vendored upstream file we want to ignore (looks like a third-
    # party copy).
    (root / "vendored").mkdir()
    (root / "vendored" / "third_party_license.md").write_text(
        "# Apache 2.0 license text\n\nIRRELEVANT BOILERPLATE\n" * 30
    )

    # A `.gitignore` that excludes the build artifacts dir.
    (root / "build").mkdir()
    (root / "build" / "generated.md").write_text("# generated, ignore me\n")
    (root / ".gitignore").write_text("build/\n")


def main() -> None:
    # Build a temp docs/ tree so the demo is self-contained.
    with tempfile.TemporaryDirectory(prefix="redhop-folder-demo-") as tmp:
        root = Path(tmp)
        setup_demo_docs(root)
        print(f"Demo directory: {root}\n")

        # ── Arm A: vanilla recursive index ────────────────────────
        # `.gitignore` is honored by default — `build/` is skipped.
        # `vendored/third_party_license.md` IS indexed (no gitignore
        # rule for it).
        print("─── Arm A · Document.from_folder(path) ─────────")
        doc_a = redhop.Document.from_folder(str(root))
        print(f"  files indexed   : {doc_a.n_files}")
        print(f"  total chunks    : {len(doc_a)}")
        print(f"  files skipped   : {len(doc_a.skipped_files)}")
        for path, reason in doc_a.skipped_files[:3]:
            print(f"    - {path}: {reason}")
        print()

        # Query that should land in the refund policy.
        ctx = doc_a.context("how long do I have to get a refund?")
        if ctx.citations:
            top = ctx.citations[0]
            print(f"  top hit source : {top['source']}")
            print(f"  top hit heading: {top['heading']}")
        print()

        # ── Arm B: add custom ignore globs ───────────────────────
        # We don't want vendored/ in the index.
        print("─── Arm B · ignore=['vendored/**'] ─────────────")
        doc_b = redhop.Document.from_folder(str(root), ignore=["vendored/**"])
        print(f"  files indexed   : {doc_b.n_files}  "
              f"(vs Arm A: {doc_a.n_files})")
        print(f"  total chunks    : {len(doc_b)}")
        print()

        # ── Arm C: persist=True (incremental on-disk index) ───────
        # First call writes `<root>/.redhop/index.json`. Second call
        # reads it back — only modified files re-parse. Useful for
        # CI/cron-rebuild workflows or local agents that re-open the
        # same docs.
        print("─── Arm C · persist=True ───────────────────────")
        import time

        t0 = time.time()
        redhop.Document.from_folder(str(root), persist=True)
        first_run_ms = (time.time() - t0) * 1000

        t0 = time.time()
        doc_c2 = redhop.Document.from_folder(str(root), persist=True)
        second_run_ms = (time.time() - t0) * 1000

        cache_path = root / ".redhop" / "index.json"
        print(f"  cache written   : {cache_path.exists()}")
        print(f"  first  run      : {first_run_ms:>5.1f} ms (cold)")
        print(f"  second run      : {second_run_ms:>5.1f} ms (warm — re-read cache)")
        print(f"  same n_files    : {doc_c2.n_files}")
        print()

        # ── Arm D: from_bytes (bytes you fetched yourself) ────────
        # If your docs live in S3 / GCS / a DB rather than on the
        # local FS, you can fetch the bytes and hand them in directly.
        # `source` is the parser hint *and* the citation label.
        print("─── Arm D · from_bytes (for S3 / GCS / blobs) ──")
        # Simulate a fetch — we have the bytes; redhop parses by
        # extension on the `source` argument.
        with open(root / "policies" / "refunds.md", "rb") as f:
            data = f.read()
        doc_d = redhop.Document.from_bytes(data, source="refunds.md")
        print(f"  indexed         : {doc_d.n_files} file, {len(doc_d)} chunks")
        ctx_d = doc_d.context("refund window")
        if ctx_d.citations:
            print(f"  citation source : {ctx_d.citations[0]['source']}")
        print()

        # Need to delete cache before tempdir cleanup or Python's
        # tempdir GC complains on some platforms.
        if cache_path.exists():
            shutil.rmtree(cache_path.parent)

    print("─── When to use what ─────────────────────────────")
    print("- from_folder(path)                 : one combined index")
    print("  over a directory. Default `recursive=True`,")
    print("  `gitignore=True`.")
    print("- ignore=[...]                      : add gitignore-style")
    print("  globs (vendored code, generated docs, etc.).")
    print("- persist=True                      : incremental cache;")
    print("  cron-rebuild / CI / hot-reload agent workflows.")
    print("- from_bytes(buffer, source='x.pdf'): bytes from S3 / GCS /")
    print("  a DB column. Same shape as from_file, just no FS read.")


if __name__ == "__main__":
    main()
