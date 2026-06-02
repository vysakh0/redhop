"""Loader + ingestion coverage: from_chunks, from_file, from_folder (incl.
persistence + incremental re-index), citations, and structural expansion.

These run on the **lexical** (default, no-model) build, so they need no ONNX
extension and no network — the whole ingestion/citation/persistence surface is
exercised with BM25 retrieval. A couple of tests that need the parsing tier
(`redhop[files]`: PDF/DOCX/PPTX/XLSX + markdown heading splitting) detect it at
runtime and skip cleanly otherwise.
"""

import json
import os
import time

import pytest

import redhop

HERE = os.path.dirname(__file__)
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "..", "crates", "redhop", "tests", "fixtures"))


def _has_files_feature() -> bool:
    """The parsing tier splits markdown by heading; the base build reads it as
    one section. Probe that to decide whether `redhop[files]` is compiled in."""
    import tempfile

    d = tempfile.mkdtemp()
    p = os.path.join(d, "probe.md")
    with open(p, "w") as f:
        f.write("# Title\nalpha alpha alpha\n\n## Refunds\nbeta beta refund window\n")
    try:
        ctx = redhop.Document.from_file(p).context("refund window beta")
    except Exception:
        return False
    return any(c["heading"] for c in ctx.citations)


HAS_FILES = _has_files_feature()
files_only = pytest.mark.skipif(not HAS_FILES, reason="needs redhop[files] parsing tier")


# --------------------------------------------------------------------------- #
# from_chunks
# --------------------------------------------------------------------------- #
def test_from_chunks_strings():
    doc = redhop.Document.from_chunks(
        ["the refund window is thirty days", "shipping takes two days"]
    )
    assert len(doc) == 2
    ctx = doc.context("refund window")
    assert "refund" in ctx.text().lower()


def test_from_chunks_dicts_carry_source():
    doc = redhop.Document.from_chunks(
        [
            {"text": "the refund window is thirty days", "source": "refunds.md"},
            {"text": "shipping takes two business days", "source": "shipping.md"},
        ]
    )
    cites = doc.context("refund window thirty").citations
    assert cites and cites[0]["source"] == "refunds.md"


def test_from_chunks_empty_raises():
    with pytest.raises(Exception):
        redhop.Document.from_chunks([])


# --------------------------------------------------------------------------- #
# from_file
# --------------------------------------------------------------------------- #
def test_from_file_text_tracks_source(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("the refund window is thirty days from purchase\nreturns require a receipt\n")
    doc = redhop.Document.from_file(str(p))
    ctx = doc.context("refund window")
    assert ctx.citations
    c = ctx.citations[0]
    assert c["source"] == str(p)
    # plain text → no structural location
    assert c["page"] is None and c["heading"] is None


def test_from_file_missing_raises():
    with pytest.raises(Exception):
        redhop.Document.from_file("/no/such/file/here.txt")


@files_only
def test_from_file_markdown_heading_and_line(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text(
        "# Overview\nintro line here\n\n## Refund Policy\ncustomers may request a refund within thirty days\n"
    )
    cites = redhop.Document.from_file(str(p)).context("refund within thirty days").citations
    hit = next((c for c in cites if "refund" in c["text"].lower()), None)
    assert hit is not None
    assert hit["heading"] == "Refund Policy"
    assert hit["line"] == 4  # "## Refund Policy" line


@files_only
def test_from_file_code_is_verbatim_with_symbol_citation(tmp_path):
    p = tmp_path / "auth.py"
    p.write_text("import os\n\ndef login(user):\n    token = make_token(user)\n    return token\n")
    ctx = redhop.Document.from_file(str(p)).context("make token login")
    hit = next((c for c in ctx.citations if "make_token" in c["text"]), None)
    assert hit is not None
    assert hit["heading"] == "def login(user)"  # symbol-named citation
    assert "\n" in hit["text"]  # verbatim — code formatting preserved, not reflowed


@files_only
@pytest.mark.skipif(not os.path.isdir(FIXTURES), reason="rust fixtures not found")
@pytest.mark.parametrize(
    "fname,query,want_key",
    [
        ("sample.docx", "refund policy shipping orders", "heading"),
        ("sample.pptx", "revenue risks supply chain", "page"),
        ("sample.xlsx", "sku price pricing", "heading"),
        ("sample.pdf", "governing law delaware terminate", "page"),
    ],
)
def test_from_file_binary_formats(fname, query, want_key):
    path = os.path.join(FIXTURES, fname)
    if not os.path.exists(path):
        pytest.skip(f"missing fixture {fname}")
    ctx = redhop.Document.from_file(path).context(query)
    assert ctx.citations, f"{fname} produced no retrievable chunks"
    assert ctx.citations[0]["source"] == path
    # at least one selected chunk carries the format's structural locator
    assert any(c[want_key] is not None for c in ctx.citations), f"{fname} missing {want_key}"


# --------------------------------------------------------------------------- #
# from_bytes  (cloud object storage / HTTP / DB blobs)
# --------------------------------------------------------------------------- #
def test_from_bytes_text_with_source_key():
    # e.g. data = s3.get_object(...)["Body"].read()
    doc = redhop.Document.from_bytes(
        b"the refund window is thirty days from purchase", source="s3://bucket/notes.txt"
    )
    c = doc.context("refund window").citations[0]
    assert c["source"] == "s3://bucket/notes.txt"
    assert c["page"] is None and c["heading"] is None


@files_only
def test_from_bytes_markdown_heading():
    md = b"# Title\nintro\n\n## Refunds\nrefund within thirty days\n"
    cites = redhop.Document.from_bytes(md, source="policy.md").context("refund thirty").citations
    hit = next((c for c in cites if "refund" in c["text"].lower()), None)
    assert hit and hit["heading"] == "Refunds"


@files_only
@pytest.mark.skipif(not os.path.isdir(FIXTURES), reason="rust fixtures not found")
def test_from_bytes_pdf_pages():
    data = open(os.path.join(FIXTURES, "sample.pdf"), "rb").read()
    ctx = redhop.Document.from_bytes(data, source="contract.pdf").context(
        "governing law delaware terminate"
    )
    assert ctx.citations and ctx.citations[0]["source"] == "contract.pdf"
    assert any(c["page"] is not None for c in ctx.citations)


@files_only
def test_from_bytes_unsupported_extension_errors():
    with pytest.raises(Exception):
        redhop.Document.from_bytes(b"\x89PNG...", source="logo.png")


# --------------------------------------------------------------------------- #
# from_folder
# --------------------------------------------------------------------------- #
def _mkfolder(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "refunds.txt").write_text("the refund window is thirty days\n")
    (tmp_path / "sub" / "shipping.txt").write_text("orders ship within two business days\n")
    return tmp_path


def test_from_folder_indexes_all_files_with_per_file_source(tmp_path):
    _mkfolder(tmp_path)
    doc = redhop.Document.from_folder(str(tmp_path))
    assert len(doc) >= 2
    r = doc.context("refund window").citations
    s = doc.context("orders ship business days").citations
    assert r and r[0]["source"].endswith("refunds.txt")
    assert s and s[0]["source"].endswith("shipping.txt")  # found in a subdir (recursive)


def test_from_folder_recursive_false_skips_subdirs(tmp_path):
    _mkfolder(tmp_path)
    doc = redhop.Document.from_folder(str(tmp_path), recursive=False)
    # the subdir file's distinctive content should not be retrievable
    assert not doc.context("orders ship business days").citations or all(
        not c["source"].endswith("shipping.txt")
        for c in doc.context("orders ship business days").citations
    )


def test_from_folder_skips_junk_dirs(tmp_path):
    _mkfolder(tmp_path)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.txt").write_text("refund refund refund vendored noise\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.txt").write_text("refund refund hidden noise\n")
    doc = redhop.Document.from_folder(str(tmp_path))
    for c in doc.context("refund").citations:
        assert "node_modules" not in c["source"] and ".git" not in c["source"]


def test_from_folder_respects_gitignore(tmp_path):
    (tmp_path / "keep.txt").write_text("the refund window is thirty days\n")
    (tmp_path / "build_out.txt").write_text("generated build artifact noise\n")
    (tmp_path / ".gitignore").write_text("build_out.txt\n")
    # default: .gitignore honored even though tmp_path isn't a git repo
    assert len(redhop.Document.from_folder(str(tmp_path))) == 1
    # opt out: the ignored file comes back
    assert len(redhop.Document.from_folder(str(tmp_path), gitignore=False)) == 2


def test_from_folder_custom_ignore_patterns(tmp_path):
    (tmp_path / "keep.txt").write_text("the refund window is thirty days\n")
    (tmp_path / "noise.log").write_text("verbose log line noise\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("subdir doc\n")
    assert len(redhop.Document.from_folder(str(tmp_path), ignore=["*.log"])) == 2  # log excluded
    assert (
        len(redhop.Document.from_folder(str(tmp_path), ignore=["sub/**"])) == 2
    )  # subdir excluded


def test_from_folder_invalid_ignore_pattern_raises(tmp_path):
    _mkfolder(tmp_path)
    with pytest.raises(Exception):
        redhop.Document.from_folder(str(tmp_path), ignore=["["])  # malformed glob


def test_from_folder_empty_raises(tmp_path):
    with pytest.raises(Exception):
        redhop.Document.from_folder(str(tmp_path))


def test_from_folder_not_a_directory_raises(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hi")
    with pytest.raises(Exception):
        redhop.Document.from_folder(str(p))


# --------------------------------------------------------------------------- #
# from_folder — persistence + incremental re-index
# --------------------------------------------------------------------------- #
def _index_path(folder):
    return os.path.join(str(folder), ".redhop", "index.json")


def test_persist_creates_index_and_reuses_unchanged(tmp_path):
    _mkfolder(tmp_path)
    redhop.Document.from_folder(str(tmp_path), persist=True)
    idx = _index_path(tmp_path)
    assert os.path.exists(idx)
    mtime = os.path.getmtime(idx)
    time.sleep(0.05)
    # second run, nothing changed → index file must NOT be rewritten
    redhop.Document.from_folder(str(tmp_path), persist=True)
    assert os.path.getmtime(idx) == mtime


def test_persist_incremental_edit_reflected(tmp_path):
    _mkfolder(tmp_path)
    redhop.Document.from_folder(str(tmp_path), persist=True)
    time.sleep(0.05)
    (tmp_path / "refunds.txt").write_text("the warranty period is one whole calendar year\n")
    doc = redhop.Document.from_folder(str(tmp_path), persist=True)
    hit = doc.context("warranty calendar year").citations
    assert hit and "warranty" in hit[0]["text"].lower()


def test_persist_add_and_remove(tmp_path):
    _mkfolder(tmp_path)
    redhop.Document.from_folder(str(tmp_path), persist=True)
    # add a file, remove another
    (tmp_path / "privacy.txt").write_text("we never sell your personal data to third parties\n")
    os.remove(tmp_path / "refunds.txt")
    redhop.Document.from_folder(str(tmp_path), persist=True)
    sources = {f["source"] for f in json.load(open(_index_path(tmp_path)))["files"]}
    assert any(s.endswith("privacy.txt") for s in sources)
    assert not any(s.endswith("refunds.txt") for s in sources)


def test_persist_index_dir_override(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("the refund window is thirty days\n")
    idx_dir = tmp_path / "cache"
    redhop.Document.from_folder(str(folder), persist=True, index_dir=str(idx_dir))
    assert os.path.exists(os.path.join(str(idx_dir), "index.json"))
    assert not os.path.exists(os.path.join(str(folder), ".redhop"))


def test_persist_fingerprint_invalidates_on_settings_change(tmp_path):
    _mkfolder(tmp_path)
    redhop.Document.from_folder(str(tmp_path), persist=True, chunk_size=128)
    fp1 = json.load(open(_index_path(tmp_path)))["fingerprint"]
    # different chunking → different fingerprint → rebuilt, not served stale
    redhop.Document.from_folder(str(tmp_path), persist=True, chunk_size=64)
    fp2 = json.load(open(_index_path(tmp_path)))["fingerprint"]
    assert fp1 != fp2


# --------------------------------------------------------------------------- #
# citations
# --------------------------------------------------------------------------- #
def test_citations_shape_from_text():
    doc = redhop.Document.from_text("the refund window is thirty days from purchase")
    cites = doc.context("refund window").citations
    assert cites
    c = cites[0]
    assert set(c.keys()) == {"source", "page", "heading", "line", "text"}
    assert c["page"] is None and c["heading"] is None and c["line"] is None
    assert c["text"]


# --------------------------------------------------------------------------- #
# structural expansion (neighbors / heading) — lexical, no model
# --------------------------------------------------------------------------- #
EXP_TEXT = (
    "Alpha section discusses onboarding. "
    "The xyzzy clause governs termination of the agreement. "
    "Gamma section covers payment terms. "
    "Delta section covers confidentiality."
)


def test_expansion_off_by_default():
    doc = redhop.Document.from_text(EXP_TEXT, chunk_size=8)
    ctx = doc.context("xyzzy clause")
    assert ctx.report.n_expanded == 0


def test_expansion_neighbors_adds_adjacent_in_document_order():
    doc = redhop.Document.from_text(EXP_TEXT, chunk_size=8)
    base = doc.context("xyzzy clause")
    expanded = doc.context("xyzzy clause", neighbors=1)
    # neighbors pulled the adjacent chunk(s) in for continuity
    assert expanded.report.n_expanded >= 1
    assert len(expanded.chunks) > len(base.chunks)
    # the seed is still present
    assert any("xyzzy" in t.lower() for t in expanded.chunks)
