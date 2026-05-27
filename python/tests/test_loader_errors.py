"""Failure-path tests for from_file / from_folder.

These assume the standard (files-enabled) build — `pip install redhop` bundles the
parsers. They verify that bad inputs produce *clear, actionable* errors and that
from_folder skips bad files while reporting them, rather than failing silently.
"""

import pytest

import redhop


def test_empty_file_errors_with_no_text(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("")
    with pytest.raises(Exception) as e:
        redhop.Document.from_file(str(p))
    assert "no text" in str(e.value).lower()


def test_no_text_message_mentions_ocr(tmp_path):
    # A whitespace-only file hits the same NoText guard a scanned/image-only PDF
    # does; its message must name OCR so the user knows the fix.
    p = tmp_path / "ws.txt"
    p.write_text("   \n\t\n")
    with pytest.raises(Exception) as e:
        redhop.Document.from_file(str(p))
    assert "ocr" in str(e.value).lower()


def test_binary_file_rejected(tmp_path):
    p = tmp_path / "data.txt"
    p.write_bytes(b"\x00\x01\x02 binary \x00 content")
    with pytest.raises(Exception) as e:
        redhop.Document.from_file(str(p))
    assert "binary" in str(e.value).lower()


def test_unsupported_extension(tmp_path):
    p = tmp_path / "archive.zip"
    p.write_bytes(b"PK\x03\x04 not really an archive")
    with pytest.raises(Exception) as e:
        redhop.Document.from_file(str(p))
    assert "support" in str(e.value).lower()


def test_from_folder_skips_bad_files_and_reports(tmp_path):
    (tmp_path / "good.md").write_text("# Title\n\nrefund within 30 days\n")
    (tmp_path / "blank.txt").write_text("")  # NoText
    (tmp_path / "bin.txt").write_bytes(b"\x00\x01\x02")  # binary
    doc = redhop.Document.from_folder(str(tmp_path))
    assert doc.n_files == 1  # only the good file indexed
    skipped = dict(doc.skipped_files)  # {path: reason}
    assert len(skipped) == 2
    # the good file is actually retrievable
    assert doc.context("refund window").text().strip()


def test_from_folder_all_bad_errors(tmp_path):
    (tmp_path / "blank.txt").write_text("")
    (tmp_path / "bin.txt").write_bytes(b"\x00\x01")
    with pytest.raises(Exception) as e:
        redhop.Document.from_folder(str(tmp_path))
    assert "no readable files" in str(e.value).lower()
