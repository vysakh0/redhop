"""Timeout-watchdog around `Document.context`.

Tests that `redhop.context_with_timeout`:
  - returns the same result as `doc.context(...)` when the call completes
    within the budget;
  - raises `TimeoutError` when the call would take longer than the budget;
  - rejects non-positive `timeout_ms` with a clear `ValueError`.

The native retrieval can't be cooperatively interrupted (Tantivy + ONNX
have no cancellation hooks), so the watchdog returns control to the
caller while the background thread keeps running until completion. That
trade-off is documented in the helper's docstring; the tests below
verify the user-visible API behaves as advertised.
"""

import pytest

import redhop


def _tiny_corpus():
    return [
        {"id": "a", "text": "the refund window is thirty days from purchase"},
        {"id": "b", "text": "customers may return items within 30 days"},
    ]


def test_returns_same_as_context_when_within_budget():
    """A generous timeout — the call completes in plenty of time and the
    returned BuiltContext is structurally identical to calling
    `doc.context()` directly."""
    doc = redhop.Document.from_chunks(_tiny_corpus())
    direct = doc.context("refund window")
    via = redhop.context_with_timeout(doc, "refund window", timeout_ms=10_000)
    assert via.text() == direct.text()
    assert list(via.chunks) == list(direct.chunks)
    assert via.report.total_tokens == direct.report.total_tokens


def test_rejects_non_positive_timeout():
    """timeout_ms must be positive — 0 and negative both reject with a
    clear ValueError naming the bad value."""
    doc = redhop.Document.from_chunks(_tiny_corpus())
    for bad in (0, -1, -1000):
        with pytest.raises(ValueError) as e:
            redhop.context_with_timeout(doc, "refund", timeout_ms=bad)
        assert "timeout_ms" in str(e.value)
        assert "positive" in str(e.value).lower()


def test_raises_timeout_error_when_budget_exhausted():
    """When the underlying `doc.context()` takes longer than the timeout,
    the watchdog raises TimeoutError with a message that names the
    budget AND explains the background-work-still-finishes limitation.

    We use a deliberately-slow fake `context` (sleeps 500ms) and a 50ms
    timeout so the test is deterministic across machines — relying on
    "a real retrieve is slower than 1ms" is too brittle on fast boxes.
    """
    import time
    from unittest.mock import MagicMock

    fake_doc = MagicMock()

    def slow_context(*args, **kwargs):
        time.sleep(0.5)  # 500ms — well over the 50ms budget below
        return "should never be returned"

    fake_doc.context = slow_context

    with pytest.raises(TimeoutError) as e:
        redhop.context_with_timeout(fake_doc, "refund window", timeout_ms=50)
    msg = str(e.value)
    assert "50ms" in msg, f"error should name the timeout; got: {msg}"
    assert "exceeded" in msg.lower()
    # The docstring promises to mention the can't-interrupt limitation in
    # the error message — verify it stays there so users aren't surprised
    # that the background work continues.
    assert "interrupt" in msg.lower() or "background" in msg.lower()


def test_forwards_optional_kwargs_to_context():
    """The watchdog wrapper forwards `budget`, `neighbors`, `include_heading`
    through to `doc.context()` — so callers using those options still get
    the timeout behavior without losing functionality."""
    doc = redhop.Document.from_chunks(_tiny_corpus())
    # Set an extreme budget so we can verify it's actually being applied
    via = redhop.context_with_timeout(doc, "refund window", timeout_ms=10_000, budget=10)
    assert via.report.token_budget == 10, (
        f"budget kwarg should reach context(); got token_budget={via.report.token_budget}"
    )
