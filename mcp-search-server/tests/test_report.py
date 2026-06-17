"""Tests for the shared error-report helper and the tool-specific hints.

These cover the recovery contract for the non-browser tools: a failed call must
come back in the same actionable shape browser_run uses — a status line, the
error text in a fenced block, and (when we can infer one) a one-line corrective
hint the model can act on.
"""

from src.tools._report import error_report
from src.tools.code_run import _code_error_hint, _stderr_hint


def test_error_report_basic_shape():
    out = error_report("boom")
    assert "**Status:** error" in out
    assert "**Error:**" in out
    assert "```text" in out
    assert "boom" in out


def test_error_report_includes_hint_when_given():
    out = error_report("boom", "try again with X")
    assert "_try again with X_" in out


def test_error_report_omits_hint_when_absent():
    out = error_report("boom")
    assert "_" not in out.split("```")[-1].strip() or "hint" not in out.lower()


def test_error_report_empty_message_has_placeholder():
    out = error_report("")
    assert "(no message)" in out


def test_error_report_custom_status():
    out = error_report("nope", status="failed")
    assert "**Status:** failed" in out


def test_code_error_hint_timeout():
    hint = _code_error_hint("Execution timed out after 30s", None)
    assert hint is not None
    assert "timeout=" in hint


def test_code_error_hint_oom_signal():
    assert _code_error_hint("killed", -9) is not None
    assert _code_error_hint("exited with code 137", 137) is not None


def test_code_error_hint_none_for_unknown():
    assert _code_error_hint("some other failure", 1) is None


def test_stderr_hint_blocked_import():
    hint = _stderr_hint("ModuleNotFoundError: No module named 'requests'")
    assert hint is not None
    assert "fetch/search/browser_run" in hint


def test_stderr_hint_none_for_clean_stderr():
    assert _stderr_hint("just a warning printed to stderr") is None
