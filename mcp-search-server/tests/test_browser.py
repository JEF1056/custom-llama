"""Tests for browser_run's error-reporting helpers.

These cover the recovery contract: when browser code fails, the model must get
back enough to fix it — a traceback whose line number matches the code it wrote,
a corrective hint for common failures, and a report that still carries page
state and the interactables list.
"""

import textwrap

from src.tools.browser import (
    _clean_browser_traceback,
    _failure_hint,
    _render_run_report,
)


def _raise_via_browser_main(code: str) -> BaseException:
    """Compile/run `code` exactly like browser_run does and return the raised exc.

    Mirrors the real wrapper: a 1-line `async def` header + indented body compiled
    under the synthetic filename ``<browser_run>``. Returns the exception object
    (with its traceback) so we can assert on the cleaned rendering.
    """
    wrapper = (
        "def __browser_main():\n" + textwrap.indent(code, "    ")
    )
    ns: dict = {}
    exec(compile(wrapper, "<browser_run>", "exec"), {}, ns)
    try:
        ns["__browser_main"]()
        raise AssertionError("code did not raise")
    except Exception as exc:  # noqa: BLE001 - we want the raised exception
        return exc


def test_clean_traceback_reports_user_code_line():
    # The failing statement is on line 2 of the user's code.
    exc = _raise_via_browser_main("x = 1\nraise ValueError('boom')\n")
    rendered = _clean_browser_traceback(exc)
    assert "ValueError: boom" in rendered
    assert "at code line 2" in rendered


def test_clean_traceback_falls_back_without_user_frame():
    exc = ValueError("standalone")  # no __browser_run__ frame
    rendered = _clean_browser_traceback(exc)
    assert rendered == "ValueError: standalone"


def test_failure_hint_timeout():
    class TimeoutError(Exception):
        pass

    hint = _failure_hint(TimeoutError("Timeout 30000ms exceeded"))
    assert hint is not None
    assert "domcontentloaded" in hint


def test_failure_hint_bad_selector():
    hint = _failure_hint(Exception("waiting for selector \"#nope\" failed"))
    assert hint is not None
    assert "interactables()" in hint


def test_failure_hint_closed_page():
    hint = _failure_hint(Exception("Target page has been closed"))
    assert hint is not None
    assert "session_id" in hint


def test_failure_hint_none_for_unknown():
    assert _failure_hint(Exception("some unrelated failure")) is None


def test_error_report_contains_page_state_and_hint():
    report = _render_run_report(
        status="error",
        sid="main",
        title="Example",
        url="https://example.com",
        body_parts=[],
        stdout_preview="printed before crash",
        stdout_hint=None,
        summary="Interactables (1 visible / 1 total):\n  [0] button: \"Go\" -> button#go",
        error_block="ValueError: boom\n  at code line 2",
        hint="do the thing differently",
    )
    assert "**Status:** error · Session: main" in report
    assert "https://example.com" in report
    assert "**Error:**" in report
    assert "at code line 2" in report
    assert "do the thing differently" in report
    assert "printed before crash" in report
    assert "button#go" in report


def test_success_report_has_no_error_block():
    report = _render_run_report(
        status="success",
        sid="main",
        title="Example",
        url="https://example.com",
        body_parts=["", "**Result:** ok"],
        stdout_preview="",
        stdout_hint=None,
        summary="",
    )
    assert "**Status:** success" in report
    assert "**Error:**" not in report
    assert "**Result:** ok" in report
