"""Sandboxed Python code execution tool for MCP server."""

import json
import logging
import multiprocessing
import os
import textwrap
from pathlib import Path
from typing import Any

from mcp.server import FastMCP

from src.config import settings

logger = logging.getLogger(__name__)

# Allowed imports for the sandbox
_ALLOWED_IMPORTS = {
    "numpy", "pandas", "matplotlib", "scipy", "sympy", "math", "statistics",
    "json", "csv", "re", "datetime", "collections", "itertools", "functools",
    "io", "os.path", "pathlib", "typing", "enum", "abc", "copy", "pprint",
    "hashlib", "base64", "struct", "urllib.parse", "urllib.request",
    "random", "string", "time", "unicodedata", "textwrap", "operator",
    "decimal", "fractions", "calendar", "tempfile", "shelve", "sqlite3",
}

# Blocked modules that must never be imported
_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib", "poplib",
    "imaplib", "telnetlib", "xmlrpc", "ctypes", "multiprocessing",
    "threading", "concurrent", "signal", "mmap", "select", "selectors",
    "asyncio", "ssl", "_thread", "posix", "nt", "winreg", "msvcrt",
    "termios", "tty", "pwd", "grp", "resource", "fcntl", "termios",
    "curses", "dbm", "gdbm", "nis", "sunau", "ossaudiodev",
    "audioop", "chunk", "colorsys", "crypt", "imghdr", "nntplib",
    "pipes", "sunau", "xdrlib", "aifc", "sndhdr", "wave",
}

# Timeout for code execution (seconds)
_CODE_EXEC_TIMEOUT = 30

# Max output length
_MAX_OUTPUT_LENGTH = 50000


def _check_import_allowed(module_name: str) -> bool:
    """Check if a module import is allowed in the sandbox."""
    # Block top-level blocked modules
    top_level = module_name.split(".")[0]
    if top_level in _BLOCKED_IMPORTS:
        return False
    # Allow if in allowed list or if it's a submodule of an allowed module
    if module_name in _ALLOWED_IMPORTS:
        return True
    # Check if top-level is in allowed imports
    if top_level in _ALLOWED_IMPORTS:
        return True
    # Allow stdlib modules not in blocked list
    try:
        from importlib import util as _util
        spec = _util.find_spec(module_name)
        if spec and spec.origin:
            # If it's in the Python stdlib (not in site-packages), allow it
            origin = spec.origin
            if "site-packages" not in origin and "dist-packages" not in origin:
                return True
    except Exception:
        pass
    return False


def _wrap_code(code: str) -> str:
    """Wrap user code with safety hooks and output capture."""
    # Create a restricted import hook
    wrapper = textwrap.dedent("""
        import builtins
        import sys
        import io
        import traceback

        # Override __import__ to block dangerous imports
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if not _check_import_allowed(name):
                raise ImportError(f"Import of '{{name}}' is not allowed in sandbox.")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = restricted_import

        # Capture stdout and stderr
        _stdout = io.StringIO()
        _stderr = io.StringIO()
        sys.stdout = _stdout
        sys.stderr = _stderr

        # Import guard function for sandbox
        _allowed_imports = {allowed_imports}
        _blocked_imports = {blocked_imports}

        def _check_import_allowed(module_name):
            top_level = module_name.split(".")[0]
            if top_level in _blocked_imports:
                return False
            if module_name in _allowed_imports:
                return True
            if top_level in _allowed_imports:
                return True
            try:
                from importlib import util as _util
                spec = _util.find_spec(module_name)
                if spec and spec.origin:
                    origin = spec.origin
                    if "site-packages" not in origin and "dist-packages" not in origin:
                        return True
            except Exception:
                pass
            return False

        try:
            exec_result = None
            try:
                exec_result = eval(compile({code_literal}, "<sandbox>", "eval"))
            except SyntaxError:
                exec(compile({code_literal}, "<sandbox>", "exec"))
        except Exception as e:
            _stderr.write(traceback.format_exc())
        finally:
            stdout_text = _stdout.getvalue()
            stderr_text = _stderr.getvalue()

            # Write results to output file
            with open({output_file}, "w") as f:
                import json as _json
                result = {{
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "result": str(exec_result) if exec_result is not None else None,
                }}
                _json.dump(result, f, indent=2, ensure_ascii=False)
    """)

    code_escaped = json.dumps(code)

    return wrapper.format(
        allowed_imports=repr(_ALLOWED_IMPORTS),
        blocked_imports=repr(_BLOCKED_IMPORTS),
        code_literal=code_escaped,
        output_file=json.dumps("/tmp/code_run_output.json"),
    )


def _exec_wrapper(code: str) -> None:
    """Execute wrapped code in subprocess (must be top-level for multiprocessing)."""
    # Use same dict for globals+locals so _check_import_allowed is in global scope
    # and accessible to restricted_import called via builtins.__import__
    ns = {}
    exec(compile(code, "<sandbox>", "exec"), ns, ns)  # type: ignore[arg-type]


def _run_code_sandbox(code: str) -> dict[str, Any]:
    """Run code in a subprocess sandbox with timeout."""
    wrapped = _wrap_code(code)
    output_file = "/tmp/code_run_output.json"
    # Remove stale output
    if os.path.exists(output_file):
        os.remove(output_file)

    proc = multiprocessing.Process(
        target=_exec_wrapper,
        args=(wrapped,),
    )
    proc.start()
    proc.join(timeout=_CODE_EXEC_TIMEOUT)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "status": "error",
            "error": f"Code execution timed out after {_CODE_EXEC_TIMEOUT}s",
            "exit_code": -1,
        }

    if proc.exitcode != 0:
        return {
            "status": "error",
            "error": f"Process exited with code {proc.exitcode}",
            "exit_code": proc.exitcode,
        }

    if not os.path.exists(output_file):
        return {
            "status": "error",
            "error": "Code execution failed — no output produced",
            "exit_code": 1,
        }

    try:
        with open(output_file, "r") as f:
            result = json.load(f)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to read execution output: {str(e)}",
            "exit_code": 1,
        }

    # Truncate output if too large
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if len(stdout) > _MAX_OUTPUT_LENGTH:
        stdout = stdout[:_MAX_OUTPUT_LENGTH] + "\n[output truncated]"
    if len(stderr) > _MAX_OUTPUT_LENGTH:
        stderr = stderr[:_MAX_OUTPUT_LENGTH] + "\n[output truncated]"

    return {
        "status": "success",
        "exit_code": 0,
        "stdout": stdout,
        "stderr": stderr,
        "result": result.get("result"),
    }


def code_run_handler(server: FastMCP) -> None:
    """Register the code_run tool."""

    @server.tool()
    async def code_run(
        code: str,
        timeout: int | None = None,
    ) -> str:
        """Execute Python code in a sandboxed subprocess. timeout defaults to 30s.

        Allowed: numpy, pandas, matplotlib, scipy, sympy, math, statistics, json, csv,
        re, datetime, collections, itertools, functools, io, os.path, pathlib.
        Blocked: os, sys, subprocess, shutil, socket, http, urllib, requests,
        threading, multiprocessing, signal, mmap.
        Use print() to return values; last expression result is also captured.
        Returns: {status, stdout, stderr, result}
        """
        global _CODE_EXEC_TIMEOUT
        if timeout:
            _CODE_EXEC_TIMEOUT = timeout

        try:
            result = _run_code_sandbox(code)
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Code run error: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            }, indent=2)

    logger.info("Registered code_run tool")
