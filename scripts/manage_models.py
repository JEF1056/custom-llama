#!/usr/bin/env python3
"""
Model management script for downloading and converting LLM models.
Supports downloading models from HuggingFace and converting to GGUF format.
"""

import contextlib
import os
import struct
import sys
import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

# HuggingFace model repositories that support GGUF downloads
# Models are organized by size for 24GB GPU compatibility:
#   - Small (<4GB at Q4_K_M): fits with large context
#   - Medium (4-12GB at Q4_K_M): fits with moderate context
#   - Large (12-18GB at Q4_K_M): fits with small context
MODELS = {
    # ============================================
    # SMALL MODELS (<4GB at Q4_K_M) - Best for 24GB GPU with large context
    # ============================================
    "qwen3.5-0.8b": {
        "hf_repo": "unsloth/Qwen3.5-0.8B-GGUF",
        "description": "Qwen 3.5 0.8B (~0.5GB)",
        "size_gb": 0.5,
    },
    "llama3.2-1b": {
        "hf_repo": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
        "description": "Llama 3.2 1B Instruct (~0.7GB)",
        "size_gb": 0.7,
    },
    "llama3.2-3b": {
        "hf_repo": "lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
        "description": "Llama 3.2 3B Instruct (~2GB)",
        "size_gb": 2,
    },
    "gemma-4-e2b": {
        "hf_repo": "lmstudio-community/gemma-4-E2B-it-GGUF",
        "description": "Gemma 4 E2B (~1.5GB)",
        "size_gb": 1.5,
        "mmproj": "mmproj-gemma-4-E2B-it-BF16.gguf",
    },
    "qwen3.5-4b": {
        "hf_repo": "unsloth/Qwen3.5-4B-GGUF",
        "description": "Qwen 3.5 4B (~2.5GB)",
        "size_gb": 2.5,
    },
    # ============================================
    # MEDIUM MODELS (4-12GB at Q4_K_M) - Fits with moderate context
    # ============================================
    "qwen2.5-coder-7b": {
        "hf_repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "description": "Qwen 2.5 Coder 7B Instruct (~4.5GB)",
        "size_gb": 4.5,
    },
    "gemma-4-e4b": {
        "hf_repo": "lmstudio-community/gemma-4-E4B-it-GGUF",
        "description": "Gemma 4 E4B (~3GB)",
        "size_gb": 3,
        "mmproj": "mmproj-gemma-4-E4B-it-BF16.gguf",
    },
    "qwen3.5-9b": {
        "hf_repo": "unsloth/Qwen3.5-9B-GGUF",
        "description": "Qwen 3.5 9B (~5.5GB)",
        "size_gb": 5.5,
    },
    "gpt-oss-20b": {
        "hf_repo": "unsloth/gpt-oss-20b-GGUF",
        "description": "GPT-OSS 20B (~11GB)",
        "size_gb": 11,
    },
    # ============================================
    # LARGE MODELS (12-18GB at Q4_K_M) - Fits with small context
    # ============================================
    "gemma-4-26b-a4b": {
        "hf_repo": "lmstudio-community/gemma-4-26B-A4B-it-GGUF",
        "description": "Gemma 4 26B-A4B (~13GB)",
        "size_gb": 13,
        "mmproj": "mmproj-gemma-4-26B-A4B-it-BF16.gguf",
    },
    "gemma-4-31b": {
        "hf_repo": "lmstudio-community/gemma-4-31B-it-GGUF",
        "description": "Gemma 4 31B (~16GB)",
        "size_gb": 16,
        "mmproj": "mmproj-gemma-4-31B-it-BF16.gguf",
    },
    "qwen3.6-27b": {
        "hf_repo": "unsloth/Qwen3.6-27B-GGUF",
        "fp16_repo": "Qwen/Qwen3.6-27B",
        "description": "Qwen 3.6 27B (~14GB)",
        "size_gb": 14,
        "mmproj": "mmproj-F16.gguf"
    },
    "qwen3.6-35b-a3b": {
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "fp16_repo": "Qwen/Qwen3.6-35B-A3B",
        "description": "Qwen 3.6 35B-A3B (~17GB)",
        "size_gb": 17,
        "mmproj": "mmproj-F16.gguf"
    },
    "qwopus3.6-35b": {
        "hf_repo": "Jackrong/Qwopus3.6-35B-A3B-v1-GGUF",
        "fp16_repo": "Jackrong/Qwopus3.6-35B-A3B-v1",
        "description": "Qwopus 3.6 35B-A3B-v1 (~17GB)",
        "size_gb": 17,
        "mmproj": "mmproj.gguf",
    },
    "qwopus3.6-27b": {
        "hf_repo": "Jackrong/Qwopus3.6-27B-v1-preview-GGUF",
        "fp16_repo": "Jackrong/Qwopus3.6-27B-v1-preview",
        "description": "Qwopus 3.6 27B-v1-preview (~14GB)",
        "size_gb": 14,
        "mmproj": "mmproj.gguf",
    },
    "minimax-m2.7": {
        "hf_repo": "unsloth/MiniMax-M2.7-GGUF",
        "fp16_repo": "unsloth/MiniMax-M2.7",
        "description": "MiniMax M2.7 (~18GB)",
        "size_gb": 18,
        "notes": "Mixture-of-Experts model. TurboQuant requires fp16 source from fp16_repo.",
        "turboquant": True,
    },
}

DEFAULT_MODELS_DIR = "./models"
DEFAULT_OUTPUT_DIR_HELP = "Output directory (default: ./models)"

# TurboQuant quantization options (extreme compression, must be quantized locally)
TQ_QUANT_OPTIONS = [
    "TQ2_0",  # 2-bit per weight - better quality while still highly compressed
    "TQ1_0",  # 1-bit per weight - extreme compression
]

# llama-quantize only accepts fp16 or bf16 as input when producing TurboQuant.
# Lower quants (Q8_0, Q6_K, …) cannot be re-quantized to TQ targets.
TQ_SOURCE_QUANTS = ["fp16", "bf16"]

# Unsloth Dynamic (UD) quantization options — pre-built download only.
# These use a mixed-precision scheme that preserves important layers at higher
# precision. llama-quantize cannot produce them; they must be downloaded directly
# from HuggingFace (typically from unsloth/*-GGUF repos).
# Naming: "UD-<VARIANT>" where VARIANT is the quantization method.
UD_QUANT_OPTIONS = [
    "UD-Q8_K_XL",   # ~8-bit (best quality)
    "UD-Q6_K_XL",   # ~6-bit
    "UD-Q5_K_XL",   # ~5-bit
    "UD-Q4_K_XL",   # ~4-bit (recommended balance)
    "UD-Q3_K_XL",   # ~3-bit
    "UD-Q2_K_XL",   # ~2-bit
    "UD-IQ3_XXS",   # ~3-bit iQuant (smaller)
    "UD-IQ2_M",     # ~2-bit iQuant (medium)
    "UD-IQ2_XXS",   # ~2-bit iQuant (smallest)
]

# Ordered list of quantization types from highest to lowest quality.
# Used when selecting the best available source file for local quantization.
QUANT_PRIORITY = [
    "fp16",
    "bf16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
    "IQ4_NL",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "IQ4_XS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ2_L",
    "IQ2 XS",
    "IQ1_S",
    "IQ1_M",
    "Q2_K",
    "Q1_K",
]

# Quantizations that cannot be re-quantized to standard (non-TQ) targets.
# For standard quants (Q4_K_M, Q6_K, …) llama-quantize accepts Q8_0 and higher
# as sources.  TurboQuant is handled separately — see TQ_SOURCE_QUANTS above.
NON_REQUANTIZABLE: set = set()


def _is_ud_quant(quant: str) -> bool:
    """Return True if the quant string is an Unsloth Dynamic (UD) quantization.

    UD quants are pre-built by Unsloth and must be downloaded directly from
    HuggingFace. llama-quantize cannot produce them.  They are identified by
    the "UD-" prefix (case-insensitive).
    """
    return quant.upper().startswith("UD-")


def _fetch_repo_files(repo_id: str) -> list:
    """Return the list of file entries from the HuggingFace /tree/main API."""
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        result = subprocess.run(
            ["curl", "-s", api_url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Warning: Failed to fetch file list from HuggingFace for {repo_id}")
            return []
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout fetching file list from HuggingFace for {repo_id}")
        return []
    except Exception as e:
        print(f"Warning: Error fetching file list from HuggingFace for {repo_id}: {e}")
        return []


def find_quant_in_repo(repo_id: str, quant: str) -> str | None:
    """Search a HuggingFace repo for a GGUF file matching the requested quantization.

    Matching is case-insensitive and requires the quant string to appear as a
    word boundary segment in the filename (e.g. "Q4_K_M" matches
    "Model-Q4_K_M.gguf" but not "SomeQ4_K_Mfoo.gguf").

    When multiple matches exist, the largest file is returned (most complete shard).

    Args:
        repo_id: HuggingFace repository ID (e.g. "unsloth/Qwen3.5-27B-GGUF")
        quant: Quantization string to search for (e.g. "Q4_K_M", "fp16", "TQ2_0")

    Returns:
        Filename (path within repo) of the best match, or None if not found.
    """
    files = _fetch_repo_files(repo_id)
    quant_lower = quant.lower()

    matches = []
    for entry in files:
        if entry.get("type") != "file":
            continue
        path = entry.get("path", "")
        if not path.endswith(".gguf"):
            continue
        name_lower = path.lower()
        # Require quant to appear surrounded by non-alphanumeric chars or at
        # string boundaries so "Q4_K_M" doesn't match inside a longer token.
        # A simple check: the quant string must appear preceded/followed by
        # '-', '_', '.', '/', or the start/end of the filename stem.
        if quant_lower in name_lower:
            idx = name_lower.find(quant_lower)
            before_ok = idx == 0 or name_lower[idx - 1] in "-_./\\"
            after_idx = idx + len(quant_lower)
            after_ok = after_idx >= len(name_lower) or name_lower[after_idx] in "-_./\\."
            if before_ok and after_ok:
                matches.append((path, entry.get("size", 0)))

    if not matches:
        return None

    # Prefer the largest file (handles sharded models — pick biggest shard,
    # but for single-file models this is just the one file).
    matches.sort(key=lambda f: -f[1])
    return matches[0][0]


def find_best_source_in_repo(repo_id: str, allowed_quants: list | None = None) -> str | None:
    """Find the highest-quality GGUF in a repo suitable as a quantization source.

    Walks QUANT_PRIORITY in order (or ``allowed_quants`` if provided), skipping
    NON_REQUANTIZABLE types, and returns the first match.  Falls back to
    NON_REQUANTIZABLE types only if nothing better is available (and no
    ``allowed_quants`` restriction is in effect).

    Args:
        repo_id: HuggingFace repository ID
        allowed_quants: Optional explicit list of quant strings to consider,
            in priority order.  When provided, only these quants are tried and
            the NON_REQUANTIZABLE fallback is skipped.  Use this to restrict
            the source to fp16/bf16 for TurboQuant targets.

    Returns:
        Filename of the best source GGUF, or None if no GGUF exists.
    """
    priority = allowed_quants if allowed_quants is not None else QUANT_PRIORITY

    for quant in priority:
        if quant.upper() in NON_REQUANTIZABLE:
            continue
        match = find_quant_in_repo(repo_id, quant)
        if match:
            return match

    if allowed_quants is not None:
        # Strict list requested — do not fall back to other quants.
        return None

    # Second pass: accept non-requantizable as last resort
    for quant in NON_REQUANTIZABLE:
        match = find_quant_in_repo(repo_id, quant)
        if match:
            return match

    return None


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (e.g. 18.9 GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _is_wsl2() -> bool:
    """Return True when the I/O environment is a WSL2-backed Windows host.

    Two signals are checked, either of which is sufficient:

    1. /proc/version contains "microsoft" — set by the WSL2 kernel when
       running directly in a WSL2 distro, and also by some Docker Desktop
       builds that expose the WSL2 kernel to containers.

    2. /sys/class/dmi/id/sys_vendor contains "microsoft" — Hyper-V (the VM
       layer Docker Desktop uses on Windows) always reports "Microsoft
       Corporation" here, even when the container kernel does not advertise
       WSL2 in /proc/version.  This covers Docker Desktop for Windows
       regardless of which WSL2 integration path it uses.

    macOS Docker Desktop uses Apple's Hypervisor framework (DMI vendor
    "Apple Inc." or "QEMU"), so it is not affected.
    """
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return True
    except OSError:
        pass

    for dmi in ("/sys/class/dmi/id/sys_vendor", "/sys/class/dmi/id/board_vendor"):
        try:
            if "microsoft" in Path(dmi).read_text().lower():
                return True
        except OSError:
            continue

    return False


def _hf_download_file(repo_id: str, filename: str, local_dir: str) -> Path:
    """Download a single file from a HuggingFace repo using the huggingface_hub API.

    Uses the Python API (not the CLI) so that progress bars display correctly
    even in non-TTY environments such as Docker detached mode.

    When CONVERT_DOWNLOAD_RATE is set (e.g. "300M"), routes through curl with
    --limit-rate instead.  Throttling write throughput reduces the disk-write
    interrupt storm that causes WSL2's vmmem to balloon and can trigger a
    CLOCK_WATCHDOG_TIMEOUT BSOD on Windows before quantization even starts.

    When hf_transfer is installed (pip install hf-transfer) and
    HF_HUB_ENABLE_HF_TRANSFER=1 is set, huggingface_hub routes all downloads
    through a Rust-based multi-connection engine for a significant speedup.
    On WSL2, hf_transfer is NOT auto-enabled because its burst writes
    overwhelm WSL2's vmmem and can crash the guest on large files.

    Falls back to aria2c (16 connections on Linux/macOS, 4 on WSL2) or curl
    when huggingface_hub is not available.

    Returns:
        Path to the downloaded file (always a flat path inside local_dir).
    """
    output_path = Path(local_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Normalize to a flat filename — HF paths may include repo subdirectories.
    dest = output_path / Path(filename).name
    if dest.exists():
        size_str = _fmt_bytes(dest.stat().st_size)
        print(f"  Already cached: {dest.name} ({size_str})")
        return dest

    _section(f"Downloading: {Path(filename).name}")
    print(f"  Repository  : {repo_id}")
    print(f"  Destination : {dest}")

    on_wsl2 = _is_wsl2()
    if on_wsl2:
        print("  Host        : WSL2/Windows — conservative I/O mode active")
        print("                (hf_transfer suppressed; aria2c capped at 4 connections)")
        print("                Set CONVERT_DOWNLOAD_RATE (e.g. 300M) to throttle further if crashes persist.")

    # When CONVERT_DOWNLOAD_RATE is set, use curl with --limit-rate to cap
    # write throughput and prevent WSL2 memory balloon / BSOD on Windows.
    download_rate = os.environ.get("CONVERT_DOWNLOAD_RATE", "").strip()
    if download_rate:
        print(f"  Rate limit  : {download_rate} (CONVERT_DOWNLOAD_RATE)")
        return _curl_download_file(repo_id, filename, dest, limit_rate=download_rate)

    # Import huggingface_hub module first; guard hf_hub_download separately from
    # enable_progress_bars so a missing/renamed helper doesn't prevent downloads.
    try:
        import huggingface_hub as _hfh
        _hf_hub_download = _hfh.hf_hub_download
    except (ImportError, AttributeError):
        print("  huggingface_hub not available — using aria2c/curl fallback.")
        return _curl_download_file(repo_id, filename, dest, on_wsl2=on_wsl2)

    # enable_progress_bars: present since 0.14.0; silently skip if absent.
    try:
        _hfh.enable_progress_bars()
    except AttributeError:
        pass

    # hf_transfer: Rust parallel downloader (pip install hf-transfer).
    # Auto-enable when the package is present and the env var isn't already set.
    # Skip auto-enable on WSL2 — hf_transfer's burst writes overwhelm WSL2's
    # vmmem and can cause the guest to crash on large files.  Users who know
    # their setup can still force it by setting HF_HUB_ENABLE_HF_TRANSFER=1.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
        try:
            import hf_transfer  # noqa: F401 — presence check only
            if on_wsl2:
                print("  hf_transfer detected but suppressed on WSL2 (set HF_HUB_ENABLE_HF_TRANSFER=1 to force).")
            else:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                print("  hf_transfer detected — enabling fast parallel download.")
        except ImportError:
            pass

    try:
        downloaded = Path(
            _hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(output_path),
            )
        )
        # hf_hub_download may place the file at a sub-path; flatten it.
        if downloaded.resolve() != dest.resolve() and downloaded.exists():
            shutil.move(str(downloaded), str(dest))
        size_str = _fmt_bytes(dest.stat().st_size)
        print(f"\n  Done: {dest.name} ({size_str})")
        return dest
    except Exception as e:
        print(f"Error: Download failed: {e}")
        sys.exit(1)


def _curl_download_file(
    repo_id: str, filename: str, dest: Path, limit_rate: str = "", on_wsl2: bool | None = None
) -> Path:
    """Download a file from HuggingFace.

    When *limit_rate* is empty and aria2c is available, downloads via
    aria2c.  On WSL2, uses 4 parallel connections to avoid the burst
    disk-write pressure that causes vmmem to balloon.  On bare-metal Linux
    or macOS, uses 16 connections to saturate the link.

    When *limit_rate* is set (WSL2 BSOD throttle path), or when aria2c is
    absent, falls back to curl.  curl's --limit-rate caps write throughput
    which prevents the WSL2 vmmem balloon that causes CLOCK_WATCHDOG_TIMEOUT
    BSODs on Windows during large file downloads.

    Args:
        limit_rate: Optional curl --limit-rate value (e.g. "300M").  When
            empty, aria2c is tried first.
        on_wsl2: Pre-computed result of _is_wsl2(); detected automatically
            when None.
    """
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"  Download URL  : {download_url}")

    hf_token = os.environ.get("HF_TOKEN", "")

    # ── aria2c path (multi-connection, no rate limit) ─────────────────────
    if not limit_rate and shutil.which("aria2c"):
        # On WSL2 reduce to 4 connections — 16 creates a disk-write burst
        # that balloons vmmem and can crash the WSL2 guest on large files.
        if on_wsl2 is None:
            on_wsl2 = _is_wsl2()
        connections = 4 if on_wsl2 else 16
        if on_wsl2:
            print(f"  Using aria2c ({connections} parallel connections — WSL2 conservative mode) ...")
        cmd = [
            "aria2c",
            f"--max-connection-per-server={connections}",
            f"--split={connections}",
            "--min-split-size=50M",
            "--continue=true",
            "--dir", str(dest.parent),
            "--out", dest.name,
        ]
        if hf_token:
            cmd += ["--header", f"Authorization: Bearer {hf_token}"]
        cmd.append(download_url)
        if not on_wsl2:
            print(f"  Using aria2c ({connections} parallel connections) ...")
        try:
            result = subprocess.run(cmd, capture_output=False, timeout=7200)
            if result.returncode == 0 and dest.exists():
                size_str = _fmt_bytes(dest.stat().st_size)
                print(f"\n  Done: {dest.name} ({size_str})")
                return dest
            print(f"\nWarning: aria2c failed (exit {result.returncode}), falling back to curl ...")
        except subprocess.TimeoutExpired:
            print("\nWarning: aria2c timed out, falling back to curl ...")
        except FileNotFoundError:
            pass  # shutil.which lied somehow; continue to curl

    # ── curl path (rate-limited or aria2c unavailable) ────────────────────
    if hf_token:
        auth_url = f"{download_url}?token={hf_token}"
    else:
        auth_url = download_url

    # Default curl progress meter shows % done, speed, and time-left (ETA).
    # -# (hash bar) is intentionally omitted — it hides the ETA column.
    # -C - resumes an interrupted download using HTTP range requests;
    # HuggingFace supports range requests so this is always safe to pass.
    cmd = ["curl", "-L", "-f", "-C", "-", "-o", str(dest)]
    if limit_rate:
        cmd += ["--limit-rate", limit_rate]
    cmd.append(auth_url)

    print("  Using curl to download ...")
    try:
        result = subprocess.run(cmd, capture_output=False, timeout=7200)
        if result.returncode != 0:
            print(f"\nError: curl download failed (exit {result.returncode})")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("\nError: curl download timed out after 2 hours")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: curl not found. Please install curl or huggingface_hub.")
        sys.exit(1)

    size_str = _fmt_bytes(dest.stat().st_size) if dest.exists() else "unknown"
    print(f"\n  Done: {dest.name} ({size_str})")
    return dest


@contextlib.contextmanager
def _keepalive(interval: int = 30, label: str = ""):
    """Print periodic heartbeat lines to prevent Docker watchdog timeouts on Windows.

    Docker Desktop on Windows (Hyper-V / WSL2) kills containers that produce no
    stdout/stderr for an extended period. Long operations like llama-quantize or
    convert_hf_to_gguf.py can run silently for many minutes, triggering the watchdog.
    This context manager starts a daemon thread that prints a timestamped line every
    ``interval`` seconds so the watchdog sees continuous activity.
    """
    stop = threading.Event()
    prefix = f"  [{label}] " if label else "  "

    def _loop():
        while not stop.wait(interval):
            pass

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1)


def _resolve_nthreads(nthreads: int | None) -> int:
    """Return the thread count to pass to llama-quantize.

    Resolution order:
    1. Explicit ``nthreads`` argument (from --threads CLI flag).
    2. ``CONVERT_THREADS`` environment variable.
    3. Half the logical CPU count (leaves the other half for the host OS).

    Capping quantization threads is critical on Windows (Docker Desktop / WSL2):
    saturating every logical CPU starves the host kernel's watchdog timer and
    can trigger a CLOCK_WATCHDOG_TIMEOUT BSOD.
    """
    if nthreads is not None and nthreads > 0:
        return nthreads

    env_val = os.environ.get("CONVERT_THREADS", "").strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)

    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count // 2)


def _quantize(source: Path, dest: Path, quant: str, nthreads: int | None = None) -> None:
    """Run llama-quantize to produce dest from source with the given quant type.

    Streams llama-quantize output directly to the console and exits on failure.

    Args:
        source: Input GGUF file (fp16, bf16, Q8_0, or any supported source quant).
        dest: Output GGUF path.
        quant: Target quantization string (e.g. "Q4_K_M", "TQ2_0").
        nthreads: Number of CPU threads to use. Falls back to CONVERT_THREADS env
            var, then cpu_count//2. Limiting threads prevents CLOCK_WATCHDOG_TIMEOUT
            BSODs on Windows when Docker saturates all host CPU cores.
    """
    threads = _resolve_nthreads(nthreads)
    source_size = _fmt_bytes(source.stat().st_size) if source.exists() else "unknown"
    _section(f"Quantizing to {quant}")
    print(f"  Source  : {source.name} ({source_size})")
    print(f"  Output  : {dest.name}")
    print(f"  Threads : {threads}")

    t0 = time.monotonic()
    try:
        with _keepalive(30, "quantize"):
            result = subprocess.run(
                ["llama-quantize", str(source), str(dest), quant, str(threads)],
                stdout=None,
                stderr=None,
            )
        if result.returncode != 0:
            print(f"\nError: llama-quantize failed (exit {result.returncode})")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: llama-quantize not found. Ensure llama.cpp is installed.")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    dest_size = _fmt_bytes(dest.stat().st_size) if dest.exists() else "unknown"
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    print(f"\n  Done: {dest.name} ({dest_size}) in {elapsed_str}")


def _section(title: str) -> None:
    """Print a visible section header."""
    bar = "─" * 60
    print(f"\n  {bar}")
    print(f"  {title}")
    print(f"  {bar}")


def _done(path: Path) -> None:
    """Print a final 'model ready' line with file size."""
    size_str = _fmt_bytes(path.stat().st_size) if path.exists() else "unknown"
    print(f"\n  ✓ Model ready: {path.name} ({size_str})")


def _run_calibration(
    model_name: str,
    model_info: dict,
    output_dir: str,
    calib_input: str,
    model_path: str | None = None,
) -> None:
    """Run triattention_calibrate.py for a model after download/convert.

    Silently skips when:
    - calib_input is empty / None (no corpus provided)
    - {output_dir}/{model_name}-triattention.bin already exists
    - The model has no safetensors repo (fp16_repo) and no local path

    On any calibration failure, prints a warning and continues — never
    exits. Calibration errors must not abort a successful download/convert.

    Args:
        model_name: Key from the MODELS dict.
        model_info: Model metadata dict.
        output_dir: Output directory (stats file written here).
        calib_input: Path to plain-text calibration corpus, or empty string.
        model_path: Optional local path to safetensors dir (e.g. st_dir from
            convert-st). When provided, avoids re-downloading weights.
    """
    if not calib_input:
        return

    stats_path = Path(output_dir) / f"{model_name}-triattention.bin"
    if stats_path.exists():
        size_str = _fmt_bytes(stats_path.stat().st_size)
        print(f"\n  TriAttention stats already exist: {stats_path.name} ({size_str}) — skipping calibration.")
        return

    # Determine which model path to pass to calibrate.
    # Prefer a local safetensors dir (no extra download).
    # Fall back to fp16_repo (HF safetensors).
    # NEVER pass a *-GGUF repo — transformers cannot load it.
    if model_path and Path(model_path).exists():
        hf_model = model_path
    elif model_info.get("fp16_repo"):
        hf_model = model_info["fp16_repo"]
    else:
        print(
            f"\n  Skipping TriAttention calibration for {model_name}: "
            "no safetensors repo (fp16_repo) available. "
            "GGUF-only repos cannot be loaded by the calibration script."
        )
        return

    script_dir = Path(__file__).parent
    calibrate_script = script_dir / "triattention_calibrate.py"
    if not calibrate_script.exists():
        print(f"\n  Skipping TriAttention calibration: {calibrate_script} not found.")
        return

    _section(f"TriAttention Calibration: {model_name}")
    print(f"  Model   : {hf_model}")
    print(f"  Corpus  : {calib_input}")
    print(f"  Output  : {stats_path}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(calibrate_script),
                "--model", str(hf_model),
                "--input", str(calib_input),
                "--output", str(stats_path),
            ],
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"\n  Warning: TriAttention calibration failed (exit {result.returncode}) — continuing.")
        elif stats_path.exists():
            size_str = _fmt_bytes(stats_path.stat().st_size)
            print(f"\n  ✓ TriAttention stats: {stats_path.name} ({size_str})")
    except subprocess.TimeoutExpired:
        print("\n  Warning: TriAttention calibration timed out after 1 hour — continuing.")
    except Exception as e:
        print(f"\n  Warning: TriAttention calibration error ({e}) — continuing.")


def download_model(model_name: str, quant: str, output_dir: str, nthreads: int | None = None, keep_intermediate: bool = False, calib_input: str = "") -> None:
    """Download (and if necessary locally quantize) a model.

    Algorithm
    ---------
    1. If the canonical output file already exists on disk → done.
    2. If the requested quant is available as a pre-built GGUF on HuggingFace
       → download it and rename to the canonical path.
    3. Otherwise → download the best available source GGUF (fp16 / bf16 / …),
       run llama-quantize to produce the canonical file, then delete the source.

    TurboQuant types (TQ1_0, TQ2_0) are never available pre-built on HF, so
    they always go through the local quantization path.

    The canonical filename is:  {output_dir}/{model_name}-{QUANT}.gguf
    This predictable name allows the entrypoint to locate the model without
    scanning the directory.

    Args:
        model_name: Key from the MODELS dict (e.g. "qwen3.5-27b")
        quant: Quantization type (e.g. "Q4_K_M", "TQ2_0")
        output_dir: Directory where the model file should be placed
    """
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print("Run 'manage_models.py list' to see available models.")
        sys.exit(1)

    model_info = MODELS[model_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    canonical = output_path / f"{model_name}-{quant}.gguf"

    _section(f"Model: {model_name}  |  Quant: {quant}")
    print(f"  Repository : {model_info['hf_repo']}")
    print(f"  Output     : {canonical}")

    # ------------------------------------------------------------------ #
    # 1. Already on disk?
    # ------------------------------------------------------------------ #
    if canonical.exists():
        size_str = _fmt_bytes(canonical.stat().st_size)
        print(f"\n  ✓ Already on disk: {canonical.name} ({size_str}) — skipping download.")
        _maybe_download_mmproj(model_info, output_path, model_name)
        _run_calibration(model_name, model_info, output_dir, calib_input)
        return

    hf_repo = model_info["hf_repo"]
    is_turboquant = quant in TQ_QUANT_OPTIONS
    is_ud = _is_ud_quant(quant)

    # ------------------------------------------------------------------ #
    # 2. Pre-built GGUF available on HuggingFace?
    # ------------------------------------------------------------------ #
    hf_file = None
    if not is_turboquant:
        print(f"  Checking HuggingFace for pre-built {quant} in {hf_repo} ...")
        hf_file = find_quant_in_repo(hf_repo, quant)

    if hf_file:
        print(f"  Found pre-built {quant}: {Path(hf_file).name}")
        downloaded = _hf_download_file(hf_repo, hf_file, str(output_path))
        if downloaded.resolve() != canonical.resolve():
            shutil.move(str(downloaded), str(canonical))
        _done(canonical)
        _maybe_download_mmproj(model_info, output_path, model_name)
        _run_calibration(model_name, model_info, output_dir, calib_input)
        return

    # ------------------------------------------------------------------ #
    # 2b. UD quant not found — cannot be produced locally
    # ------------------------------------------------------------------ #
    if is_ud:
        print(
            f"\nError: {quant} not found in {hf_repo}.\n"
            "Unsloth Dynamic (UD) quantizations are pre-built by Unsloth and cannot\n"
            "be produced by llama-quantize. They must be downloaded directly.\n\n"
            "Possible reasons:\n"
            f"  • This model does not have UD quants in {hf_repo}\n"
            "  • The model may exist in a different Unsloth GGUF repo\n\n"
            "Available UD quants can be browsed at:\n"
            f"  https://huggingface.co/{hf_repo}/tree/main\n\n"
            "To use a standard quantization instead, try:\n"
            f"  --quant Q4_K_M  (or IQ4_XS, Q3_K_L, etc.)"
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Local quantization path
    # ------------------------------------------------------------------ #
    if is_turboquant:
        print(f"  {quant} is a TurboQuant type — no pre-built GGUF exists on HuggingFace.")
        print(f"  TurboQuant requires fp16 or bf16 source; searching {hf_repo} ...")
    else:
        print(f"  No pre-built {quant} found on HuggingFace.")
    print("  Will download the best available source and quantize locally.")

    # TurboQuant (TQ1_0, TQ2_0) can only be produced from fp16 or bf16 input.
    # Lower quants such as Q8_0 are not accepted by llama-quantize for TQ targets.
    source_allowed = TQ_SOURCE_QUANTS if is_turboquant else None
    source_file = find_best_source_in_repo(hf_repo, allowed_quants=source_allowed)

    if source_file is None and is_turboquant:
        fp16_repo = model_info.get("fp16_repo", hf_repo)
        print(
            f"Error: No fp16 or bf16 GGUF found in {hf_repo}.\n"
            "TurboQuant (TQ1_0 / TQ2_0) can only be quantized from fp16 or bf16 GGUF.\n"
            "The source weights may be available as safetensors and need converting first.\n"
            "Use the convert image:\n"
            f"  docker compose run --rm llama-convert convert-st {model_name} --quant {quant}\n"
            f"  (downloads from {fp16_repo}, converts to fp16 GGUF, then quantizes)"
        )
        sys.exit(1)

    if source_file is None and "fp16_repo" in model_info:
        print(
            f"Error: No suitable GGUF source found in {hf_repo} and "
            f"{model_info['fp16_repo']} is a safetensors repo which requires "
            f"llama-convert (not available in this image).\n"
            "Please add a GGUF variant for this model or build with llama-convert support."
        )
        sys.exit(1)

    if source_file is None:
        print(f"Error: No suitable GGUF source found in {hf_repo} for local quantization.")
        sys.exit(1)

    print(f"  Best available source: {Path(source_file).name}")

    actual_source = _hf_download_file(hf_repo, source_file, str(output_path))

    _quantize(actual_source, canonical, quant, nthreads=nthreads)

    if keep_intermediate:
        print(f"\n  Keeping source GGUF  : {actual_source.name}")
    else:
        print(f"\n  Removing source GGUF : {actual_source.name}")
        actual_source.unlink(missing_ok=True)

    _done(canonical)
    _maybe_download_mmproj(model_info, output_path, model_name)
    _run_calibration(model_name, model_info, output_dir, calib_input)


def _find_mmproj_in_repo(repo_id: str) -> str | None:
    """Search a HuggingFace repo for an mmproj GGUF file.

    Returns the filename of the first file whose name contains "mmproj", or None.
    """
    files = _fetch_repo_files(repo_id)
    for entry in files:
        if entry.get("type") != "file":
            continue
        path = entry.get("path", "")
        if path.endswith(".gguf") and "mmproj" in path.lower():
            return path
    return None


def _maybe_download_mmproj(model_info: dict, output_path: Path, model_name: str = "") -> None:
    """Download the multimodal projector file if this model requires one.

    The downloaded file is always renamed to ``{model_name}-mmproj.gguf`` so
    that the entrypoint can locate it by model name without needing an explicit
    ``LLAMA_MMPROJ`` path.

    The ``mmproj`` field in the model entry can be:
    - A specific filename string → download that exact file.
    - ``True`` → auto-detect the mmproj filename by searching the GGUF repo.
    - Absent / falsy → model has no mmproj, skip.
    """
    mmproj_value = model_info.get("mmproj")
    if not mmproj_value:
        return

    repo_id = model_info["hf_repo"]

    if mmproj_value is True:
        print("\nSearching for mmproj file in repo ...")
        mmproj_filename = _find_mmproj_in_repo(repo_id)
        if not mmproj_filename:
            print(f"Warning: No mmproj file found in {repo_id} — vision will not be available.")
            return
        print(f"  Found: {mmproj_filename}")
    else:
        mmproj_filename = mmproj_value

    canonical_name = f"{model_name}-mmproj.gguf" if model_name else Path(mmproj_filename).name
    canonical_path = output_path / canonical_name

    if canonical_path.exists():
        size_str = _fmt_bytes(canonical_path.stat().st_size)
        print(f"  Multimodal projector already exists: {canonical_name} ({size_str})")
        return

    print(f"\nDownloading multimodal projector: {Path(mmproj_filename).name} ...")
    downloaded = _hf_download_file(repo_id, mmproj_filename, str(output_path))
    if downloaded.resolve() != canonical_path.resolve():
        shutil.move(str(downloaded), str(canonical_path))
    print(f"  Multimodal projector ready: {canonical_name}")


def _skip_gguf_value(f, vtype: int):
    """Skip a single GGUF metadata value based on its type tag.
    Returns the value for integer types, None otherwise.

    Type IDs: 0-7 = UINT8..BOOL, 8 = STRING, 9 = ARRAY, 10-12 = UINT64..FLOAT64.
    """
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in fixed:
        data = f.read(fixed[vtype])
        if vtype in (4, 5):  # UINT32, INT32
            return struct.unpack("<I" if vtype == 4 else "<i", data)[0]
        if vtype in (10, 11):  # UINT64, INT64
            return struct.unpack("<Q" if vtype == 10 else "<q", data)[0]
        return None
    elif vtype == 8:  # STRING
        slen = struct.unpack("<Q", f.read(8))[0]
        f.read(slen)
        return None
    elif vtype == 9:  # ARRAY
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            _skip_gguf_value(f, elem_type)
        return None
    return None


def _gguf_read_header(gguf_path: Path) -> dict:
    """Read GGUF metadata and tensor names from a file header.

    Returns dict with keys: 'metadata' (dict of kv pairs we care about),
    'tensor_names' (list of str), 'n_tensors' (int).
    """
    result: dict = {"metadata": {}, "tensor_names": [], "n_tensors": 0}
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return result
        _version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        result["n_tensors"] = n_tensors

        # Read KV pairs, capture ones we care about
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8")
            vtype = struct.unpack("<I", f.read(4))[0]
            val = _skip_gguf_value(f, vtype)
            if "nextn_predict_layers" in key or "block_count" in key:
                result["metadata"][key] = val

        # Read tensor names
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode("utf-8")
            result["tensor_names"].append(name)
            n_dims = struct.unpack("<I", f.read(4))[0]
            f.read(8 * n_dims)  # dimensions
            f.read(4)           # type
            f.read(8)           # offset
    return result


def convert_safetensors(
    model_name: str,
    quant: str,
    output_dir: str,
    nthreads: int | None = None,
    mtp: bool = False,
    keep_intermediate: bool = False,
    calib_input: str = "",
) -> None:
    """Convert a safetensors model to a quantized GGUF via fp16 GGUF intermediate.

    Pipeline
    --------
    1.  Download the safetensors repo (fp16_repo, or hf_repo if fp16_repo absent).
    2.  Run convert_hf_to_gguf.py → fp16 GGUF (includes MTP tensors when present).
    3.  Run llama-quantize → target quant GGUF.
    4.  Clean up the safetensors download and the fp16 intermediate.

    When ``mtp=True``:
    - The prebuilt-GGUF shortcut is skipped (prebuilts strip MTP tensors).
    - Output is named ``{model_name}-{quant}-mtp.gguf``.
    - Enable with ``LLAMA_SPEC_TYPE=mtp`` in .env.

    Args:
        model_name: Key from the MODELS dict (e.g. "qwen3.5-27b")
        quant: Target quantization (e.g. "TQ2_0", "Q4_K_M", "IQ4_XS")
        output_dir: Directory where the final GGUF should be placed
        mtp: When True, produce a GGUF with MTP tensors included.
    """
    # Locate convert_hf_to_gguf.py — expected in the same directory as this script
    # when running inside the convert Docker image.
    script_dir = Path(__file__).parent
    convert_script = script_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(
            "Error: convert_hf_to_gguf.py not found.\n"
            "The 'convert-st' command requires the convert Docker image.\n"
            "Build it with:  docker compose build llama-convert\n"
            "Then run:       docker compose run --rm llama-convert convert-st "
            f"{model_name} --quant {quant}"
        )
        sys.exit(1)

    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print("Run 'manage_models.py list' to see available models.")
        sys.exit(1)

    model_info = MODELS[model_name]
    source_repo = model_info.get("fp16_repo") or model_info["hf_repo"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if mtp:
        canonical = output_path / f"{model_name}-{quant}-mtp.gguf"
    else:
        canonical = output_path / f"{model_name}-{quant}.gguf"

    mtp_label = "  MTP    : enabled (nextn tensors will be included)" if mtp else ""
    _section(f"Model: {model_name}  |  Quant: {quant}  |  Source: {source_repo}")
    print(f"  Output : {canonical}")
    if mtp_label:
        print(mtp_label)

    if canonical.exists():
        size_str = _fmt_bytes(canonical.stat().st_size)
        print(f"\n  ✓ Already on disk: {canonical.name} ({size_str}) — skipping.")
        _maybe_download_mmproj(model_info, output_path, model_name)
        _run_calibration(model_name, model_info, output_dir, calib_input)
        return

    # ------------------------------------------------------------------ #
    # 0. Short-circuit: check if a pre-built GGUF already exists in hf_repo
    #    (same logic as the `download` command). Skip for TurboQuant targets
    #    and for MTP builds — prebuilt GGUFs strip MTP tensors.
    # ------------------------------------------------------------------ #
    is_turboquant = quant in TQ_QUANT_OPTIONS

    if _is_ud_quant(quant):
        print(
            f"\nError: UD quants ({quant}) cannot be produced via convert-st.\n"
            "Unsloth Dynamic (UD) quantizations are pre-built and must be downloaded.\n"
            "Use the download command instead:\n"
            f"  docker compose run --rm llama-convert download {model_name} --quant {quant}"
        )
        sys.exit(1)
    hf_repo = model_info["hf_repo"]
    if not is_turboquant and not mtp:
        print(f"  Checking HuggingFace for pre-built {quant} in {hf_repo} ...")
        hf_file = find_quant_in_repo(hf_repo, quant)
        if hf_file:
            print(f"  Found pre-built {quant}: {Path(hf_file).name}")
            print("  Skipping safetensors download — downloading GGUF directly.")
            downloaded = _hf_download_file(hf_repo, hf_file, str(output_path))
            if downloaded.resolve() != canonical.resolve():
                shutil.move(str(downloaded), str(canonical))
            _done(canonical)
            _maybe_download_mmproj(model_info, output_path, model_name)
            _run_calibration(model_name, model_info, output_dir, calib_input)
            return
        print(f"  No pre-built {quant} found — falling back to safetensors conversion.")
    elif mtp:
        print("  Skipping pre-built GGUF check — prebuilts strip MTP tensors.")

    # ------------------------------------------------------------------ #
    # 1. Check for existing fp16 GGUF before downloading anything
    # ------------------------------------------------------------------ #
    # Use a distinct filename when MTP is requested so plain and MTP fp16s
    # can coexist and we never mistake one for the other.
    fp16_stem = f"{model_name}-fp16-mtp" if mtp else f"{model_name}-fp16"
    fp16_gguf = output_path / f"{fp16_stem}.gguf"

    # Track the best local model path for calibration (safetensors dir > HF repo).
    _calib_model_path: str | None = None

    if fp16_gguf.exists():
        fp16_size = _fmt_bytes(fp16_gguf.stat().st_size)
        _section("Converting safetensors → fp16 GGUF")
        print(f"  ✓ fp16 GGUF already on disk: {fp16_gguf.name} ({fp16_size}) — skipping download & conversion.")
    else:
        # -------------------------------------------------------------- #
        # 1a. Download safetensors
        # -------------------------------------------------------------- #
        _section(f"Downloading safetensors: {source_repo}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("Error: huggingface_hub is not installed.")
            sys.exit(1)

        hf_token = os.environ.get("HF_TOKEN") or None

        # Download into the HF native cache (HF_HOME/hub) rather than a flat
        # local_dir.  This means triattention_calibrate.py's from_pretrained()
        # call will find the weights already cached and skip a second download.
        # With local_dir the two paths use incompatible layouts and can't share.
        # max_workers=1 serializes shard downloads to avoid WSL2 vmmem balloon.
        st_dir = Path(snapshot_download(
            repo_id=source_repo,
            token=hf_token,
            ignore_patterns=["*.md", "*.txt", "*.json.lock", "*.gguf"],
            max_workers=1,
        ))
        print(f"  Cached : {st_dir}")
        # from_pretrained(source_repo) will be a cache hit — no re-download.
        _calib_model_path = source_repo

        # -------------------------------------------------------------- #
        # 1b. Convert safetensors → fp16 GGUF
        # -------------------------------------------------------------- #
        _section("Converting safetensors → fp16 GGUF")
        print(f"  Source : {st_dir}")
        print(f"  Output : {fp16_gguf.name}")

        with _keepalive(30, "convert"):
            result = subprocess.run(
                [
                    "python3", str(convert_script),
                    str(st_dir),
                    "--outtype", "f16",
                    "--outfile", str(fp16_gguf),
                ],
                stdout=None,
                stderr=None,
            )
        if result.returncode != 0:
            print(f"\nError: convert_hf_to_gguf.py failed (exit {result.returncode})")
            sys.exit(1)

        fp16_size = _fmt_bytes(fp16_gguf.stat().st_size) if fp16_gguf.exists() else "unknown"
        print(f"\n  Done: {fp16_gguf.name} ({fp16_size})")

        # Weights are in HF cache — no intermediate dir to clean up here.
        # Calibration runs below using source_repo (cache hit via HF_HOME).

    # ------------------------------------------------------------------ #
    # 3. Quantize fp16 GGUF → target quant
    # ------------------------------------------------------------------ #
    _quantize(fp16_gguf, canonical, quant, nthreads=nthreads)

    # ------------------------------------------------------------------ #
    # 3b. Verify MTP tensors in output GGUF
    # ------------------------------------------------------------------ #
    if mtp:
        _section("Verifying MTP tensors in output GGUF")
        try:
            header = _gguf_read_header(canonical)
            tensor_names = header["tensor_names"]
            metadata = header["metadata"]

            # Check nextn_predict_layers metadata (the authoritative MTP indicator)
            nextn_key = next((k for k in metadata if "nextn_predict_layers" in k), None)
            nextn_val = metadata.get(nextn_key) if nextn_key else None
            block_key = next((k for k in metadata if "block_count" in k), None)
            block_val = metadata.get(block_key) if block_key else None

            print(f"  Total tensors in GGUF: {len(tensor_names)}")
            print(f"  Metadata: {nextn_key} = {nextn_val}")
            print(f"  Metadata: {block_key} = {block_val}")

            if nextn_val and nextn_val > 0:
                print(f"  ✓ GGUF declares {nextn_val} MTP prediction layer(s)")
            else:
                print("  ✗ WARNING: nextn_predict_layers not set or zero — MTP metadata missing.")

            # Look for nextn-specific tensors (blk.N.nextn.*)
            nextn_tensors = [n for n in tensor_names if ".nextn." in n]
            if nextn_tensors:
                print(f"  ✓ Found {len(nextn_tensors)} nextn tensor(s):")
                for t in nextn_tensors:
                    print(f"      {t}")
            else:
                # Also check for MTP block tensors (blk.N.attn_* at index >= base layers)
                base_layers = (block_val - nextn_val) if block_val and nextn_val else 0
                if base_layers > 0:
                    mtp_block = f"blk.{base_layers}"
                    mtp_found = [n for n in tensor_names if n.startswith(mtp_block + ".")]
                    if mtp_found:
                        print(f"  ✓ Found {len(mtp_found)} tensor(s) at {mtp_block}.*:")
                        for t in mtp_found:
                            print(f"      {t}")
                    else:
                        print(f"\n  ✗ WARNING: No MTP tensors found (no .nextn. or {mtp_block}.* tensors).")
                        print(f"    The server may crash with: missing tensor '{mtp_block}.attn_norm.weight'")
                        print(f"    File kept: {canonical.name}")
                else:
                    print("  ✗ WARNING: Cannot determine MTP block index — check manually.")
        except Exception as e:
            print(f"  Warning: MTP verification failed ({e}) — proceeding anyway.")

    # ------------------------------------------------------------------ #
    # 4. Clean up fp16 intermediate
    # ------------------------------------------------------------------ #
    if keep_intermediate:
        print(f"\n  Keeping fp16 GGUF : {fp16_gguf.name}")
    else:
        print(f"\n  Removing fp16 GGUF : {fp16_gguf.name}")
        fp16_gguf.unlink(missing_ok=True)

    _done(canonical)
    _maybe_download_mmproj(model_info, output_path, model_name)
    # _calib_model_path is the HF repo ID (source_repo) — from_pretrained will
    # find it in HF_HOME (/models/.hf-cache) without re-downloading.
    _run_calibration(model_name, model_info, output_dir, calib_input, model_path=_calib_model_path)

    if mtp:
        print(
            "\n  ── MTP GGUF ready ────────────────────────────────────────────\n"
            f"  To use this GGUF, add to your .env:\n"
            f"    LLAMA_MODEL=/models/{canonical.name}\n"
            f"    LLAMA_SPEC_TYPE=mtp\n"
            f"    LLAMA_SPEC_DRAFT_N_MAX=3\n"
            f"    LLAMA_PARALLEL=1\n"
            "  Then restart the server:  docker compose up -d\n"
            "  ─────────────────────────────────────────────────────────────"
        )


def convert_model(
    model_path: str,
    quant_method: str = "Q4_K_M",
    output_dir: str = DEFAULT_MODELS_DIR,
    nthreads: int | None = None,
) -> None:
    """Re-quantize an existing GGUF to a different quantization using llama-quantize."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem
    parts = model_name.rsplit("-", 1)
    if len(parts) > 1 and parts[1][0].isdigit():
        base_name = parts[0]
        output_file = f"{base_name}-{quant_method}.gguf"
    else:
        output_file = f"{model_name}-{quant_method}.gguf"

    output_file_path = output_path / output_file
    print(f"Converting {model_path} to {quant_method} quantization...")
    print(f"Output: {output_file_path}")
    _quantize(Path(model_path), output_file_path, quant_method, nthreads=nthreads)
    print(f"Model converted to: {output_file_path}")


def turboquant_model(
    model_path: str,
    quant_method: str = "TQ2_0",
    output_dir: str = DEFAULT_MODELS_DIR,
    nthreads: int | None = None,
) -> None:
    """Convert a GGUF model to TurboQuant format using llama-quantize."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem
    output_file = f"{model_name}-{quant_method}.gguf"
    output_file_path = output_path / output_file

    print(f"Converting {model_path} to TurboQuant {quant_method}...")
    print(f"Output: {output_file_path}")
    print("Note: TurboQuant models are ~2-bit or ~1-bit per weight for extreme compression")
    print("Important: For best quality, convert from FP16 if available, or Q8_0")
    _quantize(Path(model_path), output_file_path, quant_method, nthreads=nthreads)
    print(f"Model converted to TurboQuant: {output_file_path}")


def list_models() -> None:
    """List available models."""
    print("Available models:")
    print("=" * 70)
    print()
    tiers = [
        ("SMALL MODELS (<4GB at Q4_K_M) - Best for 24GB GPU with large context:", lambda s: s < 4),
        ("MEDIUM MODELS (4-12GB at Q4_K_M) - Fits with moderate context:", lambda s: 4 <= s <= 12),
        ("LARGE MODELS (12-18GB at Q4_K_M) - Fits with small context:", lambda s: s > 12),
    ]
    for title, size_filter in tiers:
        print(title)
        print("-" * 70)
        for key, info in MODELS.items():
            if not size_filter(info.get("size_gb", 0)):
                continue
            tags = []
            if info.get("turboquant"):
                tags.append("TurboQuant")
            if "mmproj" in info:
                tags.append("Multimodal")
            if info.get("mtp_capable") or info.get("mtp_graft_from"):
                tags.append("MTP ready")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            print(f"  {key:25s} - {info['description']}{tag_str}")
        print()

    print("Supported quantizations (pass to --quant):")
    print("  Standard   :", ", ".join(QUANT_PRIORITY[:8]), "...")
    print("  TurboQuant :", ", ".join(TQ_QUANT_OPTIONS))
    print("  Unsloth UD :", ", ".join(UD_QUANT_OPTIONS))
    print()
    print("  UD (Unsloth Dynamic) quants use mixed precision to preserve important")
    print("  layers at higher bit depth. Download-only — cannot be produced locally.")
    print("  Available from unsloth/*-GGUF repos on HuggingFace.")
    print()
    print("Multimodal models automatically download mmproj.gguf for vision support.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage LLM models for llama.cpp server"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List models
    subparsers.add_parser("list", help="List available models")

    _threads_help = (
        "Number of CPU threads for llama-quantize (default: CONVERT_THREADS env var, "
        "or cpu_count//2). Limiting threads prevents CLOCK_WATCHDOG_TIMEOUT BSODs "
        "on Windows when Docker Desktop saturates all host CPU cores."
    )

    # Download model
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument(
        "model", help="Model name (e.g. qwen3.5-27b)"
    )
    download_parser.add_argument(
        "-q",
        "--quant",
        default="Q3_K_L",
        help=(
            "Quantization type (default: Q3_K_L). "
            "If the requested quant is not available on HuggingFace, the script "
            "will download the best available source (FP16/BF16) and quantize locally, "
            "then clean up the source file. "
            f"Standard options: {', '.join(QUANT_PRIORITY[:8])} ... "
            f"TurboQuant options: {', '.join(TQ_QUANT_OPTIONS)}. "
            f"Unsloth Dynamic (UD) options: {', '.join(UD_QUANT_OPTIONS)} "
            "(download-only, mixed-precision, better quality per bit)."
        ),
    )
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_MODELS_DIR,
        help=DEFAULT_OUTPUT_DIR_HELP,
    )
    download_parser.add_argument(
        "-t", "--threads", type=int, default=None, metavar="N", help=_threads_help
    )
    download_parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        default=False,
        help=(
            "Do not delete the downloaded source GGUF after local quantization. "
            "Useful to avoid re-downloading if you need to requantize."
        ),
    )
    download_parser.add_argument(
        "--calib-input",
        default="",
        metavar="FILE",
        help=(
            "Plain-text corpus file for TriAttention calibration "
            "(e.g. wikitext-2-raw/wiki.test.raw). "
            "Also read from TRIATTENTION_INPUT env var. "
            "Skipped when not provided. Requires fp16_repo to be set for the model."
        ),
    )

    # Convert model (re-quantize an existing GGUF)
    convert_parser = subparsers.add_parser("convert", help="Re-quantize an existing GGUF")
    convert_parser.add_argument(
        "model_path", help="Path to the GGUF model to convert"
    )
    convert_parser.add_argument(
        "-q",
        "--quant",
        default="Q3_K_L",
        choices=QUANT_PRIORITY[:10],
        help="Quantization method (default: Q3_K_L)",
    )
    convert_parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_MODELS_DIR,
        help=DEFAULT_OUTPUT_DIR_HELP,
    )
    convert_parser.add_argument(
        "-t", "--threads", type=int, default=None, metavar="N", help=_threads_help
    )

    # TurboQuant model
    tq_parser = subparsers.add_parser("turboquant", help="Convert a GGUF to TurboQuant format")
    tq_parser.add_argument(
        "model_path", help="Path to the GGUF model to convert"
    )
    tq_parser.add_argument(
        "-q",
        "--quant",
        default="TQ2_0",
        choices=TQ_QUANT_OPTIONS,
        help="TurboQuant quantization method (default: TQ2_0)",
    )
    tq_parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_MODELS_DIR,
        help=DEFAULT_OUTPUT_DIR_HELP,
    )
    tq_parser.add_argument(
        "-t", "--threads", type=int, default=None, metavar="N", help=_threads_help
    )

    # Convert safetensors → fp16 GGUF → quantized GGUF (convert image only)
    cst_parser = subparsers.add_parser(
        "convert-st",
        help=(
            "Download safetensors weights, convert to fp16 GGUF via "
            "convert_hf_to_gguf.py, then quantize. "
            "Requires the convert Docker image (docker compose build llama-convert)."
        ),
    )
    cst_parser.add_argument(
        "model", help="Model name (e.g. qwen3.5-27b)"
    )
    cst_parser.add_argument(
        "-q",
        "--quant",
        default="TQ2_0",
        help=(
            "Target quantization (default: TQ2_0). "
            f"TurboQuant: {', '.join(TQ_QUANT_OPTIONS)}. "
            f"Standard: {', '.join(QUANT_PRIORITY[:8])} ..."
        ),
    )
    cst_parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_MODELS_DIR,
        help=DEFAULT_OUTPUT_DIR_HELP,
    )
    cst_parser.add_argument(
        "-t", "--threads", type=int, default=None, metavar="N", help=_threads_help
    )
    cst_parser.add_argument(
        "--mtp",
        action="store_true",
        default=False,
        help=(
            "Include MTP (Multi-Token Prediction) head tensors in the output GGUF. "
            "For fine-tunes, MTP weights are auto-grafted from the base model "
            "(resolved from mtp_graft_from or config._name_or_path). "
            "Base models with mtp_capable=True use their own weights. "
            "Skips the prebuilt-GGUF shortcut. Output: {model}-{quant}-mtp.gguf. "
            "Enable with LLAMA_SPEC_TYPE=mtp in .env."
        ),
    )
    cst_parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        default=False,
        help=(
            "Do not delete the fp16 GGUF or the downloaded safetensors cache after "
            "quantization. Useful to avoid re-downloading/re-converting if you need "
            "to requantize to a different format."
        ),
    )
    cst_parser.add_argument(
        "--calib-input",
        default="",
        metavar="FILE",
        help=(
            "Plain-text corpus file for TriAttention calibration "
            "(e.g. wikitext-2-raw/wiki.test.raw). "
            "Also read from TRIATTENTION_INPUT env var. "
            "When provided, calibration runs against the local safetensors dir "
            "before cleanup (avoids re-downloading weights)."
        ),
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        calib_input = getattr(args, "calib_input", "") or os.environ.get("TRIATTENTION_INPUT", "")
        download_model(args.model, args.quant, args.output_dir, nthreads=args.threads, keep_intermediate=args.keep_intermediate, calib_input=calib_input)
    elif args.command == "convert":
        convert_model(args.model_path, args.quant, args.output_dir, nthreads=args.threads)
    elif args.command == "turboquant":
        turboquant_model(args.model_path, args.quant, args.output_dir, nthreads=args.threads)
    elif args.command == "convert-st":
        calib_input = getattr(args, "calib_input", "") or os.environ.get("TRIATTENTION_INPUT", "")
        convert_safetensors(args.model, args.quant, args.output_dir, nthreads=args.threads, mtp=args.mtp, keep_intermediate=args.keep_intermediate, calib_input=calib_input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
