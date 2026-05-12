#!/usr/bin/env python3
"""
Model management script for downloading and converting LLM models.
Supports downloading models from HuggingFace and converting to GGUF format.
"""

import contextlib
import os
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
        "description": "Qwen 3.6 27B (~14GB)",
        "size_gb": 14,
    },
    "qwen3.5-27b": {
        "hf_repo": "unsloth/Qwen3.5-27B-GGUF",
        "description": "Qwen 3.5 27B (~14GB)",
        "size_gb": 14,
    },
    "qwen3.6-35b-a3b": {
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "description": "Qwen 3.6 35B-A3B (~17GB)",
        "size_gb": 17,
    },
    "qwopus3.6-35b": {
        "hf_repo": "Jackrong/Qwopus3.6-35B-A3B-v1-GGUF",
        "fp16_repo": "Jackrong/Qwopus3.6-35B-A3B-v1",
        "description": "Qwopus 3.6 35B-A3B-v1 (~17GB)",
        "size_gb": 17,
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

    Falls back to aria2c (16 parallel connections) or curl when
    huggingface_hub is not available.

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
        return _curl_download_file(repo_id, filename, dest)

    # enable_progress_bars: present since 0.14.0; silently skip if absent.
    try:
        _hfh.enable_progress_bars()
    except AttributeError:
        pass

    # hf_transfer: Rust parallel downloader (pip install hf-transfer).
    # Auto-enable when the package is present and the env var isn't already set.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
        try:
            import hf_transfer  # noqa: F401 — presence check only
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
    repo_id: str, filename: str, dest: Path, limit_rate: str = ""
) -> Path:
    """Download a file from HuggingFace.

    When *limit_rate* is empty and aria2c is available, downloads via
    aria2c with 16 parallel connections — this saturates typical network
    links far better than a single TCP stream.

    When *limit_rate* is set (WSL2 BSOD throttle path), or when aria2c is
    absent, falls back to curl.  curl's --limit-rate caps write throughput
    which prevents the WSL2 vmmem balloon that causes CLOCK_WATCHDOG_TIMEOUT
    BSODs on Windows during large file downloads.

    Args:
        limit_rate: Optional curl --limit-rate value (e.g. "300M").  When
            empty, aria2c is tried first.
    """
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"  Download URL  : {download_url}")

    hf_token = os.environ.get("HF_TOKEN", "")

    # ── aria2c path (multi-connection, no rate limit) ─────────────────────
    if not limit_rate and shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=50M",
            "--continue=true",
            "--dir", str(dest.parent),
            "--out", dest.name,
        ]
        if hf_token:
            cmd += ["--header", f"Authorization: Bearer {hf_token}"]
        cmd.append(download_url)
        print("  Using aria2c (16 parallel connections) ...")
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
    cmd = ["curl", "-L", "-f", "-o", str(dest)]
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


def download_model(model_name: str, quant: str, output_dir: str, nthreads: int | None = None) -> None:
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
        _maybe_download_mmproj(model_info, output_path)
        return

    hf_repo = model_info["hf_repo"]
    is_turboquant = quant in TQ_QUANT_OPTIONS

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
        _maybe_download_mmproj(model_info, output_path)
        return

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

    print(f"\n  Removing source GGUF : {actual_source.name}")
    actual_source.unlink(missing_ok=True)

    _done(canonical)
    _maybe_download_mmproj(model_info, output_path)


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


def _maybe_download_mmproj(model_info: dict, output_path: Path) -> None:
    """Download the multimodal projector file if this model requires one.

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

    mmproj_path = output_path / Path(mmproj_filename).name
    if mmproj_path.exists():
        print(f"Multimodal projector already exists: {mmproj_path}")
        return

    print(f"\nDownloading multimodal projector: {mmproj_filename} ...")
    _hf_download_file(repo_id, mmproj_filename, str(output_path))
    print(f"Multimodal projector ready: {mmproj_path}")


def convert_safetensors(model_name: str, quant: str, output_dir: str, nthreads: int | None = None) -> None:
    """Convert a safetensors model to a quantized GGUF via fp16 GGUF intermediate.

    Pipeline
    --------
    1. Download the safetensors repo (fp16_repo, or hf_repo if fp16_repo absent).
    2. Run convert_hf_to_gguf.py → fp16 GGUF.
    3. Run llama-quantize → target quant GGUF.
    4. Clean up the safetensors download and the fp16 intermediate.

    This command is only available in the ``convert`` Docker image, which ships
    convert_hf_to_gguf.py and its Python dependencies (torch, transformers, …).

    Args:
        model_name: Key from the MODELS dict (e.g. "qwopus3.6-35b")
        quant: Target quantization (e.g. "TQ2_0", "Q4_K_M")
        output_dir: Directory where the final GGUF should be placed
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

    canonical = output_path / f"{model_name}-{quant}.gguf"

    _section(f"Model: {model_name}  |  Quant: {quant}  |  Source: {source_repo}")
    print(f"  Output : {canonical}")

    if canonical.exists():
        size_str = _fmt_bytes(canonical.stat().st_size)
        print(f"\n  ✓ Already on disk: {canonical.name} ({size_str}) — skipping.")
        _maybe_download_mmproj(model_info, output_path)
        return

    # ------------------------------------------------------------------ #
    # 0. Short-circuit: check if a pre-built GGUF already exists in hf_repo
    #    (same logic as the `download` command). Skip for TurboQuant targets
    #    since those are never pre-built on HuggingFace.
    # ------------------------------------------------------------------ #
    is_turboquant = quant in TQ_QUANT_OPTIONS
    hf_repo = model_info["hf_repo"]
    if not is_turboquant:
        print(f"  Checking HuggingFace for pre-built {quant} in {hf_repo} ...")
        hf_file = find_quant_in_repo(hf_repo, quant)
        if hf_file:
            print(f"  Found pre-built {quant}: {Path(hf_file).name}")
            print("  Skipping safetensors download — downloading GGUF directly.")
            downloaded = _hf_download_file(hf_repo, hf_file, str(output_path))
            if downloaded.resolve() != canonical.resolve():
                shutil.move(str(downloaded), str(canonical))
            _done(canonical)
            _maybe_download_mmproj(model_info, output_path)
            return
        print(f"  No pre-built {quant} found — falling back to safetensors conversion.")

    # ------------------------------------------------------------------ #
    # 1. Download safetensors
    # ------------------------------------------------------------------ #
    _section(f"Downloading safetensors: {source_repo}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or None
    st_dir = output_path / f".{model_name}-safetensors"

    # max_workers=1 serializes shard downloads to avoid the simultaneous disk-
    # write pressure that causes WSL2's vmmem to balloon on Windows.
    snapshot_download(
        repo_id=source_repo,
        local_dir=str(st_dir),
        token=hf_token,
        ignore_patterns=["*.md", "*.txt", "*.json.lock", "*.gguf"],
        max_workers=1,
    )

    # ------------------------------------------------------------------ #
    # 2. Convert safetensors → fp16 GGUF
    # ------------------------------------------------------------------ #
    fp16_gguf = output_path / f"{model_name}-fp16.gguf"
    _section("Converting safetensors → fp16 GGUF")

    if fp16_gguf.exists():
        fp16_size = _fmt_bytes(fp16_gguf.stat().st_size)
        print(f"  ✓ fp16 GGUF already on disk: {fp16_gguf.name} ({fp16_size}) — skipping conversion.")
    else:
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

    # ------------------------------------------------------------------ #
    # 3. Quantize fp16 GGUF → target quant
    # ------------------------------------------------------------------ #
    _quantize(fp16_gguf, canonical, quant, nthreads=nthreads)

    # ------------------------------------------------------------------ #
    # 4. Clean up intermediate files (only on success — preserves resumability)
    # ------------------------------------------------------------------ #
    print(f"\n  Removing fp16 GGUF : {fp16_gguf.name}")
    fp16_gguf.unlink(missing_ok=True)
    print(f"  Removing safetensors cache : {st_dir.name}")
    shutil.rmtree(st_dir, ignore_errors=True)

    _done(canonical)
    _maybe_download_mmproj(model_info, output_path)


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
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            print(f"  {key:25s} - {info['description']}{tag_str}")
        print()

    print("Supported quantizations (pass to --quant):")
    print("  Standard:", ", ".join(QUANT_PRIORITY[:8]), "...")
    print("  TurboQuant:", ", ".join(TQ_QUANT_OPTIONS))
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
        default="Q4_K_M",
        help=(
            "Quantization type (default: Q4_K_M). "
            "If the requested quant is not available on HuggingFace, the script "
            "will download the best available source (FP16/BF16) and quantize locally, "
            "then clean up the source file. "
            f"Standard options: {', '.join(QUANT_PRIORITY[:8])} ... "
            f"TurboQuant options: {', '.join(TQ_QUANT_OPTIONS)}"
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

    # Convert model (re-quantize an existing GGUF)
    convert_parser = subparsers.add_parser("convert", help="Re-quantize an existing GGUF")
    convert_parser.add_argument(
        "model_path", help="Path to the GGUF model to convert"
    )
    convert_parser.add_argument(
        "-q",
        "--quant",
        default="Q4_K_M",
        choices=QUANT_PRIORITY[:10],
        help="Quantization method (default: Q4_K_M)",
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
        "model", help="Model name (e.g. qwopus3.6-35b)"
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

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        download_model(args.model, args.quant, args.output_dir, nthreads=args.threads)
    elif args.command == "convert":
        convert_model(args.model_path, args.quant, args.output_dir, nthreads=args.threads)
    elif args.command == "turboquant":
        turboquant_model(args.model_path, args.quant, args.output_dir, nthreads=args.threads)
    elif args.command == "convert-st":
        convert_safetensors(args.model, args.quant, args.output_dir, nthreads=args.threads)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
