#!/usr/bin/env python3
"""
Model management script for downloading and converting LLM models.
Supports downloading models from HuggingFace and converting to GGUF format.
"""

import os
import sys
import argparse
import json
import shutil
import subprocess
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
        "notes": "Mixture-of-Experts model. Use Q8_0 quantization for best quality.",
        "turboquant": True,
        "turboquant_source": "Q8_0",
    },
}

DEFAULT_MODELS_DIR = "./models"
DEFAULT_OUTPUT_DIR_HELP = "Output directory (default: ./models)"

# TurboQuant quantization options (extreme compression, must be quantized locally)
TQ_QUANT_OPTIONS = [
    "TQ2_0",  # 2-bit per weight - better quality while still highly compressed
    "TQ1_0",  # 1-bit per weight - extreme compression
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

# Quantizations that are acceptable as re-quantization sources.
# All standard quants including Q8_0 are valid inputs to llama-quantize.
# (The historical concern about Q8_0 only applied to certain older llama.cpp builds
# and does not affect TurboQuant targets.)
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


def find_best_source_in_repo(repo_id: str) -> str | None:
    """Find the highest-quality GGUF in a repo suitable as a quantization source.

    Walks QUANT_PRIORITY in order, skipping NON_REQUANTIZABLE types, and
    returns the first match.  Falls back to NON_REQUANTIZABLE types only if
    nothing better is available.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Filename of the best source GGUF, or None if no GGUF exists.
    """
    # First pass: prefer quants that can be re-quantized
    for quant in QUANT_PRIORITY:
        if quant.upper() in NON_REQUANTIZABLE:
            continue
        match = find_quant_in_repo(repo_id, quant)
        if match:
            return match

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

    try:
        from huggingface_hub import hf_hub_download, enable_progress_bars
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # Force progress bars on even without a TTY (Docker -d mode, log output, etc.)
    enable_progress_bars()

    _section(f"Downloading: {Path(filename).name}")
    print(f"  Repository  : {repo_id}")
    print(f"  Destination : {dest}")

    try:
        downloaded = Path(
            hf_hub_download(
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


def _quantize(source: Path, dest: Path, quant: str) -> None:
    """Run llama-quantize to produce dest from source with the given quant type.

    Streams llama-quantize output directly to the console and exits on failure.
    """
    source_size = _fmt_bytes(source.stat().st_size) if source.exists() else "unknown"
    _section(f"Quantizing to {quant}")
    print(f"  Source : {source.name} ({source_size})")
    print(f"  Output : {dest.name}")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["llama-quantize", str(source), str(dest), quant],
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


def download_model(model_name: str, quant: str, output_dir: str) -> None:
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
    else:
        print(f"  No pre-built {quant} found on HuggingFace.")
    print("  Will download the best available source and quantize locally.")

    source_file = find_best_source_in_repo(hf_repo)

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

    _quantize(actual_source, canonical, quant)

    print(f"\n  Removing source file: {actual_source.name}")
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


def convert_model(
    model_path: str,
    quant_method: str = "Q4_K_M",
    output_dir: str = DEFAULT_MODELS_DIR,
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
    _quantize(Path(model_path), output_file_path, quant_method)
    print(f"Model converted to: {output_file_path}")


def turboquant_model(
    model_path: str,
    quant_method: str = "TQ2_0",
    output_dir: str = DEFAULT_MODELS_DIR,
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
    _quantize(Path(model_path), output_file_path, quant_method)
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

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        download_model(args.model, args.quant, args.output_dir)
    elif args.command == "convert":
        convert_model(args.model_path, args.quant, args.output_dir)
    elif args.command == "turboquant":
        turboquant_model(args.model_path, args.quant, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
