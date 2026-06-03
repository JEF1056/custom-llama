#!/usr/bin/env python3
"""
Model management script for downloading AutoRound INT4 safetensors models.
"""

import os
import sys
import argparse
from pathlib import Path

# AutoRound INT4 safetensors — loaded by vLLM with --quantization auto-round.
# Preferred over AWQ on RTX 3090: 19–21 GB vs AWQ's 21.56 GB which forces
# --enforce-eager (kills CUDA graphs, 78% decode overhead).
# MTP head kept in BF16 → ~90% draft acceptance with NEXTN speculative decoding.
# Vision tower at original precision.
MODELS = {
    "qwen3.6-27b-autoround": {
        "hf_repo": "Lorbus/Qwen3.6-27B-int4-AutoRound",
        "description": "Qwen 3.6 27B AutoRound INT4 safetensors, MTP+Vision (~19GB)",
        "size_gb": 19,
    },
    "qwen3.6-35b-a3b-autoround": {
        "hf_repo": "shieldstar/Qwen3.6-35B-A3B-int4-AutoRound-EC",
        "description": "Qwen 3.6 35B-A3B AutoRound INT4 safetensors, MTP+Vision (~21GB)",
        "size_gb": 21,
    },
}

DEFAULT_MODELS_DIR = "./models"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n  {bar}")
    print(f"  {title}")
    print(f"  {bar}")


def download_model(model_name: str, output_dir: str) -> None:
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print("Run 'manage_models.py list' to see available models.")
        sys.exit(1)

    model_info = MODELS[model_name]
    hf_repo = model_info["hf_repo"]
    model_dir = Path(output_dir) / model_name

    _section(f"Model: {model_name}  (AutoRound INT4 safetensors)")
    print(f"  Repository  : {hf_repo}")
    print(f"  Destination : {model_dir}")
    print(f"  Size        : ~{model_info['size_gb']}GB")

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"\n  ✓ Already on disk: {model_dir} — skipping download.")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
            sys.exit(1)

        hf_token = os.environ.get("HF_TOKEN") or None
        model_dir.mkdir(parents=True, exist_ok=True)
        print("\n  Downloading safetensors (this may take a while) ...")
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(model_dir),
            token=hf_token,
            ignore_patterns=["*.md", "*.txt"],
            max_workers=1,
        )
        total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        print(f"\n  ✓ Downloaded: {model_dir} ({_fmt_bytes(total)})")

    print()
    print(f"  Add to your .env:")
    print(f"    LLM_MODEL_PATH=/models/{model_name}")
    print(f"    LLM_QUANTIZATION=auto-round")
    print()
    print(f"  Then start the server:")
    print(f"    docker compose up vllm-server")


def list_models() -> None:
    print("Available models:")
    print("=" * 70)
    print()
    for key, info in MODELS.items():
        print(f"  {key:35s} {info['description']}")
    print()
    print("Download: manage_models.py download <name>")
    print("vLLM:     --quantization auto-round  (set LLM_QUANTIZATION=auto-round)")


def main():
    parser = argparse.ArgumentParser(
        description="Download AutoRound INT4 safetensors models for vLLM"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("list", help="List available models")

    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument(
        "model", help="Model name (e.g. qwen3.6-27b-autoround)"
    )
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Output directory (default: {DEFAULT_MODELS_DIR})",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        download_model(args.model, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
