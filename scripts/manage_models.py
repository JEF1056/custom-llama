#!/usr/bin/env python3
"""
Model management script for downloading and converting LLM models.
Supports downloading models from HuggingFace and converting to GGUF format.
"""

import os
import sys
import argparse
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
        "default_file": "Qwen3.5-0.8B-Q4_K_M.gguf",
        "description": "Qwen 3.5 0.8B (4-bit quantized, ~0.5GB)",
        "size_gb": 0.5,
    },
    "llama3.2-1b": {
        "hf_repo": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
        "default_file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.2 1B Instruct (4-bit quantized, ~0.7GB)",
        "size_gb": 0.7,
    },
    "llama3.2-3b": {
        "hf_repo": "lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
        "default_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.2 3B Instruct (4-bit quantized, ~2GB)",
        "size_gb": 2,
    },
    "gemma-4-e2b": {
        "hf_repo": "lmstudio-community/gemma-4-E2B-it-GGUF",
        "default_file": "gemma-4-E2B-it-Q4_K_M.gguf",
        "description": "Gemma 4 E2B (4-bit quantized, ~1.5GB)",
        "size_gb": 1.5,
        "mmproj": "mmproj-gemma-4-E2B-it-BF16.gguf",  # Multimodal projector for vision support
    },
    "qwen3.5-4b": {
        "hf_repo": "unsloth/Qwen3.5-4B-GGUF",
        "default_file": "Qwen3.5-4B-Q4_K_M.gguf",
        "description": "Qwen 3.5 4B (4-bit quantized, ~2.5GB)",
        "size_gb": 2.5,
    },
    # ============================================
    # MEDIUM MODELS (4-12GB at Q4_K_M) - Fits with moderate context
    # ============================================
    "qwen2.5-coder-7b": {
        "hf_repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "default_file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 Coder 7B Instruct (4-bit quantized, ~4.5GB)",
        "size_gb": 4.5,
    },
    "gemma-4-e4b": {
        "hf_repo": "lmstudio-community/gemma-4-E4B-it-GGUF",
        "default_file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "description": "Gemma 4 E4B (4-bit quantized, ~3GB)",
        "size_gb": 3,
        "mmproj": "mmproj-gemma-4-E4B-it-BF16.gguf",  # Multimodal projector for vision support
    },
    "qwen3.5-9b": {
        "hf_repo": "unsloth/Qwen3.5-9B-GGUF",
        "default_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "description": "Qwen 3.5 9B (4-bit quantized, ~5.5GB)",
        "size_gb": 5.5,
    },
    "gpt-oss-20b": {
        "hf_repo": "unsloth/gpt-oss-20b-GGUF",
        "default_file": "gpt-oss-20b-Q4_K_M.gguf",
        "description": "GPT-OSS 20B (4-bit quantized, ~11GB)",
        "size_gb": 11,
    },
    # ============================================
    # LARGE MODELS (12-18GB at Q4_K_M) - Fits with small context
    # ============================================
    "gemma-4-26b-a4b": {
        "hf_repo": "lmstudio-community/gemma-4-26B-A4B-it-GGUF",
        "default_file": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "description": "Gemma 4 26B-A4B (4-bit quantized, MoE, ~13GB)",
        "size_gb": 13,
        "mmproj": "mmproj-gemma-4-26B-A4B-it-BF16.gguf",  # Multimodal projector for vision support
    },
    "gemma-4-31b": {
        "hf_repo": "lmstudio-community/gemma-4-31B-it-GGUF",
        "default_file": "gemma-4-31B-it-Q4_K_M.gguf",
        "description": "Gemma 4 31B (4-bit quantized, ~16GB)",
        "size_gb": 16,
        "mmproj": "mmproj-gemma-4-31B-it-BF16.gguf",  # Multimodal projector for vision support
    },
    "qwen3.6-27b": {
        "hf_repo": "unsloth/Qwen3.6-27B-GGUF",
        "default_file": "Qwen3.6-27B-Q4_K_M.gguf",
        "description": "Qwen 3.6 27B (4-bit quantized, ~14GB)",
        "size_gb": 14,
    },
    "qwen3.5-27b": {
        "hf_repo": "unsloth/Qwen3.5-27B-GGUF",
        "default_file": "Qwen3.5-27B-Q4_K_M.gguf",
        "description": "Qwen 3.5 27B (4-bit quantized, ~14GB)",
        "size_gb": 14,
    },
    "qwen3.6-35b-a3b": {
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "default_file": "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "description": "Qwen 3.6 35B-A3B (4-bit quantized, MoE, ~17GB)",
        "size_gb": 17,
    },
    "minimax-m2.7": {
        "hf_repo": "unsloth/MiniMax-M2.7-GGUF",
        "default_file": "MiniMax-M2.7-Q8_0-00001-of-00006.gguf",
        "description": "MiniMax M2.7 (8-bit quantized, MoE, ~18GB)",
        "size_gb": 18,
        "notes": "Mixture-of-Experts model. Use Q8_0 quantization for best quality. BF16 and MXFP4_MOE variants also available.",
        "turboquant": True,
        "turboquant_source": "Q8_0",
    },
}

# Quantization options for download (Q4_K_M is recommended for best quality/size tradeoff)
QUANT_OPTIONS = [
    "Q4_K_M",  # Recommended - best quality for size
    "Q5_K_M",  # Slightly better quality
    "Q6_K",    # High quality
    "Q8_0",    # 8-bit - best source for TurboQuant conversion
    "Q4_0",    # Basic 4-bit
    "Q3_K_M",  # Smaller size
    "IQ4_XS",  # Even smaller
]

# TurboQuant quantization options (extreme compression)
TQ_QUANT_OPTIONS = [
    "TQ2_0",   # 2-bit per weight - better quality while still highly compressed
    "TQ1_0",   # 1-bit per weight - extreme compression
]


def download_model(model_name: str, output_dir: str, quant: str = "Q4_K_M") -> None:
    """Download a model from HuggingFace."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_info = MODELS[model_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_name = model_info["default_file"]
    if quant != "Q4_K_M":
        # Replace the quantization suffix
        parts = file_name.rsplit(".", 1)
        file_name = f"{parts[0]}-{quant}.{parts[1]}"

    file_path = output_path / file_name

    if file_path.exists():
        print(f"Model already exists: {file_path}")
    else:
        print(f"Downloading {model_info['description']}...")
        print(f"Source: {model_info['hf_repo']}/{file_name}")
        print(f"Destination: {file_path}")

        # Use huggingface-cli to download
        import subprocess

        try:
            result = subprocess.run(
                [
                    "huggingface-cli",
                    "download",
                    model_info["hf_repo"],
                    file_name,
                    "--local-dir", str(output_path),
                    "--local-dir-use-symlinks", "False",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Error downloading model:")
                print(result.stderr)
                sys.exit(1)
            print(f"Model downloaded to: {file_path}")
        except FileNotFoundError:
            print("huggingface-cli not found. Install with: pip install huggingface_hub")
            sys.exit(1)

    # Download mmproj.gguf if this is a multimodal model
    if "mmproj" in model_info:
        mmproj_file = model_info["mmproj"]
        mmproj_path = output_path / mmproj_file
        if not mmproj_path.exists():
            print(f"\nDownloading multimodal projector: {mmproj_file}...")
            print(f"Source: {model_info['hf_repo']}/{mmproj_file}")
            print(f"Destination: {mmproj_path}")
            try:
                result = subprocess.run(
                    [
                        "huggingface-cli",
                        "download",
                        model_info["hf_repo"],
                        mmproj_file,
                        "--local-dir", str(output_path),
                        "--local-dir-use-symlinks", "False",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"Error downloading mmproj:")
                    print(result.stderr)
                else:
                    print(f"Multimodal projector downloaded to: {mmproj_path}")
            except FileNotFoundError:
                print("huggingface-cli not found. Please download mmproj.gguf manually.")
        else:
            print(f"Multimodal projector already exists: {mmproj_path}")


def convert_model(
    model_path: str,
    quant_method: str = "Q4_K_M",
    output_dir: str = "./models",
) -> None:
    """Convert a model to GGUF format using llama-quantize."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the model name from the file
    model_name = Path(model_path).stem

    # Generate output filename
    parts = model_name.rsplit("-", 1)
    if len(parts) > 1 and parts[1][0].isdigit():
        base_name = parts[0]
        quant_suffix = parts[1]
        # Replace quantization suffix
        output_file = f"{base_name}-{quant_method}.gguf"
    else:
        output_file = f"{model_name}-{quant_method}.gguf"

    output_file_path = output_path / output_file

    print(f"Converting {model_path} to {quant_method} quantization...")
    print(f"Output: {output_file_path}")

    # Use llama-quantize to convert
    import subprocess

    try:
        result = subprocess.run(
            [
                "llama-quantize",
                model_path,
                str(output_file_path),
                quant_method,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error converting model:")
            print(result.stderr)
            sys.exit(1)
        print(f"Model converted to: {output_file_path}")
    except FileNotFoundError:
        print("llama-quantize not found. Ensure llama.cpp is installed.")
        sys.exit(1)


def turboquant_model(
    model_path: str,
    quant_method: str = "TQ2_0",
    output_dir: str = "./models",
) -> None:
    """Convert a GGUF model to TurboQuant format using llama-quantize."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the model name from the file
    model_name = Path(model_path).stem

    # Generate output filename with TurboQuant suffix
    output_file = f"{model_name}-{quant_method}.gguf"
    output_file_path = output_path / output_file

    print(f"Converting {model_path} to TurboQuant {quant_method}...")
    print(f"Output: {output_file_path}")
    print(f"Note: TurboQuant models are ~2-bit or ~1-bit per weight for extreme compression")
    print(f"Important: For best quality, convert from FP16 if available, or Q8_0")

    # Use llama-quantize to convert to TurboQuant
    import subprocess

    try:
        result = subprocess.run(
            [
                "llama-quantize",
                model_path,
                str(output_file_path),
                quant_method,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error converting model to TurboQuant:")
            print(result.stderr)
            sys.exit(1)
        print(f"Model converted to TurboQuant: {output_file_path}")
    except FileNotFoundError:
        print("llama-quantize not found. Ensure llama.cpp is installed.")
        sys.exit(1)


def list_models() -> None:
    """List available models."""
    print("Available models:")
    print("=" * 70)
    print()
    print("SMALL MODELS (<4GB at Q4_K_M) - Best for 24GB GPU with large context:")
    print("-" * 70)
    for key, info in MODELS.items():
        size_gb = info.get("size_gb", 0)
        if size_gb < 4:
            if info.get("turboquant"):
                print(f"  {key:25s} - {info['description']} [TurboQuant]")
            elif "mmproj" in info:
                print(f"  {key:25s} - {info['description']} [Multimodal]")
            else:
                print(f"  {key:25s} - {info['description']}")
    print()
    print("MEDIUM MODELS (4-12GB at Q4_K_M) - Fits with moderate context:")
    print("-" * 70)
    for key, info in MODELS.items():
        size_gb = info.get("size_gb", 0)
        if 4 <= size_gb <= 12:
            if info.get("turboquant"):
                print(f"  {key:25s} - {info['description']} [TurboQuant]")
            elif "mmproj" in info:
                print(f"  {key:25s} - {info['description']} [Multimodal]")
            else:
                print(f"  {key:25s} - {info['description']}")
    print()
    print("LARGE MODELS (12-18GB at Q4_K_M) - Fits with small context:")
    print("-" * 70)
    for key, info in MODELS.items():
        size_gb = info.get("size_gb", 0)
        if size_gb > 12:
            if info.get("turboquant"):
                print(f"  {key:25s} - {info['description']} [TurboQuant]")
            elif "mmproj" in info:
                print(f"  {key:25s} - {info['description']} [Multimodal]")
            else:
                print(f"  {key:25s} - {info['description']}")
    print()
    print("Available quantization options:")
    for q in QUANT_OPTIONS:
        print(f"  {q}")
    print()
    print("TurboQuant quantization options:")
    for q in TQ_QUANT_OPTIONS:
        print(f"  {q}")
    print()
    print("  TQ1_0: ~1-bit per weight - extreme compression")
    print("  TQ2_0: ~2-bit per weight - better quality while still highly compressed")
    print()
    print("Multimodal models require mmproj.gguf file for vision/image support.")
    print("See README.md for instructions on enabling image input.")


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
        "model", help="Model name (e.g., llama3.2-8b)"
    )
    download_parser.add_argument(
        "-q",
        "--quant",
        default="Q4_K_M",
        choices=QUANT_OPTIONS,
        help="Quantization method (default: Q4_K_M)",
    )
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default="./models",
        help="Output directory (default: ./models)",
    )

    # Convert model
    convert_parser = subparsers.add_parser("convert", help="Convert a model to GGUF")
    convert_parser.add_argument(
        "model_path", help="Path to the model to convert"
    )
    convert_parser.add_argument(
        "-q",
        "--quant",
        default="Q4_K_M",
        choices=QUANT_OPTIONS,
        help="Quantization method (default: Q4_K_M)",
    )
    convert_parser.add_argument(
        "-o",
        "--output-dir",
        default="./models",
        help="Output directory (default: ./models)",
    )

    # TurboQuant model
    tq_parser = subparsers.add_parser("turboquant", help="Convert a model to TurboQuant format")
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
        default="./models",
        help="Output directory (default: ./models)",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "download":
        download_model(args.model, args.output_dir, args.quant)
    elif args.command == "convert":
        convert_model(args.model_path, args.quant, args.output_dir)
    elif args.command == "turboquant":
        turboquant_model(args.model_path, args.quant, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
