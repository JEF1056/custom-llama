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
MODELS = {
    "llama3.1-8b": {
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "default_file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.1 8B Instruct (4-bit quantized)",
    },
    "llama3.1-70b": {
        "hf_repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "default_file": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.1 70B Instruct (4-bit quantized)",
    },
    "llama3.1-405b": {
        "hf_repo": "bartowski/Meta-Llama-3.1-405B-Instruct-GGUF",
        "default_file": "Meta-Llama-3.1-405B-Instruct-Q4_K_M.gguf",
        "description": "Meta Llama 3.1 405B Instruct (4-bit quantized)",
    },
    "phi3-mini": {
        "hf_repo": "bartowski/Phi-3-mini-4k-instruct-GGUF",
        "default_file": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "description": "Microsoft Phi-3 Mini 4K Instruct (4-bit quantized)",
    },
    "mistral-7b": {
        "hf_repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "default_file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "description": "Mistral 7B Instruct (4-bit quantized)",
    },
    "mixtral-8x7b": {
        "hf_repo": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "default_file": "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
        "description": "Mistral MoE 8x7B Instruct (4-bit quantized)",
    },
    "qwopus3.6-35b": {
        "hf_repo": "Jackrong/Qwopus3.6-35B-A3B-v1-GGUF",
        "default_file": "Qwopus3.6-35B-A3B-v1-Q8_0.gguf",
        "description": "Qwopus 3.6 35B-A3B v1 (8-bit quantized GGUF - best for TurboQuant conversion)",
        "notes": "Mixture-of-Experts model: 35B total params with 3.1B active. Multimodal - supports images via mmproj.gguf. For TurboQuant, always convert from FP16 if available (check for FP16 files in the repo), otherwise use Q8_0. Available quantizations: Q3_K_L, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, IQ4_XS.",
        "turboquant": True,  # This model supports TurboQuant conversion
        "turboquant_source": "Q8_0",  # Recommended source for TurboQuant conversion (FP16 preferred if available)
        "fp16_file": None,  # Check repo for FP16 availability - update if found
        "mmproj": "mmproj.gguf",  # Multimodal projector for vision/image support
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
    print("-" * 60)
    for key, info in MODELS.items():
        tq_note = " [TurboQuant]" if info.get("turboquant") else ""
        mm_note = " [Multimodal]" if "mmproj" in info else ""
        print(f"  {key:20s} - {info['description']}{tq_note}{mm_note}")
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
    print("Multimodal models require mmproj.gguf file for vision support.")
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
        "model", help="Model name (e.g., llama3.1-8b)"
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