#!/usr/bin/env python3
"""
Model management script for downloading and converting LLM models.
Supports downloading models from HuggingFace and converting to GGUF format.
"""

import os
import sys
import argparse
import json
import subprocess
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
        "mmproj": "mmproj-gemma-4-E2B-it-BF16.gguf",  # Multimodal projector for vision support
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
        "mmproj": "mmproj-gemma-4-E4B-it-BF16.gguf",  # Multimodal projector for vision support
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
        "mmproj": "mmproj-gemma-4-26B-A4B-it-BF16.gguf",  # Multimodal projector for vision support
    },
    "gemma-4-31b": {
        "hf_repo": "lmstudio-community/gemma-4-31B-it-GGUF",
        "description": "Gemma 4 31B (~16GB)",
        "size_gb": 16,
        "mmproj": "mmproj-gemma-4-31B-it-BF16.gguf",  # Multimodal projector for vision support
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
        "description": "Qwopus 3.6 35B-A3B-v1 (~17GB)",
        "size_gb": 17,
    },
    "minimax-m2.7": {
        "hf_repo": "unsloth/MiniMax-M2.7-GGUF",
        "description": "MiniMax M2.7 (~18GB)",
        "size_gb": 18,
        "notes": "Mixture-of-Experts model. Use Q8_0 quantization for best quality. BF16 and MXFP4_MOE variants also available.",
        "turboquant": True,
        "turboquant_source": "Q8_0",
    },
}

# TurboQuant quantization options (extreme compression)
TQ_QUANT_OPTIONS = [
    "TQ2_0",   # 2-bit per weight - better quality while still highly compressed
    "TQ1_0",   # 1-bit per weight - extreme compression
]

# Quantization priority for selecting largest quantized version
# FP16 > BF16 > Q8_0 > Q6_K > Q5_K_M > Q4_K_M > others
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


def get_largest_quantized_file(repo_id: str) -> str:
    """Fetch the largest quantized file (FP16 or Q8_0 preferred) from HuggingFace repo using the HuggingFace API."""
    # Use HuggingFace API to list files
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    
    try:
        result = subprocess.run(
            ["curl", "-s", api_url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Warning: Failed to fetch files from HuggingFace API for {repo_id}")
            return None
        
        files = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout fetching files from HuggingFace API for {repo_id}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse HuggingFace API response for {repo_id}")
        return None
    except Exception as e:
        print(f"Warning: Error fetching files from HuggingFace API for {repo_id}: {e}")
        return None
    
    # Filter for .gguf files
    gguf_files = [(f["path"], f.get("size", 0)) for f in files if f.get("type") == "file" and f.get("path", "").endswith(".gguf")]
    
    if not gguf_files:
        print(f"Warning: No .gguf files found in {repo_id}")
        return None
    
    # Sort by quantization priority (FP16 > BF16 > Q8_0 > ...)
    def quant_priority(filepath):
        path_lower = filepath.lower()
        for i, quant in enumerate(QUANT_PRIORITY):
            if quant.lower() in path_lower:
                return i
        return len(QUANT_PRIORITY)  # Files without known quantization go to the end
    
    # Sort by priority (lower = better), then by file size (larger first)
    gguf_files.sort(key=lambda f: (quant_priority(f[0]), -f[1]))
    
    return gguf_files[0][0]


def download_model(model_name: str, output_dir: str) -> None:
    """Download a model from HuggingFace."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    
    model_info = MODELS[model_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fetch the largest quantized file from HuggingFace
    file_name = get_largest_quantized_file(model_info["hf_repo"])
    if not file_name:
        print(f"Error: Could not determine largest quantized file for {model_info['hf_repo']}")
        sys.exit(1)
    
    file_path = output_path / file_name
    
    if file_path.exists():
        print(f"Model already exists: {file_path}")
    else:
        print(f"Downloading {model_info['description']}...")
        print(f"Source: {model_info['hf_repo']}/{file_name}")
        print(f"Destination: {file_path}")
        
        # Use hf CLI to download
        import subprocess
        
        try:
            # Stream output to console so progress bar is visible
            result = subprocess.run(
                [
                    "hf",
                    "download",
                    model_info["hf_repo"],
                    file_name,
                    "--local-dir", str(output_path),
                ],
                stdout=None,  # Stream to console
                stderr=None,  # Stream to console
            )
            if result.returncode != 0:
                print(f"\nError downloading model")
                sys.exit(1)
            print(f"\nModel downloaded to: {file_path}")
        except FileNotFoundError:
            print("hf CLI not found. Install with: pip install huggingface_hub")
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
                # Stream output to console so progress bar is visible
                result = subprocess.run(
                    [
                        "hf",
                        "download",
                        model_info["hf_repo"],
                        mmproj_file,
                        "--local-dir", str(output_path),
                    ],
                    stdout=None,  # Stream to console
                    stderr=None,  # Stream to console
                )
                if result.returncode != 0:
                    print(f"\nError downloading mmproj")
                else:
                    print(f"\nMultimodal projector downloaded to: {mmproj_path}")
            except FileNotFoundError:
                print("hf CLI not found. Please download mmproj.gguf manually.")
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
    print("Quantization priority (largest first):")
    for q in QUANT_PRIORITY:
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
        choices=QUANT_PRIORITY[:10],  # Only allow common quantization options
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
        download_model(args.model, args.output_dir)
    elif args.command == "convert":
        convert_model(args.model_path, args.quant, args.output_dir)
    elif args.command == "turboquant":
        turboquant_model(args.model_path, args.quant, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
