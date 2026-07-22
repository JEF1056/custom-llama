#!/usr/bin/env bash
#
# K-quant quantization script for Qwen3.6-35B-A3B MLX safetensors.
#
# Applies the "Docker 262K-Balanced" K-quant policy using mlx-kquant:
#   MTP layer (L40):       FP16
#   Router (ffn_gate_inp): FP16
#   Embedding (token_embd): Q4_K_M
#   Output (lm_head):      Q6_K
#   Shared expert (shexp): Q8_0
#   Edge layers (L0-4, L35-39) attention:  Q5_K_M
#   Edge layers (L0-4, L35-39) experts:    Q4_K_M
#   Middle layers (L5-34) attention:       Q5_K_M
#   Middle layers (L5-34) experts:         Q3_K
#   Norms/biases:                          FP16
#
# Usage:
#   bash quantize-mlx.sh --input <mlx-fp16-safetensors-dir> --output <quantized-dir> [--kv-bits 4]
#
# Requires: mlx-kquant with mlx==0.31.2
set -euo pipefail

# ---- Parse arguments --------------------------------------------------------
INPUT_DIR=""
OUTPUT_DIR=""
KV_BITS=${KV_BITS:-4}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)  INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --kv-bits) KV_BITS="$2"; shift 2 ;;
        *) echo "[quantize-mlx] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "[quantize-mlx] ERROR: --input is required (MLX FP16 safetensors directory)." >&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "[quantize-mlx] ERROR: --output is required (quantized model output directory)." >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[quantize-mlx] ERROR: Input directory not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[quantize-mlx] Input:  $INPUT_DIR"
echo "[quantize-mlx] Output: $OUTPUT_DIR"
echo "[quantize-mlx] KV bits: $KV_BITS"
echo "[quantize-mlx] Policy: Docker 262K-Balanced K-quant"

# ---- Run the K-quant conversion ---------------------------------------------
# Use mlx-kquant's quantize module with a custom preset.
# The preset is defined inline as a Python dict matching the layer-tier policy.
export INPUT_DIR OUTPUT_DIR KV_BITS
python3 << 'PYTHON_SCRIPT'
import sys
import os
import json
from pathlib import Path

input_dir  = Path(os.environ.get("INPUT_DIR", ""))
output_dir = Path(os.environ.get("OUTPUT_DIR", ""))
kv_bits    = int(os.environ.get("KV_BITS", "4"))

if not input_dir.is_dir():
    print(f"[quantize-mlx] ERROR: Input directory not found: {input_dir}", file=sys.stderr)
    sys.exit(1)

# Build the K-quant custom preset matching the Docker 262K-Balanced recipe.
# mlx-kquant uses codec names: fp16, q4_k_m, q5_k_m, q3_k, q6_k, q8_0, etc.
# The preset maps tensor name patterns to target codecs.

preset = {
    # MTP layer (L40): FP16 — keep full precision for the MTP block
    "blk.40.attn_q.weight":      "fp16",
    "blk.40.attn_k.weight":      "fp16",
    "blk.40.attn_v.weight":      "fp16",
    "blk.40.attn_output.weight": "fp16",
    "blk.40.ffn_gate.weight":    "fp16",
    "blk.40.ffn_up.weight":      "fp16",
    "blk.40.ffn_down.weight":    "fp16",
    "blk.40.ffn_gate_up_shexp.weight": "fp16",
    "blk.40.ffn_gate_exps.weight":     "fp16",
    "blk.40.ffn_up_exps.weight":       "fp16",
    "blk.40.ffn_down_exps.weight":     "fp16",
    "blk.40.norm.*":                   "fp16",
    "blk.40.ffn_gate_inp.weight":      "fp16",

    # Router (ffn_gate_inp): FP16
    "ffn_gate_inp.weight": "fp16",

    # Embedding: Q4_K_M
    "token_embd.weight": "q4_k_m",

    # Output (lm_head): Q6_K
    "lm_head.weight": "q6_k",

    # Shared expert (shexp): Q8_0
    "*ffn_gate_up_shexp.weight": "q8_0",

    # Edge layers (L0-4, L35-39) attention: Q5_K_M
    "blk.[0-4].attn_q.weight":      "q5_k_m",
    "blk.[0-4].attn_k.weight":      "q5_k_m",
    "blk.[0-4].attn_v.weight":      "q5_k_m",
    "blk.[0-4].attn_output.weight": "q5_k_m",
    "blk.[35-39].attn_q.weight":      "q5_k_m",
    "blk.[35-39].attn_k.weight":      "q5_k_m",
    "blk.[35-39].attn_v.weight":      "q5_k_m",
    "blk.[35-39].attn_output.weight": "q5_k_m",

    # Edge layers (L0-4, L35-39) experts: Q4_K_M
    "blk.[0-4].ffn_gate_exps.weight":    "q4_k_m",
    "blk.[0-4].ffn_up_exps.weight":      "q4_k_m",
    "blk.[0-4].ffn_down_exps.weight":    "q4_k_m",
    "blk.[35-39].ffn_gate_exps.weight":    "q4_k_m",
    "blk.[35-39].ffn_up_exps.weight":      "q4_k_m",
    "blk.[35-39].ffn_down_exps.weight":    "q4_k_m",

    # Middle layers (L5-34) attention: Q5_K_M
    "blk.[5-9].attn_q.weight":      "q5_k_m",
    "blk.[5-9].attn_k.weight":      "q5_k_m",
    "blk.[5-9].attn_v.weight":      "q5_k_m",
    "blk.[5-9].attn_output.weight": "q5_k_m",
    "blk.[10-19].attn_q.weight":      "q5_k_m",
    "blk.[10-19].attn_k.weight":      "q5_k_m",
    "blk.[10-19].attn_v.weight":      "q5_k_m",
    "blk.[10-19].attn_output.weight": "q5_k_m",
    "blk.[20-29].attn_q.weight":      "q5_k_m",
    "blk.[20-29].attn_k.weight":      "q5_k_m",
    "blk.[20-29].attn_v.weight":      "q5_k_m",
    "blk.[20-29].attn_output.weight": "q5_k_m",
    "blk.[30-34].attn_q.weight":      "q5_k_m",
    "blk.[30-34].attn_k.weight":      "q5_k_m",
    "blk.[30-34].attn_v.weight":      "q5_k_m",
    "blk.[30-34].attn_output.weight": "q5_k_m",

    # Middle layers (L5-34) experts: Q3_K
    "blk.[5-9].ffn_gate_exps.weight":    "q3_k",
    "blk.[5-9].ffn_up_exps.weight":      "q3_k",
    "blk.[5-9].ffn_down_exps.weight":    "q3_k",
    "blk.[10-19].ffn_gate_exps.weight":    "q3_k",
    "blk.[10-19].ffn_up_exps.weight":      "q3_k",
    "blk.[10-19].ffn_down_exps.weight":    "q3_k",
    "blk.[20-29].ffn_gate_exps.weight":    "q3_k",
    "blk.[20-29].ffn_up_exps.weight":      "q3_k",
    "blk.[20-29].ffn_down_exps.weight":    "q3_k",
    "blk.[30-34].ffn_gate_exps.weight":    "q3_k",
    "blk.[30-34].ffn_up_exps.weight":      "q3_k",
    "blk.[30-34].ffn_down_exps.weight":    "q3_k",

    # Norms/biases: FP16 (all norm layers)
    "blk.*.norm.weight":     "fp16",
    "blk.*.norm1.weight":    "fp16",
    "blk.*.norm2.weight":    "fp16",
    "blk.*.post_attention_norm.weight": "fp16",
    "blk.*.post_mlp_norm.weight":     "fp16",
}

# Save preset for reference
preset_path = output_dir / "kquant_preset.json"
with open(preset_path, "w") as f:
    json.dump(preset, f, indent=2)
print(f"[quantize-mlx] K-quant preset saved to: {preset_path}")

# Now run the actual quantization using mlx-kquant
try:
    from mlx_kquant import quantize, dequantize
    from mlx.core import load, save
    import mlx.core as mx

    # List all .safetensors files in the input directory
    safetensors_files = list(input_dir.glob("*.safetensors"))
    if not safetensors_files:
        print(f"[quantize-mlx] ERROR: No .safetensors files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[quantize-mlx] Found {len(safetensors_files)} safetensors file(s)")

    # Load tensors from all safetensors files
    all_tensors = {}
    for sf in safetensors_files:
        print(f"[quantize-mlx] Loading: {sf.name}")
        # Use safetensors loader
        try:
            import safetensors.torch
            tensors = safetensors.torch.load_file(str(sf))
            for name, tensor in tensors.items():
                all_tensors[name] = tensor
        except Exception as e:
            print(f"[quantize-mlx] Warning: failed to load {sf.name}: {e}", file=sys.stderr)

    if not all_tensors:
        print(f"[quantize-mlx] ERROR: No tensors loaded from {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[quantize-mlx] Loaded {len(all_tensors)} tensors")

    # Apply K-quant quantization
    print(f"[quantize-mlx] Applying K-quant policy (kv_bits={kv_bits})...")

    # Group tensors by codec target
    import re
    def expand_range_group(m):
        start, end = int(m.group(1)), int(m.group(2))
        return f"(?:{'|'.join(str(i) for i in range(start, end + 1))})"

    codec_groups = {}
    for name, tensor in all_tensors.items():
        matched_codec = "fp16"  # default
        for pattern, codec in preset.items():
            # Simple glob matching: convert pattern to regex
            regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
            regex_pattern = re.sub(r'\[(\d+)-(\d+)\]', expand_range_group, regex_pattern)
            if re.match(f"^{regex_pattern}$", name):
                matched_codec = codec
                break
        if matched_codec not in codec_groups:
            codec_groups[matched_codec] = []
        codec_groups[matched_codec].append(name)

    print(f"[quantize-mlx] Codec distribution:")
    for codec, names in sorted(codec_groups.items()):
        print(f"  {codec}: {len(names)} tensors")

    # Quantize using mlx-kquant
    # mlx-kquant quantize function takes a model dict and preset
    quantized_tensors = quantize(all_tensors, preset, kv_bits=kv_bits)

    # Save quantized model as .safetensors
    output_file = output_dir / "model.safetensors"
    print(f"[quantize-mlx] Saving quantized model to: {output_file}")
    safetensors.torch.save_file(quantized_tensors, str(output_file))

    # Copy config.json
    config_src = input_dir / "config.json"
    config_dst = output_dir / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy2(config_src, config_dst)
        print(f"[quantize-mlx] Copied config.json")

    # Copy tokenizer files
    for tokenizer_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]:
        src = input_dir / tokenizer_file
        dst = output_dir / tokenizer_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[quantize-mlx] Copied {tokenizer_file}")

    print(f"[quantize-mlx] Quantization complete: {output_dir}")

except ImportError:
    print("[quantize-mlx] mlx-kquant not available, using fallback manual quantization", file=sys.stderr)
    print("[quantize-mlx] Install: pip install mlx-kquant", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

echo "[quantize-mlx] Done."
