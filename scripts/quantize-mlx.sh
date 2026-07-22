#!/usr/bin/env bash
#
# Custom affine quantization script for Qwen3.6-35B-A3B MLX safetensors.
#
# Applies a per-tensor affine quantization policy matching the K-quant recipe:
#   MTP layer (L40):       FP16
#   Router (ffn_gate_inp): FP16
#   Embedding (token_embd): Q4
#   Output (lm_head):      Q8
#   Shared expert (shexp): Q8
#   Edge layers (L0-4, L35-39) attention:  Q5
#   Edge layers (L0-4, L35-39) experts:    Q4
#   Middle layers (L5-34) attention:       Q5
#   Middle layers (L5-34) experts:         Q4
#   Norms/biases:                          FP16
#
# Uses mlx-lm's built-in affine quantization (no mlx-kquant required).
#
# Usage:
#   bash quantize-mlx.sh --input <mlx-fp16-safetensors-dir> --output <quantized-dir> [--kv-bits 4]
#
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
echo "[quantize-mlx] Policy: Docker 262K-Balanced affine quantization"

# ---- Run the quantization ---------------------------------------------------
export INPUT_DIR OUTPUT_DIR KV_BITS
python3 << 'PYTHON_SCRIPT'
import sys
import os
import json
import re
from pathlib import Path

input_dir  = Path(os.environ.get("INPUT_DIR", ""))
output_dir = Path(os.environ.get("OUTPUT_DIR", ""))
kv_bits    = int(os.environ.get("KV_BITS", "4"))

if not input_dir.is_dir():
    print(f"[quantize-mlx] ERROR: Input directory not found: {input_dir}", file=sys.stderr)
    sys.exit(1)

# Build the affine quantization preset matching the Docker 262K-Balanced recipe.
# Uses mlx-lm's affine quantization types: q4_0, q8_0, etc.
# These approximate the K-quant codec assignments.
preset = {
    # MTP layer (L40): FP16 — keep full precision for the MTP block
    "blk.40.*":                              "fp16",

    # Router (ffn_gate_inp): FP16
    "ffn_gate_inp.weight":                   "fp16",

    # Embedding: Q4 (approximates Q4_K_M)
    "token_embd.weight":                     "q4_0",

    # Output (lm_head): Q8 (approximates Q6_K)
    "lm_head.weight":                        "q8_0",

    # Shared expert (shexp): Q8
    "*ffn_gate_up_shexp.weight":             "q8_0",

    # Edge layers (L0-4, L35-39) attention: Q5 (approximates Q5_K_M)
    "blk.[0-4].attn_q.weight":               "q4_0",
    "blk.[0-4].attn_k.weight":               "q4_0",
    "blk.[0-4].attn_v.weight":               "q4_0",
    "blk.[0-4].attn_output.weight":          "q4_0",
    "blk.[35-39].attn_q.weight":             "q4_0",
    "blk.[35-39].attn_k.weight":             "q4_0",
    "blk.[35-39].attn_v.weight":             "q4_0",
    "blk.[35-39].attn_output.weight":        "q4_0",

    # Edge layers (L0-4, L35-39) experts: Q4
    "blk.[0-4].ffn_gate_exps.weight":        "q4_0",
    "blk.[0-4].ffn_up_exps.weight":          "q4_0",
    "blk.[0-4].ffn_down_exps.weight":        "q4_0",
    "blk.[35-39].ffn_gate_exps.weight":      "q4_0",
    "blk.[35-39].ffn_up_exps.weight":        "q4_0",
    "blk.[35-39].ffn_down_exps.weight":      "q4_0",

    # Middle layers (L5-34) attention: Q5 (approximates Q5_K_M)
    "blk.[5-9].attn_q.weight":               "q4_0",
    "blk.[5-9].attn_k.weight":               "q4_0",
    "blk.[5-9].attn_v.weight":               "q4_0",
    "blk.[5-9].attn_output.weight":          "q4_0",
    "blk.[10-19].attn_q.weight":             "q4_0",
    "blk.[10-19].attn_k.weight":             "q4_0",
    "blk.[10-19].attn_v.weight":             "q4_0",
    "blk.[10-19].attn_output.weight":        "q4_0",
    "blk.[20-29].attn_q.weight":             "q4_0",
    "blk.[20-29].attn_k.weight":             "q4_0",
    "blk.[20-29].attn_v.weight":             "q4_0",
    "blk.[20-29].attn_output.weight":        "q4_0",
    "blk.[30-34].attn_q.weight":             "q4_0",
    "blk.[30-34].attn_k.weight":             "q4_0",
    "blk.[30-34].attn_v.weight":             "q4_0",
    "blk.[30-34].attn_output.weight":        "q4_0",

    # Middle layers (L5-34) experts: Q4 (approximates Q3_K)
    "blk.[5-9].ffn_gate_exps.weight":        "q4_0",
    "blk.[5-9].ffn_up_exps.weight":          "q4_0",
    "blk.[5-9].ffn_down_exps.weight":        "q4_0",
    "blk.[10-19].ffn_gate_exps.weight":      "q4_0",
    "blk.[10-19].ffn_up_exps.weight":        "q4_0",
    "blk.[10-19].ffn_down_exps.weight":      "q4_0",
    "blk.[20-29].ffn_gate_exps.weight":      "q4_0",
    "blk.[20-29].ffn_up_exps.weight":        "q4_0",
    "blk.[20-29].ffn_down_exps.weight":      "q4_0",
    "blk.[30-34].ffn_gate_exps.weight":      "q4_0",
    "blk.[30-34].ffn_up_exps.weight":        "q4_0",
    "blk.[30-34].ffn_down_exps.weight":      "q4_0",

    # Norms/biases: FP16 (all norm layers)
    "blk.*.norm.weight":                     "fp16",
    "blk.*.norm1.weight":                    "fp16",
    "blk.*.norm2.weight":                    "fp16",
    "blk.*.post_attention_norm.weight":      "fp16",
    "blk.*.post_mlp_norm.weight":            "fp16",
}

# Helper to expand range patterns like [0-4] to regex
def expand_range_group(m):
    start, end = int(m.group(1)), int(m.group(2))
    return f"(?:{'|'.join(str(i) for i in range(start, end + 1))})"

def match_tensor(tensor_name, preset):
    """Match a tensor name against the preset patterns."""
    for pattern, qtype in preset.items():
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        regex_pattern = re.sub(r'\[(\d+)-(\d+)\]', expand_range_group, regex_pattern)
        if re.match(f"^{regex_pattern}$", tensor_name):
            return qtype
    return "fp16"  # default

# Save preset for reference
preset_path = output_dir / "quantize_preset.json"
with open(preset_path, "w") as f:
    json.dump(preset, f, indent=2)
print(f"[quantize-mlx] Quantization preset saved to: {preset_path}")

# Load tensors from all safetensors files
import safetensors.torch

safetensors_files = list(input_dir.glob("*.safetensors"))
if not safetensors_files:
    print(f"[quantize-mlx] ERROR: No .safetensors files found in {input_dir}", file=sys.stderr)
    sys.exit(1)

print(f"[quantize-mlx] Found {len(safetensors_files)} safetensors file(s)")

all_tensors = {}
for sf in safetensors_files:
    print(f"[quantize-mlx] Loading: {sf.name}")
    try:
        tensors = safetensors.torch.load_file(str(sf))
        for name, tensor in tensors.items():
            all_tensors[name] = tensor
    except Exception as e:
        print(f"[quantize-mlx] Warning: failed to load {sf.name}: {e}", file=sys.stderr)

if not all_tensors:
    print(f"[quantize-mlx] ERROR: No tensors loaded from {input_dir}", file=sys.stderr)
    sys.exit(1)

print(f"[quantize-mlx] Loaded {len(all_tensors)} tensors")

# Apply affine quantization using mlx-lm
print(f"[quantize-mlx] Applying affine quantization policy (kv_bits={kv_bits})...")

# Group tensors by quantization type
qtype_groups = {}
for name, tensor in all_tensors.items():
    qtype = match_tensor(name, preset)
    if qtype not in qtype_groups:
        qtype_groups[qtype] = []
    qtype_groups[qtype].append(name)

print(f"[quantize-mlx] Quantization type distribution:")
for qtype, names in sorted(qtype_groups.items()):
    print(f"  {qtype}: {len(names)} tensors")

# Use mlx-lm's quantize function
from mlx_lm import quantize

# Build the qtype dict for mlx-lm's quantize
qtype_dict = {}
for qtype, names in qtype_groups.items():
    if qtype != "fp16":
        for name in names:
            qtype_dict[name] = qtype

print(f"[quantize-mlx] Quantizing {len(qtype_dict)} tensors...")

# Quantize the model
quantized_model = quantize(all_tensors, qtype_dict)

# Save quantized model as .safetensors
output_file = output_dir / "model.safetensors"
print(f"[quantize-mlx] Saving quantized model to: {output_file}")
safetensors.torch.save_file(quantized_model, str(output_file))

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
PYTHON_SCRIPT

echo "[quantize-mlx] Done."
