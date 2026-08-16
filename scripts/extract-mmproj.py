#!/usr/bin/env python3
"""
Convert Qwen3 / Qwen3.5 / Qwen3.8 vision encoder & multimodal projector
tensors from HuggingFace safetensors into a standalone GGUF mmproj file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
    sys.path.insert(1, '/opt/iqllama/gguf-py')
from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Qwen vision weights to mmproj GGUF")
    parser.add_argument("model_dir", type=str, help="Directory containing config.json and *.safetensors")
    parser.add_argument("output_file", type=str, help="Output mmproj GGUF path")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_file = Path(args.output_file)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} does not exist", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    vision_config = config.get("vision_config", {})
    if not vision_config:
        print("Error: No vision_config found in config.json", file=sys.stderr)
        sys.exit(1)

    writer = GGUFWriter(str(out_file), arch="clip")

    # Set metadata for PROJECTOR_TYPE_QWEN3VL in clip.cpp
    writer.add_name("Qwen3-VL Vision Tower")
    writer.add_description("Extracted Qwen3.5/3.8 vision projector and transformer")
    writer.add_string("clip.projector_type", "qwen3vl_merger")

    # Extract hyperparameters
    hidden_size = vision_config.get("hidden_size", 1152)
    depth = vision_config.get("depth", 27)
    num_heads = vision_config.get("num_heads", 16)
    intermediate_size = vision_config.get("intermediate_size", 4304)
    patch_size = vision_config.get("patch_size", 16)
    spatial_merge_size = vision_config.get("spatial_merge_size", 2)
    image_token_id = config.get("image_token_id", 151655)

    writer.add_uint32("clip.vision.embedding_length", hidden_size)
    writer.add_uint32("clip.vision.block_count", depth)
    writer.add_uint32("clip.vision.feed_forward_length", intermediate_size)
    writer.add_uint32("clip.vision.attention.head_count", num_heads)
    writer.add_uint32("clip.vision.projection_dim", config.get("hidden_size", 5120))
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_uint32("clip.vision.image_size", 384)
    writer.add_uint32("clip.vision.patch_size", patch_size)
    writer.add_uint32("clip.vision.spatial_merge_size", spatial_merge_size)
    writer.add_uint32("clip.vision.image_token_id", image_token_id)
    writer.add_array("clip.vision.image_mean", [0.48145466, 0.4578275, 0.40821073])
    writer.add_array("clip.vision.image_std", [0.26862954, 0.26130258, 0.27577711])
    writer.add_bool("clip.has_vision_encoder", True)

    # Locate and map safetensors
    safetensor_files = list(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"Error: No *.safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[extract-mmproj] Scanning {len(safetensor_files)} safetensors files...")
    visual_tensors = {}

    for sf_path in sorted(safetensor_files):
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                if "visual" in k or "merger" in k:
                    tensor = f.get_tensor(k).to(torch.float32).numpy()
                    visual_tensors[k] = tensor

    print(f"[extract-mmproj] Found {len(visual_tensors)} visual/projector tensors.")

    # Tensor name mapping: HF -> GGUF clip.cpp conventions
    # Qwen3-VL:
    # model.visual.blocks.{i}.attn.qkv.weight -> v.blk.{i}.attn_qkv.weight
    # model.visual.blocks.{i}.attn.proj.weight -> v.blk.{i}.attn_out.weight
    # model.visual.blocks.{i}.mlp.linear_fc1.weight -> v.blk.{i}.ffn_up.weight
    # model.visual.blocks.{i}.mlp.linear_fc2.weight -> v.blk.{i}.ffn_down.weight
    # model.visual.blocks.{i}.norm1.weight -> v.blk.{i}.ln1.weight
    # model.visual.blocks.{i}.norm2.weight -> v.blk.{i}.ln2.weight
    # model.visual.merger.linear_fc1.weight -> mm.0.weight
    # model.visual.merger.linear_fc2.weight -> mm.2.weight
    # model.visual.patch_embed.proj.weight -> v.patch_embd.weight

    added = 0
    for hf_name, tensor in visual_tensors.items():
        gguf_name = hf_name
        gguf_name = gguf_name.replace("model.visual.", "v.")
        gguf_name = gguf_name.replace("visual.", "v.")
        gguf_name = gguf_name.replace("blocks.", "blk.")
        gguf_name = gguf_name.replace(".attn.proj.", ".attn_out.")
        gguf_name = gguf_name.replace(".attn.qkv.", ".attn_qkv.")
        gguf_name = gguf_name.replace(".mlp.linear_fc1.", ".ffn_up.")
        gguf_name = gguf_name.replace(".mlp.linear_fc2.", ".ffn_down.")
        gguf_name = gguf_name.replace(".norm1.", ".ln1.")
        gguf_name = gguf_name.replace(".norm2.", ".ln2.")
        gguf_name = gguf_name.replace("v.merger.linear_fc1.", "mm.0.")
        gguf_name = gguf_name.replace("v.merger.linear_fc2.", "mm.2.")
        gguf_name = gguf_name.replace("v.patch_embed.proj.", "v.patch_embd.")
        gguf_name = gguf_name.replace("patch_embed.proj.", "patch_embd.")
        gguf_name = gguf_name.replace("v.pos_embed.", "v.position_embd.")
        gguf_name = gguf_name.replace("pos_embed.", "position_embd.")
        if "v.patch_embd.weight" in gguf_name and tensor.ndim == 5:
            # tensor is [out_dim, in_ch, t_patch=2, p_h=16, p_w=16]
            # ik_llama clip.cpp splits Conv3D into two Conv2D kernels:
            # v.patch_embd.weight (slice 0) and v.patch_embd.weight.1 (slice 1)
            # each with shape [out_dim, in_ch, p_h, p_w]
            t0 = tensor[:, :, 0, :, :].copy()
            t1 = tensor[:, :, 1, :, :].copy()
            writer.add_tensor("v.patch_embd.weight", t0)
            writer.add_tensor("v.patch_embd.weight.1", t1)
            added += 2
            continue

        writer.add_tensor(gguf_name, tensor)
        added += 1

    print(f"[extract-mmproj] Writing {added} tensors to {out_file}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"[extract-mmproj] Successfully created {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
