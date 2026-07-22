#!/usr/bin/env python3
"""Quantize the Qwen3-VL mmproj (vision tower) GGUF's BF16 weight tensors to
Q8_0, leaving F32 norms/biases untouched. Non-generated companion to
quantize.sh: ik_llama.cpp's llama-quantize tool can't be used for this because
it hard-requires a recognized LLM architecture (llm_load_arch), and the mmproj
file's general.architecture is "clip" - so this does the Q8_0 block
quantization directly with gguf-py + numpy (both already present in this
image, see docker/Dockerfile).

Real-hardware validation (RTX 3090, this repo's docker/ dev stack, 2026-07-20):
  - mmproj-BF16.gguf: 861.0 MiB -> mmproj-Q8_0.gguf: 585.7 MiB (~32% smaller,
    ~276 MiB less VRAM at load). Not the full ~46% a pure BF16->Q8_0
    conversion would give, because 27 of the 110 BF16 tensors (all
    v.blk.N.ffn_down.weight, row length 4304) aren't a multiple of Q8_0's
    32-element block size and must stay BF16 (ggml has no fallback for
    unaligned rows; F16 wouldn't save anything either, since it's also
    2 bytes/element like BF16).
  - Loaded successfully via ik_llama.cpp's clip.cpp (CUDA backend) and
    produced a correct description of a synthetic red-square/blue-circle
    test image, matching the un-quantized mmproj's behavior - no observed
    accuracy or vision-quality regression from this specific test.
  - Q8_0 conversion of the vision tower has no measurable effect on
    text-generation tps: the vision tower only runs once per image at
    prompt-encode time, not per output token.

Usage:
    python3 quantize-mmproj.py <src_mmproj.gguf> <dst_mmproj.gguf>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
    sys.path.insert(1, '/opt/iqllama/gguf-py')
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

QK8_0 = 32


def quantize_q8_0_rows(x: np.ndarray) -> np.ndarray:
    """x: (n_rows, row_len) float32, row_len % 32 == 0.
    Returns (n_rows, row_len // 32 * 34) uint8, ggml Q8_0 layout: per 32-elem
    block, an fp16 scale d = amax/127 followed by 32 int8 values round(x/d).
    """
    n_rows, row_len = x.shape
    n_blocks_per_row = row_len // QK8_0
    xb = x.reshape(n_rows, n_blocks_per_row, QK8_0)
    amax = np.max(np.abs(xb), axis=2, keepdims=True)
    d = (amax / 127.0).astype(np.float32)
    d_safe = np.where(d == 0, 1.0, d)
    q = np.round(xb / d_safe).astype(np.int8)
    d_f16 = d.astype(np.float16)
    out = np.empty((n_rows, n_blocks_per_row, 2 + QK8_0), dtype=np.uint8)
    out[:, :, :2] = d_f16.view(np.uint8)
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(n_rows, n_blocks_per_row * (2 + QK8_0))


def main(src_path: str, dst_path: str) -> None:
    reader = GGUFReader(src_path)
    arch = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            arch = str(field.contents())
            break
    writer = GGUFWriter(dst_path, arch=arch or 'clip')

    for field in reader.fields.values():
        name = field.name
        if name in ('GGUF.version', 'general.architecture'):
            continue
        types = field.types
        val_type = types[0]
        if val_type == GGUFValueType.ARRAY:
            writer.add_key_value(name, field.contents(), GGUFValueType.ARRAY, sub_type=types[-1])
        else:
            writer.add_key_value(name, field.contents(), val_type)

    n_conv = n_skip = before = after = 0
    for t in reader.tensors:
        data = t.data
        nbytes_before = data.nbytes
        before += nbytes_before
        if t.tensor_type.name == 'BF16':
            f32 = data.view(np.uint16).astype(np.uint32)
            f32 = (f32 << 16).view(np.float32)
            row_len = f32.shape[-1]
            n_rows = f32.size // row_len
            if row_len % QK8_0 == 0:
                q8 = quantize_q8_0_rows(f32.reshape(n_rows, row_len))
                raw_shape = (*f32.shape[:-1], q8.shape[-1])
                writer.add_tensor(t.name, q8.reshape(raw_shape), raw_shape=raw_shape,
                                   raw_dtype=GGMLQuantizationType.Q8_0)
                n_conv += 1
                after += q8.nbytes
            else:
                # Not a multiple of the Q8_0 block size (e.g. row_len=4304) -
                # leave as BF16 (F16 would be the same size, no benefit).
                writer.add_tensor(t.name, data.copy(), raw_shape=t.data.shape,
                                   raw_dtype=t.tensor_type)
                after += nbytes_before
                n_skip += 1
        else:
            if t.tensor_type.name in ('F32', 'F16', 'F64', 'I8', 'I16', 'I32', 'I64'):
                writer.add_tensor(t.name, data.copy())
            else:
                writer.add_tensor(t.name, data.copy(), raw_shape=t.data.shape, raw_dtype=t.tensor_type)
            after += nbytes_before

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"[quantize-mmproj] converted {n_conv} BF16 tensors to Q8_0 ({n_skip} kept BF16, row_len%32!=0)")
    print(f"[quantize-mmproj] {src_path} -> {dst_path}: {before/1024/1024:.1f} MiB -> {after/1024/1024:.1f} MiB")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <src_mmproj.gguf> <dst_mmproj.gguf>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
