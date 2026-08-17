#!/usr/bin/env python3
"""
Phase 2: Ultra-Fast Dedicated DFlash Drafter Training.

Standard PyTorch Batched Multi-Slice Architecture:
- Extracts all valid draft blocks across sequences in a shard (~60,000 blocks = 468 full batches per shard).
- Overlapped background prefetching in DataLoader worker completely hides disk I/O.
- Eliminates per-item IPC serialization overhead (Single-handle IPC yield).
- batch_size=128: Maximizes Tensor Core GEMM matrix saturation on RTX 3090.
- Linear Scaled LR (8e-4 with 500-step Warmup and Cosine Annealing to 1e-5).
- max_steps=10000 (Equivalent to 40,000 steps at batch 32 = 20+ full dataset epochs).
- 100% on-die GDDR6X VRAM compute.
"""

import argparse
import glob
import os
import sys
import time
import math
import random
import gc
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import bitsandbytes as bnb
from model import DFlashConfig, DFlashDraftModel


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def purge_ram():
    """Forces both PyTorch and Linux glibc to release all allocated memory arenas."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash drafter on extracted features")
    parser.add_argument("--features-dir", type=str, default="/workspace-data/features")
    parser.add_argument("--output-dir", type=str, default="/output/Qwen3.8-27B-heretic-dflash")
    parser.add_argument("--batch-size", type=int, default=128, help="Parallel draft blocks per training step")
    parser.add_argument("--block-size", type=int, default=4, help="Matching 3 draft tokens per step")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=200)
    return parser.parse_args()


class HighThroughputBatchedDataset(IterableDataset):
    """Slices all blocks across sequences per shard, yielding full 128-item batch tensors via single-handle IPC."""
    def __init__(self, features_dir: str, batch_size: int = 128, block_size: int = 4):
        super().__init__()
        self.shard_files = sorted(glob.glob(os.path.join(features_dir, "shard_*.pt")))
        if not self.shard_files:
            raise RuntimeError(f"No shard files found in {features_dir}")
        print(f"[Dataset] Found {len(self.shard_files)} shards in {features_dir} (High-Throughput Multi-Slice Engine)", flush=True)
        self.batch_size = batch_size
        self.block_size = block_size

    def __iter__(self):
        k = self.block_size
        while True:
            shuffled_shards = list(self.shard_files)
            random.shuffle(shuffled_shards)

            for sf in shuffled_shards:
                try:
                    samples = torch.load(sf, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(f"[DataLoaderError] {sf}: {e}", flush=True)
                    continue

                blocks = []
                contexts = []
                ctx_positions = []
                prop_positions = []

                # Extract all valid blocks from the shard (dense training coverage)
                for s in samples:
                    tokens = s["tokens"]
                    hidden = s["hidden"]
                    S = tokens.shape[0]

                    if S < k + 2:
                        continue

                    num_blocks = (S - 1) // k
                    for b in range(num_blocks):
                        start = b * k
                        block_tok = tokens[start:start + k]
                        h_ctx = hidden[start:start + 1].to(torch.bfloat16)

                        ctx_pos = torch.tensor([start], dtype=torch.long)
                        prop_pos = torch.arange(start, start + k, dtype=torch.long)

                        blocks.append(block_tok)
                        contexts.append(h_ctx)
                        ctx_positions.append(ctx_pos)
                        prop_positions.append(prop_pos)

                del samples
                purge_ram()

                # Shuffle all extracted blocks within the shard
                total = len(blocks)
                if total >= self.batch_size:
                    perm = torch.randperm(total).tolist()
                    for idx in range(0, total - self.batch_size + 1, self.batch_size):
                        batch_indices = perm[idx:idx + self.batch_size]
                        
                        b_tensor = torch.stack([blocks[i] for i in batch_indices], dim=0)
                        h_tensor = torch.stack([contexts[i] for i in batch_indices], dim=0)
                        cp_tensor = torch.stack([ctx_positions[i] for i in batch_indices], dim=0)
                        pp_tensor = torch.stack([prop_positions[i] for i in batch_indices], dim=0)

                        yield b_tensor, h_tensor, cp_tensor, pp_tensor

                del blocks, contexts, ctx_positions, prop_positions
                purge_ram()


def get_lr(step: int, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 2: Ultra-Fast Dedicated Drafter Training on {device.upper()} ===", flush=True)
    print(f"Features Dir: {args.features_dir}", flush=True)
    print(f"Output Dir:   {args.output_dir}", flush=True)
    print(f"Batch Size:   {args.batch_size} | Block Size: {args.block_size} | LR: {args.lr:.1e} | Max Steps: {args.max_steps}", flush=True)

    proj_file = os.path.join(args.features_dir, "projection_weights.pt")
    if not os.path.exists(proj_file):
        print(f"Error: Projection weights {proj_file} not found. Run Phase 1 first.", flush=True)
        sys.exit(1)

    print("Loading embed_tokens and lm_head projection weights onto GPU...", flush=True)
    projections = torch.load(proj_file, map_location="cpu", weights_only=True)
    embed_weights = projections["embed_tokens"].to(device=device, dtype=torch.bfloat16)  # [248320, 5120]
    lm_head_weights = projections["lm_head"].to(device=device, dtype=torch.bfloat16)      # [248320, 5120]
    del projections
    purge_ram()

    config = DFlashConfig(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        target_layer_ids=[1, 16, 31, 46, 61],
        num_target_layers=64,
        block_size=args.block_size,
        mask_token_id=248070,
        vocab_size=248320,
        sliding_window=2048,
        rope_theta=10000000.0,
    )

    drafter = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
    drafter.train()

    optimizer = bnb.optim.AdamW8bit(drafter.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Trainable drafter parameters: {sum(p.numel() for p in drafter.parameters()):,}", flush=True)
    print(f"Total Dedicated VRAM Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB (100% on-die GDDR6X)", flush=True)

    dataset = HighThroughputBatchedDataset(args.features_dir, batch_size=args.batch_size, block_size=args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    k = args.block_size
    mask_id = config.mask_token_id
    step = 0
    t0 = time.time()
    loss_acc = 0.0

    print("=== Training Loop Active (Multi-Slice Continuous Tensor Core Compute) ===", flush=True)

    data_iter = iter(dataloader)

    while step < args.max_steps:
        cur_lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        block_tok, h_ctx, ctx_pos, prop_pos = next(data_iter)
        block_tok = block_tok.to(device, non_blocking=True)
        h_ctx = h_ctx.to(device, non_blocking=True)
        ctx_pos = ctx_pos.to(device, non_blocking=True)
        prop_pos = prop_pos.to(device, non_blocking=True)

        # Mask positions 1..k-1
        masked_tok = block_tok.clone()
        masked_tok[:, 1:] = mask_id

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Lookup embeddings
            draft_embeds = F.embedding(masked_tok, embed_weights)  # [B, k, 5120]

            # Forward pass through drafter with RoPE position embeddings
            drafter_out = drafter(draft_embeds, h_ctx, ctx_pos, prop_pos)  # [B, k, 5120]
            
            # Single fused GEMM projection on CUDA Tensor Cores
            draft_logits = F.linear(drafter_out[:, 1:], lm_head_weights)  # [B, k-1, 248320]
            targets = block_tok[:, 1:].contiguous()                       # [B, k-1]

            loss = F.cross_entropy(
                draft_logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_acc += loss.item()
        step += 1
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0
            avg_loss = loss_acc / (args.eval_interval if step > 1 else 1)
            print(f"Step {step}/{args.max_steps} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | Speed: {rate:.1f} steps/s | Elapsed: {elapsed:.1f}s", flush=True)
            loss_acc = 0.0

    print("=== Phase 2 Complete! Exporting Official DFlash Safetensors ===", flush=True)
    drafter.export_mlx_safetensors(args.output_dir)
    print(f"DFlash Drafter successfully saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
