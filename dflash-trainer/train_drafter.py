#!/usr/bin/env python3
"""
Phase 2: Ultra-Fast Dedicated DFlash Drafter Training.

Memory-efficient streaming dataset from shard files on disk.
Zero host RAM ballooning: loads shards on demand or in mini-batches.
"""

import argparse
import glob
import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from model import DFlashConfig, DFlashDraftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash drafter on extracted features")
    parser.add_argument("--features-dir", type=str, default="/workspace-data/features")
    parser.add_argument("--output-dir", type=str, default="/output/Qwen3.8-27B-heretic-dflash")
    parser.add_argument("--block-size", type=int, default=16, help="Official DFlash block size")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    return parser.parse_args()


class ShardDataset:
    """Streams and cycles through feature shard files without keeping all in RAM."""
    def __init__(self, features_dir: str):
        self.shard_files = sorted(glob.glob(os.path.join(features_dir, "shard_*.pt")))
        if not self.shard_files:
            raise RuntimeError(f"No shard files found in {features_dir}")
        print(f"[Dataset] Found {len(self.shard_files)} shard files on disk.", flush=True)
        self.current_shard_idx = -1
        self.current_samples = []
        self._load_next_shard()

    def _load_next_shard(self):
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
        sf = self.shard_files[self.current_shard_idx]
        self.current_samples = torch.load(sf, map_location="cpu", weights_only=True)
        random.shuffle(self.current_samples)

    def sample(self):
        if not self.current_samples:
            self._load_next_shard()
        return self.current_samples.pop()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 2: Ultra-Fast Drafter Training on {device.upper()} ===", flush=True)
    print(f"Features Dir: {args.features_dir}", flush=True)
    print(f"Output Dir:   {args.output_dir}", flush=True)
    print(f"Block Size:   {args.block_size} | Max Steps: {args.max_steps}", flush=True)

    proj_file = os.path.join(args.features_dir, "projection_weights.pt")
    if not os.path.exists(proj_file):
        print(f"Error: Projection weights {proj_file} not found. Run Phase 1 first.", flush=True)
        sys.exit(1)

    print("Loading embed_tokens and lm_head projection weights onto GPU...", flush=True)
    projections = torch.load(proj_file, map_location="cpu", weights_only=True)
    embed_weights = projections["embed_tokens"].to(device=device, dtype=torch.bfloat16)  # [248320, 5120]
    lm_head_weights = projections["lm_head"].to(device=device, dtype=torch.bfloat16)      # [248320, 5120]

    # Model configuration matching official z-lab/Qwen3.6-27B-DFlash
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
    )

    drafter = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
    drafter.train()

    optimizer = bnb.optim.AdamW8bit(drafter.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Trainable drafter parameters: {sum(p.numel() for p in drafter.parameters()):,}", flush=True)
    print(f"Total Dedicated VRAM Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB (100% on-die GDDR6X)", flush=True)

    dataset = ShardDataset(args.features_dir)

    k = args.block_size
    mask_id = config.mask_token_id
    step = 0
    t0 = time.time()

    print("=== Training Loop Active (Zero Spill, Full Speed) ===", flush=True)

    while step < args.max_steps:
        sample = dataset.sample()
        tokens = sample["tokens"]  # [S]
        hidden = sample["hidden"]  # [S, 5 * 5120]
        S = tokens.shape[0]

        if S < k + 2:
            continue

        num_blocks = (S - 1) // k
        if num_blocks <= 0:
            continue

        # Choose a random anchor position in sequence
        start = random.randint(0, (num_blocks - 1)) * k
        block_tok = tokens[start:start + k].unsqueeze(0).to(device)  # [1, k]
        h_ctx = hidden[start:start + 1].unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, 1, 25600]

        # Mask tokens [t0, MASK, MASK, ...]
        masked_tok = block_tok.clone()
        masked_tok[:, 1:] = mask_id

        # Lookup embeddings
        draft_embeds = F.embedding(masked_tok, embed_weights)  # [1, k, 5120]

        # Forward pass through drafter
        drafter_out = drafter(draft_embeds, h_ctx)  # [1, k, 5120]
        
        # Linear projection to vocabulary
        draft_logits = F.linear(drafter_out[:, 1:], lm_head_weights)  # [1, k-1, 248320]

        loss = F.cross_entropy(
            draft_logits.reshape(-1, config.vocab_size),
            block_tok[:, 1:].reshape(-1),
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0
            print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f} | Speed: {rate:.1f} steps/s | Elapsed: {elapsed:.1f}s", flush=True)

    print("=== Phase 2 Complete! Exporting Official DFlash Safetensors ===", flush=True)
    drafter.export_mlx_safetensors(args.output_dir)
    print(f"DFlash Drafter successfully saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
