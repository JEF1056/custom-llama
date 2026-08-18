#!/usr/bin/env python3
"""
Phase 2: Ultra-Fast Dedicated DFlash Drafter Training with Full Prefix Context Attention.

Exact Architectural Parity with Official DFlash (arXiv:2602.06036):
1. Target Context is the sequence of hidden states up to the bonus token: `h_ctx = hidden[max(0, start - ctx_len) : start]`.
2. Proposal block is `[tokens[start], MASK, MASK, MASK]`, predicting `tokens[start+1 : start+k]`.
3. Standard PyTorch DataLoader with Multi-Worker prefetching on RTX 3090.
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
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per forward step")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch size = 8 * 4 = 32)")
    parser.add_argument("--block-size", type=int, default=16, help="DFlash draft block size (15 prediction targets)")
    parser.add_argument("--ctx-len", type=int, default=4086, help="Target context sequence length (4k prefix history)")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate for 1.7B drafter")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient norm clipping threshold")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing (keeps VRAM strictly under 14.5 GB)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile kernel fusion")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1000, help="Evaluate on held-out unseen shards every N steps")
    parser.add_argument("--val-steps", type=int, default=50, help="Number of validation batches to evaluate")
    return parser.parse_args()


class FullContextBatchedDataset(IterableDataset):
    """Extracts draft blocks alongside full causal prefix context with multi-worker prefetching."""
    def __init__(self, shard_files: list, batch_size: int = 64, block_size: int = 16, ctx_len: int = 16, is_val: bool = False):
        super().__init__()
        self.shard_files = sorted(shard_files)
        if not self.shard_files:
            raise RuntimeError("No shard files provided to dataset")
        mode = "Validation (Held-Out)" if is_val else "Training"
        print(f"[Dataset] {mode}: {len(self.shard_files)} shards (ctx_len={ctx_len}, block_size={block_size})", flush=True)
        self.batch_size = batch_size
        self.block_size = block_size
        self.ctx_len = ctx_len
        self.is_val = is_val

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Partition shards across DataLoader background worker processes
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards_for_worker = [self.shard_files[i] for i in range(len(self.shard_files)) if i % num_workers == worker_id]
        else:
            shards_for_worker = list(self.shard_files)

        k = self.block_size
        C = self.ctx_len
        bsz = self.batch_size

        while True:
            shuffled_shards = list(shards_for_worker)
            if not self.is_val:
                random.shuffle(shuffled_shards)

            for sf in shuffled_shards:
                try:
                    samples = torch.load(sf, map_location="cpu", weights_only=True, mmap=True)
                except Exception as e:
                    print(f"[DataLoaderError] {sf}: {e}", flush=True)
                    continue

                indices = list(range(len(samples)))
                if not self.is_val:
                    random.shuffle(indices)

                cur_b = []
                cur_h = []
                cur_cp = []
                cur_pp = []

                for idx in indices:
                    s = samples[idx]
                    tokens = s["tokens"]
                    hidden = s["hidden"]
                    S = tokens.shape[0]

                    if S < k + 2:
                        continue

                    # Sample draft block anchors across sequence (Random Anchors for train, Strided for val)
                    min_start = max(1, min(16, S - k - 1))
                    max_starts = S - k - 1
                    if max_starts < min_start:
                        continue

                    if not self.is_val:
                        all_possible_starts = list(range(min_start, max_starts + 1))
                        num_anchors = max(1, min(len(all_possible_starts), max(1, (S - 1) // k)))
                        start_positions = random.sample(all_possible_starts, num_anchors)
                    else:
                        start_positions = list(range(min_start, max_starts + 1, k))

                    for start in start_positions:
                        cur_b.append(tokens[start : start + k])
                        cur_pp.append(torch.arange(start, start + k, dtype=torch.long))

                        ctx_start = max(0, start - C)
                        h_slice = hidden[ctx_start : start].to(torch.bfloat16)
                        L_avail = h_slice.shape[0]

                        if L_avail == C:
                            cur_h.append(h_slice)
                            cur_cp.append(torch.arange(start - C, start, dtype=torch.long))
                        else:
                            h_padded = torch.zeros((C, h_slice.shape[1]), dtype=torch.bfloat16)
                            h_padded[C - L_avail :] = h_slice
                            cp_padded = torch.zeros(C, dtype=torch.long)
                            cp_padded[C - L_avail :] = torch.arange(ctx_start, start, dtype=torch.long)
                            cur_h.append(h_padded)
                            cur_cp.append(cp_padded)

                        if len(cur_b) == bsz:
                            b_tensor = torch.stack(cur_b, dim=0)
                            h_tensor = torch.stack(cur_h, dim=0)
                            cp_tensor = torch.stack(cur_cp, dim=0)
                            pp_tensor = torch.stack(cur_pp, dim=0)
                            cur_b.clear()
                            cur_h.clear()
                            cur_cp.clear()
                            cur_pp.clear()
                            yield b_tensor, h_tensor, cp_tensor, pp_tensor

                del cur_b, cur_h, cur_cp, cur_pp, samples, indices
                purge_ram()


def get_lr(step: int, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def evaluate_val_set(drafter, val_shards, embed_weights, lm_head_weights, config, device, args):
    """Evaluates cross-entropy loss and top-1 accuracy on strictly held-out unseen dataset on demand."""
    drafter.eval()
    k = config.block_size
    mask_id = config.mask_token_id
    total_loss = 0.0
    total_acc_first = 0.0
    total_acc_all = 0.0
    num_batches = args.val_steps
    
    # Instantiate on demand with mmap streaming and purge immediately
    val_dataset = FullContextBatchedDataset(val_shards, batch_size=args.batch_size, block_size=args.block_size, ctx_len=args.ctx_len, is_val=True)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0, pin_memory=True)
    val_iter = iter(val_loader)

    with torch.no_grad():
        for _ in range(num_batches):
            block_tok, h_ctx, ctx_pos, prop_pos = next(val_iter)
            block_tok = block_tok.to(device, non_blocking=True)
            h_ctx = h_ctx.to(device, non_blocking=True)
            ctx_pos = ctx_pos.to(device, non_blocking=True)
            prop_pos = prop_pos.to(device, non_blocking=True)

            masked_tok = block_tok.clone()
            masked_tok[:, 1:] = mask_id

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                draft_embeds = F.embedding(masked_tok, embed_weights)
                drafter_out = drafter(draft_embeds, h_ctx, ctx_pos, prop_pos)
                draft_logits = F.linear(drafter_out[:, 1:], lm_head_weights)
                gamma = 0.85
                pos_weights = torch.tensor([gamma ** i for i in range(k - 1)], device=device, dtype=torch.bfloat16)
                pos_weights = pos_weights / pos_weights.sum()

                loss_unreduced = F.cross_entropy(
                    draft_logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                ).view(-1, k - 1)
                loss = (loss_unreduced * pos_weights).sum(dim=-1).mean()

                preds = torch.argmax(draft_logits, dim=-1)
                acc_first = (preds[:, 0] == targets[:, 0]).float().mean().item()
                acc_all = (preds == targets).float().mean().item()

                total_loss += loss.item()
                total_acc_first += acc_first
                total_acc_all += acc_all

    del val_iter, val_loader, val_dataset
    purge_ram()
    drafter.train()
    return (total_loss / num_batches), (total_acc_first / num_batches * 100), (total_acc_all / num_batches * 100)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 2: Ultra-Fast Dedicated Drafter Training on {device.upper()} ===", flush=True)
    print(f"Features Dir: {args.features_dir}", flush=True)
    print(f"Output Dir:   {args.output_dir}", flush=True)
    print(f"Batch Size:   {args.batch_size} | Block Size: {args.block_size} | Context Len: {args.ctx_len} | LR: {args.lr:.1e} | Max Steps: {args.max_steps}", flush=True)

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
    if args.gradient_checkpointing:
        drafter.gradient_checkpointing_enable()
        print("Gradient Checkpointing: ENABLED (decoder layers memory-optimized)", flush=True)
    else:
        print("Gradient Checkpointing: DISABLED (full activation caching for maximum speed)", flush=True)

    raw_drafter = drafter
    if args.compile:
        try:
            print("torch.compile: JIT compiling drafter with PyTorch 2.6 Inductor...", flush=True)
            drafter = torch.compile(drafter)
            print("torch.compile: ENABLED (fused kernels active)", flush=True)
        except Exception as e:
            print(f"torch.compile note: {e}, running in eager mode.", flush=True)
            drafter = raw_drafter

    drafter.train()

    optimizer = bnb.optim.AdamW8bit(drafter.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Trainable drafter parameters: {sum(p.numel() for p in drafter.parameters()):,}", flush=True)
    print(f"Total Dedicated VRAM Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB (100% on-die GDDR6X)", flush=True)

    all_shards = sorted(glob.glob(os.path.join(args.features_dir, "shard_*.pt")))
    if len(all_shards) < 2:
        train_shards = all_shards
        val_shards = all_shards
    else:
        # Strictly hold out the last 2 shards (shard_0018, shard_0019) for validation only
        val_shards = all_shards[-2:]
        train_shards = all_shards[:-2]

    train_dataset = FullContextBatchedDataset(train_shards, batch_size=args.batch_size, block_size=args.block_size, ctx_len=args.ctx_len, is_val=False)
    
    # Zero-copy in-process streaming with mmap to avoid shared memory IPC bus errors
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
    )
    train_iter = iter(train_loader)

    k = args.block_size
    mask_id = config.mask_token_id
    step = 0
    t0 = time.time()
    loss_acc = 0.0
    acc_first_acc = 0.0
    acc_all_acc = 0.0
    eval_count = 0

    # Decaying position weights: prioritize earlier tokens (which drive speculative acceptance)
    gamma = 0.85
    pos_weights = torch.tensor([gamma ** i for i in range(k - 1)], device=device, dtype=torch.bfloat16)
    pos_weights = pos_weights / pos_weights.sum()

    print(f"=== Training Loop Active (Train Shards: {len(train_shards)}, Held-Out Val Shards: {len(val_shards)}) ===", flush=True)

    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        cur_lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        # Gradient accumulation inner loop
        accum_loss = 0.0
        accum_acc_first = 0.0
        accum_acc_all = 0.0

        for _ in range(args.grad_accum_steps):
            block_tok, h_ctx, ctx_pos, prop_pos = next(train_iter)
            block_tok = block_tok.to(device, non_blocking=True)
            h_ctx = h_ctx.to(device, non_blocking=True)
            ctx_pos = ctx_pos.to(device, non_blocking=True)
            prop_pos = prop_pos.to(device, non_blocking=True)

            # Mask proposal positions 1..k-1 (position 0 is the bonus token)
            masked_tok = block_tok.clone()
            masked_tok[:, 1:] = mask_id

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Lookup embeddings
                draft_embeds = F.embedding(masked_tok, embed_weights)  # [B, k, 5120]

                # Forward pass through drafter with RoPE position embeddings across [ctx_len + k]
                drafter_out = drafter(draft_embeds, h_ctx, ctx_pos, prop_pos)  # [B, k, 5120]
                
                # Single fused GEMM projection on CUDA Tensor Cores
                draft_logits = F.linear(drafter_out[:, 1:], lm_head_weights)  # [B, k-1, 248320]
                targets = block_tok[:, 1:].contiguous()                       # [B, k-1]

                loss_unreduced = F.cross_entropy(
                    draft_logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                ).view(-1, k - 1)
                loss = (loss_unreduced * pos_weights).sum(dim=-1).mean()
                scaled_loss = loss / args.grad_accum_steps

            scaled_loss.backward()
            accum_loss += loss.item() / args.grad_accum_steps

            with torch.no_grad():
                preds = torch.argmax(draft_logits, dim=-1)
                accum_acc_first += (preds[:, 0] == targets[:, 0]).float().mean().item() * 100 / args.grad_accum_steps
                accum_acc_all += (preds == targets).float().mean().item() * 100 / args.grad_accum_steps

        # Gradient clipping to prevent optimization collapse
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), max_norm=args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        loss_acc += accum_loss
        acc_first_acc += accum_acc_first
        acc_all_acc += accum_acc_all
        eval_count += 1
        
        # Step training progress log (smoothed over eval_interval)
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0
            avg_loss = loss_acc / eval_count
            avg_acc_first = acc_first_acc / eval_count
            avg_acc_all = acc_all_acc / eval_count
            
            print(f"Step {step}/{args.max_steps} | Train Loss: {avg_loss:.4f} | Train Acc (First): {avg_acc_first:.1f}% | Train Acc (All {k-1}): {avg_acc_all:.1f}% | LR: {cur_lr:.2e} | Speed: {rate:.1f} steps/s", flush=True)
            loss_acc = 0.0
            acc_first_acc = 0.0
            acc_all_acc = 0.0
            eval_count = 0

        # Evaluation step on strictly held-out validation set
        if step % args.val_interval == 0:
            val_loss, val_acc_first, val_acc_all = evaluate_val_set(
                drafter, val_shards, embed_weights, lm_head_weights, config, device, args
            )
            print(f"\n=======================================================", flush=True)
            print(f" [HELD-OUT EVAL @ STEP {step}] Val Loss: {val_loss:.4f} | Val Acc (First): {val_acc_first:.1f}% | Val Acc (All {k-1}): {val_acc_all:.1f}%", flush=True)
            print(f"=======================================================\n", flush=True)

    print("=== Phase 2 Complete! Exporting Official DFlash Safetensors ===", flush=True)
    raw_drafter.export_mlx_safetensors(args.output_dir)
    print(f"DFlash Drafter successfully saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
