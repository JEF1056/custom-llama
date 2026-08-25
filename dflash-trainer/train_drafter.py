#!/usr/bin/env python3
"""
Phase 2: Ultra-Fast Dedicated DFlash 2 Drafter Training with Full Prefix Context Attention.

Exact Architectural Parity with Official DFlash 2 (z-lab/Qwen3.8-27B-DFlash2):
1. Target Context is the sequence of hidden states up to bonus token from layers [5, 19, 33, 47, 61].
2. 2-Tap Dynamic Convolutions with kernel size 2 across proposal blocks.
3. Candidate Path Selector Codebooks for transition scoring.
4. Persistent Fixed Validation Dataset support (--fixed-val-dir).
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
import io
import zstandard
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import bitsandbytes as bnb
from model import DFlash2Config, DFlash2DraftModel


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
    parser = argparse.ArgumentParser(description="Train DFlash 2 drafter on extracted features")
    parser.add_argument("--features-dir", type=str, default="/workspace-data/features")
    parser.add_argument("--fixed-val-dir", type=str, default="/workspace-data/fixed_val_features", help="Persistent fixed validation set directory")
    parser.add_argument("--output-dir", type=str, default="/output/Qwen3.8-27B-heretic-dflash")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per forward step")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch size = 8 * 4 = 32)")
    parser.add_argument("--block-size", type=int, default=8, help="DFlash 2 draft block size (default 8, 7 prediction targets)")
    parser.add_argument("--ctx-len", type=int, default=2048, help="Target context sequence length (2k prefix history matching z-lab sliding window)")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate for 1.7B drafter")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient norm clipping threshold")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing (keeps VRAM strictly under 14.5 GB)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile kernel fusion")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=500, help="Evaluate on held-out benchmark set every N steps")
    parser.add_argument("--pretrained-drafter", type=str, default="z-lab/Qwen3.8-27B-DFlash2", help="Pretrained drafter model ID or local directory to initialize from")
    parser.add_argument("--val-steps", type=int, default=40, help="Number of validation batches to evaluate")
    return parser.parse_args()


class FullContextBatchedDataset(IterableDataset):
    """Extracts draft blocks alongside full causal prefix context with multi-worker prefetching."""
    def __init__(self, shard_files: list, batch_size: int = 64, block_size: int = 8, ctx_len: int = 2048, is_val: bool = False):
        super().__init__()
        self.shard_files = sorted(shard_files)
        if not self.shard_files:
            raise RuntimeError("No shard files provided to dataset")
        mode = "Validation (Persistent Benchmark)" if is_val else "Training"
        print(f"[Dataset] {mode}: {len(self.shard_files)} shards (ctx_len={ctx_len}, block_size={block_size})", flush=True)
        self.batch_size = batch_size
        self.block_size = block_size
        self.ctx_len = ctx_len
        self.is_val = is_val

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
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

            dctx = zstandard.ZstdDecompressor()
            for sf in shuffled_shards:
                if sf.endswith(".zst"):
                    def sample_stream(filepath):
                        try:
                            with open(filepath, "rb") as f:
                                with dctx.stream_reader(f) as reader:
                                    while True:
                                        len_b = reader.read(4)
                                        if not len_b or len(len_b) < 4:
                                            break
                                        l = int.from_bytes(len_b, "big")
                                        d = reader.read(l)
                                        yield torch.load(io.BytesIO(d), weights_only=True)
                        except Exception as e:
                            print(f"[DataLoaderError] {filepath}: {e}", flush=True)

                    cur_b = []
                    cur_h = []
                    cur_cp = []
                    cur_pp = []

                    for s in sample_stream(sf):
                        tokens = s["tokens"]
                        hidden = s["hidden"]
                        S_cur = tokens.shape[0]

                        if S_cur < k + 2:
                            continue

                        min_start = max(1, min(16, S_cur - k - 1))
                        max_starts = S_cur - k - 1
                        if max_starts < min_start:
                            continue

                        if not self.is_val:
                            all_possible_starts = list(range(min_start, max_starts + 1))
                            num_anchors = max(1, min(len(all_possible_starts), max(1, (S_cur - 1) // k)))
                            start_positions = random.sample(all_possible_starts, num_anchors)
                        else:
                            start_positions = list(range(min_start, max_starts + 1, k))

                        for start in start_positions:
                            cur_b.append(tokens[start : start + k].to(torch.long))
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

                    del cur_b, cur_h, cur_cp, cur_pp
                    purge_ram()
                else:
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
                            cur_b.append(tokens[start : start + k].to(torch.long))
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
                drafter_out, selector_feat = drafter(draft_embeds, h_ctx, ctx_pos, prop_pos)
                draft_logits = F.linear(drafter_out[:, 1:], lm_head_weights)
                targets = block_tok[:, 1:].contiguous()
                
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
    print(f"=== Phase 2: Ultra-Fast Dedicated DFlash 2 Training on {device.upper()} ===", flush=True)
    print(f"Features Dir: {args.features_dir}", flush=True)
    print(f"Fixed Val Dir:{args.fixed_val_dir}", flush=True)
    print(f"Output Dir:   {args.output_dir}", flush=True)
    print(f"Batch Size:   {args.batch_size} | Block Size: {args.block_size} | Context Len: {args.ctx_len} | LR: {args.lr:.1e} | Max Steps: {args.max_steps}", flush=True)

    proj_file = os.path.join(args.features_dir, "projection_weights.pt")
    if not os.path.exists(proj_file):
        # Fallback to fixed_val_dir if projection_weights.pt is stored there
        alt_proj = os.path.join(args.fixed_val_dir, "projection_weights.pt")
        if os.path.exists(alt_proj):
            proj_file = alt_proj
        else:
            print(f"Error: Projection weights {proj_file} not found. Run Phase 1 first.", flush=True)
            sys.exit(1)

    print("Loading embed_tokens and lm_head projection weights onto GPU...", flush=True)
    projections = torch.load(proj_file, map_location="cpu", weights_only=True)
    embed_weights = projections["embed_tokens"].to(device=device, dtype=torch.bfloat16)  # [248320, 5120]
    lm_head_weights = projections["lm_head"].to(device=device, dtype=torch.bfloat16)      # [248320, 5120]
    del projections
    purge_ram()

    config = DFlash2Config(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        target_layer_ids=[5, 19, 33, 47, 61],
        num_target_layers=64,
        block_size=args.block_size,
        mask_token_id=248070,
        vocab_size=248320,
        sliding_window=2048,
        rope_theta=10000000.0,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
    )

    drafter = DFlash2DraftModel(config).to(device=device, dtype=torch.bfloat16)

    if args.pretrained_drafter:
        print(f"Loading pretrained drafter weights from '{args.pretrained_drafter}'...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            if os.path.exists(args.pretrained_drafter):
                model_file = os.path.join(args.pretrained_drafter, "model.safetensors")
            else:
                model_file = hf_hub_download(repo_id=args.pretrained_drafter, filename="model.safetensors")

            state_dict = load_file(model_file)
            incompatible = drafter.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded successfully! Missing keys: {len(incompatible.missing_keys)}, Unexpected keys: {len(incompatible.unexpected_keys)}", flush=True)
        except Exception as e:
            print(f"Warning: Could not load pretrained drafter ({e}), initializing from scratch.", flush=True)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing across all DFlash 2 layers...", flush=True)
        drafter.gradient_checkpointing_enable()

    train_shards = sorted(glob.glob(os.path.join(args.features_dir, "shard_*.pt*")))
    val_shards = sorted(glob.glob(os.path.join(args.fixed_val_dir, "shard_*.pt*")))

    if not val_shards:
        print(f"Warning: No shards found in fixed_val_dir {args.fixed_val_dir}. Falling back to last train shards.", flush=True)
        if len(train_shards) >= 2:
            num_val = max(1, min(2, len(train_shards) // 10))
            val_shards = train_shards[-num_val:]
            train_shards = train_shards[:-num_val]
        else:
            val_shards = train_shards

    print(f"Train Shards: {len(train_shards)} | Fixed Validation Shards: {len(val_shards)}", flush=True)

    train_dataset = FullContextBatchedDataset(
        shard_files=train_shards,
        batch_size=args.batch_size,
        block_size=args.block_size,
        ctx_len=args.ctx_len,
        is_val=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2, pin_memory=True)
    train_iter = iter(train_loader)

    optimizer = bnb.optim.AdamW8bit(drafter.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    drafter.train()
    step = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_acc_first = 0.0
    accum_acc_all = 0.0
    k = config.block_size
    mask_id = config.mask_token_id

    gamma = 0.85
    pos_weights = torch.tensor([gamma ** i for i in range(k - 1)], device=device, dtype=torch.bfloat16)
    pos_weights = pos_weights / pos_weights.sum()

    print(f"\n=== Commencing DFlash 2 Phase 2 Drafter Training ===", flush=True)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        cur_step_loss = 0.0
        cur_step_acc_first = 0.0
        cur_step_acc_all = 0.0

        for micro_step in range(args.grad_accum_steps):
            try:
                block_tok, h_ctx, ctx_pos, prop_pos = next(train_iter)
            except Exception as e:
                print(f"[DatasetExhausted/Error] {e}, re-initializing iterator...", flush=True)
                train_iter = iter(train_loader)
                block_tok, h_ctx, ctx_pos, prop_pos = next(train_iter)

            block_tok = block_tok.to(device, non_blocking=True)
            h_ctx = h_ctx.to(device, non_blocking=True)
            ctx_pos = ctx_pos.to(device, non_blocking=True)
            prop_pos = prop_pos.to(device, non_blocking=True)

            masked_tok = block_tok.clone()
            masked_tok[:, 1:] = mask_id

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                draft_embeds = F.embedding(masked_tok, embed_weights)
                drafter_out, selector_feat = drafter(draft_embeds, h_ctx, ctx_pos, prop_pos)
                draft_logits = F.linear(drafter_out[:, 1:], lm_head_weights)
                targets = block_tok[:, 1:].contiguous()

                loss_unreduced = F.cross_entropy(
                    draft_logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                ).view(-1, k - 1)
                loss_ce = (loss_unreduced * pos_weights).sum(dim=-1).mean()

                # DFlash 2 Candidate Selector Loss
                if k > 2:
                    # selector_feat: [bsz, k, rank]
                    # predecessor is the actual token that generated the next candidate
                    # The token sequence for the block is block_tok. 
                    # block_tok[:, 0] is the anchor, block_tok[:, 1:] are the drafted tokens.
                    # For predicting targets[:, 1:] (which is block_tok[:, 2:]), the predecessor is block_tok[:, 1:-1].
                    predecessor_ids = block_tok[:, 1:-1].contiguous() # [bsz, k-2]
                    pred_emb = drafter.candidate_selector.predecessor_codebook[predecessor_ids] # [bsz, k-2, rank]
                    
                    # Compute the transition features
                    transition_feat = pred_emb * selector_feat[:, 1:-1] # [bsz, k-2, rank]
                    
                    # Compute logits over vocab
                    selector_logits = transition_feat @ drafter.candidate_selector.successor_codebook.T # [bsz, k-2, vocab_size]
                    
                    loss_selector = F.cross_entropy(
                        selector_logits.reshape(-1, config.vocab_size),
                        targets[:, 1:].reshape(-1),
                    )
                    loss = loss_ce + 0.1 * loss_selector
                else:
                    loss = loss_ce
                loss_scaled = loss / args.grad_accum_steps

            loss_scaled.backward()

            cur_step_loss += loss.item() / args.grad_accum_steps
            with torch.no_grad():
                preds = torch.argmax(draft_logits, dim=-1)
                acc_first = (preds[:, 0] == targets[:, 0]).float().mean().item()
                acc_all = (preds == targets).float().mean().item()
                cur_step_acc_first += acc_first / args.grad_accum_steps
                cur_step_acc_all += acc_all / args.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.max_grad_norm)

        cur_lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        optimizer.step()
        step += 1

        accum_loss += cur_step_loss
        accum_acc_first += cur_step_acc_first
        accum_acc_all += cur_step_acc_all

        if step % args.eval_interval == 0:
            dt = time.time() - t0
            avg_loss = accum_loss / args.eval_interval
            avg_acc_first = accum_acc_first / args.eval_interval * 100
            avg_acc_all = accum_acc_all / args.eval_interval * 100
            speed = args.eval_interval / dt
            print(
                f"Step {step}/{args.max_steps} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Acc (First): {avg_acc_first:.1f}% | "
                f"Train Acc (All {k-1}): {avg_acc_all:.1f}% | "
                f"LR: {cur_lr:.2e} | "
                f"Speed: {speed:.1f} steps/s",
                flush=True,
            )
            accum_loss = 0.0
            accum_acc_first = 0.0
            accum_acc_all = 0.0
            t0 = time.time()

        if step % args.val_interval == 0 or step == args.max_steps:
            if val_shards:
                val_loss, val_acc_first, val_acc_all = evaluate_val_set(
                    drafter=drafter,
                    val_shards=val_shards,
                    embed_weights=embed_weights,
                    lm_head_weights=lm_head_weights,
                    config=config,
                    device=device,
                    args=args,
                )
                print(f"\n=======================================================", flush=True)
                print(f" [HELD-OUT EVAL @ STEP {step}] Val Loss: {val_loss:.4f} | Val Acc (First): {val_acc_first:.1f}% | Val Acc (All {k-1}): {val_acc_all:.1f}%", flush=True)
                print(f"=======================================================\n", flush=True)

    print("\n=== Phase 2 Complete! Exporting Official DFlash 2 Safetensors ===", flush=True)
    drafter.export_mlx_safetensors(args.output_dir)
    print(f"DFlash 2 Drafter successfully saved to {args.output_dir}")
    print("=======================================================")
    print("   DFlash 2 Training Successfully Finished!")
    print(f"   Exported model in: {args.output_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
