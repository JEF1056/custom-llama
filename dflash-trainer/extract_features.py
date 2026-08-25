#!/usr/bin/env python3
"""
Phase 1: High-Throughput Zero-Leak Multi-Language & Reasoning Feature Extraction for DFlash 2.

Features:
- DFlash 2 Target Layers: [5, 19, 33, 47, 61]
- Zstandard Level 3 Compression directly to disk
- Persistent Fixed Validation Set extraction (--fixed-val-dir)
"""

import os
import sys
import gc
import ctypes
import queue
import threading
import time
import argparse
import itertools
import io
import zstandard

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers.modeling_utils as mu

mu.caching_allocator_warmup = lambda *args, **kwargs: None

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def purge_system_memory():
    """Purges unreferenced memory from both PyTorch allocator and Linux glibc heap."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Language & Reasoning Feature Extraction for DFlash 2")
    parser.add_argument("--model-id", type=str, default="/models/qwen38-src/hf_safetensors")
    parser.add_argument("--output-dir", type=str, default="/workspace-data/features")
    parser.add_argument("--fixed-val-dir", type=str, default="/workspace-data/fixed_val_features", help="Directory for permanent fixed validation set")
    parser.add_argument("--cache-dir", type=str, default="/workspace-data/dataset-cache")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--skip-samples", type=int, default=0, help="Number of initial sequences to skip for iterative multi-round training")
    parser.add_argument("--chunk-size", type=int, default=150)
    parser.add_argument("--seq-len", type=int, default=2064)
    return parser.parse_args()


def build_interleaved_stream(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN", None)
    print(f"[Prompts] Streaming multi-language & reasoning datasets (Auth: {'ENABLED' if token else 'ANONYMOUS'}, Cache: {cache_dir})...", flush=True)

    def load_stream(repo_name, *args, **kwargs):
        for attempt in range(1, 6):
            try:
                return load_dataset(repo_name, *args, streaming=True, cache_dir=cache_dir, token=token, **kwargs)
            except Exception as e:
                print(f"Attempt {attempt}/5: Error streaming {repo_name} ({e}), retrying in {attempt * 3}s...", flush=True)
                time.sleep(attempt * 3)
        return load_dataset(repo_name, *args, streaming=True, cache_dir=cache_dir, **kwargs)

    cf_ds = load_stream("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    evol_ds = load_stream("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    mag_ds = load_stream("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    math_ds = load_stream("openai/gsm8k", "main", split="train")
    chat_ds = load_stream("HuggingFaceH4/ultrachat_200k", split="train_sft")

    def generator():
        cf_iter = iter(cf_ds)
        evol_iter = iter(evol_ds)
        mag_iter = iter(mag_ds)
        math_iter = iter(math_ds)
        chat_iter = iter(chat_ds)

        pattern = [
            "code_feedback", "evol_code", "magicoder", "code_feedback", "math",
            "code_feedback", "evol_code", "chat", "code_feedback", "magicoder"
        ]
        for domain in itertools.cycle(pattern):
            try:
                if domain == "code_feedback":
                    item = next(cf_iter)
                    q = item.get("query", "").strip()
                    a = item.get("answer", "").strip()
                    if q and a:
                        yield [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": f"<think>\nAnalyze and implement the code solution.\n</think>\n\n{a}"}
                        ]
                elif domain == "evol_code":
                    item = next(evol_iter)
                    q = item.get("instruction", "").strip()
                    a = item.get("output", "").strip()
                    if q and a:
                        yield [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": f"<think>\nPlan the architecture and implement the required system component.\n</think>\n\n{a}"}
                        ]
                elif domain == "magicoder":
                    item = next(mag_iter)
                    prob = item.get("problem", "").strip()
                    sol = item.get("solution", "").strip()
                    if prob and sol:
                        yield [
                            {"role": "user", "content": prob},
                            {"role": "assistant", "content": f"<think>\nAnalyze algorithmic requirements and edge cases.\n</think>\n\n{sol}"}
                        ]
                elif domain == "math":
                    item = next(math_iter)
                    q = item.get("question", "").strip()
                    a = item.get("answer", "").strip()
                    if q and a:
                        yield [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": f"<think>\nWork through the mathematical solution step by step.\n</think>\n\n{a}"}
                        ]
                elif domain == "chat":
                    item = next(chat_iter)
                    msgs = item.get("messages", [])
                    if msgs and len(msgs) >= 2:
                        yield msgs
            except StopIteration:
                break
            except Exception:
                continue

    return generator()


class BackgroundPrefetcher(threading.Thread):
    """Prefetches, tokenizes, and packs continuous sequences up to seq_len."""
    def __init__(self, stream_gen, tokenizer, seq_len: int = 2064, max_queue_size: int = 16, skip_samples: int = 0):
        super().__init__(daemon=True)
        self.stream_gen = stream_gen
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.skip_samples = skip_samples
        self.stopped = False

    def run(self):
        token_buffer = []
        skipped = 0

        if self.skip_samples > 0:
            print(f"[Prefetcher] Fast-forwarding {self.skip_samples} prior sequences...", flush=True)

        for conversation in self.stream_gen:
            if self.stopped:
                break
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                continue

            tok_ids = self.tokenizer.encode(formatted_text, add_special_tokens=True)
            token_buffer.extend(tok_ids)

            while len(token_buffer) >= self.seq_len:
                if self.stopped:
                    break
                seq = token_buffer[: self.seq_len]
                token_buffer = token_buffer[self.seq_len :]

                if skipped < self.skip_samples:
                    skipped += 1
                    continue
                elif skipped == self.skip_samples and self.skip_samples > 0:
                    print(f"[Prefetcher] Fast-forward complete ({self.skip_samples} skipped)! Starting streaming of new data...", flush=True)
                    skipped += 1

                t_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
                self.queue.put(t_tensor, block=True)

        self.queue.put(None)

    def stop(self):
        self.stopped = True
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass


def extract_slice_to_dir(target_model, tokenizer, target_layers, output_dir, cache_dir, num_samples, skip_samples, chunk_size, seq_len, device, label="Training"):
    os.makedirs(output_dir, exist_ok=True)
    hidden_states = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook_fn

    model_layers = target_model.model.layers if hasattr(target_model, "model") else target_model.layers
    hooks = [model_layers[l_idx].register_forward_hook(make_hook(l_idx)) for l_idx in target_layers]

    stream = build_interleaved_stream(cache_dir)
    prefetcher = BackgroundPrefetcher(stream, tokenizer, seq_len, max_queue_size=16, skip_samples=skip_samples)
    prefetcher.start()

    sample_count = 0
    shard_idx = 0
    in_shard_count = 0
    cctx = zstandard.ZstdCompressor(level=3)
    t0 = time.time()

    def open_shard(s_idx):
        path = os.path.join(output_dir, f"shard_{s_idx:04d}.pt.zst")
        f = open(path, "wb")
        writer = cctx.stream_writer(f)
        return f, writer

    print(f"=== [{label}] Feature Extraction Active ({num_samples} samples, {chunk_size}/shard, Zstd-Compressed BF16) ===", flush=True)

    shard_file, shard_writer = open_shard(shard_idx)

    while sample_count < num_samples:
        tokens_cpu = prefetcher.queue.get()
        if tokens_cpu is None:
            break

        tokens = tokens_cpu.to(device, non_blocking=True)
        hidden_states.clear()
        with torch.no_grad():
            if hasattr(target_model, "model"):
                _ = target_model.model(tokens)
            else:
                _ = target_model(tokens)
            concat_h = torch.cat([hidden_states[l] for l in target_layers], dim=-1).cpu().to(torch.bfloat16)
            hidden_states.clear()

        sample_dict = {
            "tokens": tokens_cpu.squeeze(0).to(torch.int32),
            "hidden": concat_h.squeeze(0).to(torch.bfloat16),
        }
        buf = io.BytesIO()
        torch.save(sample_dict, buf)
        raw_bytes = buf.getvalue()
        shard_writer.write(len(raw_bytes).to_bytes(4, "big") + raw_bytes)

        del sample_dict, buf, raw_bytes, tokens_cpu, concat_h
        in_shard_count += 1
        sample_count += 1

        if sample_count % 50 == 0 or sample_count == 1:
            elapsed = time.time() - t0
            rate = sample_count / elapsed if elapsed > 0 else 0
            print(f"[{label}] Extracted {sample_count}/{num_samples} sequences (Shard {shard_idx:04d}: {in_shard_count}/{chunk_size}) | Rate: {rate:.1f} seq/s | Elapsed: {elapsed:.1f}s", flush=True)
            purge_system_memory()

        if in_shard_count >= chunk_size:
            shard_writer.flush()
            shard_writer.close()
            shard_file.close()
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt.zst")
            print(f"[Zstd Shard Saved] {shard_path} finalized ({in_shard_count} samples).", flush=True)
            purge_system_memory()
            shard_idx += 1
            in_shard_count = 0
            if sample_count < num_samples:
                shard_file, shard_writer = open_shard(shard_idx)

    if in_shard_count > 0:
        shard_writer.flush()
        shard_writer.close()
        shard_file.close()
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt.zst")
        print(f"[Zstd Shard Saved] Final {shard_path} finalized ({in_shard_count} samples).", flush=True)
        purge_system_memory()

    prefetcher.stop()
    for h in hooks:
        h.remove()
    return sample_count


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 1: Zero-Leak Feature Extraction for DFlash 2 on {device.upper()} ===", flush=True)
    print(f"Target Model:  {args.model_id}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Fixed Val Dir: {args.fixed_val_dir}")
    print(f"Num Samples:   {args.num_samples} | Chunk Size: {args.chunk_size} | Seq Len: {args.seq_len}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fixed_val_dir, exist_ok=True)

    print("Loading tokenizer and target model in 4-bit NF4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    purge_system_memory()
    print(f"Target model loaded! VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    target_layers = [5, 19, 33, 47, 61]

    # Save embed_tokens and lm_head projection weights
    print("Saving embed_tokens and lm_head projection weights for Phase 2...", flush=True)
    with torch.no_grad():
        embed_weights = target_model.get_input_embeddings().weight.data.cpu().to(torch.bfloat16)
        lm_head_weights = target_model.get_output_embeddings().weight.data.cpu().to(torch.bfloat16)
    
    proj_dict = {"embed_tokens": embed_weights, "lm_head": lm_head_weights}
    torch.save(proj_dict, os.path.join(args.output_dir, "projection_weights.pt"))
    torch.save(proj_dict, os.path.join(args.fixed_val_dir, "projection_weights.pt"))
    del embed_weights, lm_head_weights, proj_dict
    purge_system_memory()

    # Check if Fixed Validation Benchmark is already extracted
    existing_val_shards = [f for f in os.listdir(args.fixed_val_dir) if f.startswith("shard_") and f.endswith(".pt.zst")]
    if len(existing_val_shards) < 2:
        print(f"\n>>> Extracting persistent fixed validation benchmark (300 samples) to {args.fixed_val_dir}...", flush=True)
        extract_slice_to_dir(
            target_model=target_model,
            tokenizer=tokenizer,
            target_layers=target_layers,
            output_dir=args.fixed_val_dir,
            cache_dir=args.cache_dir,
            num_samples=300,
            skip_samples=0,
            chunk_size=150,
            seq_len=args.seq_len,
            device=device,
            label="Fixed-Val Benchmark"
        )
        print(f">>> Persistent fixed validation benchmark successfully created!\n", flush=True)
    else:
        print(f">>> Persistent fixed validation benchmark already ready ({len(existing_val_shards)} shards found in {args.fixed_val_dir}).", flush=True)

    # Extract Training Shards
    # Note: offset training skip by 300 to avoid overlapping with fixed val set
    effective_skip = args.skip_samples + 300
    extract_slice_to_dir(
        target_model=target_model,
        tokenizer=tokenizer,
        target_layers=target_layers,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        skip_samples=effective_skip,
        chunk_size=args.chunk_size,
        seq_len=args.seq_len,
        device=device,
        label=f"Round-Training (Skip: {effective_skip})"
    )

    print(f"=== Phase 1 Extraction Complete! ===", flush=True)


if __name__ == "__main__":
    main()
