#!/usr/bin/env python3
"""
Phase 1: High-Throughput Zero-Leak Multi-Language & Reasoning Feature Extraction.

Memory Optimizations:
- ctypes malloc_trim(0) at every 50-sequence interval and shard boundary.
- Detached CPU tensor storage without retaining autograd/hook graph closures.
- Bounded prefetch queue and asynchronous single-shard writer.
- Explicit torch.cuda.empty_cache() to return unreferenced virtual memory.
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

# Configure PyTorch memory allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers.modeling_utils as mu

# Disable aggressive transformers warmup allocation
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
    parser = argparse.ArgumentParser(description="Multi-Language & Reasoning Feature Extraction for DFlash")
    parser.add_argument("--model-id", type=str, default="/models/qwen38-src/hf_safetensors")
    parser.add_argument("--output-dir", type=str, default="/workspace-data/features")
    parser.add_argument("--cache-dir", type=str, default="/workspace-data/dataset-cache")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=4128)
    return parser.parse_args()


def build_interleaved_stream(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Prompts] Streaming multi-language & reasoning datasets (Cache: {cache_dir})...", flush=True)

    cf_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True, cache_dir=cache_dir)
    evol_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True, cache_dir=cache_dir)
    mag_ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True, cache_dir=cache_dir)
    math_ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True, cache_dir=cache_dir)
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, cache_dir=cache_dir)

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
    def __init__(self, stream, tokenizer, seq_len: int, max_queue_size: int = 16):
        super().__init__(daemon=True)
        self.stream = stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False

    def run(self):
        accum_tokens = []
        for messages in self.stream:
            if self.stopped:
                break
            if not messages:
                continue
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                toks = self.tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0]
                accum_tokens.append(toks)
                total_len = sum(t.shape[0] for t in accum_tokens)

                while total_len >= self.seq_len:
                    cat_toks = torch.cat(accum_tokens, dim=0)
                    chunk = cat_toks[: self.seq_len].unsqueeze(0)
                    rem = cat_toks[self.seq_len :]
                    accum_tokens = [rem] if rem.shape[0] > 0 else []
                    total_len = sum(t.shape[0] for t in accum_tokens)
                    self.queue.put(chunk, block=True)
            except Exception:
                continue
        self.queue.put(None)

    def stop(self):
        self.stopped = True


class BackgroundShardSaver(threading.Thread):
    """Saves shards to disk asynchronously and forces immediate OS memory release."""
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=1)
        self.stopped = False

    def run(self):
        while not self.stopped:
            item = self.queue.get()
            if item is None:
                break
            shard_path, shard_data = item
            try:
                torch.save(shard_data, shard_path)
                print(f"[AsyncIO] Saved {shard_path} ({len(shard_data)} samples). System memory trimmed.", flush=True)
            except Exception as e:
                print(f"[AsyncIO] Error saving {shard_path}: {e}", flush=True)
            finally:
                del shard_data
                del item
                purge_system_memory()
                self.queue.task_done()

    def save_shard_async(self, shard_path: str, shard_data: list):
        self.queue.put((shard_path, shard_data), block=True)

    def wait_all(self):
        self.queue.join()

    def stop(self):
        self.stopped = True
        self.queue.put(None)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 1: Zero-Leak Feature Extraction on {device.upper()} ===", flush=True)
    print(f"Target Model: {args.model_id}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Num Samples:  {args.num_samples} | Chunk Size: {args.chunk_size} | Seq Len: {args.seq_len}")

    os.makedirs(args.output_dir, exist_ok=True)

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
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    purge_system_memory()
    print(f"Target model loaded! VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    target_layers = [1, 16, 31, 46, 61]
    hidden_states = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook_fn

    model_layers = target_model.model.layers if hasattr(target_model, "model") else target_model.layers
    for l_idx in target_layers:
        model_layers[l_idx].register_forward_hook(make_hook(l_idx))

    stream = build_interleaved_stream(args.cache_dir)
    prefetcher = BackgroundPrefetcher(stream, tokenizer, args.seq_len, max_queue_size=16)
    prefetcher.start()

    saver = BackgroundShardSaver()
    saver.start()

    sample_count = 0
    shard_idx = 0
    current_shard = []
    t0 = time.time()

    print("=== Feature Extraction Active (malloc_trim + Purged RAM) ===", flush=True)

    while sample_count < args.num_samples:
        tokens_cpu = prefetcher.queue.get()
        if tokens_cpu is None:
            break

        tokens = tokens_cpu.to(device, non_blocking=True)
        hidden_states.clear()
        with torch.no_grad():
            _ = target_model(tokens)
            concat_h = torch.cat([hidden_states[l] for l in target_layers], dim=-1).cpu().to(torch.bfloat16)
            hidden_states.clear()

        # Detach and store clone to decouple from PyTorch autograd tracking
        current_shard.append({
            "tokens": tokens_cpu.squeeze(0).clone(),
            "hidden": concat_h.squeeze(0).clone(),
        })

        sample_count += 1
        if sample_count % 50 == 0 or sample_count == 1:
            elapsed = time.time() - t0
            rate = sample_count / elapsed if elapsed > 0 else 0
            print(f"Extracted {sample_count}/{args.num_samples} sequences | Rate: {rate:.1f} seq/s | Elapsed: {elapsed:.1f}s", flush=True)
            purge_system_memory()

        if len(current_shard) >= args.chunk_size:
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:04d}.pt")
            saver.save_shard_async(shard_path, current_shard)
            shard_idx += 1
            current_shard = []
            purge_system_memory()

    if current_shard:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:04d}.pt")
        saver.save_shard_async(shard_path, current_shard)
        current_shard = []
        purge_system_memory()

    print("Waiting for asynchronous background disk writes to complete...", flush=True)
    saver.wait_all()
    saver.stop()
    prefetcher.stop()

    print("Saving embed_tokens and lm_head projection weights for Phase 2...", flush=True)
    with torch.no_grad():
        embed_weights = target_model.get_input_embeddings().weight.data.cpu().to(torch.bfloat16)
        lm_head_weights = target_model.get_output_embeddings().weight.data.cpu().to(torch.bfloat16)
    
    torch.save(
        {"embed_tokens": embed_weights, "lm_head": lm_head_weights},
        os.path.join(args.output_dir, "projection_weights.pt")
    )
    purge_system_memory()
    print(f"=== Phase 1 Extraction Complete! Total Samples: {sample_count} ===", flush=True)


if __name__ == "__main__":
    main()
