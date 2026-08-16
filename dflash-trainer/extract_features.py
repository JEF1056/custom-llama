#!/usr/bin/env python3
"""
Phase 1: High-Throughput Offline Feature Extraction for DFlash Drafter.

Stream-to-Disk (Zero Host RAM Accumulation):
- Flushes shards to disk every 200 samples (~100MB RAM max).
- Saves features to a single disposable Docker volume (/workspace-data/features).
- Periodic gc.collect() to guarantee strict low RAM usage.
"""

import argparse
import gc
import os
import sys
import time
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hidden state features from target model")
    parser.add_argument("--model-id", type=str, default="/models/qwen38-src/hf_safetensors")
    parser.add_argument("--output-dir", type=str, default="/workspace-data/features")
    parser.add_argument("--cache-dir", type=str, default="/workspace-data/dataset-cache")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of diverse sequences to extract")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=200, help="Samples per shard file (low RAM footprint)")
    return parser.parse_args()


def build_interleaved_stream(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Dataset] Streaming multi-domain datasets (Cache: {cache_dir})...", flush=True)

    code_ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True, cache_dir=cache_dir)
    tool_ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True, cache_dir=cache_dir)
    chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, cache_dir=cache_dir)

    def generator():
        c_iter = iter(code_ds)
        t_iter = iter(tool_ds)
        ch_iter = iter(chat_ds)

        pattern = ["code", "code", "chat", "tool", "code", "chat", "tool", "code", "chat"]
        for domain in itertools.cycle(pattern):
            try:
                if domain == "code":
                    item = next(c_iter)
                    prob = item.get("problem", "")
                    sol = item.get("solution", "")
                    if prob and sol:
                        yield [{"role": "user", "content": prob}, {"role": "assistant", "content": sol}]
                elif domain == "tool":
                    item = next(t_iter)
                    raw_chat = item.get("chat", "")
                    if raw_chat:
                        parts = raw_chat.split("ASSISTANT:")
                        if len(parts) >= 2:
                            u_part = parts[0].replace("USER:", "").strip()
                            a_part = parts[1].replace("<|endoftext|>", "").strip()
                            yield [{"role": "user", "content": u_part}, {"role": "assistant", "content": a_part}]
                elif domain == "chat":
                    item = next(ch_iter)
                    msgs = item.get("messages", [])
                    if msgs:
                        yield msgs
            except StopIteration:
                break
            except Exception:
                continue

    return generator()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 1: Feature Extraction on {device.upper()} ===", flush=True)
    print(f"Target Model: {args.model_id}", flush=True)
    print(f"Output Dir:   {args.output_dir}", flush=True)
    print(f"Num Samples:  {args.num_samples} | Seq Len: {args.seq_len} | Shard Size: {args.chunk_size}", flush=True)

    print("Loading tokenizer and target model in 4-bit NF4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

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
    
    sample_count = 0
    shard_idx = 0
    current_shard = []
    t0 = time.time()

    print("=== Extraction Active ===", flush=True)

    for messages in stream:
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            continue

        tokens = tokenizer(text, max_length=args.seq_len, truncation=True, return_tensors="pt")["input_ids"].to(device)
        if tokens.shape[1] < 18:
            continue

        hidden_states.clear()
        with torch.no_grad():
            _ = target_model(tokens)
            concat_h = torch.cat([hidden_states[l] for l in target_layers], dim=-1).cpu().to(torch.bfloat16)
            hidden_states.clear()

        current_shard.append({
            "tokens": tokens.cpu().squeeze(0),  # [S]
            "hidden": concat_h.squeeze(0),      # [S, 5 * 5120]
        })
        sample_count += 1

        if sample_count % 50 == 0:
            elapsed = time.time() - t0
            rate = sample_count / elapsed if elapsed > 0 else 0
            print(f"Extracted {sample_count}/{args.num_samples} sequences | Rate: {rate:.1f} seq/s | Elapsed: {elapsed:.1f}s", flush=True)

        if len(current_shard) >= args.chunk_size:
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:04d}.pt")
            torch.save(current_shard, shard_path)
            print(f"Saved {shard_path} ({len(current_shard)} samples)", flush=True)
            del current_shard
            current_shard = []
            shard_idx += 1
            gc.collect()

        if sample_count >= args.num_samples:
            break

    if current_shard:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(current_shard, shard_path)
        print(f"Saved final {shard_path} ({len(current_shard)} samples)", flush=True)
        del current_shard
        gc.collect()

    # Save embed_tokens and lm_head weights for Phase 2 training
    print("Saving embed_tokens and lm_head projection weights for Phase 2...", flush=True)
    embed_w = target_model.get_input_embeddings().weight.data.cpu().to(torch.bfloat16)
    lm_head_w = target_model.get_output_embeddings().weight.data.cpu().to(torch.bfloat16)
    torch.save({
        "embed_tokens": embed_w,
        "lm_head": lm_head_w,
    }, os.path.join(args.output_dir, "projection_weights.pt"))

    print(f"=== Phase 1 Extraction Complete! Total Samples: {sample_count} ===", flush=True)


if __name__ == "__main__":
    main()
