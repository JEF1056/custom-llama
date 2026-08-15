#!/usr/bin/env python3
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from model import DFlashConfig, DFlashDraftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash drafter for Qwen3.8-27B-heretic-ara")
    parser.add_argument("--model-id", type=str, default="heretic-org/Qwen3.8-27B-heretic-ara")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--output-dir", type=str, default="/output/Qwen3.8-27B-heretic-dflash")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting DFlash Drafter Training on {device.upper()} ===")
    print(f"Target Model: {args.model_id}")
    print(f"Output Dir:   {args.output_dir}")

    print("Loading tokenizer and target model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Load frozen target model in 8-bit / 4-bit / bfloat16
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    target_layers = [4, 16, 28, 40, 52]
    config = DFlashConfig(
        hidden_size=target_model.config.hidden_size,
        intermediate_size=10240,
        num_hidden_layers=5,
        num_attention_heads=40,
        num_key_value_heads=8,
        target_layer_ids=target_layers,
        num_target_layers=target_model.config.num_hidden_layers,
        block_size=args.block_size,
        mask_token_id=248077,
        vocab_size=target_model.config.vocab_size,
    )

    drafter = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
    drafter.train()
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Trainable drafter parameters: {sum(p.numel() for p in drafter.parameters() if p.requires_grad):,}")

    # Hook target model intermediate layers
    hidden_states = {}
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook_fn

    model_layers = target_model.model.layers if hasattr(target_model, "model") else target_model.layers
    for l_idx in target_layers:
        model_layers[l_idx].register_forward_hook(make_hook(l_idx))

    print(f"Loading training dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train_sft", streaming=True)

    step = 0
    t0 = time.time()
    embed_tokens = target_model.get_input_embeddings()
    lm_head = target_model.get_output_embeddings()

    for item in dataset:
        messages = item.get("messages", [])
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            continue

        tokens = tokenizer(text, max_length=args.seq_len, truncation=True, return_tensors="pt")["input_ids"].to(device)
        if tokens.shape[1] < args.block_size + 2:
            continue

        with torch.no_grad():
            _ = target_model(tokens)
            # Concatenate extracted hidden states: [B, Seq_Len, len(target_layers) * hidden_size]
            concat_h = torch.cat([hidden_states[l] for l in target_layers], dim=-1)

        B, S, _ = concat_h.shape
        # Create block-masked targets for speculative prediction
        mask_id = config.mask_token_id
        k = args.block_size

        # Slide blocks across sequence
        for start_pos in range(0, S - k, k):
            block_tokens = tokens[:, start_pos:start_pos + k]
            # Draft input: [t0, MASK, MASK, MASK]
            masked_input = block_tokens.clone()
            masked_input[:, 1:] = mask_id
            
            with torch.no_grad():
                draft_embeds = embed_tokens(masked_input)
                h_ctx = concat_h[:, :start_pos + 1]

            drafter_out = drafter(draft_embeds, h_ctx)
            with torch.no_grad():
                draft_logits = lm_head(drafter_out[:, 1:])

            loss = F.cross_entropy(
                draft_logits.reshape(-1, config.vocab_size),
                block_tokens[:, 1:].reshape(-1),
            )

            (loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    print("=== Training Complete! Exporting Model ===")
    drafter.export_mlx_safetensors(args.output_dir)
    print(f"DFlash Drafter successfully saved to {args.output_dir}")

if __name__ == "__main__":
    main()
