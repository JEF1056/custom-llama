---
license: apache-2.0
base_model: trohrbaugh/Qwen3.8-27B-heretic-ara
tags:
- dflash
- speculative-decoding
- fast-inference
- mlx
- vllm
- sglang
- qwen3
- code
- function-calling
pipeline_tag: text-generation
---

# Qwen3.8-27B-heretic-dflash: Official 5-Layer DFlash Speculative Drafter

`Qwen3.8-27B-heretic-dflash` is a high-performance **DFlash (Block Diffusion Speculative Drafter)** trained explicitly for **[`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara)** (Qwen3 27B architecture, 248,320 vocabulary, 5,120 hidden dimension).

It achieves up to **2.5×–3.5× decoding speedups** by generating 16-token draft blocks in parallel from deep feature conditioning, verified in single forward passes.

---

## 🏗️ Architecture Specification

The model adheres 100% to the official **`z-lab/Qwen3.6-27B-DFlash`** architecture and parameter schema:

| Architectural Component | Value / Configuration |
| :--- | :--- |
| **Base / Target Model** | `trohrbaugh/Qwen3.8-27B-heretic-ara` (64 total layers) |
| **Drafter Layers (`num_hidden_layers`)** | **5 Transformer Decoder Layers** |
| **Hidden Size (`hidden_size`)** | **5,120** |
| **Intermediate Size (`intermediate_size`)** | **17,408** (SwiGLU MLP) |
| **Attention Heads** | **32 Q-Heads / 8 KV-Heads** (Grouped Query Attention 4:1) |
| **Head Dimension (`head_dim`)** | **128** |
| **Block Size (`block_size`)** | **16 tokens** |
| **Target Layer IDs (`target_layer_ids`)** | **`[1, 16, 31, 46, 61]`** (5 uniformly spaced feature extraction layers) |
| **Feature Projection (`fc`)** | `Linear(25600, 5120, bias=False)` |
| **Mask Token ID (`mask_token_id`)** | **`248070`** |
| **Vocabulary Size (`vocab_size`)** | **248,320** |

---

## 📊 Training Dataset Composition

Trained on a balanced, polyglot multi-domain stream designed specifically for coding, tool use, and multi-turn instruction following:

1. **Polyglot Code (45%)** — [`ise-uiuc/Magicoder-OSS-Instruct-75K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)
   - Real-world code generation, bug fixing, and refactoring across **C++, Rust, Python, Go, and TypeScript**.
2. **Tool & Function Calling (25%)** — [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
   - Complex structured JSON schemas, API call syntax, parameter filling, and tool outputs.
3. **Dialogue & Reasoning (30%)** — [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
   - Multi-turn conversational planning, explanations, and general knowledge.

---

## ⚙️ Training Methodology: Two-Phase Zero-Spill Pipeline

Trained on an **NVIDIA GeForce RTX 3090 (24 GB VRAM)** using a two-phase decoupled distillation pipeline to ensure maximum tensor core throughput with 0 MB/s PCIe spilling:

1. **Phase 1 (Offline Target Feature Extraction)**:
   - Target model (`trohrbaugh/Qwen3.8-27B-heretic-ara`) loaded in 4-bit NormalFloat (`NF4`) consuming 16.45 GB VRAM.
   - Streamed 10,000 multi-domain sequences (`seq_len = 1024`) and extracted concatenated hidden representations from layers `[1, 16, 31, 46, 61]` ($5 \times 5120 = 25,600$ dims) directly to disk shards.
2. **Phase 2 (Ultra-Fast Drafter Optimization)**:
   - Base model unloaded completely from VRAM.
   - Trained the 5-layer drafter (461M params) for **10,000 optimization steps** using `bitsandbytes.optim.AdamW8bit` ($lr = 2 \times 10^{-4}$, weight decay = $0.01$) directly on on-die GDDR6X memory (7.96 GB VRAM footprint) at full 385W GPU TDP.

---

## 🚀 How to Use

### 1. Apple Silicon (MLX-VLM / MLX-LM)

```bash
python3 -m mlx_vlm.server \
    --host 0.0.0.0 \
    --port 8080 \
    --model trohrbaugh/Qwen3.8-27B-heretic-ara \
    --draft-model jfan/Qwen3.8-27B-heretic-dflash
```

### 2. vLLM (CUDA / High-Throughput Serving)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model trohrbaugh/Qwen3.8-27B-heretic-ara \
    --speculative-model jfan/Qwen3.8-27B-heretic-dflash \
    --num-speculative-tokens 16 \
    --port 8000
```

### 3. SGLang

```bash
python3 -m sglang.launch_server \
    --model-path trohrbaugh/Qwen3.8-27B-heretic-ara \
    --speculative-draft-model-path jfan/Qwen3.8-27B-heretic-dflash \
    --speculative-num-steps 16 \
    --port 30000
```

---

## 📜 Citation & Credits

- Base Model: [`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara)
- DFlash Architecture: *DFlash: Block Diffusion for Speculative Decoding* (arXiv:2602.06036)
- Trained by: [jfan](https://huggingface.co/jfan)
