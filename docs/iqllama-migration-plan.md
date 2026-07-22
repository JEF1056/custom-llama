# Migration plan: ditch 1-bit Qwen3.6-27B, serve Qwen3.6-35B-A3B on stock ik_llama.cpp

Status: **IMPLEMENTED (Phases 1, 3, 4, 5 as code; Phase 2 as scripts; Phases 6-7
run against a bring-up quant on real hardware — see
[qwen36-bench-results.md](qwen36-bench-results.md)).** `docker/Dockerfile`,
`docker/entrypoint.sh`, both `docker-compose.yml` files, `docker/.env(.example)`,
`router/config.yaml` and `README.md` now target stock `ik_llama.cpp` +
Qwen3.6-35B-A3B (Hadamard KV, MTP, vision, prompt cache). `scripts/
download-source-gguf.sh`, `compute-imatrix.sh` and `quantize.sh` implement the
offline Phase 2 weights pipeline (Unsloth BF16 source -> our own imatrix -> the
262K-Balanced recipe); the `--custom-q` regexes in `quantize.sh` have been
verified against the real BF16 GGUF's tensor names (via direct GGUF header
inspection, see docs/qwen36-bench-results.md) and are no longer placeholders.
Phases 6 (functional bring-up) and 7 (benchmarking) have been run against
Unsloth's public pre-quantized IQ3_XXS bring-up GGUF on a real RTX 3090: basic
completion, vision, prompt-cache reuse (54x), and long-context (262144, 26K-token
prompt) all PASS; MTP is BLOCKED — root-caused to Unsloth's GGUF conversion
dropping the original checkpoint's `mtp.*` tensors entirely (verified by direct
GGUF header inspection), a gap that also affects our own Phase 2 pipeline since it
is built on the same Unsloth BF16 source. Full details, numbers, and follow-up
items in qwen36-bench-results.md. An earlier pass through this
document incorrectly concluded stock ik_llama.cpp couldn't serve this model
(wrong: it had checked the wrong clip.cpp path, stopped its MTP tensor search too
early, and only checked 2 of many architectures' graph builders for Hadamard). That
conclusion was corrected via direct re-verification against a freshly-refreshed
`ikawrakow/ik_llama.cpp@main` clone (commit `9d07d86`, 2026-07-18) plus GitHub
PR/issue history and real HF `config.json` for `Qwen/Qwen3.6-35B-A3B`. See
"## 0. Verified findings" below for the corrected, source-checked picture. Section
2's feasibility table is largely accurate after all (the specific PR numbers it
cited don't exist in ik_llama's history, but the underlying features do, under
different PR numbers/mechanisms — see section 0).

## 0. Verified findings (source-checked, supersedes earlier "BLOCKED" note)

Verified by re-cloning ik_llama.cpp fresh, reading `src/graphs/build_qwen3next.cpp`,
`src/llama-build-context.cpp`, `src/llama-load-tensors.cpp`, `src/llama-model.h`,
and `examples/mtmd/clip.cpp` directly, cross-checked against ik_llama's real GitHub
PR/issue history and the real HF `config.json` for `Qwen/Qwen3.6-35B-A3B` (real HF
class: `Qwen3_5MoeForConditionalGeneration`, `model_type: qwen3_5_moe`,
`mtp_num_hidden_layers: 1`, `full_attention_interval: 4` -> 10 full-attn / 30
linear-attn(deltanet) of 40 layers, `num_key_value_heads: 2`, `head_dim: 256`, real
vision tower).

**1. Hadamard KV — CONFIRMED WORKING.** `build_qwen3next.cpp`'s full-attention
layers call the shared `llm_build_context::build_std_attention()` helper
(`src/llama-build-context.cpp:2775`), which itself contains the generic
`cparams.k_cache_hadamard`/`v_cache_hadamard` -> `ggml_hadamard(...)` logic (around
lines 2915-2926). The block size comes from `model.hadamard_size_k(il)`/
`hadamard_size_v(il)` (`src/llama-model.h:564-572`), which is purely
`head_dim`-based (`hadamard_size()` just needs a power-of-2 or a divisor of
64-512) with **no architecture exclusion list** — only MLA models
(DeepSeek2/GLM-DSA/Mistral4) get special-cased to a fixed size 64. Qwen3.6's
attention head_dim is 256 (power of 2), so `-khad`/`-vhad` apply cleanly. (My
earlier conclusion only checked `build_minimaxm2.cpp`/`build_deepseek2.cpp` for
*direct* `ggml_hadamard` calls and missed that most architectures, including this
one, get it for free via the shared helper.)

**2. MTP — CONFIRMED WORKING.** `create_qwen35moe_tensors()`
(`src/llama-load-tensors.cpp:1589`) has real, complete MTP wiring: computes
`is_mtp_layer` from `hparams.nextn_predict_layers` (the standard NextN/MTP layer
count metadata), loads an `output_extra.weight` MTP output head, and loads
`LLM_TENSOR_NEXTN_EH_PROJ`/`NEXTN_ENORM`/`NEXTN_HNORM`/`NEXTN_SHARED_HEAD_NORM` for
the trailing MTP layer(s) — a DeepSeek-V3-style single-extra-transformer-layer MTP
design (not the Gemma4-specific centroid-based `LLM_TENSOR_MTP_*` tensors I
initially searched for and wrongly concluded was the only MTP mechanism). Runtime
flags: `--spec-type mtp:n_max=N,p_min=0.0` (self-speculative, no separate draft
file needed) and `-mtprot`/`--mtp-requantize-output-tensor TYPE` to quantize the
MTP output head independently. This is actively maintained specifically for Qwen35
(PRs `#1987` "Fix Qwen35 mtp warmup", `#1979` "Fix qwen MTP accept rate
regression", `#1894`, `#2027` "Split mode graph for dense Qwen35 MTP" — all real,
merged, on `ikawrakow/ik_llama.cpp`). A real user (`sayap`, PR #1987 comments)
reports 127 t/s @ 97.99% MTP acceptance on Qwen3.6-27B, single 3090, vs 116 t/s on
mainline llama.cpp for the same test.

**3. Vision — CONFIRMED WORKING.** `examples/mtmd/clip.cpp` (not `tools/mtmd/` —
that path doesn't exist, which is why my first search found nothing) has a
complete `PROJECTOR_TYPE_QWEN3VL` implementation: `build_qwen3vl()`, M-RoPE
position handling, deepstack multi-level feature fusion support (`build_qwen2vl()`/
`PROJECTOR_TYPE_QWEN25VL` also present for the wider Qwen-VL family). Qwen3.6's
vision tower is the same Qwen3-VL-lineage ViT (just with
`deepstack_visual_indexes: []`, i.e. deepstack disabled for this checkpoint), so
this is directly usable.

**4. Conversion — SOLVED (use a pre-converted GGUF as the quantization source,
not a from-scratch HF→GGUF converter).** ik_llama's own `convert_hf_to_gguf.py`
still has no `@Model.register("Qwen3_5MoeForConditionalGeneration")`, so we cannot
convert the raw HF safetensors ourselves with ik_llama's bundled script. But this
doesn't block us: GitHub issue `ikawrakow/ik_llama.cpp#2041` ("Qwen3.6-35B-A3B ...
fails to load: tensor 'blk.0.ssm_dt.bias' not found") confirms — via ik_llama
collaborator `joelfarthing` — that **"ik_llama does support and load `qwen35moe`
GGUFs, including Qwen3.6-35B-A3B variants"**; that specific bug report's failure
was only because the user had an **Ollama-format** GGUF with nonstandard tensor
names (`ssm_dt` without the required `.bias` suffix, 3-section instead of
4-section `mrope` dims). `joelfarthing`'s stated fix: use a GGUF "produced in the
usual llama.cpp/ik_llama-compatible layout, for example the Bartowski or Unsloth
quants." Mainline `ggml-org/llama.cpp` has real, merged Qwen3.5/Qwen3.6 support
(e.g. PR `#25222` "Qwen3.6-27b-Q8_0 prefill speed up", `#25141` "register
t_layer_inp for qwen3next"), and `unsloth/Qwen3.6-35B-A3B-GGUF` on HF ships
exactly what we need as a quantization source: `BF16/` full-precision shards,
`imatrix_unsloth.gguf_file` (a precomputed imatrix we can reuse or recompute),
and separate `mmproj-BF16.gguf`/`mmproj-F16.gguf`/`mmproj-F32.gguf` vision
projector files. **Plan: download Unsloth's BF16 shards + mmproj, then run our
own `llama-imatrix` (optionally seeded/compared against Unsloth's) and
`llama-quantize --custom-q` with our 262K-Balanced recipe (section 4c) using
ik_llama's own quantizer** — this keeps ik_llama as the actual serving/quantizing
engine (satisfies requirement 1) while sidestepping the missing Python converter.

**Net effect:** all 4 originally-questioned requirements (Hadamard KV, MTP, vision,
self-quantization of this exact model) are achievable with stock `ik_llama.cpp`.
Proceeding with the original plan (sections 1-7 below), with one adjustment: the
weights pipeline (Phase 2) starts from Unsloth's BF16 GGUF + mmproj rather than
from raw HF safetensors via a from-scratch converter.

## 1. Goal

Replace the current Qwen3.6-27B (1-bit, TurboQuant+ fork + DSpark) stack with:

- **Stock `ik_llama.cpp`** (ikawrakow/ik_llama.cpp, `main`) as the only inference engine.
- **`Qwen/Qwen3.6-35B-A3B`** (hybrid MoE: gated delta-net + attention, with MTP heads),
  quantized in-house with an APEX-aligned mixed-precision recipe mapped onto
  ik_llama's CUDA IQK quants, sized (~16.5 GB) to fit **262K context + vision** on 24 GB
  (edge experts `iq4_ks`, middle experts `iq3_k`; see section 4c).
- 4-bit KV cache with **Hadamard rotation** ("hammond").
- **Vision**, **MTP speculative decoding**, and **prompt caching** all enabled at once
  on the hybrid model.
- **Maximized context length** on a single RTX 3090 (24 GB, Ampere sm_86).
- Full build optimization for this machine.
- Router: keep the existing **LiteLLM** router (see requirement 9 note below).

## 2. Feasibility summary (confirmed — see "## 0. Verified findings" above for the corrected mechanism/evidence behind rows 2/5/6, since the original PR numbers below don't exist but the features do)

| Requirement | Verdict on stock ik_llama.cpp | Evidence |
|---|---|---|
| 1. iqllama stock | Yes — build from `ikawrakow/ik_llama.cpp@main` | repo/README |
| 2. 4-bit KV + Hadamard | Yes — `-ctk q4_0 -ctv q4_0` + Hadamard K-cache (PR 1033/1034) and V-cache (PR 1527) | README "Features" |
| 3. Full machine optimization | Yes — CUDA sm_86, fused MoE (`-fmoe`), flash attn, native SIMD | README |
| 4. Quantize Qwen3.6-35B with `iq4_ks`/`iq3_ks` | Yes — both exist w/ CUDA kernels; `llama-quantize --custom-q regex=type` + per-class flags | `examples/quantize/quantize.cpp` (`IQ3_KS`=3.19 bpw, `IQ4_KS`=4.25 bpw) |
| 5. MTP head at `iq4_ks` | Yes — MTP decode for Qwen3.5/3.6 (PR 1698/1745); `--mtp-requantize-output-tensor` (PR 1809) | README |
| 6. Vision | Yes — vision in `llama-server` (PR 901) via mmproj | README |
| 7. prompt-cache + hybrid + vision + MTP together | Likely — recurrent-model checkpoints (PR 1310/1398) = prompt cache for hybrids; fused delta-net (PR 1315+); vision + MTP present. **Must be validated together at runtime** | README |
| 8. Max context | Yes - hybrid keeps only attention-layer KV growing; delta-net layers use fixed recurrent state; 4-bit + Hadamard KV shrinks it further; ik_llama VRAM auto-fit (PR 1501/1504/1872). 262K KV @ q4 = only ~1.5 GB (10 attn layers x 2 KV heads); the binding constraint is weight size, so the recipe targets ~16.5 GB (section 4b/4c) | README |
| 9. Router mode + `.ini` | **Not in ik_llama.cpp** — this is a mainline-only feature (`--models-preset`, `--models-dir`, launch with no model). Per instruction ("check if it exists; if not, skip"): **skip native router; keep LiteLLM**. | mainline server README vs ik_llama docs (no `server-router` doc) |

### Requirement 9 detail

Mainline `llama.cpp` has a native router: run `llama-server` with no model, configure
models via an **`.ini`** preset (`--models-preset my-models.ini`, sections = models,
keys = CLI args without dashes, `[*]` global, `load-on-startup`, etc.) or `--models-dir`.
`ik_llama.cpp` is on an older merge base and does not carry this. Adopting it would mean
giving up ik_llama's IQK quants, Hadamard KV, fused delta-net and Qwen3.6 MTP — which are
the whole point of requirements 1-5. So we **keep LiteLLM** as the router and skip the
native `.ini` router. (If a native router later matters more than the IQK/Hadamard stack,
the alternative is mainline llama.cpp — a different project, out of scope here.)

## 3. Key design decisions

- **One engine, two roles**: build ik_llama.cpp once; use its `llama-quantize` +
  `llama-imatrix` to make the GGUF, and its `llama-server` to serve.
- **Quant recipe = APEX-aligned, ik_llama IQK types.** APEX's caveat that
  "IQ formats underperform K-quants for MoE experts" is about mainline **codebook** IQ
  (e.g. `IQ3_S`). ik_llama's `iqN_ks`/`iqN_k` are **non-linear** quants that perform like
  k-quants and have CUDA kernels. APEX's key lesson - "**precision allocation matters more
  than uniform bit-width**" - drives the recipe: to fit **262K + vision** on 24 GB the
  weights must be ~16.5 GB, so we spend bits on the sensitive tensors (edge experts
  `iq4_ks`, shared expert `q8_0`, attention `iq5_ks`, routing `f16`, output `q6_K`) and
  save them on the sparse middle experts (`iq3_k`, the best fast iq3), recovering quality
  with a diverse imatrix. See sections 4b-4d.
- **KV**: 4-bit (`q4_0`) K and V with Hadamard transforms enabled (rotation reduces
  quantization error, per PR 1033/1034/1527).
- **Vision**: ship a separate `mmproj` GGUF. See risk R1 — the given repo name has no
  `-VL`; confirm the base is multimodal or use the matching VL sibling for the tower.
- **Backends**: keep the LiteLLM router and the Mac/MLX deployment entry, but note the
  Qwen3.6 ik_llama build is **CUDA-only** (Mac cannot run it). The Mac slot remains a
  separate backend; only the CUDA backend is retargeted here.

## 4. Specialized quants available in ik_llama.cpp (verified from `quantize.cpp`)

ik_llama adds two SOTA quant families on top of stock llama.cpp legacy/K/codebook-IQ
types. All the ones below have CUDA kernels.

**Non-linear "KS / K / KL" family** (k-quant-competitive, fast on CUDA - the workhorses):

| Type | bpw | Type | bpw | Type | bpw |
|---|---|---|---|---|---|
| `iq2_ks` | 2.19 | `iq3_ks` | 3.19 | `iq4_kss` | 4.00 |
| `iq2_k`  | 2.375| `iq3_k`  | 3.44 | `iq4_ks`  | 4.25 |
| `iq2_kl` | 2.69 | `iq3_kl` | 4.00 | `iq4_k`   | 4.50 |
|          |      |          |      | `iq5_ks`  | 5.25 |
|          |      |          |      | `iq5_k`   | 5.50 |
|          |      |          |      | `iq6_k`   | 6.60 |

**Trellis "KT" family** (QTIP-style; best quality-per-bit, but heavier dequant ->
slower prefill/generation): `iq1_kt` 1.75, `iq2_kt` 2.125, `iq3_kt` 3.125, `iq4_kt` 4.00.

**CUDA support verified** (against ik_llama source, `main`): every type used in the
recipes has both a quantized matmul (MMQ, used in prefill) and a vector kernel
(MMVQ/GEMV, used in generation), plus a dequant kernel:
- `iq4_ks`, `iq5_ks`, `iq3_k`, `iq6_k`, `iq4_kss`, `iq3_ks` -> `ggml/src/ggml-cuda/mmq.cu`
  (MMQ), `iqk_mmvq.cu` (`mul_mat_vec_iq*_q8_1_cuda`), `convert.cu` (`dequantize_row_iq*`).
- `q4_0`, `q8_0`, `q6_K`, `f16` -> core CUDA (mmq.cu / mmvq.cu).
- `iq3_kt`/`iq4_kt` (trellis) -> MMQ + MMVQ + dmmv all present, **but** `iqk_mul_mat.cpp`
  `is_dequant_better()` routes KT through a `Q8_0_R8` dequant path for batches
  `nrc_y >= 16` (i.e. **prefill dequantizes first**) - the concrete reason KT is slower
  for prefill/gen. The KS/K family have direct MMQ kernels with no such fallback.

Net: the primary recipe (`iq4_ks` / `iq3_k` / `iq5_ks` / `q8_0` / `q6_K`, `q4_0` KV) is
**fully CUDA-accelerated on both the prefill and generation paths** with no dequant
fallback; only the optional `iq3_kt` alternative incurs the prefill dequant path.

Also useful: `q6_0` (6.5), `q8_KV` (8.0, for KV), `iq4_nl` (4.5), plus `_R4`/`_R8`
row-interleaved repacks (CPU-side speed only). Per-class flags exist for all of them:
`--output-tensor-type`, `--token-embedding-type`, `--per-layer-token-embedding-type`,
`--ffn-gate-inp-type`, `--attn-{q,k,v,qkv,output}-type`, `--ffn-{gate,down,up}-type`,
and `--custom-q regex=type`. Tool notes: recommended output tensor = `q6_K`; `attn_q`
typically one tier below the ffn type, `attn_v` one tier above.

### Empirical quality curve (ik_llama, Qwen3-235B-A22B MoE, wikitext PPL; bf16 = 4.308)

| Quant | bpw | PPL | delta vs bf16 |
|---|---|---|---|
| `q8_0`   | 8.50 | 4.314 | +0.006 |
| `iq5_k`  | 5.91 | 4.335 | +0.027 |
| `iq4_k`  | 4.90 | 4.367 | +0.059 |
| `iq4_kss`| 4.21 | 4.402 | +0.094 |
| `iq4_ks` | 4.28 | 4.416 | +0.108 |
| `iq3_k`  | 3.90 | 4.456 | +0.148 |
| `iq3_ks` | 3.70 | 4.492 | +0.184 |
| `iq2_kl` | 2.99 | 4.791 | +0.483 |

This reproduces APEX's core finding on ik's own quants: quality holds through the
`iq5`/`iq4` band, then the cost rises sharply into `iq3` and collapses at `iq2`. The knee
is exactly where APEX puts it ("below Q5_K degrades"). Note this is the **non-linear** IQK
family, not the mainline **codebook** IQ (`IQ3_S`/`IQ3_XXS`) that APEX warns against - so
using `iqN_ks` is faithful to APEX, not in conflict with it.

## 4b. Context / VRAM budget (why the recipe must target ~16-17 GB)

Requirement 8 is **maximize context** (262144 native). The good news: the hybrid
architecture makes the KV cache tiny, so context length is **not** the binding
constraint - the model weight is.

KV grows only in the **10 attention layers** (the 30 DeltaNet layers use a fixed-size
recurrent state, independent of context). Attention = 2 KV heads x head_dim 256:

```
KV bytes/token = n_attn_layers(10) x n_kv_heads(2) x head_dim(256) x 2(K+V) x bytes/elem
q4_0 (0.5625 B/elem): 10 x 2 x 256 x 2 x 0.5625 = 5,760 B/token
  262,144 tokens -> ~1.51 GB      (q8_0 KV -> ~2.85 GB, f16 KV -> ~5.37 GB)
```

So 262K at 4-bit Hadamard KV costs only **~1.5 GB**. The real 24 GB budget at 262K:

| Component | @262K, vision on | Notes |
|---|---|---|
| KV cache (q4_0 + Hadamard) | ~1.5 GB | only 10 attn layers, 2 KV heads |
| DeltaNet recurrent state | ~0.3 GB | fixed, context-independent |
| Vision mmproj (resident) | ~1.0 GB | + transient spikes when an image is sent |
| Prefill compute buffers (`-fa`, `-ub 2048`) | ~2.0 GB | bounded by ubatch, not context (flash attn) |
| MTP draft overhead | ~0.2 GB | |
| **Non-weight subtotal** | **~5.0 GB** | leaves **~19 GB** for weights |

A ~20 GB model therefore **cannot** hold 262K + vision + a usable ubatch (5 + 20 = 25 GB
> 24). To fit with margin, weights must be **~16-17 GB**. Since experts are ~92% of the
model, that forces the middle expert layers into the `iq3` band - the APEX "below Q5_K"
zone. We accept that only for the least-sensitive middle experts and buy the quality back
with ik-specific tricks (below), keeping every sensitive tensor high.

## 4c. Recommended recipe: "262K-Balanced" (~16.5 GB, fast) [PRIMARY]

APEX's real lesson is "**precision allocation matters more than uniform bit-width**", so
we spend bits where they matter (edges, shared expert, attention, routing, output) and
save them on the sparse middle experts. ik_llama lets us beat mainline at equal size:
`iq3_k` (3.44 bpw, +0.148 ppl) is better-behaved than mainline `Q3_K` and than `iq3_ks`
(3.19, +0.184), and stays CUDA-fast (no trellis).

| Tensor class | ik_llama type | bpw | Rationale |
|---|---|---|---|
| routed experts, **edge** L0-4 & L35-39 | `iq4_ks` | 4.25 | most sensitive layers held at the knee |
| routed experts, **middle** L5-34 | `iq3_k` | 3.44 | sparse bulk; best fast iq3 (> `iq3_ks`) |
| shared expert (`ffn_*_shexp`) | `q8_0` | 8.5 | heavy-tailed (kurtosis ~13); APEX min |
| attention (`attn_q/k/v/output`), 10 layers | `iq5_ks` | 5.25 | tiny param cost, quality-critical |
| router (`ffn_gate_inp`) | `q8_0`/f16 | - | routing errors cascade; near-lossless |
| `token_embd` | `iq4_ks` | 4.25 | large; kept in RAM even on full offload |
| `output` (lm_head) | `q6_K` | 6.5 | ik_llama's recommended output type |
| MTP head tensors | `iq4_ks` (req 5) | 4.25 | opt `iq5_ks` raises draft accept -> faster gen |

**Size**: edge 8.05B @ 4.25 = 4.28 GB; middle 24.15B @ 3.44 = 10.4 GB; shared/attn/
deltanet/embed/output/router/MTP ~1.8 GB -> **~16.5 GB**. With the ~5 GB non-weight
budget above: **~21.5 GB total at 262K + vision -> fits 24 GB with ~2.5 GB margin.**

Use a **strong diverse imatrix** (chat/code/reasoning/tool-calling, no Wikipedia): APEX
found the imatrix gain is **largest exactly on aggressive quants**, which is where the
middle-expert `iq3_k` sits - this is the main quality recovery lever.

## 4d. Alternatives (pick per priority)

- **"262K-Quality" (~15.5 GB, slightly slower TG)**: middle experts -> `iq3_kt` (trellis,
  3.125 bpw) instead of `iq3_k`. Trellis gives better quality-per-bit and frees ~1 GB more
  headroom (larger ubatch / longer ctx), at the cost of ~10-20% token-gen speed. Best if
  you want to push toward ~512K or maximize quality within 24 GB.
- **3-tier gradient (~17.2 GB)**: split middle into near-edge L5-9 & L30-34 = `iq4_ks`
  and deep-middle L10-29 = `iq3_k`. Finer APEX-style gradient; a bit more quality, ~0.7 GB
  larger. Fits 262K + vision but with less margin.
- **"Quality, no full context" (~20 GB)**: edges `iq5_ks` / middle `iq4_ks` (the earlier
  balanced recipe). Best quality, but only reaches ~262K **text-only with a small ubatch**
  and cannot also host vision comfortably. Use only if 262K+vision is dropped.

**Context beyond 262K**: KV is cheap here, so ~512K fits by raising ctx (KV -> ~3 GB) with
the ~16.5 GB model. ~1M (YaRN) needs ~5.7 GB KV -> pair with `iq3_kt` weights and/or a
smaller ubatch.

**Higher-quality variant if it fits at reduced context**: middle `iq5_ks`, edges/attn
`iq6_k` -> ~21-22 GB, approaching q8_0 quality; expect to trade some context length.

Implementation uses `--imatrix <diverse.imatrix>` plus `--custom-q` regex rules (and/or
the per-class `--attn-*-type` / `--ffn-*-type` / `--token-embedding-type` /
`--output-tensor-type` / `--per-layer-token-embedding-type` / `--ffn-gate-inp-type` /
`--mtp-requantize-output-tensor` flags). Exact tensor-name regexes are derived from the
converted GGUF's tensor list at implementation time.

**Imatrix**: compute with ik_llama `llama-imatrix` on a **diverse** corpus (chat, code,
reasoning, tool-calling; no Wikipedia) per APEX I-variant, from the bf16 GGUF. APEX found
this trades a tiny wikitext-PPL increase for better downstream accuracy and lower KL.

## 5. Phased task breakdown

### Phase 0 — Prep & verification (no code) — MOSTLY RESOLVED (see "## 0. Verified findings")
- Layer count, hybrid block layout, MTP tensor names/mechanism, and vision tower are all
  RESOLVED via direct source read + the real HF `config.json` (section 0).
- Remaining, cheap to do once Phase 1's binary exists: re-confirm exact flag spellings on
  the *actual built binary* (`./build/bin/llama-server --help 2>&1 | grep -iE
  "khad|vhad|hadamard|spec-type|mtprot|mtp-requantize|mmproj"`) — the repo merges several
  times a day, so the flag names found in source on 2026-07-18 could be renamed by the
  time we actually build. Do not assume; re-grep `--help` before writing Phase 3's
  entrypoint flags.

### Phase 1 — Build ik_llama.cpp (image)
- Rewrite `docker/Dockerfile`:
  - source = `https://github.com/ikawrakow/ik_llama.cpp.git`, ref `main`
    (build args `LLAMA_REPO` / `LLAMA_REF`; keep `git`/`local` `BUILD_MODE`).
  - `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
    -DGGML_CUDA_F16=ON -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_SERVER=ON`
    (+ any ik_llama IQK/FA build toggles for sm_86; verify names). Build targets
    `llama-server`, `llama-quantize`, `llama-imatrix`.
  - CUDA 12.x base (ik_llama's supported/tested CUDA line) — verify vs the `cu12` images.
- Update `docker/docker-compose.yml` and root `docker-compose.yml`: image/service names
  (`qwen36-cuda` or similar), build args.

### Phase 2 — Weights pipeline (source: Unsloth's pre-converted GGUF, per section 0 item 4 — not a from-scratch HF conversion)
- Add `scripts/` (in custom-llama) to:
  1. **Download source artifacts** from `unsloth/Qwen3.6-35B-A3B-GGUF` (HF) instead of
     raw HF safetensors — ik_llama's own `convert_hf_to_gguf.py` has no registration for
     `Qwen3_5MoeForConditionalGeneration`, so we use an already-correct, standard-layout
     GGUF as the conversion source instead:
     ```
     hf download unsloth/Qwen3.6-35B-A3B-GGUF \
       --include "BF16/*" "mmproj-BF16.gguf" "imatrix_unsloth.gguf_file" \
       --local-dir /models/qwen36-src
     ```
     (~70GB+ for the BF16 shards + mmproj; verify actual size/shard count at download time.)
  2. **Merge shards** if the BF16 GGUF is split (`llama-gguf-split --merge` or verify
     ik_llama's loader auto-follows `*-00001-of-0000N.gguf` naming without a manual merge).
  3. **Compute our own diverse imatrix** from the merged BF16 GGUF (chat/code/reasoning/
     tool-calling corpus, no Wikipedia, per APEX):
     ```
     ./build/bin/llama-imatrix -m qwen36-bf16.gguf -f diverse_corpus.txt \
       -o qwen36.imatrix --chunks 200 -ngl 999
     ```
     Compare its downstream quality against Unsloth's shipped `imatrix_unsloth.gguf_file`
     in Phase 7e's perplexity check; keep whichever scores better.
  4. **Quantize** with the section 4c "262K-Balanced" recipe:
     ```
     ./build/bin/llama-quantize --imatrix qwen36.imatrix \
       --custom-q "blk\.([0-4]|3[5-9])\..*ffn_(gate|up|down)_exps\.weight=iq4_ks" \
       --custom-q "blk\.([5-9]|[12][0-9]|3[0-4])\..*ffn_(gate|up|down)_exps\.weight=iq3_k" \
       --custom-q ".*ffn_(gate|up|down)_shexp\.weight=q8_0" \
       --attn-q-type iq5_ks --attn-k-type iq5_ks --attn-v-type iq5_ks --attn-output-type iq5_ks \
       --ffn-gate-inp-type q8_0 --token-embedding-type iq4_ks --output-tensor-type q6_K \
       --mtp-requantize-output-tensor iq4_ks \
       qwen36-bf16.gguf qwen36-262k-balanced.gguf iq4_ks
     ```
     (placeholder regexes — finalize against the real tensor list via `gguf-dump`/
     `llama-gguf` once the BF16 GGUF is in hand; layer ranges/tensor names must match
     exactly or `--custom-q` rules silently no-op.)
  5. **Vision projector**: reuse Unsloth's `mmproj-BF16.gguf` as-is first (small, ~1GB,
     likely fine at full precision); only requantize it if VRAM pressure demands it.
- Decide storage: produce these GGUFs **offline** (outside the Docker build, this is a
  multi-hour/large-disk operation) and mount via the existing `/models` volume; the
  Dockerfile/entrypoint only need to know final filenames via `.env`.

### Phase 3 — Server entrypoint & config
- Rewrite `docker/entrypoint.sh` for ik_llama flags:
  - `-m <qwen36 gguf> -ngl 999 -fa on -fmoe`
  - `-c 0`/auto-fit for max context (`-fit on` equivalent if present) or explicit model max
  - `-ctk q4_0 -ctv q4_0` + Hadamard enable flags
  - MTP speculative flags (draft = MTP head)
  - `--mmproj <file>` (or `--no-mmproj` when `ENABLE_VISION=0`)
  - `--cache-ram <MiB>` prompt cache; `--parallel`, `--jinja`, sampling defaults,
    `--slot-save-path` (optional)
  - **Do NOT pass `-rtr` / do not use `_R4`/`_R8` quants.** Run-time repacking targets
    row-interleaved CPU-SIMD layouts and only helps tensors computed on the CPU. With the
    model fully offloaded to the single GPU it yields no gain, disables mmap, and can
    force matmuls onto the CPU (repacked k-quants have no CUDA row-interleaved kernel) ->
    slower prefill. Keep the non-interleaved CUDA quants (`iq4_ks`/`iq3_k`/...) and full
    `-ngl`. (`-rtr` would only make sense in a CPU/hybrid `-ot ...exps=CPU` config.)
- Rewrite `docker/.env` + `docker/.env.example`: drop all DSpark/Qwen3.6-27B knobs; add
  `HF_REPO`, `HF_FILE`, `MMPROJ_FILE`, `KV_TYPE(_K/_V)`, `KV_HADAMARD`, `ENABLE_MTP`,
  `ENABLE_VISION`, `CTX`, `CACHE_RAM_MIB`, `N_PARALLEL`, sampling.

### Phase 4 — Quant recipe as code
- Encode the section-4 recipe into the quantize script (regex rules resolved against the
  actual tensor list); parameterize `NUM_LAYERS` and edge width.

### Phase 5 — Router & docs
- `router/config.yaml`: rename `qwen3.6-27b` -> `qwen3.6-35b` (keep CUDA + Mac deployments,
  latency routing). Keep LiteLLM (native router skipped per req 9).
- Rewrite `README.md` for the new stack; keep `docs/dspark-integration-plan.md` as history.

### Phase 6 — Bring-up & functional verification
- Start the server with the final recipe GGUF + mmproj, `-ctk q4_0 -ctv q4_0` + Hadamard
  flags, `--spec-type mtp:n_max=4,p_min=0.0`, full context.
- Functional smoke tests (curl against `/v1/chat/completions` etc.) — each must pass
  before moving to Phase 7 benchmarking; a failure here is stop-the-line:
  1. **Basic completion**: short prompt, confirms the model loads and generates coherent
     text at all.
  2. **Vision**: send a request with an attached image (base64 `image_url`); confirm the
     response demonstrates the model actually looked at the image content, not just text.
  3. **MTP engaging**: check server logs / bench stats for an MTP acceptance rate > 0
     (confirms the speculative path is really firing, not silently falling back to plain
     decode).
  4. **Prompt cache hit**: send the same long (~8K+ token) prefix twice in a row; confirm
     the second request's prefill/time-to-first-token is dramatically faster than the
     first (cache hit), not just similar.
  5. **Long-context stability**: push a prompt near the 262144-token max (synthetic filler
     is fine); confirm no OOM/crash — validates the recurrent DeltaNet state at scale.
  6. **All-four-together (R3, the actual requirement-7 acceptance bar)**: repeat test 4
     (long cached prefix) **while** the conversation also includes an image **and** MTP
     is enabled — single combined request/session proving prompt-cache + hybrid + vision
     + MTP genuinely coexist, not just each in isolation.

### Phase 7 — Performance benchmarking (tokens/sec)
- **7a. Raw throughput** via ik_llama's built-in `llama-bench` (no HTTP overhead), swept
  over context depth and batch size:
  ```
  ./build/bin/llama-bench -m qwen36-262k-balanced.gguf -ngl 999 -fa 1 -fmoe 1 \
    -ctk q4_0 -ctv q4_0 -p 512,4096,32768,131072,262144 -n 128 -b 2048 -ub 2048 -r 3
  ```
  Records prompt-processing (`pp`) and token-generation (`tg`) tok/s at each depth — the
  direct measurement for requirement 8 (does max context cost generation speed?).
- **7b. MTP speculative speedup**: repeat the `tg` benchmark with
  `--spec-type mtp:n_max=4,p_min=0.0` vs without, same prompts; report the tok/s delta and
  MTP acceptance rate. Sanity-check against the real-world reference already gathered in
  research (`sayap`, PR #1987: 127-180 tok/s @ 93-98% acceptance on a single/dual 3090 for
  Qwen3.6-27B) — our 35B-A3B numbers should land in a broadly similar range.
- **7c. Vision decode latency**: time an end-to-end `/v1/chat/completions` request with
  one attached image (curl + `time`, or a short Python script timing the streamed
  response); separate image-encode time (time-to-first-token) from post-image generation
  tok/s.
- **7d. Prompt-cache reuse speedup**: repeat the same long-prefix request twice; record
  time-to-first-token and effective tok/s for the cached vs. uncached run — the speedup
  ratio is the requirement-7 prompt-cache metric.
- **7e. Quant-quality sanity**: `llama-perplexity -m qwen36-262k-balanced.gguf -f
  wikitext-2-raw/wiki.test.raw`, compared against a `q8_0` quantization of the same base
  GGUF; delta should land in the same few-tenths-of-a-point ballpark as the section-4 PPL
  curve, not multiples (a large delta means a broken `--custom-q` regex, not just "a
  smaller quant").
- Record every run in `docs/qwen36-bench-results.md`: GGUF file, git commit of ik_llama,
  GPU/driver/CUDA version, exact command, `pp`/`tg` per context depth, MTP acceptance %,
  PPL delta — so future recipe/flag changes can be compared like-for-like instead of
  re-benchmarked from memory.
- **Acceptance bar** (sanity thresholds, not hard science): generation throughput should
  stay in the double-digit-to-low-hundreds tok/s range at every tested context depth on a
  single 3090 for a 35B-A3B MoE; MTP acceptance rate should be >80% (below that, the
  speculative overhead may not pay for itself — disable `--spec-type` if so); PPL delta vs
  `q8_0` should be a few tenths, not multiples.

## 6. Risks / to verify at implementation

- **R1 Vision**: RESOLVED - the model card confirms `Qwen/Qwen3.6-35B-A3B` is a Causal LM
  **with a built-in vision encoder** (`image-text-to-text`, arch `qwen3_5_moe`), and
  `examples/mtmd/clip.cpp` has a complete `PROJECTOR_TYPE_QWEN3VL` implementation (section
  0 item 3). We don't need our own converter to emit the mmproj — Unsloth already ships
  `mmproj-BF16.gguf` for this exact model (section 0 item 4, Phase 2). Remaining task:
  Phase 6 test 2 confirms it actually loads and produces sane output on our build.
- **R2 Hadamard flags**: RESOLVED in source (section 0 item 1 — generic `k_cache_hadamard`/
  `v_cache_hadamard` via `build_std_attention()`, head_dim 256 is power-of-2 compatible).
  Remaining task: Phase 0 re-confirms the exact CLI flag spelling (`-khad`/`-vhad` or
  renamed) against the actual built binary's `--help` before writing Phase 3's entrypoint.
- **R3 All-four-together**: prompt cache + hybrid + vision + MTP in a single ik_llama
  server instance — the core validation of requirement 7 — is addressed by Phase 6 test 6
  (combined functional smoke test) and Phase 7b-7d (combined performance measurement); not
  yet run against a real server, but no longer an open source-level question (all four
  mechanisms are individually confirmed real in section 0).
- **R4 Layer count**: RESOLVED - **40 layers** (hybrid 10x(3x DeltaNet->MoE + 1x
  Attention->MoE); only the 10 attention layers carry KV). Edges L0-4 & L35-39, middle L5-34.
- **R5 24 GB fit**: at 262K + vision the non-weight budget is ~5 GB (KV ~1.5, deltanet
  state ~0.3, mmproj ~1.0, prefill buffers ~2.0, MTP ~0.2), so weights must be ~16-17 GB.
  The 262K-Balanced recipe (~16.5 GB) fits with ~2.5 GB margin; tune quant tier / ubatch /
  KV type if a larger ubatch or >262K context is wanted.
- **R6 Mac backend**: cannot run this ik_llama build; the router keeps it as a separate
  backend only.

## 7. Explicitly out of scope / dropped

- 1-bit Qwen3.6-27B (Q1_0), the TurboQuant+ fork, DSpark drafter, and all DSpark env/flags.
- Native `llama.cpp` router `.ini` mode (not in ik_llama; requirement 9 skipped).
