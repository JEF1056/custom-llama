# Qwen3.6-35B-A3B bring-up & benchmark results

First real-hardware run of the iqllama migration (see
[iqllama-migration-plan.md](iqllama-migration-plan.md)). This run uses
Unsloth's public pre-quantized IQ3_XXS quant as a bring-up/smoke-test quant,
**not** our own 262K-Balanced recipe — the `model-prep` pipeline (BF16
download -> imatrix -> `quantize.sh`) has not been run yet. Numbers here
characterize the engine/architecture, not the final quant's quality. Two
variants of the bring-up quant were used: an initial one from
`unsloth/Qwen3.6-35B-A3B-GGUF` (no MTP tensors), later replaced by the same
quant from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (has MTP tensors) once that
repo was found — see "MTP status" below.

## Environment
- GPU: RTX 3090, 24GB VRAM, driver 610.43.02, CUDA 12.6.3 (WSL2 + Docker
  Desktop, `--gpus all` passthrough).
- Engine: stock `ikawrakow/ik_llama.cpp@main`, built with
  `GGML_CUDA=ON -DGGML_CUDA_F16=ON`, sm_86.
- Model: `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (13.1-14.1GB depending on repo) +
  `mmproj-BF16.gguf` (861MB vision tower), from Unsloth.
- Server flags: `-fa on --jinja -ctk q4_0 -ctv q4_0 -khad -vhad --cache-ram 8192
  --parallel 1 -ub 1024`, `CTX=262144` (native, no truncation), fused-MoE
  default-on (no `-fmoe`/`-no-fmoe` override), `--spec-type mtp:n_max=4,p_min=0.0`.
- Memory at load: 12,464.76 MiB reported for model tensors + non-KV buffers;
  41/41 layers offloaded to GPU (CUDA0 buffer 12,190.81 MiB); ~22,220 MiB
  available for compute/KV on top of that.

## Phase 6 — Functional smoke tests

| # | Test | Result |
|---|------|--------|
| 1 | Basic completion | **PASS** — coherent response, reasoning_content populated (thinking mode) |
| 2 | Vision (image input) | **PASS** — CLIP encoder loads (`has vision encoder`, CUDA0 backend); model produces a coherent description referencing the attached image |
| 3 | MTP engaging | **PASS** — 76.2% draft acceptance rate (112/147 tokens), generation throughput 130.8 tok/s (up from ~105-120 without MTP); see "MTP status" below |
| 4 | Prompt-cache hit | **PASS** — 54x prompt_ms speedup on identical-prefix replay (see Phase 7d) |
| 5 | Long-context stability | **PASS** — 26,451-token prompt processed successfully at `n_ctx=262144`, no OOM/crash |
| 6 | All-four-together (R3) | **PASS** (each leg independently demonstrated: long cached prefix, vision, MTP, all against the same server/quant) |

### MTP status: available, from a separate Unsloth repo (not the default one)
First attempt: re-enabling `ENABLE_MTP=1` against the original bring-up quant
(from `unsloth/Qwen3.6-35B-A3B-GGUF`) failed:
`"MTP speculative stage requested, but the server was not started with MTP
support"`. Root-caused, not just retried:
- The **original** `Qwen/Qwen3.6-35B-A3B` HF safetensors checkpoint genuinely
  has MTP weights (`mtp.fc.weight`, `mtp.layers.0.*`, `mtp.norm.weight`,
  `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`) —
  confirmed via `model.safetensors.index.json`.
- But **`unsloth/Qwen3.6-35B-A3B-GGUF`'s conversion drops these `mtp.*`
  tensors entirely** — confirmed by parsing the raw GGUF header (via HTTP
  range request, no full download needed) of both its bring-up IQ3_XXS quant
  (733 tensors) and its BF16 source GGUF (534+199=733 tensors): zero
  MTP-related tensor names in either.
- **Searching HF's model listing for other Qwen3.6-35B-A3B GGUF repos (not
  just re-checking the one already in use) turned up
  `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`** — a separate repo, same file layout,
  but WITH the MTP layer included as an extra `blk.40.*` transformer block
  (not a separate `mtp.` namespace): `blk.40.nextn.eh_proj.weight`,
  `blk.40.nextn.enorm.weight`, `blk.40.nextn.hnorm.weight`,
  `blk.40.nextn.shared_head_norm.weight` plus standard attn/ffn tensors for
  that block. Confirmed in both its BF16 source (219 vs 199 tensors) and its
  pre-quantized IQ3_XXS (753 vs 733 tensors).
- Downloaded `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` from the MTP repo, set
  `GGUF_FILE=qwen36-bringup-mtp.gguf` + `ENABLE_MTP=1`, restarted — logs show
  `common_speculative_state_mtp: MTP context ready` and `speculative decoding
  context initialized`; a real completion request returned
  `draft acceptance rate = 0.76190 (112 accepted / 147 generated)` in the
  server logs and `draft_n`/`draft_n_accepted` in the per-request `timings`
  JSON.
- `scripts/download-source-gguf.sh`'s default `SRC_REPO` and
  `docker/.env(.example)`'s default `HF_REPO` were updated to
  `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` so our own Phase 2 pipeline produces an
  MTP-capable 262K-Balanced quant too.

### MTP output-tensor precision: q8_0 vs bf16 (A/B tested on real hardware)
The MTP block's non-output tensors (attn/ffn) are kept at BF16 in our
recipe (matching the vision-tower policy of not quantizing small
accuracy-sensitive auxiliary components). The specific output-projection
tensor (`blk.40.nextn.eh_proj.weight`, controlled at runtime by
`MTP_REQUANTIZE_OUTPUT_TYPE` / `--mtp-requantize-output-tensor`) was A/B
tested directly rather than assumed, 3 runs each, same prompt:

| variant | avg predicted tok/s | draft acceptance rate |
|---|---|---|
| default (baked-in, already Q8_0 in this quant) | ~127.97 | 0.735-0.783 |
| explicit q8_0 | **~131.5** | 0.735-0.750 |
| explicit bf16 | ~125.1 (slower) | 0.690-0.778 |

**q8_0 wins** — consistently faster with no meaningful acceptance-rate
benefit from the extra bf16 precision (acceptance-rate variation across all
three is noise-level, not a trend). Set as the runtime default
(`MTP_REQUANTIZE_OUTPUT_TYPE=q8_0`) and baked into `scripts/quantize.sh`
(`--mtp-requantize-output-tensor q8_0`).

## Phase 7 — Performance benchmarking

### 7a. Raw throughput (measured via server HTTP timings, not standalone `llama-bench`
`llama-bench` was not included in this build's targets (added to `docker/Dockerfile`
for the *next* rebuild); these numbers come from the `timings` field the
server returns per-request, which reports the same `prompt_per_second`/
`predicted_per_second` metrics `llama-bench` would, with a small constant HTTP
overhead.

| prompt words | prompt_n (tokens) | prompt tok/s | predicted_n | gen tok/s | wall (s) |
|---|---|---|---|---|---|
| 64    | 86    | 209.11  | 64 | 114.96 | 1.00 |
| 512   | 549   | 1913.29 | 64 | 106.75 | 1.05 |
| 2048  | 2134  | 2425.33 | 64 | 110.08 | 1.52 |
| 8192  | 6428  | 2536.64 | 64 | 104.14 | 3.19 |

Generation throughput holds essentially flat (~105-115 tok/s) from an 86-token
to a 6,428-token prompt — no measurable falloff with context depth, consistent
with the hybrid architecture (only 10/40 layers pay full-attention KV cost;
the 30 DeltaNet layers use a fixed-size recurrent state independent of
context length). Prompt processing throughput scales up with batch size as
expected (209 -> 2537 tok/s).

### 7a-bis. Physical batch size (`-ub`) VRAM/speed sweep
Swept by restarting the server with different `-ub` values (via `EXTRA_ARGS`)
and measuring `nvidia-smi` memory + prompt-processing tok/s on an 8192-word
prompt (same IQ3_XXS quant):

| `-ub` | VRAM used | prompt tok/s |
|---|---|---|
| 256  | ~15.8GB | ~1818 |
| 512 (ik_llama.cpp default) | ~16.1GB | ~2580 |
| **1024 (chosen default)** | **~16.6GB** | **~3192** |
| 2048 (= `-b` default) | ~17.7GB | ~3567 |

Generation (decode) throughput is unaffected by `-ub` at any value tested
(~104-110 tok/s throughout) — `-ub` only affects prompt-processing (prefill)
parallelism. **1024** was chosen as the new default: +24% prompt tok/s over
the ik_llama.cpp stock default of 512 for only ~500MiB extra VRAM (best
tok/s-per-MiB step in the sweep; 1024->2048 has worse marginal efficiency),
while leaving ~8GB headroom on this 13GB quant for a larger production quant
later. Added as a proper `UBATCH_SIZE` env var in `docker/entrypoint.sh`
(previously only tested via the `EXTRA_ARGS` hack).

### 7b. MTP speculative speedup
Same prompt (photosynthesis explanation, 150 max tokens, temp=0.2), with vs.
without `--spec-type mtp:n_max=4,p_min=0.0`:

| mode | predicted tok/s | draft acceptance rate |
|---|---|---|
| without MTP | ~105-120 (Phase 7a range) | n/a |
| with MTP | **~127-134** (avg ~131.5 with q8_0 output tensor) | **~74-78%** |

MTP gives a real, measurable speedup (roughly +10-25% generation tok/s) at a
~76% draft acceptance rate. This is below the migration plan's ">80%"
acceptance-rate sanity threshold, plausibly because this is Unsloth's
aggressive IQ3_XXS bring-up quant (heavy MoE-expert quantization, Q2_K/Q3_K)
rather than our own recipe's more conservative expert quantization — worth
revisiting once the real 262K-Balanced quant exists.

### 7c. Vision decode latency
Single data point from the Phase 6 vision smoke test: a 1x1 test image +
short text prompt (32 total prompt tokens including image tokens) processed
in `prompt_ms=1382.8` (`23.1 tok/s` effective, dominated by CLIP encode fixed
cost, not token count) with generation at `119.3 tok/s` afterward — consistent
with the text-only generation throughput above, confirming vision encoding
adds a fixed latency cost but doesn't slow subsequent token generation.

### 7d. Prompt-cache reuse speedup
12,402-token prompt sent twice back-to-back:

| run | prompt_n | prompt_ms | cached_tokens | wall (s) |
|---|---|---|---|---|
| cold  | 4,215 | 1970.7 | 8,192  | 2.33 |
| warm  | 5     | 36.4   | 12,402 | 0.34 |

**54.1x** prompt_ms speedup on full cache hit (cold run already had partial
cache reuse of 8,192 tokens from prior sweep requests sharing filler-text
n-grams, hence `prompt_n=4215` instead of the full 12,402 — the warm run's
`prompt_n=5` shows only the trailing few tokens needed re-processing).

### 7e. Quant-quality sanity (perplexity)
**Not run this pass** — would require downloading a second quant (e.g. Q8_0,
~37GB) of the same base model for a delta comparison, or running
`llama-perplexity` against wikitext-2 with only the single IQ3_XXS quant on
hand (an absolute PPL number without a same-recipe baseline isn't very
actionable). Deferred until our own 262K-Balanced GGUF exists, at which point
the natural comparison is 262K-Balanced vs. a Q8_0 quant of the same BF16
source, per the original plan.

### Long-context stability (bonus, folded into Phase 6 test 5 / Phase 7a context)
A 26,451-token prompt (well beyond the 8,192-token cache tests) processed
successfully at `n_ctx=262144` (the model's full native context, not a
truncated bring-up value): `prompt_per_second=2627.49`,
`predicted_per_second=84.84`, wall time 10.90s for prompt + 32 generated
tokens. No OOM/crash. Confirms the hybrid DeltaNet+attention design keeps
long-context prompt processing fast and generation throughput stable even
approaching the 262K window on a single 24GB card, for this 13GB quant.

## Summary / acceptance bar check
- Generation throughput (~105-134 tok/s across all tested depths, higher with
## Final setup benchmark (task suite + 100K+ long-context sweep)

Run against the final tuned configuration: `qwen36-bringup-mtp.gguf`
(IQ3_XXS, MTP-capable), `ENABLE_MTP=1`, `MTP_REQUANTIZE_OUTPUT_TYPE=q8_0`,
`UBATCH_SIZE=1024`, `CTX=262144`, `-khad -vhad -ctk q4_0 -ctv q4_0`.

### Task suite (128 max tokens, temp=0.2, short prompts)

| task | prompt_n | prefill tok/s | gen_n | gen tok/s | draft acc |
|---|---|---|---|---|---|
| factual_qa | 25 | 44.48 | 128 | 121.66 | 0.715 |
| code_gen | 28 | 366.77 | 128 | 120.60 | 0.676 |
| summarization | 23 | 298.83 | 128 | 109.17 | 0.593 |
| math_reasoning | 48 | 529.22 | 128 | 139.99 | 0.845 |
| creative_writing | 20 | 289.06 | 128 | 119.10 | 0.692 |

Generation throughput is consistently in the ~109-140 tok/s range across task
types, with MTP draft acceptance varying by content predictability (math
reasoning's more templated working-out accepts drafts best at 84.5%; more
open-ended prose tasks land in the 59-72% range). Prefill tok/s for these
short prompts is noisy (dominated by fixed per-request overhead, not batch
parallelism) and not meaningfully comparable across tasks at this length.

### Long-context sweep, 4K-172K tokens (cold prefill, unique content per depth)
First attempt at this sweep used a repeating small word-cycle across depths,
which the server's prompt cache correctly recognized as a shared prefix and
mostly served from cache — inflating apparent throughput and yielding
misleadingly low `prompt_n` at higher depths despite `n_past` correctly
reaching the full requested length with `truncated=false` throughout (no
data corruption, just an invalid *cold-prefill* benchmark methodology). Fixed
by using unique random content per depth and `"cache_prompt": false` in each
request.

**A second methodology bug produced a real crash**: an interim run used a
fabricated out-of-vocabulary token pool (`tok12345` style strings) which the
tokenizer expanded ~5-6x beyond the target word count via BPE fragmentation,
silently exceeding `n_ctx=262144` for the larger depths. The server did not
reject this with a clean HTTP 4xx — it crashed (confirmed via
`docker inspect server --format '{{.State.Restarting}} {{.RestartCount}}'`
showing `RestartCount=1`), and Docker's `restart: unless-stopped` policy
auto-recovered it (model reload took ~20s). **Flagged as a real finding, not
just a benchmark artifact**: ik_llama.cpp's server should ideally reject
prompts exceeding `n_ctx` with a clean error rather than crashing; worth a
closer look (or an upstream issue) before relying on this in production
without an application-level token-count guard in front of it.

Final clean sweep (realistic word-based content, ~1.0-1.3 tokens/word,
`cache_prompt: false`, 32 max generated tokens, temp=0.0):

| target words | prompt_n (tokens) | prefill tok/s | gen_n | gen tok/s | draft acc | wall (s) |
|---|---|---|---|---|---|---|
| 4,000   | 4,070   | 2051.56 | 32 | 127.71 | 0.793 | 2.29 |
| 20,000  | 20,288  | 2598.59 | 32 | 122.69 | 0.821 | 8.27 |
| 50,000  | 50,687  | 2316.75 | 32 | 121.32 | 0.923 | 22.64 |
| 80,000  | 80,997  | 2055.01 | 32 | 106.30 | 0.889 | 40.77 |
| 110,000 | 111,453 | 1812.98 | 32 | 89.41  | 0.885 | 63.24 |
| 140,000 | 141,838 | 1612.36 | 32 | 89.35  | 0.923 | 89.96 |
| 170,000 | 172,133 | 1436.37 | 32 | 81.66  | 0.889 | 122.17 |

Both prefill and generation throughput decline gradually with context depth
(prefill: 2599 -> 1436 tok/s from 20K to 172K tokens; generation: 123 -> 82
tok/s over the same range) rather than collapsing — expected, since the 10
full-attention layers' KV lookups scale with context length even though the
30 DeltaNet layers stay O(1) per token. MTP draft acceptance stayed high
(79-92%) at every depth tested, actually trending upward at mid-depths,
confirming MTP remains effective deep into long-context generation, not just
at short prompts. No OOM/crash in this final clean sweep at any depth up to
172K tokens (the earlier crash was specifically the malformed-tokenization
bug above, not a real 172K-token failure).

## Summary / acceptance bar check
- Generation throughput (~82-140 tok/s depending on task/context depth, with
  MTP) is well within the plan's "double-digit-to-low-hundreds tok/s" sanity
  bar for a 35B-A3B MoE on a single 3090.
- Prompt-cache reuse (54x) and long-context stability (262144 native, no
  crash on well-formed prompts up to 172K tokens tested) both clearly pass.
- Vision works end-to-end.
- MTP works end-to-end (~76-92% draft acceptance depending on content/depth,
  real speedup) once pointed at the correct Unsloth repo (`-MTP-GGUF`
  variant); the output-tensor precision (q8_0 vs bf16) was A/B tested and
  q8_0 kept as it's faster with no acceptance-rate cost.
- ubatch size (`-ub`) was empirically tuned to 1024 (see the ubatch sweep
  section above / repo memory) — balances VRAM headroom against
  prompt-processing throughput.
- **Open item**: the server crashes (rather than cleanly rejecting) on a
  prompt whose tokenized length exceeds `n_ctx`; add an application-level
  token-count guard or investigate an upstream fix before production use
  with untrusted/unbounded input.

## Follow-up work
1. Add `llama-bench` to the next image rebuild (already added to
   `docker/Dockerfile`) and re-run 7a with it for HTTP-overhead-free numbers.
2. Run `model-prep` (full BF16 -> imatrix -> 262K-Balanced quantize) against
   `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (now the default `SRC_REPO`/`HF_REPO`)
   once the `--custom-q` regexes in `scripts/quantize.sh` are verified
   against real tensor names (`llama-gguf-dump` once the BF16 GGUF is
   downloaded).
3. Re-run this full benchmark suite (including the MTP acceptance rate and
   ubatch sweep) against the real 262K-Balanced quant for an apples-to-apples
   comparison with these bring-up numbers.
4. Perplexity comparison (7e) still deferred — needs a second quant of the
   same base model to compare against.
5. Investigate/guard against the n_ctx-exceeded crash found during long-context
   benchmarking (see "Long-context sweep" above): add a request-side token
   count check in the router/client, or find/file an upstream ik_llama.cpp
   issue for a graceful rejection instead of a crash.

