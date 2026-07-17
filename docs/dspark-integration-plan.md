# DSpark integration plan (turboquant fork)

Goal: port PrismML's DSpark block-diffusion drafter into
`JEF1056/llama-cpp-turboquant` (branch `bonsai`) and integrate it with the
fork's **multi-sequence** speculative framework so that speculative decoding and
prompt/prefix caching run **at the same time** - eliminating the vendor demo's
"spec OR cache, pick one" trade-off and its 16K context cap.

This is the "fix the spec-decoding + prompt-cache issue" half of the task. The
Docker/base-Bonsai half is already done (see `docker/` + notes below).

> Why this is a plan and not a patch: it is ~6,800 lines across ~85 files and
> introduces a whole new arch + quant type + drafter subsystem. Per the fork's
> `AGENTS.md`, large new-pattern changes must be reviewed and owned by a human,
> and every step must keep the `bonsai` branch build-clean. It also cannot be
> compile-verified without the CUDA toolchain + a full fork build. Land it in
> the ordered, independently-buildable milestones below.

## Current state (already true in the fork)

- Base Bonsai inference WORKS: `Q1_0` (enum 41, `QK1_0=128`) is byte-identical
  to the vendor block and has full CUDA support (`mmvq`/`mmq`/`vecdotq`/`convert`
  /`getrows` + `template-instances/mmq-instance-q1_0.cu`). `LLM_ARCH_QWEN35` /
  `QWEN35MOE` are fully wired in `src/llama-arch.cpp` + `src/llama-model.cpp`,
  and `src/models/qwen35.cpp` exists. The vendor `Bonsai-27B-Q1_0.gguf` loads
  as-is.
- Multi-seq speculative framework exists: `common_speculative_init(params,
  n_seq)`, per-seq `common_speculative_get_draft_params`,
  `common_speculative_process(batch)`, `common_speculative_need_embd` /
  `need_embd_nextn`. Spec types: none, draft-simple, draft-eagle3, draft-mtp,
  ngram-*. There is NO draft-dspark yet.
- `cache_prompt` / `--cache-reuse N` and speculative are INDEPENDENT server
  knobs here (the coupling that disables the cache is a PrismML artifact, not
  present in this fork). This is exactly why the fix is achievable.
- MISSING: `Q2_0` weight quant (enum 42, needed only for the DSpark verify
  path), `LLM_ARCH_DSPARK`, `src/models/dspark.cpp`, the drafter kernels, and
  the server tap-capture wiring.

## Prism commits to port (branch `prism`, merge-base `8a963fc1`)

Q2_0 weight quant + kernels:
- `984bf9723` core Q2_0 type, `0e0353d7a`/`4d88cd4eb` CPU (AVX512 VNNI),
  `9c0edeadd` CUDA, `34dc5812c` Vulkan, `f51b5aa19` Metal,
  Hopper set `21eff18c9`/`6f25c34b0`/`02860ff13`/`fdb74a023`/`3ff0bfc71`,
  `41e362dac` CUDA Q1_0 extract.

DSpark drafter:
- `ba62b7023` (#55) core drafter + CUDA Markov resample,
- `b28d513e4` (#57) converter log-SNR embedding,
- `972086d74`/`da9c580e3` (#63) independent unmasked-capture path,
- `c024aa26e` (#67) server tap-capture.
- SKIP for now: `3560f10cc` (#59) + `887f007b0` (#64) Metal (Mac uses MLX, not
  llama.cpp) and CI/test-only commits.

Optional KV mean-center (only if TurboQuant KV regresses on Bonsai long ctx):
- `afc74b756` (#51), `a18e55e49`/`f28050ecf`/`3277aaab2` (#52), `a5527fc87`
  (#53).

## Milestones (each must build clean before the next)

### M1 - Q2_0 quant type (additive, cross-backend)
- `ggml/src/ggml-common.h`: add `block_q2_0` (mirror prism; `QK2_0=128`, 2-bit).
- `ggml/include/ggml.h` + `ggml/src/ggml.c`: add `GGML_TYPE_Q2_0` (enum 42) +
  complete `type_traits` entry (blck size, quant/dequant refs, vec_dot).
- CPU ref quant/dequant so the type is usable without a GPU, then the AVX512
  path.
- CUDA: `convert.cu`, `vecdotq.cuh`, `mmvq.cu`, `mmq.cu(h)` +
  `template-instances/mmq-instance-q2_0.cu`.
- Vulkan last (only if you build Vulkan). Verify: `test-quantize-fns` /
  `test-backend-ops` for Q2_0 on CPU + CUDA.

### M2 - DSpark arch definitions (additive, no behaviour yet)
- `src/llama-arch.{h,cpp}`: add `LLM_ARCH_DSPARK`, its name, the 9 KVs
  (`dspark.block_size`, `mask_token_id`, `target_layers`, `markov_rank`,
  `confidence_head`, `confidence_head_with_markov`, `log_snr_conditioning`,
  `min_log_snr`, `max_log_snr`) and the 7 tensors (`dspark.fc`,
  `dspark.hidden_norm`, `dspark.markov_head_a`/`_b`, `dspark.confidence_head`,
  `dspark.log_snr_fc1`/`_fc2`). Definitions only - compiles, dormant until a
  dspark GGUF is loaded.

### M3 - DSpark model load + graph
- `src/llama-model.cpp`: hparams load + tensor load for `LLM_ARCH_DSPARK`
  (prism `+102`).
- `src/models/dspark.cpp` (new, prism `+408`): the block-diffusion build graph
  (log-SNR embedding -> fc -> Markov resample heads -> confidence head).
- Register it in the model dispatch. Verify: load a dspark GGUF headers-only
  (no decode yet).

### M4 - target hidden-state tap (unmasked capture)
- `src/models/qwen35.cpp` (prism `+120`): expose the target's hidden states at
  `dspark.target_layers` as a graph output.
- `src/llama-graph.cpp` / `llama-context.cpp` / `llama-kv-cache.cpp`: this is
  the HIGH-CONFLICT part - the fork diverged most here. Re-implement prism's
  capture path against the fork's structures; do NOT cherry-pick. Keep the
  capture as an extra output that coexists with the normal (cached) decode.

### M5 - integrate as a `draft-dspark` speculative type (THE FIX)
- `common/speculative.{h,cpp}` + `common/arg.cpp`: add `draft-dspark` to the
  spec-type list. Implement it against the fork's multi-seq API:
  `need_embd`/`need_embd_nextn` requests the target hidden states, the drafter
  produces a block of candidates, `common_speculative_process(batch)` verifies.
  Because the fork's cache and spec are independent, the tap-capture happens
  DURING normal cached decode - so prompt caching stays on. This is the
  behavioural difference from the vendor demo.
- `tools/server/server-context.cpp`: wire the tap output through the fork's
  (refactored) server path (prism `+173`, but against the NEW structure).
  Verify: run with `--spec-type draft-dspark` AND `--cache-reuse 256` together;
  confirm a cache hit on a repeated prefix while spec decode is active.

### M6 - Docker + validation
- Flip `docker/entrypoint.sh` to also download the DSpark drafter GGUF and pass
  `--spec-type draft-dspark` (+ its `--model-draft`/tap flags) once M5 lands.
  Today the entrypoint intentionally omits DSpark and always keeps the cache on.
- A/B vs the vendor demo: decode speedup, cache reuse works, tool calling, 262K
  context, output-quality parity on Bonsai.

## Guardrails (from the fork AGENTS.md)
- ASCII only in fork code/comments (no emdash/arrow/unicode); concise comments.
- Keep `bonsai` building clean at every milestone; never half-finish across
  files.
- Do NOT `git push` / open a PR / commit on the user's behalf. If asked to
  commit, use `Assisted-by:` (never `Co-authored-by:`).
- Private fork => exempt from the no-AI-PR policy, but the human must be able to
  explain and maintain every line.
