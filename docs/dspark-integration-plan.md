# DSpark integration plan (turboquant fork)

Goal: port PrismML's DSpark block-diffusion drafter into
`JEF1056/llama-cpp-turboquant` (branch `bonsai`) and integrate it with the
fork's **multi-sequence** speculative framework so that speculative decoding and
prompt/prefix caching run **at the same time** - eliminating the vendor demo's
"spec OR cache, pick one" trade-off and its 16K context cap.

This is the "fix the spec-decoding + prompt-cache issue" half of the task. The
Docker/base-Bonsai half is already done (see `docker/` + notes below), and
vision is now wired into the Docker too (see the vision section).

> Why this is a plan and not a patch: DSpark itself is ~2,100 lines across 12
> core files; with the KV-mean-center bundle, tests and CI it is several
> thousand lines. The shipped drafter is `Q4_1` (a standard, already-supported
> type), so the cross-backend `Q2_0` quant work is very likely NOT needed - see
> M0/M1. It adds a new arch + host-side drafter loop. Per the fork's `AGENTS.md`,
> large new-pattern changes must be reviewed and owned by a human, kept
> build-clean at every step, and this cannot be compile-verified here (no CUDA
> toolchain / full build). Land it in the ordered milestones below.

## What DSpark actually is (from prism `src/models/dspark.cpp`)

DSpark is a **separate, tiny drafter model** (a plain dense Qwen3-style stack,
standard `attn_*`/`ffn_*` tensors) - NOT a modification of the Bonsai target.
The novelty is its attention: at every drafter layer it attends over two
concatenated K/V sources -

1. a small window of the **target's captured multi-layer tap features**
   (`n_dspark_target_layers` target layers concatenated per position, projected
   `n_capture*n_embd -> n_embd` once via `dspark.fc` + `dspark.hidden_norm`),
   re-projected fresh through each layer's own `k_proj`/`v_proj`; and
2. the **draft block's own residual stream** (`block_size` positions, seeded
   from `mask_token_id` + last accepted token).

The two are concatenated and attended fully **non-causally** (caller must
`llama_set_causal_attn(ctx, false)`); only the draft-block rows issue queries,
and the leading context rows are sliced off after `build_attn`. Optional GIDD
**log-SNR** conditioning (`LogSnrEmbed`, host-precomputed sinusoids -> fc1 ->
silu -> fc2) is added to the draft embedding in-graph.

The forward graph produces only the BASE trunk logits. The actual
block-diffusion draft loop - **markov resample** heads (low-rank, optionally
`Q2_0`), **confidence** head, advance-by-`n_accepted`, and cropping the target
cache back with a partial `llama_memory_seq_rm` - is host-side in
`common/speculative.cpp` (`common_speculative_impl_draft_dspark`).

## Current state (already true in the `bonsai` fork)

- Base Bonsai inference WORKS: `Q1_0` byte-identical + full CUDA; `LLM_ARCH_QWEN35`
  / `QWEN35MOE` fully wired; `src/models/qwen35.cpp` present; vendor GGUF loads.
- **Framework parity is HIGH.** prism and `bonsai` share merge-base
  `8a963fc1`, which ALREADY has the multi-seq speculative framework
  (`common_speculative_process`, `need_embd`/`need_embd_nextn`), the `draft-mtp`
  block-verify type, the `QWEN35MOE` hybrid GDN/attention memory, and the
  recurrent-state rollback ring (`need_n_rs_seq` / `llm_arch_supports_rs_rollback`)
  that DSpark's post-verify cache crop depends on. All are still in `bonsai`.
  So DSpark slots into the SAME API prism used - this is a merge/re-apply, not a
  rewrite against a foreign framework.
- `cache_prompt` / `--cache-reuse N` and speculative are INDEPENDENT knobs here.
  This is why spec + cache can coexist (the fix).
- MISSING in `bonsai`: `Q2_0` quant, `LLM_ARCH_DSPARK`, `src/models/dspark.cpp`,
  the tap-capture API (`llama_set_capture_layers` / `llama_get_embeddings_capture_ith`
  / `llama_dspark_ctx` / `llama_set_dspark_ctx` - all prism-new, confirmed absent),
  `common/dspark-markov.*`, and the `draft-dspark` speculative impl.

## The gap prism itself LEFT OPEN (this is the real deliverable)

prism's own note in `common/speculative.cpp`: *"the generic CLI (`--spec-type`)
and server paths do NOT yet engage capture, so selecting `draft-dspark` there
currently"* falls back - only the test harness `tests/test-dspark-real-eval.cpp`
engages `llama_set_capture_layers` and stages the tap. **So even in prism,
`--spec-type draft-dspark` does not work end-to-end on the server.** Finishing
that wiring (engage capture + stage `llama_dspark_ctx` inside the server's
speculative loop, alongside the prompt cache) IS the fix the task asks for.

## File inventory to port (prism vs merge-base `8a963fc1`)

DSpark core (2,131 insertions / 12 files):
- new: `src/models/dspark.cpp` (+408), `common/dspark-markov.{h,cu}` (CUDA/BLAS
  host resample), `conversion/dspark.py` (drafter converter), `docs/dspark-scope.md`.
- changed: `common/speculative.{h,cpp}` (+747, the draft loop), `common/common.h`
  (+`DRAFT_DSPARK`, `need_n_rs_seq`), `src/llama-arch.{h,cpp}` (+arch/KVs/tensors),
  `src/llama-context.{h,cpp}` (+457 capture API), `src/llama-graph.{h,cpp}`
  (+`llm_graph_input_dspark_ctx`/`_logsnr`, capture output), `src/llama-model.{h,cpp}`
  (+`llama_model_dspark`), `src/models/qwen35.cpp` (+120 tap emit), `src/llama-ext.h`,
  `src/llama-cparams.h`, `src/llama-hparams.h`, `include/llama.h`,
  `tools/server/server-context.cpp`, `convert_hf_to_gguf.py`, `gguf-py/*`.

`Q2_0` quant type (very likely NOT needed - the shipped drafter is `Q4_1`):
- new: `ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu`,
  `ggml/src/ggml-cuda/mmq-hopper-q1.cu`, `.../vulkan-shaders/dequant_q2_0.comp`.
- changed: `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-common.h`,
  `ggml/src/ggml-quants.{c,h}`, CPU (`ggml-cpu/*`, arm/x86 `quants.c`), CUDA
  (`convert.cu`, `vecdotq.cuh`, `mmvq.cu`, `mmq.cu(h)`, `getrows.cu`, `common.cuh`,
  `dequantize.cuh`), Metal, Vulkan, `src/llama-quant.cpp`, `tools/quantize/quantize.cpp`.

OPTIONAL - KV mean-center (only if TurboQuant KV regresses on Bonsai long ctx):
- new: `common/kv-mean-center.{h,cpp}`, `tools/kv-mean-center/*`, `docs/kv-mean-center.md`.

SKIP: Metal DSpark (Mac uses MLX, not llama.cpp), `.github/workflows/release-prism.yml`,
prism-only tests unless you want them as ports.

## Prompt-cache compatibility (HARD REQUIREMENT)

The port must keep prompt caching working WITH DSpark. On this fork that is
achievable because both ride the same machinery:

- Bonsai is a hybrid (GDN + attention) arch. `llama_memory_can_shift()` is false
  for it, so the server auto-disables `--cache-reuse` (mid-prompt KV shifting)
  and `ctx_shift` on it. That path is NOT how Bonsai caches.
- `common_context_can_seq_rm(ctx)` returns `COMMON_CONTEXT_SEQ_RM_TYPE_RS`
  ("bounded partial sequence removal") for Bonsai, because `llama_n_rs_seq() > 0`
  - the recurrent-state rollback ring. Prompt reuse then works via **context
  checkpoints + prompt-state cache** (`llama_state_seq_get/set_data_ext`,
  `server_prompt_cache`, `n_ctx_checkpoints=32`, `cache_ram_mib=8192` by
  default). The server explicitly logs "speculative decoding will use
  checkpoints".
- DSpark's block-verify crops the target cache with a PARTIAL
  `llama_memory_seq_rm`, which only works because `need_n_rs_seq()` sized the
  same RS rollback ring (prism already adds `DRAFT_DSPARK` to `need_n_rs_seq`).
  So DSpark and prompt caching use the SAME RS/checkpoint path -> compatible by
  construction, no re-prefill-every-request coupling.

Requirements this imposes on the port:
1. Keep `DRAFT_DSPARK` in `need_n_rs_seq()` (prism does).
2. The M6 server wiring must engage capture WITHOUT forcing a full re-prefill,
   and must leave `cache_ram_mib`/`n_ctx_checkpoints` untouched (do NOT copy any
   vendor "disable cache when spec on" logic - it does not exist here, keep it
   that way).
3. Verify explicitly: a second request sharing a prefix restores from a
   checkpoint (prompt-cache hit) WHILE `--spec-type draft-dspark` is active.

Docker already reflects this: `entrypoint.sh` uses `--cache-ram` (checkpoint
prompt-state cache), not `--cache-reuse`.

## Vision tower (SUPPORTED - wired into Docker)

Bonsai-27B is a reasoning + vision model, and the vision tower loads on this
fork today with NO new code:

- The vendor ships the tower as a separate mmproj GGUF alongside the text model
  in `prism-ml/Bonsai-27B-gguf`: `Bonsai-27B-mmproj-Q8_0.gguf` (~629 MB, the HQQ
  4-bit tower in a `Q8_0` container) and `Bonsai-27B-mmproj-BF16.gguf` (~931 MB
  reference). ~0.46B params, 27 blocks.
- Bonsai derives from Qwen3.6-27B, and its visual encoder maps onto the fork's
  existing `qwen3vl_merger` projector plus the layerwise `qf_proj_blocks` /
  `vision_feature_layer` / `proj_spatial_offsets` path already in
  `tools/mtmd/clip.cpp`. prism adds NO Bonsai-specific vision code - its only
  clip.cpp diff is a 2-line cosmetic fix (a `printf` with `%s ... %d %d`, a
  missing `__func__` arg and `size_t` args -> `%zu`). That same latent bug sits
  in `bonsai` at `clip.cpp:1660`; worth porting for hygiene, but it only fires
  on the vision_feature_layer/proj_spatial_offsets size-mismatch error path, so
  it does not block vision.
- Loading: `llama-server -hf` auto-downloads a sibling mmproj (HF-authenticated),
  or you pass it explicitly. The Docker pins it deterministically with
  `--mmproj-url .../Bonsai-27B-mmproj-Q8_0.gguf`; the URL download reuses
  `HF_TOKEN` via `sub_opts.bearer_token`, and giving an explicit file avoids the
  ambiguous auto-pick between the two mmproj packs. `ENABLE_VISION=0` ->
  `--no-mmproj`.
- Vision + cache: the server disables `n_cache_reuse` (KV-shift) for multimodal
  ("cache_reuse is not supported by multimodal"), and per-request images are
  re-encoded (not prefix-cached). But Bonsai does not use cache-reuse anyway -
  the checkpoint/prompt-state cache still applies to the text prefix, so vision
  and prompt caching coexist for the text portion. Verify checkpoints are not
  also disabled under multimodal on this fork.

Status: DONE in Docker (`entrypoint.sh` `ENABLE_VISION` / `MMPROJ_FILE`).
Optional follow-up: port prism's 2-line `clip.cpp` printf fix into the fork.

## Milestones (each must build clean before the next)

### M0 - get the drafter GGUF + decide on Q2_0
- The Bonsai DSpark **drafter** is published in `prism-ml/Bonsai-27B-gguf` as
  `Bonsai-27B-dspark-Q4_1.gguf` (~1.79 GB, default) and
  `Bonsai-27B-dspark-bf16.gguf` (~7.29 GB, reference) - separate from the 27B
  target. No conversion needed; download it (HF-authenticated).
- Both ship as `Q4_1` / `bf16`, which are already fully supported types. So
  **skip M1** - `Q2_0` is not used by the shipped drafter. Only revisit M1 if a
  future drafter build ships `Q2_0` weights.

### M1 - Q2_0 quant type (SKIP unless a future drafter ships Q2_0; additive)
- `ggml/src/ggml-common.h` `block_q2_0`; `ggml/include/ggml.h` + `ggml/src/ggml.c`
  `GGML_TYPE_Q2_0` enum + full `type_traits`; `ggml-quants.{c,h}` ref quant.
- CPU (arm/x86 `quants.c`, `ggml-cpu/*`), then CUDA (`convert.cu`, `vecdotq.cuh`,
  `mmvq.cu`, `mmq.cu(h)` + `mmq-instance-q2_0.cu`, `getrows.cu`). Vulkan only if
  built. `src/llama-quant.cpp` + `tools/quantize` + `gguf-py`.
- Verify: `test-quantize-fns` / `test-backend-ops` for Q2_0 (CPU + CUDA).

### M2 - DSpark arch + model load (additive; dormant until a dspark GGUF loads)
- `src/llama-arch.{h,cpp}`: `LLM_ARCH_DSPARK`, its 9 KVs (`dspark.block_size`,
  `mask_token_id`, `target_layers`, `markov_rank`, `confidence_head`,
  `confidence_head_with_markov`, `log_snr_conditioning`, `min_log_snr`,
  `max_log_snr`) and 7 tensors (`dspark.fc`, `dspark.hidden_norm`,
  `dspark.markov_head_a`/`_b`, `dspark.confidence_head`, `dspark.log_snr_fc1`/`_fc2`).
- `src/llama-hparams.h` fields; `src/llama-model.{h,cpp}` `llama_model_dspark`
  (`load_arch_hparams` / `load_arch_tensors` per prism); register in dispatch.
- `convert_hf_to_gguf.py` + `conversion/dspark.py`. Verify: load a dspark GGUF
  (headers/tensors only, no decode).

### M3 - DSpark forward graph
- `src/models/dspark.cpp` (+408): the non-causal two-source attention graph
  above (fc/hidden_norm tap projection, optional log-SNR embed, per-layer
  concat(target_ctx, draft) -> `build_qkv`/`build_attn` -> slice to draft rows).
- `src/llama-graph.{h,cpp}`: `llm_graph_input_dspark_ctx` + `_logsnr` inputs and
  the staged-ctx plumbing (`llama_dspark_ctx`, `params.dspark_ctx`). Verify with
  `tests/test-dspark-forward.cpp` (port it) against a tiny converted drafter.

### M4 - target multi-layer tap capture
- `src/models/qwen35.cpp` (+120): emit hidden states at `dspark.target_layers`
  as a capture output during the TARGET's normal decode.
- `src/llama-context.{h,cpp}` (+457): `llama_set_capture_layers` /
  `llama_get_embeddings_capture_ith` + buffers. This is the **highest-conflict**
  file vs TurboQuant KV work - merge carefully, keep capture as an ADDITIVE
  output that does not perturb the cached decode. Verify: capture N layers on a
  qwen35 forward, check shapes.

### M5 - `draft-dspark` speculative impl + the host loop
- `common/speculative.{h,cpp}` (+747): port `common_speculative_impl_draft_dspark`
  and register `{"draft-dspark", COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK}` in
  `common/common.h` + `common/arg.cpp`. Includes `common/dspark-markov.{h,cu}`
  (CUDA/BLAS markov resample; env `LLAMA_DSPARK_MARKOV_CUDA`), the block-draft
  loop, confidence head, and the partial `llama_memory_seq_rm` post-verify crop
  (relies on `need_n_rs_seq` already in `bonsai`). Verify: `tests/test-dspark-loop.cpp`
  + `test-dspark-real-eval.cpp` (ports) - the synthetic-target harness runs the
  loop deterministically with no GPU.

### M6 - FINISH THE SERVER WIRING (the actual fix prism left open)
- `tools/server/server-context.cpp`: in the speculative path, when the drafter
  is dspark, call `llama_set_capture_layers` on the target, request logits on
  the rows that feed the tap, and stage `llama_set_dspark_ctx` before drafting -
  the piece prism explicitly notes the server does NOT do yet. Keep the
  checkpoint/prompt-state cache ON throughout (it is an independent knob; do NOT
  copy any "disable cache when spec on" logic).
- Verify: `--spec-type draft-dspark` + `--cache-ram 8192` together; confirm a
  prefix checkpoint hit WHILE spec decode runs (this is the trade-off removal).

### M7 - Docker + A/B validation
- Update `docker/entrypoint.sh` to also fetch the dspark drafter GGUF
  (`Bonsai-27B-dspark-Q4_1.gguf`, ~1.79 GB) and pass
  `--model-draft ... --spec-type draft-dspark` once M6 works. Today it
  intentionally omits DSpark and keeps the cache on. Vision (`--mmproj`) is
  already wired and independent of this.
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
