# Speculative-Decoding + Context/Parallel Config Sweep

Reproducible, resumable tuning of the `qwopus3.6-27b` decode path. Optimizes
**decode throughput (t/s)** at **mid (~25k)** and **long (~160k)** contexts for
**general text** and **code**, and separately decides between **maximizing
single-request context** vs **running two parallel slots** (≥100k tokens each).

Tool: [scripts/spec_sweep/](../scripts/spec_sweep/) — run with
`python -m scripts.spec_sweep`.

## How config is applied

`llama-server` reads all of these settings **only** from
[config/models.ini](../config/models.ini), bind-mounted read-only at
`/etc/llama-server/models.ini` ([docker-compose.yml](../docker-compose.yml) line 93).
[entrypoint.sh](../entrypoint.sh) launches with `--models-preset` and does **not**
consume `LLAMA_SPEC_*` env vars. So the sweep **edits the host ini and
force-recreates** the container (the read-only mount only blocks container-side
writes; host edits are re-read on restart):

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml \
  up -d --force-recreate llama-server
```

Edits are **section-aware**: a key is replaced only inside its declared section
(`[*]` global vs `[qwopus3.6-27b]`), so e.g. the 35B block's `ctx-size` is never
touched. New keys (e.g. `spec-draft-backend-sampling`) are appended to the
correct section automatically.

## Parameters swept

| Param | Section | Baseline | Why it affects decode speed |
| --- | --- | --- | --- |
| `spec-type` | `[*]` | `draft-mtp,ngram-mod` | Which speculative paths run and in what order |
| `spec-draft-p-min` | `[*]` | `0.2` | Cutoff that stops extending low-confidence draft chains |
| `spec-ngram-mod-n-match` | `[*]` | `8` | Min matching ngram length to fire the CPU path |
| `spec-ngram-mod-n-min` | `[*]` | `8` | Min draft length allowed |
| `spec-ngram-mod-n-max` | `[*]` | `16` | Max draft chain length (over-drafting wastes verify) |
| `spec-draft-backend-sampling` | `[*]` | (binary default) | GPU vs CPU sampling of the MTP draft head |
| `spec-default` | `[*]` | `true` | Likely inert (false does not actually disable spec) |
| `triattention-interval` | `[qwopus3.6-27b]` | `128` | CPU eviction-scoring stall frequency at long ctx |
| `ctx-size` / `rope-scale` / `parallel` | mixed | 196608 / 6 / 1 | Single-request context vs concurrent slots |

### Other params considered (and why excluded)

Surveyed in the fork's [common/arg.cpp](../../llama-cpp-turboquant/common/arg.cpp):

- `cache-type-k` / `cache-type-v` (currently `turbo4`/`turbo2`): lowering further
  (turbo2/turbo1) speeds long-ctx attention but measurably hurts quality — held
  fixed by the quality constraint.
- `triattention-budget` (50): lower = faster long-ctx decode but quality cost;
  **held fixed at 50** per the standing constraint. `triattention-interval`
  (raising 128→256) is the safe, near-zero-quality-impact alternative and **is**
  swept.
- `defrag-thold`: **deprecated/no-op** in this fork.
- `spec-draft-n-max` (2): already tuned optimal in prior work (n_max=2 → best
  53–56 t/s; wider dilutes acceptance), so it is held at 2.
- `spec-draft-n-min`, `spec-draft-p-split`, ngram-simple/ngram-map variants:
  secondary; left at defaults to bound the sweep.
- `flash-attn` (on), `kv-unified` (on), `backend-sampling` (on): already optimal.

## Maximizing context length

`qwen3.6` is natively trained with YaRN `rope-scale 8` at `yarn-orig-ctx 32768`
(→ **262144** native context). `rope-scale = ctx-size / 32768` keeps RoPE
extension within the trained range, so all of these are native-quality:

| Config | parallel | ctx-size | rope-scale | Effective |
| --- | --- | --- | --- | --- |
| `single196` (current) | 1 | 196608 | 6 | one 196K request |
| `maxctx256` | 1 | 262144 | 8 | one 256K request (native max) |
| `parallel2` | 2 | 204800 | 6.25 | two ~102K slots concurrently |

Stage C measures all three at ~90k (per slot) for text+code and picks the winner
by **aggregate** decode throughput (sum across slots for the parallel case).

## Methodology — staged greedy

A full grid would be 100+ runs (each 160k run ≈ 6 min of prompt eval). Instead
the sweep is staged-greedy on the cheap 25k context, then validates at 160k and
decides context/parallel last. Each stage holds prior winners fixed; score =
`mean(text_tg, code_tg)`.

| Stage | Ctx | Configs | Picks |
| --- | --- | --- | --- |
| A1 `spec-type` | 25k | `draft-mtp,ngram-mod` / `ngram-mod,draft-mtp` / `draft-mtp` | best spec-type |
| A2 `spec-draft-p-min` | 25k | `0.1` / `0.2` / `0.3` (if winner has MTP) | best p-min |
| A3 ngram | 25k | n-max {8,16,32}, n-match {4,8} (if winner has ngram) | best ngram trio |
| A4 draft backend sampling | 25k | on / off (if winner has MTP) | best sampling loc |
| B 160k validation | 160k | winner vs baseline vs winner+`tri-interval=256` | spec_final |
| C ctx / parallel | ~90k | `single196` / `maxctx256` / `parallel2` | ctx_winner |

> `ngram-mod` *alone* is intentionally excluded from A1: on this fork the ngram
> speculative path still asserts a draft context (`ctx_dft`), which only exists
> when a draft model (`draft-mtp`) is loaded. Running `spec-type=ngram-mod`
> with no draft model aborts the server with `GGML_ASSERT(ctx_dft)`, so every A1
> candidate keeps `draft-mtp` present.

## Reproducibility

- Prompt payloads are generated from **in-repo corpora**:
  prose from [calibration-data/wikitext-2-raw-test.txt](../calibration-data/wikitext-2-raw-test.txt);
  code from the repo's own python sources (`scripts/*.py`,
  `mcp-search-server/src/**/*.py`, `sync-env.py`). Each prompt appends
  `/no_think` so the budget produces real content, not reasoning.
- Throughput is computed from inter-token arrival times in the SSE stream over a
  tail window (first 40 tokens skipped to exclude prefill/eviction warmup), so it
  reflects steady-state decode. For `parallel2`, N identical streams run
  concurrently and the **aggregate** t/s is reported (with per-stream values).
- **Repeats:** every `(ctx, workload)` measurement is run multiple times
  (`config.REPEATS` = 3 at 25k, 2 at 90k/160k) and the **median** decode t/s is
  recorded along with the spread (`tg_runs` and the coefficient of variation
  `tg_cv` %), so run-to-run GPU variance is visible and rankings aren't decided
  by a single noisy sample.
- **TTFT (time to first token):** the cold first-token latency is captured for
  every measurement (`ttft` column) and summarised for the long-context and
  per-slot winners in `summary.json`. Only the *first* repeat is cold — later
  repeats hit the server prompt cache — so TTFT is read from that cold run and
  reflects real prefill cost. TTFT at a fixed context is governed mainly by
  `ctx-size` / `rope-scale` / `triattention-*` (Stage C and B), not by the
  spec-decode draft type; decode t/s remains the primary ranking metric.

## Resume & ETA

- Every completed config and every stage decision is persisted to
  `benchmark/results/spec-sweep/state.json`; results stream to `results.csv`.
- Re-running `python -m scripts.spec_sweep run` **skips finished configs** and
  resumes from the last incomplete stage — safe after a crash, OOM, or Ctrl-C.
- After each config the tool prints elapsed time and an **ETA** for the remaining
  work, auto-calibrated from measured per-weight time.
- `config/models.ini` is backed up to
  `benchmark/results/spec-sweep/models.ini.backup` before the first edit;
  restore with `python -m scripts.spec_sweep restore`.

## Usage

```bash
cd custom-llama

python -m scripts.spec_sweep payloads   # (re)build prompt payloads
python -m scripts.spec_sweep run        # full staged sweep (resumable)
python -m scripts.spec_sweep status     # progress + decisions so far
python -m scripts.spec_sweep reset      # clear state so the next run starts fresh
python -m scripts.spec_sweep restore    # restore models.ini from backup
```

Outputs (under `benchmark/results/spec-sweep/`, gitignored):

- `results.csv` — one row per (config, ctx, workload): median `tg`, spread
  (`tg_cv` %, `tg_runs`), cold `ttft`, and all tracked params.
- `state.json` — resume state + stage decisions.
- `summary.json` — final chosen spec config, ctx/parallel winner, applied
  values, and the winners' TTFT.

The sweep leaves the winning config **live** in `config/models.ini`. After it
finishes, update the explanatory comments in the `[*]` and `[qwopus3.6-27b]`
blocks to document the chosen values and their measured rationale (matching the
existing comment style).

## Baseline reference (live config, ~25k)

| Workload | Decode t/s |
| --- | --- |
| text | ~46.0 |
| code | ~46.3 |
