# APC in `_run_speculative()` — Exploration Report

## Reference Code Snippets

### 1. `_store_apc_exact_checkpoints` (from `mlx_vlm/generate/ar.py`, line 1782)

```python
def _store_apc_exact_checkpoints(self) -> None:
    if self._apc_manager is None or self._apc_mode != "exact":
        return
    for batch_idx, meta in enumerate(self._apc_meta):
        if meta is None or meta.get("checkpoint_done"):
            continue
        checkpoint_len = int(meta.get("checkpoint_len") or 0)
        if checkpoint_len <= 0:
            continue
        if self._row_real_tokens_processed(batch_idx) != checkpoint_len:
            continue
        prompt_cache = self._apc_prompt_cache_for_store(batch_idx)
        if prompt_cache is None:
            continue
        self._apc_manager.store_exact_cache(
            meta["full_input_ids"][:checkpoint_len],
            prompt_cache,
            extra_hash=meta.get("extra_hash", 0),
        )
        meta["checkpoint_done"] = True
```

**`_apc_prompt_cache_for_store(batch_idx)`** (line 1779):
```python
def _apc_prompt_cache_for_store(self, batch_idx: int) -> Optional[List[Any]]:
    return _apc.snapshot_prompt_cache_row(self.prompt_cache, batch_idx)
```

### 2. Key `_row_real_tokens_processed` helper (line 1765):

```python
def _row_real_tokens_processed(self, batch_idx: int) -> int:
    meta = self._apc_meta[batch_idx]
    prefix_len = int(meta.get("prefix_len", 0) or 0)
    if self._right_pad_per_row is not None:
        suffix_done = min(...)
        return prefix_len + suffix_done
    real_done = (
        self._processed_prompt_columns - self._left_padding_per_row[batch_idx]
    )
    return prefix_len + min(self._suffix_lens[batch_idx], max(0, real_done))
```

### 3. How `BatchGenerator` creates prompt_cache with speculative proto (line ~1642):

```python
elif draft_model is not None and draft_kind is not None:
    self.prompt_cache = make_speculative_prompt_cache(
        model,
        draft_kind=draft_kind,
        batch_size=len(input_ids),
        left_padding=left_padding,
        make_cache=lambda lm, lp: _make_cache(
            lm,
            lp,
            kv_bits=kv_bits,                  # <-- kv_bits from ResponseGenerator
            kv_group_size=kv_group_size,      # <-- from ResponseGenerator
            kv_quant_scheme=kv_quant_scheme,  # <-- from ResponseGenerator
            quantized_kv_start=quantized_kv_start,  # <-- from ResponseGenerator
            prefill_length=max_length,
        ),
    )
```

### 4. `_run_speculative` prefill section (generation.py, lines 1991-1997):

```python
prompt_cache = make_speculative_prompt_cache(
    lm,
    draft_kind=draft_kind,
    batch_size=B,
    left_padding=left_padding,
    make_cache=_make_cache,    # <-- bare reference, NO kv params closure
)
```

### 5. `make_speculative_prompt_cache` (speculative/utils.py, line 105):

```python
def make_speculative_prompt_cache(
    lm,
    *,
    draft_kind: str,
    batch_size: int,
    left_padding,
    make_cache: Callable,
):
    if draft_kind == "mtp" and batch_size == 1:
        return cache.make_prompt_cache(lm)
    return make_cache(lm, left_padding)  # bare _make_cache — no kv params!
```

### 6. `_make_cache` function signature (generate/ar.py, line 724):

```python
def _make_cache(
    model,
    left_padding,
    kv_bits=None,              # default None!
    kv_group_size=64,
    kv_quant_scheme=DEFAULT_KV_QUOT_SCHEME,  # "uniform"
    quantized_kv_start=0,
    prefill_length=0,
):
```

### 7. `_run` wiring through `BatchGenerator` (lines 1684-1702):

```python
batch_gen = BatchGenerator(
    self.model.language_model,
    self.processor,
    stop_tokens=self.stop_tokens,
    sampler=self._make_sampler(args),
    kv_bits=self.kv_bits,            # from ResponseGenerator.__init__
    kv_group_size=self.kv_group_size,
    kv_quant_scheme=self.kv_quant_scheme,
    quantized_kv_start=self.quantized_kv_start,
    ...
    apc_manager=self.apc_manager,     # <-- Critical for APC
    draft_model=self.draft_model,
    draft_kind=self.draft_kind,
    ...
)
```

### 8. `ResponseGenerator.__init__` kv params (lines 1044-1047):

```python
self.kv_bits = kv_bits
self.kv_group_size = kv_group_size
self.kv_quant_scheme = kv_quant_scheme
self.quantized_kv_start = quantized_kv_start
```

### 9. `app.py` — APC manager creation (lines 626-635):

```python
runtime.apc_manager = _apc.from_env(
    model_namespace=_apc.apc_disk_namespace(
        model_path,
        adapter_path=adapter_path,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_quant_scheme=kv_quant_scheme,
        quantized_kv_start=quantized_kv_start,
    )
)
```

## Architecture Diagram: `_run` vs `_run_speculative`

**`_run()` — non-speculative or MTP path:**

```
collect requests → BatchGenerator.insert() 
    → PromptProcessingBatch (with APC hooks)
        → make_speculative_prompt_cache(make_cache overridden with kv params)
            → _make_cache with kv_bits/kv_group_size/kv_quant_scheme
                → KV cache created with proper quantization
        → prompt_step() 【chunked prefill, calls _store_apc_exact_checkpoints()】
        → generate() → GenerationBatch.next()
            → next() calls _store_apc_exact_checkpoints() after each step
```

**`_run_speculative()` — current DFlash/EAGLE-3 path:**

```
collect requests → _gpu_embed()
    → _merge_prefill_prompt_kwargs()
    → make_speculative_prompt_cache(lm, draft_kind, batch_size, left_padding, make_cache=_make_cache)
        → make_cache=_make_cache  (BARE — NO kv params)
            → _make_cache(lm, left_padding)  ← uses defaults: kv_bits=None, kv_group_size=64
    → _run_chunked_speculative_prefill()  ← NO _store_apc_exact checkpoints!
    → run_speculative_server_rounds()  ← NO _store_apc_exact checkpoints!
    → stream tokens out
```

## Root Cause Analysis

### Issue 1: `make_cache` is passed raw, not as a lambda with kv params

In `_run` → `BatchGenerator.__init__` → `PromptProcessingBatch.__init__` (line 1642-1657):
```python
make_cache=lambda lm, lp: _make_cache(
    lm, lp,
    kv_bits=kv_bits, kv_group_size=kv_group_size, ...
)
```

In `_run_speculative` → line 1991-1997:
```python
make_cache=_make_cache  # raw — no kv_bits/kv_group_size/kv_quant_scheme/quantized_kv_start
```

**Result**: `_run_speculative` creates BatchKVCache (unquantized) instead of BatchQuantizedKVCache when KV quantization is configured via env vars.

### Issue 2: No APC metadata tracking in `_run_speculative`

`_run` uses `BatchGenerator.insert()` which:
- Tracks `self._unprocessed_sequences` (per-request metadata)
- Creates `_apc_meta` in `PromptProcessingBatch.__init__`

`_run_speculative` directly iterates `pending` requests:
```python
for request in pending:
    uid = id(rqueue)
    uids.append(uid)
    # NO _apc_meta created
```

No `_apc_meta` means no `_store_apc_exact_checkpoints()` can fire.

### Issue 3: No post-prefill checkpoint storage

In `_run`, `PromptProcessingBatch.prompt_step()` calls `self._store_apc_exact_checkpoints()` (line 1833):
```python
def prompt_step(self) -> int:
    ...
    self._store_apc_exact_checkpoints()  # ← after each prefill chunk
    ...
```

But this is only called because `_run` uses `BatchGenerator` which calls `_step()` which calls `prompt_step()` on `PromptProcessingBatch`.

In `_run_speculative`, after `_run_chunked_speculative_prefill()` completes (line 2013-2022), there is no equivalent call.

### Issue 4: `prompt_cache` created differently

```python
# _run (batched, per-requests):
self.prompt_cache = make_speculative_prompt_cache(
    model, draft_kind=draft_kind, batch_size=len(input_ids),
    left_padding=left_padding,
    make_cache=lambda lm, lp: _make_cache(
        lm, lp, kv_bits=kv_bits, ...
    ),
)

# _run_speculative (raw batch):
prompt_cache = make_speculative_prompt_cache(
    lm, draft_kind=draft_kind, batch_size=B,
    left_padding=left_padding,
    make_cache=_make_cache,   # no kv params!
)
```

However, `make_speculative_prompt_cache` DOES fall through to `make_cache(lm, left_padding)` which calls `_make_cache` with correct structure — but without the kv_bits closure, it uses defaults `None` for kv_bits, `64` for group_size, etc.

## Concrete Plan: Adding APC Storage to `_run_speculative`

### Step A: Fix kv parameter propagation (block A)

**Location**: `generation.py`, line 1991-1997

**Change**: Override `make_cache` to close over `self.kv_bits` etc:

```python
prompt_cache = make_speculative_prompt_cache(
    lm,
    draft_kind=draft_kind,
    batch_size=B,
    left_padding=left_padding,
    make_cache=lambda lm, lp: _make_cache(
        lm, lp,
        kv_bits=self.kv_bits,
        kv_group_size=self.kv_group_size,
        kv_quant_scheme=self.kv_quant_scheme,
        quantized_kv_start=self.quantized_kv_start,
        prefill_length=max_len,
    ),
)
```

This is a **drop-in fix** — no structural changes to `_run_speculative`.

### Step B: Build APC metadata after speculative prefill (block B)

**Location**: After `_run_chunked_speculative_prefill()` (line 2013-2022), before `run_speculative_server_rounds()`.

**New code**: Build `_apc_meta` per-request:

```python
ApC_meta = []    # per request
if self.apc_manager is not None:
    for i, (uid, input_ids_list, gen_kwargs) in enumerate(
        zip(uids, all_input_ids, prompt_kwargs_list)
    ):
        extra_hash = _apc.semantic_extra_hash(
            tenant=gen_kwargs.get("_apc_tenant"),
            image_hash=gen_kwargs.get("_apc_image_hash"),
            media={"audio": gen_kwargs.get("input_features")},
            model=self.model,
            processor=self.processor,
        )
        full_input_ids = input_ids_list  # e.g. [101, 5234, ..., 1]
        media_ids = _apc.multimodal_token_ids_from_config(self.model.config)
        checkpoint_len = _apc.adjust_prefix_to_text_suffix_boundary(
            full_input_ids,
            len(full_input_ids) - self.apc_manager.exact_cache_guard_tokens,
            media_ids,
            max_prefix_tokens=len(full_input_ids) - 1,
        )
        ApC_meta.append({
            "full_input_ids": full_input_ids,
            "prefix_len": 0,      # cold start, no warm prefix
            "extra_hash": extra_hash,
            "apc_blocks": [],
            "checkpoint_len": checkpoint_len,  # 0 if no exact checkpoint
        })
else:
    ApC_meta = [None] * B
```

### Step C: Store APC checkpoint after speculative prefill (block C)

**Location**: Still in the same prefill section, after `_run_chunked_speculative_prefill()` completes.

**New code**:

```python
if self.apc_manager is not None and self.apc_mode == "exact":
    for batch_idx, meta in enumerate(ApC_meta):
        if meta is None or meta.get("checkpoint_done"):
            continue
        checkpoint_len = int(meta.get("checkpoint_len") or 0)
        if checkpoint_len <= 0:
            continue
        # Snapshot just the relevant rows from the batch cache
        row_snapshot = _apc.snapshot_prompt_cache_row(
            prompt_cache, batch_idx
        )
        if row_snapshot is not None:
            self.apc_manager.store_exact_cache(
                meta["full_input_ids"][:checkpoint_len],
                row_snapshot,
                extra_hash=meta.get("extra_hash", 0),
            )
            meta["checkpoint_done"] = True
```

This mirrors exactly what `_store_apc_exact_checkpoints` does (lines 1782-1801), but extracted into the `_run_speculative` path.

### Step D: Handle speculative rounds — store checkpoints for "bonus" tokens

**Location**: After `run_speculative_server_rounds()` (line 2099), in the decode loop (lines 2119-2159).

**Logic**: After each speculative block is accepted, store the extended KV cache:

```python
for tok_list, _ in rounds_iter:
    for j, tok in enumerate(tok_list):
        ...
        # After accepting the token, store APC
        if self.apc_manager is not None:
            row_snapshot = _apc.snapshot_prompt_cache_row(prompt_cache, j)
            stored_tokens = len(all_input_ids[j]) + current_gen_step
            self.apc_manager.store_block(
                all_input_ids[j],  # full token sequence so far
                row_snapshot,
            )
```

Whether to store incrementally after each accepted token or after the block completes depends on whether "exact" or "block" APC mode applies.

### Alternative: Extract into a helper class

If the speculative path becomes too complex inline, one could create a `SpeculativeBatchGenerator` class that wraps the UIDs, prompt_cache, and APC metadata, exposing an `insert()`/`next()` interface similar to `BatchGenerator` but for the speculative path.

## Key Limitation: `snapshot_prompt_cache_row` Requires Row-Level Cache

The `make_cache` function (when kv_bits is None → default path) returns `BatchKVCache(left_padding)`. These objects:

1. Accept a `left_padding` list `[pad0, pad1, ...]`
2. Have per-row cache entries
3. Support `_apc.snapshot_prompt_cache_row(cache, row_idx)` which extracts a single row

Since `make_speculative_prompt_cache` calls `make_cache(lm, left_padding)` when `batch_size > 1`, and `_make_cache` creates `BatchKVCache(left_padding)` for each layer, the cache DOES support `snapshot_prompt_cache_row`.

**Verification needed**: Check if `BatchKVCache`, `BatchQuantizedKVCache`, and `BatchTurboQuantKVCache` all implement `snapshot_prompt_cache_row` correctly.

## Summary of Changes Required

| # | Change | Lines | Impact |
|---|--------|-------|--------|
| A | Override `make_cache` with kv params | 1991-1997 (generation.py) | Ensures quantized KV cache in speculative path |
| B | Build `_apc_meta` per-request | After line 2022 | Enables checkpoint tracking |
| C | Call `store_exact_cache` after prefill | After line 2022 | Behold the cached KV! |
| D | Store after speculative rounds | Line 2119+ | Grow cache for future requests |

Changes A is independent of B-D. Changes B-D depend on `self.apc_manager` being available (it is — stored on ResponseGenerator at line 1049 and used in `_run` at line 1696).

### Parameters available in `_run_speculative`:

- `self.apc_manager` — ✓ (line 1049 in ResponseGenerator)
- `self.kv_bits` — ✓ (line 1044)
- `self.kv_group_size` — ✓ (line 1045)
- `self.kv_quant_scheme` — ✓ (line 1046)
- `self.quantized_kv_start` — ✓ (line 1047)

All parameters are accessible. The only missing piece is wiring them into the `make_cache` closure (Change A) and adding the metadata tracking + store calls (Changes B-D).
