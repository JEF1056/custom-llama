import sys
import importlib
import mlx.core as mx
import asyncio
import threading
import time
import logging

# 1. Load the actual mlx_lm.generate module dynamically using importlib to bypass parent package namespace function collision
generate_module = importlib.import_module("mlx_lm.generate")

# 2. Define the ActiveBatchWrapper class to adapt GenerationBatch to the old Batch dataclass structure
class ActiveBatchWrapper:
    def __init__(self, batch):
        self._batch = batch

    def __len__(self):
        return len(self._batch)

    @property
    def uids(self):
        return self._batch.uids
    @uids.setter
    def uids(self, val):
        self._batch.uids = val

    @property
    def tokens(self):
        return self._batch.tokens
    @tokens.setter
    def tokens(self, val):
        self._batch.tokens = val

    @property
    def samplers(self):
        return self._batch.samplers
    @samplers.setter
    def samplers(self, val):
        self._batch.samplers = val

    @property
    def logits_processors(self):
        return self._batch.logits_processors
    @logits_processors.setter
    def logits_processors(self, val):
        self._batch.logits_processors = val

    @property
    def max_tokens(self):
        return self._batch.max_tokens
    @max_tokens.setter
    def max_tokens(self, val):
        self._batch.max_tokens = val

    @property
    def cache(self):
        return self._batch.prompt_cache
    @cache.setter
    def cache(self, val):
        self._batch.prompt_cache = val

    @property
    def y(self):
        return self._batch._next_tokens
    @y.setter
    def y(self, val):
        self._batch._next_tokens = val

    @property
    def logprobs(self):
        return self._batch._next_logprobs
    @logprobs.setter
    def logprobs(self, val):
        self._batch._next_logprobs = val

    def filter(self, keep):
        self._batch.filter(keep)

    def extend(self, other):
        raw_other = other._batch if isinstance(other, ActiveBatchWrapper) else other
        self._batch.extend(raw_other)

    def extract_cache(self, idx):
        return self._batch.extract_cache(idx)

# Inject the active_batch property into BatchGenerator to support vllm-mlx active_batch accesses
@property
def active_batch(self):
    if not hasattr(self, "_generation_batch") or len(self._generation_batch) == 0:
        return None
    return ActiveBatchWrapper(self._generation_batch)

@active_batch.setter
def active_batch(self, val):
    if val is None:
        if hasattr(self, "_generation_batch") and len(self._generation_batch) > 0:
            self._generation_batch.filter([])

generate_module.BatchGenerator.active_batch = active_batch

# 3. Define and inject the missing BatchGenerator._step method with DRY sampling support
def BatchGenerator_step(self, input_tokens, prompt_cache, samplers, logits_processors, tokens):
    logits = self.model(input_tokens, cache=prompt_cache)
    logits = logits[:, -1, :]  # shape [B, V]
    
    import numpy as np
    
    next_tokens = []
    next_logprobs = []
    for i in range(input_tokens.shape[0]):
        seq_logits = logits[i : i + 1]
        
        # Apply logits processors if present
        if logits_processors and i < len(logits_processors) and logits_processors[i] is not None:
            seq_logits = logits_processors[i](tokens[i], seq_logits)
            
        # Apply DRY sampling penalty to seq_logits (allowed_length=128, penalty_last_n=2048, multiplier=0.4, base=1.75)
        if len(tokens[i]) > 128:
            seq_logits_np = np.array(seq_logits)
            ctx = tokens[i][-2048:]
            ctx_len = len(ctx)
            for j in range(128 - 1, ctx_len - 1):
                L = 0
                while L <= j and ctx[j - L] == ctx[ctx_len - 1 - L]:
                    L += 1
                if L >= 128:
                    next_tok = ctx[j + 1]
                    penalty = 0.4 * (1.75 ** (L - 128))
                    seq_logits_np[0, next_tok] -= penalty
            seq_logits = mx.array(seq_logits_np)

        # Sample next token
        y = samplers[i](seq_logits)
        
        # Calculate log probabilities
        logprob = seq_logits[0, y] - mx.logsumexp(seq_logits, axis=-1)
        
        next_tokens.append(y)
        next_logprobs.append(logprob)
        
    return mx.concatenate(next_tokens), next_logprobs

generate_module.BatchGenerator._step = BatchGenerator_step
print("[COMPAT] Successfully injected missing BatchGenerator._step back into mlx_lm.generate.BatchGenerator with DRY sampling", file=sys.stderr, flush=True)

# 4. Patch default values in SamplingParams class
import vllm_mlx.request
orig_init = vllm_mlx.request.SamplingParams.__init__

def patched_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    # If parameters remain at their original defaults, re-assign them to user custom defaults
    if self.temperature == 0.7:
        self.temperature = 0.6
    if self.top_p == 0.9:
        self.top_p = 0.95
    if self.top_k == 0:
        self.top_k = 10
    if self.min_p == 0.0:
        self.min_p = 0.05
    if self.presence_penalty == 0.0:
        self.presence_penalty = 0.0
    if self.repetition_penalty == 1.0:
        self.repetition_penalty = 1.0

vllm_mlx.request.SamplingParams.__init__ = patched_init
print("[COMPAT] Injected custom SamplingParams defaults (temp=0.6, top_p=0.95, top_k=10, min_p=0.05)", file=sys.stderr, flush=True)

import vllm_mlx.server
import vllm_mlx.patches.qwen3_next_mtp
import vllm_mlx.patches.qwen3_5_mtp

# 5. Fix the MTP validation bug in batched.py by routing validation for Qwen 3.5/3.6 models
vllm_mlx.patches.qwen3_next_mtp.validate_mtp_support = vllm_mlx.patches.qwen3_5_mtp.validate_mtp_support
print("[COMPAT] Redirected Qwen3-Next MTP validator to Qwen3.5 MTP validator", file=sys.stderr, flush=True)

# 6. Monkeypatch the model name validator to bypass validation and log the requested name
def bypass_validate(request_model):
    print(f"\n[DIAGNOSTIC] Intercepted request model name: '{request_model}' - auto-allowing!", file=sys.stderr, flush=True)
    return

vllm_mlx.server._validate_model_name = bypass_validate

# 6.5 Patch bind_generation_streams to use thread-local stream caching to prevent exhausting Metal command queue limits
import vllm_mlx.mlx_streams
import vllm_mlx.mllm_scheduler

_stream_thread_local = threading.local()

def patched_bind_generation_streams(module_names=("mlx_lm.generate", "mlx_vlm.generate")):
    if not hasattr(_stream_thread_local, "stream"):
        _stream_thread_local.stream = mx.new_stream(mx.default_device())
    
    default_stream = _stream_thread_local.stream
    mx.set_default_stream(default_stream)
    
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "generation_stream"):
            setattr(module, "generation_stream", default_stream)
            
    return default_stream

vllm_mlx.mlx_streams.bind_generation_streams = patched_bind_generation_streams
vllm_mlx.mllm_scheduler.bind_generation_streams = patched_bind_generation_streams
print("[COMPAT] Patched bind_generation_streams to use thread-local stream caching and prevent command queue leaks", file=sys.stderr, flush=True)

# 7. Patch asyncio.Queue.put_nowait to be thread-safe for background worker dispatch execution
orig_put_nowait = asyncio.Queue.put_nowait

def thread_safe_put_nowait(self, item):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return orig_put_nowait(self, item)
    
    # Delegate to the main thread's event loop
    loop.call_soon_threadsafe(orig_put_nowait, self, item)

asyncio.Queue.put_nowait = thread_safe_put_nowait
print("[COMPAT] Patched asyncio.Queue.put_nowait to support thread-safe background scheduler execution", file=sys.stderr, flush=True)

# 8. Monkeypatch MLLMScheduler._process_loop to run step() in a thread pool, preventing event loop starvation
import vllm_mlx.mllm_scheduler

async def patched_process_loop(self) -> None:
    from vllm_mlx.mllm_scheduler import bind_generation_streams
    mllm_logger = logging.getLogger("vllm_mlx.mllm_scheduler")
    
    streams_bound = False

    def _ensure_streams_bound() -> None:
        nonlocal streams_bound
        if not streams_bound:
            bind_generation_streams()
            streams_bound = True

    loop = asyncio.get_running_loop()

    while self._running:
        try:
            # --- Early preprocessing phase ---
            bg = self.batch_generator
            if bg is not None:
                for req in list(getattr(bg, "unprocessed_requests", ())) :
                    if (
                        req.input_ids is None
                        and not req.images
                        and not req.videos
                        and not req.audio
                    ):
                        try:
                            tic = time.perf_counter()
                            await loop.run_in_executor(
                                None, bg._preprocess_request, req
                            )
                            elapsed = time.perf_counter() - tic
                            if elapsed > 1.0:
                                n_tok = (
                                    req.input_ids.size
                                    if req.input_ids is not None
                                    else 0
                                )
                                mllm_logger.info(
                                    f"Preprocessing {req.request_id[:12]}"
                                    f": {n_tok} tokens in {elapsed:.2f}s"
                                )
                        except Exception as e:
                            mllm_logger.error(
                                f"Early preprocessing failed for "
                                f"{req.request_id}: {e}"
                            )

            # --- Step phase ---
            if self.has_requests():
                _ensure_streams_bound()
                tic = time.perf_counter()
                
                # Execute self.step() in a background thread to prevent blocking Uvicorn's event loop
                def step_wrapper():
                    bind_generation_streams()
                    self.step()
                    
                await loop.run_in_executor(None, step_wrapper)
                
                elapsed = time.perf_counter() - tic
                if elapsed > 2.0:
                    mllm_logger.warning(
                        f"Slow MLLM step: {elapsed:.2f}s "
                        f"(waiting={len(self.waiting)}, "
                        f"running={len(self.running)})"
                    )
                n_yields = 10 if elapsed > 1.0 else 5
                for _ in range(n_yields):
                    await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            mllm_logger.error(f"Error in MLLM process loop: {e}", exc_info=True)
            await asyncio.sleep(0.1)

vllm_mlx.mllm_scheduler.MLLMScheduler._process_loop = patched_process_loop
print("[COMPAT] Patched MLLMScheduler._process_loop to execute generation steps in a background thread pool", file=sys.stderr, flush=True)

# 9. Define ThreadLocalStreamDescriptor to ensure all worker threads have a valid thread-local stream mapping
class ThreadLocalStreamDescriptor:
    def __init__(self):
        self._local = threading.local()

    def __get__(self, instance, owner):
        if not hasattr(self._local, "stream"):
            self._local.stream = mx.new_stream(mx.default_device())
        return self._local.stream

    def __set__(self, instance, value):
        if not hasattr(self._local, "stream"):
            self._local.stream = mx.new_stream(mx.default_device())
        self._local.stream = value

# Replace MLLMBatchGenerator._stream class attribute with our descriptor
import vllm_mlx.mllm_batch_generator
vllm_mlx.mllm_batch_generator.MLLMBatchGenerator._stream = ThreadLocalStreamDescriptor()
print("[COMPAT] Replaced MLLMBatchGenerator._stream with thread-local stream descriptor", file=sys.stderr, flush=True)

# 10. Monkeypatch MLLMBatchGenerator._run_chunked_text_prefill with dynamic step sizing to eliminate GPU OOM failures
def patched_run_chunked_text_prefill(self, request, cache) -> mx.array:
    from vllm_mlx.mllm_batch_generator import _eval_prompt_cache, PrefillAbortedError
    mllm_logger = logging.getLogger("vllm_mlx.mllm_batch_generator")
    
    input_ids = request.input_ids
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    total = input_ids.shape[1]
    
    processed = 0
    chunk_count = 0
    output = None
    
    mllm_logger.info(
        f"[chunked_prefill] Starting {request.request_id[:12]}: "
        f"{total} tokens, dynamic chunking enabled"
    )

    while processed < total:
        # Determine step size dynamically based on the processed context length
        # to ensure the intermediate self-attention buffer size does not cause Metal OOMs,
        # while keeping processing speed as fast as possible for shorter context bounds.
        if processed < 15000:
            step = 2048
        elif processed < 30000:
            step = 1024
        elif processed < 60000:
            step = 512
        else:
            step = 256
            
        step = min(step, total - processed)

        # Check for abort between chunks
        if request.request_id in self._aborted_request_ids:
            self._aborted_request_ids.discard(request.request_id)
            mllm_logger.info(
                f"[chunked_prefill] Aborted {request.request_id} at "
                f"{processed}/{total} tokens"
            )
            raise PrefillAbortedError(request.request_id)

        chunk = input_ids[:, processed : processed + step]
        output = self.language_model(chunk, cache=cache)
        
        # Evict intermediate arrays from cache to break the lazy graph
        _eval_prompt_cache(cache)
        
        processed += step
        chunk_count += 1
        self._prefill_progress[request.request_id] = (processed, total)

        if chunk_count % 10 == 0 or processed == total:
            mllm_logger.info(
                f"[chunked_prefill] {request.request_id[:12]}: "
                f"chunk {chunk_count}, {processed}/{total} tokens (step={step})"
            )

        # Clear Metal allocator cache on every single chunk iteration to prevent memory accumulation
        mx.clear_cache()
            
    if hasattr(output, "logits"):
        return output.logits
    return output

vllm_mlx.mllm_batch_generator.MLLMBatchGenerator._run_chunked_text_prefill = patched_run_chunked_text_prefill
print("[COMPAT] Patched MLLMBatchGenerator._run_chunked_text_prefill with dynamic chunking OOM prevention", file=sys.stderr, flush=True)

# 11. Start the server
import vllm_mlx.cli

if __name__ == "__main__":
    vllm_mlx.cli.main()
