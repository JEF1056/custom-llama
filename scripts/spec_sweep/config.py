"""Static configuration for the spec-decode / context-parallel sweep.

All paths are derived from this file's location so the tool is runnable from
anywhere. The sweep matrix is data-only here; the staging logic lives in
runner.py.
"""
from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]          # …/custom-llama
INI_PATH = REPO_ROOT / "config" / "models.ini"
PROSE_CORPUS = REPO_ROOT / "calibration-data" / "wikitext-2-raw-test.txt"
RESULTS_DIR = REPO_ROOT / "benchmark" / "results" / "spec-sweep"
PAYLOAD_DIR = RESULTS_DIR / "payloads"
RESULTS_CSV = RESULTS_DIR / "results.csv"
STATE_JSON = RESULTS_DIR / "state.json"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
INI_BACKUP = RESULTS_DIR / "models.ini.backup"

# ── Server / docker ───────────────────────────────────────────────────────────
MODEL = "qwopus3.6-27b"
API_BASE = "http://localhost:8080"
COMPOSE = [
    "docker", "compose",
    "-f", "docker-compose.yml",
    "-f", "docker-compose.override.yml",
]
HEALTH_TIMEOUT = 600          # seconds to wait for model to serve after restart
MAX_TOKENS = 320              # generation length per measurement
WARMUP_SKIP = 40              # tokens skipped before measuring steady-state tg

# Repeat each (ctx, workload) measurement N times and report median + spread, so
# run-to-run GPU variance is visible. Cheap 25k stages repeat more; the
# prefill-heavy long stages repeat fewer to keep wall-clock sane. The *first*
# repeat is the only cold-cache one, so it provides the representative TTFT.
REPEATS = {"25k": 3, "90k": 2, "160k": 2}
REPEATS_DEFAULT = 1

# ── Code corpus (reproducible, in-repo) ───────────────────────────────────────
# Concatenated python sources used as the "code" workload. Globbed at build time;
# listed explicitly-by-glob so the payload is deterministic given the repo state.
CODE_GLOBS = [
    "scripts/*.py",
    "mcp-search-server/src/**/*.py",
    "sync-env.py",
]

# Measured char/token ratios on this model's tokenizer (prose ≈4.15, code ≈3.86).
PROSE_CHARS_PER_TOK = 4.15
CODE_CHARS_PER_TOK = 3.86

# ── Config keys we edit, and which ini section they live in ───────────────────
# section "*" → [*] global block; "qwopus3.6-27b" → that model's block.
KEY_SECTION = {
    "spec-type": "*",
    "spec-draft-n-max": "*",
    "spec-draft-p-min": "*",
    "spec-ngram-mod-n-match": "*",
    "spec-ngram-mod-n-min": "*",
    "spec-ngram-mod-n-max": "*",
    "spec-default": "*",
    "spec-draft-backend-sampling": "*",
    "parallel": "*",
    "ctx-size": "qwopus3.6-27b",
    "rope-scale": "qwopus3.6-27b",
    "triattention-budget": "qwopus3.6-27b",
    "triattention-window": "qwopus3.6-27b",
    "triattention-interval": "qwopus3.6-27b",
}

# Columns recorded for every run (config snapshot read back from the ini).
TRACKED_KEYS = list(KEY_SECTION.keys())

# ── Baseline (current production values) ──────────────────────────────────────
BASELINE = {
    "spec-type": "draft-mtp,ngram-mod",
    "spec-draft-n-max": 2,
    "spec-draft-p-min": 0.2,
    "spec-ngram-mod-n-match": 8,
    "spec-ngram-mod-n-min": 8,
    "spec-ngram-mod-n-max": 16,
    "spec-default": "true",
    "parallel": 1,
    "ctx-size": 196608,
    "rope-scale": 6,
    "triattention-budget": 50,
    "triattention-window": 256,
    "triattention-interval": 128,
}

# ── Context buckets (target prompt token counts) ──────────────────────────────
CTX_MID = 25_000
CTX_LONG = 160_000
CTX_SLOT = 90_000        # used for the parallel-vs-maxctx stage (fits a 102K slot)

# ── Context / parallel comparison configs (Stage C) ───────────────────────────
# qwen3.6 is natively trained with yarn rope-scale 8 at orig-ctx 32768
# (→ 262144 native). rope-scale = ctx-size / 32768 keeps extension at/under the
# trained range, so these are all native-quality.
#   single196 : current production (1 slot, 196K)
#   maxctx256 : 1 slot at native 256K (rope-scale 8) — maximises single-request ctx
#   parallel2 : 2 slots @ ~102K each (parallel=2, ctx 204800) — concurrency throughput
CTX_PARALLEL_CONFIGS = {
    "single196": {"parallel": 1, "ctx-size": 196608, "rope-scale": 6},
    "maxctx256": {"parallel": 1, "ctx-size": 262144, "rope-scale": 8},
    "parallel2": {"parallel": 2, "ctx-size": 204800, "rope-scale": 6.25},
}
