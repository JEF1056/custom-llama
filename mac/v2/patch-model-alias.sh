#!/usr/bin/env bash
# Patch mlx-vlm to add --served-model-name aliasing.
# When a request comes in for "qwen3.6-35b-a3b", resolve it to the preloaded local model.

set -e

MLX_VLM_SITE="$(python3 -c "import mlx_vlm, os; print(os.path.dirname(mlx_vlm.__file__))")"
CLI_FILE="$MLX_VLM_SITE/server/cli.py"
APP_FILE="$MLX_VLM_SITE/server/app.py"

# Check if already patched
if grep -q "MLX_VLM_SERVED_MODEL_NAME" "$CLI_FILE" 2>/dev/null; then
    echo "[PATCH] mlx-vlm already patched. Skipping."
    exit 0
fi

echo "[PATCH] Patching mlx-vlm server for model name aliasing..."

# Use Python for reliable multiline patching
python3 << 'PYEOF'
import os

mlx_site = os.path.dirname(__import__("mlx_vlm").__file__)

cli_file = os.path.join(mlx_site, "server", "cli.py")
app_file = os.path.join(mlx_site, "server", "app.py")

# ---- Patch cli.py: add --served-model-name argument ----
with open(cli_file, "r") as f:
    content = f.read()

# Insert the argument after --model definition
marker = '''        "--model",
        type=str,
        default=None,
        help="Pre-load a language model at startup'''

insert = '''        "--model",
        type=str,
        default=None,
        help="Pre-load a language model at startup'''

# The argument block for --model ends with the closing paren; find it
# and insert our new argument block after it
import re

# Find the --model arg block and the closing ), then insert after
pattern = r'(parser\.add_argument\(\s*["\x27]--model["\x27].*?\n\s*\))'
match = re.search(pattern, content, re.DOTALL)
if match:
    insert_pos = match.end()
    new_arg = '''
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Alias for the pre-loaded model used in API requests (e.g. qwen3.6-35b-a3b).",
    )
'''
    content = content[:insert_pos] + new_arg + content[insert_pos:]
    with open(cli_file, "w") as f:
        f.write(content)
    print(f"  Padded {cli_file}: added --served-model-name argument")
else:
    print(f"  WARNING: Could not find --model argument in {cli_file}")

# Also add the env var assignment after the --model block
env_marker = 'if args.model:'
if env_marker in content:
    replacement = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    if args.served_model_name:
        os.environ["MLX_VLM_SERVED_MODEL_NAME"] = args.served_model_name'''
    # Find the existing block
    old_block = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path'''
    if old_block in content:
        content = content.replace(old_block, replacement)
        with open(cli_file, "w") as f:
            f.write(content)
        print(f"  Padded {cli_file}: added MLX_VLM_SERVED_MODEL_NAME env var")

# ---- Patch app.py: resolve served name in get_cached_model ----
with open(app_file, "r") as f:
    content = f.read()

# Find the get_cached_model function and inject the resolution logic
old_sig = '''def get_cached_model(
    model_path: str,
    adapter_path=_INHERIT_ADAPTER,
    *,
    model_kind: str = "auto",
):'''

new_sig = '''def get_cached_model(
    model_path: str,
    adapter_path=_INHERIT_ADAPTER,
    *,
    model_kind: str = "auto",
):
    # Resolve served-model-name alias to actual preloaded path
    _served_name = os.environ.get("MLX_VLM_SERVED_MODEL_NAME")
    _preload_model = os.environ.get("MLX_VLM_PRELOAD_MODEL")
    if _served_name and _preload_model and model_path == _served_name:
        model_path = _preload_model'''

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    with open(app_file, "w") as f:
        f.write(content)
    print(f"  Padded {app_file}: added model name resolution in get_cached_model")
else:
    print(f"  WARNING: Could not find get_cached_model signature in {app_file}")

print("[PATCH] Done. mlx-vlm now supports --served-model-name aliasing.")
PYEOF

chmod +x "$CLI_FILE" "$APP_FILE" 2>/dev/null || true

echo "[PATCH] Complete."
