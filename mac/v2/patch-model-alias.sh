#!/usr/bin/env bash
# Patch mlx-vlm to add --served-model-name aliasing.
# When a request comes in for "qwen3.6-35b-a3b", resolve it to the preloaded local model.

set -e

python3 << 'PYEOF'
import os
import re
import mlx_vlm

mlx_site = os.path.dirname(mlx_vlm.__file__)
cli_file = os.path.join(mlx_site, "server", "cli.py")
app_file = os.path.join(mlx_site, "server", "app.py")

PATCH_MARKER_CLI = "MLX_VLM_SERVED_MODEL_NAME"
PATCH_MARKER_APP = "Resolve served-model-name alias"

# ============================================================
# 1. Patch cli.py: add --served-model-name argument + env var
# ============================================================
with open(cli_file, "r") as f:
    cli_content = f.read()

if PATCH_MARKER_CLI in cli_content:
    print("  [PATCH] cli.py already patched.")
else:
    # Insert --served-model-name argument after the --model block
    arg_block = '''    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Alias for the pre-loaded model used in API requests (e.g. qwen3.6-35b-a3b).",
    )
'''
    # Find the closing paren of the --model argument block
    model_arg_pattern = r'(parser\.add_argument\(\s*["\x27]--model["\x27].*?\n\s*\))'
    match = re.search(model_arg_pattern, cli_content, re.DOTALL)
    if match:
        insert_pos = match.end()
        cli_content = cli_content[:insert_pos] + "\n" + arg_block + cli_content[insert_pos:]

    # Add env var assignment after the --model env block
    old_env = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path'''
    new_env = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    if args.served_model_name:
        os.environ["MLX_VLM_SERVED_MODEL_NAME"] = args.served_model_name  # MLX_VLM_SERVED_MODEL_NAME'''
    cli_content = cli_content.replace(old_env, new_env)

    with open(cli_file, "w") as f:
        f.write(cli_content)
    print("  [PATCH] cli.py: added --served-model-name argument + env var")

# ============================================================
# 2. Patch app.py: resolve served name in get_cached_model
# ============================================================
with open(app_file, "r") as f:
    app_content = f.read()

if PATCH_MARKER_APP in app_content:
    print("  [PATCH] app.py already patched.")
else:
    # Find the get_cached_model function and inject resolution logic
    # The function signature:
    #   def get_cached_model(
    #       model_path: str,
    #       adapter_path=_INHERIT_ADAPTER,
    #       *,
    #       model_kind: str = "auto",
    #   ):
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
    # Resolve served-model-name alias to actual preloaded path  # PATCH_MARKER_APP
    _served_name = os.environ.get("MLX_VLM_SERVED_MODEL_NAME")
    _preload_model = os.environ.get("MLX_VLM_PRELOAD_MODEL")
    if _served_name and _preload_model and model_path == _served_name:
        model_path = _preload_model'''

    if old_sig in app_content:
        app_content = app_content.replace(old_sig, new_sig)
        with open(app_file, "w") as f:
            f.write(app_content)
        print("  [PATCH] app.py: added model name resolution in get_cached_model")
    else:
        print("  [WARNING] Could not find get_cached_model signature in app.py")

print("[PATCH] Complete.")
PYEOF
