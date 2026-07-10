#!/usr/bin/env bash
# Patch mlx-vlm to add --served-model-name aliasing.
# Always re-applies to ensure correctness.

set -e

python3 << 'PYEOF'
import os
import mlx_vlm

mlx_site = os.path.dirname(mlx_vlm.__file__)
cli_file = os.path.join(mlx_site, "server", "cli.py")
app_file = os.path.join(mlx_site, "server", "app.py")

print(f"  Patching mlx-vlm at: {mlx_site}")

# ============================================================
# 1. Patch cli.py: add --served-model-name argument + env var
# ============================================================
with open(cli_file, "r") as f:
    cli_content = f.read()

# Remove any previous patch attempts to ensure clean state
cli_lines = []
skip = False
for line in cli_content.split("\n"):
    if 'MLX_VLM_SERVED_MODEL_NAME' in line:
        continue
    if '"--served-model-name"' in line or "'--served-model-name'" in line:
        skip = True
        continue
    if skip and line.strip() == ")":
        skip = False
        continue
    if skip and (line.strip().startswith("parser.add_argument") or line.strip() == "):"):
        skip = False
        continue
    cli_lines.append(line)

cli_content = "\n".join(cli_lines)

# Now add the argument after --model block
arg_block = '''    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Alias for the pre-loaded model used in API requests (e.g. qwen3.6-35b-a3b).",
    )
'''

model_pattern = '("--model"'
if model_pattern in cli_content:
    # Find the end of the --model argument block
    idx = cli_content.find(model_pattern)
    # Find the closing paren of this parser.add_argument call
    paren_start = cli_content.rfind("parser.add_argument(", 0, idx + 200)
    paren_count = 1
    pos = paren_start + len("parser.add_argument(")
    while pos < len(cli_content) and paren_count > 0:
        if cli_content[pos] == '(':
            paren_count += 1
        elif cli_content[pos] == ')':
            paren_count -= 1
        pos += 1
    # Insert after the closing paren
    cli_content = cli_content[:pos] + "\n" + arg_block + cli_content[pos:]
    print("  [PATCH] cli.py: added --served-model-name argument")

# Add env var assignment
old_env = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path'''
new_env = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    if args.served_model_name:
        os.environ["MLX_VLM_SERVED_MODEL_NAME"] = args.served_model_name'''

if old_env in cli_content:
    cli_content = cli_content.replace(old_env, new_env)
    print("  [PATCH] cli.py: added MLX_VLM_SERVED_MODEL_NAME env var")
elif "args.served_model_name" in cli_content:
    print("  [PATCH] cli.py: env var already present")

with open(cli_file, "w") as f:
    f.write(cli_content)

# ============================================================
# 2. Patch app.py: resolve served name in get_cached_model
# ============================================================
with open(app_file, "r") as f:
    app_content = f.read()

# Remove any previous resolution patch
app_lines = []
skip_resolution = False
for line in app_content.split("\n"):
    if "Resolve served-model-name alias" in line or "MLX_VLM_SERVED_MODEL_NAME" in line:
        skip_resolution = False
        continue
    if "_served_name" in line or "_preload_model" in line:
        skip_resolution = False
        continue
    app_lines.append(line)

app_content = "\n".join(app_lines)

# Now add the resolution logic
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
    _served = os.environ.get("MLX_VLM_SERVED_MODEL_NAME")
    _preload = os.environ.get("MLX_VLM_PRELOAD_MODEL")
    if _served and _preload and model_path == _served:
        model_path = _preload'''

if old_sig in app_content:
    app_content = app_content.replace(old_sig, new_sig)
    print("  [PATCH] app.py: added model name resolution in get_cached_model")
else:
    print("  [WARNING] Could not find get_cached_model signature in app.py")

with open(app_file, "w") as f:
    f.write(app_content)

print("[PATCH] Complete.")
PYEOF
