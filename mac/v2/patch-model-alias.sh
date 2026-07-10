#!/usr/bin/env bash
# Patch mlx-vlm to add --served-model-name aliasing.

set -e

python3 << 'PYEOF'
import os
import mlx_vlm

mlx_site = os.path.dirname(mlx_vlm.__file__)
cli_file = os.path.join(mlx_site, "server", "cli.py")
app_file = os.path.join(mlx_site, "server", "app.py")

print(f"  Patching mlx-vlm at: {mlx_site}")

# ============================================================
# 1. Patch cli.py
# ============================================================
with open(cli_file, "r") as f:
    cli_text = f.read()

# Strip previous --served-model-name attempt
lines = cli_text.split("\n")
cleaned = []
i = 0
while i < len(lines):
    if '"--served-model-name"' in lines[i] or "'--served-model-name'" in lines[i]:
        # Skip this entire parser.add_argument block
        depth = 0
        started = False
        while i < len(lines):
            for ch in lines[i]:
                if ch == '(':
                    depth += 1
                    started = True
                elif ch == ')':
                    depth -= 1
                    if started and depth == 0:
                        i += 1
                        break
            if started and depth == 0:
                break
            i += 1
        i += 1
        continue
    if "MLX_VLM_SERVED_MODEL_NAME" in lines[i] or "args.served_model_name" in lines[i]:
        i += 1
        continue
    cleaned.append(lines[i])
    i += 1
cli_text = "\n".join(cleaned)

# Insert --served-model-name after --model block
parts = []
in_model_block = False
depth = 0
inserted = False

for line in cli_text.split("\n"):
    parts.append(line)
    if not inserted and '"--model"' in line:
        in_model_block = True
    if in_model_block and not inserted:
        for ch in line:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and in_model_block:
                # End of --model block, insert after
                parts.extend([
                    "",
                    '    parser.add_argument(',
                    '        "--served-model-name",',
                    '        type=str,',
                    '        default=None,',
                    '        help="Alias for the pre-loaded model used in API requests.",',
                    '    )',
                ])
                inserted = True
                in_model_block = False
                break

if inserted:
    print("  [PATCH] cli.py: added --served-model-name argument")
else:
    print("  [WARNING] Could not insert --served-model-name argument")

# Add env var assignment
old_env = '''if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path'''

new_env = old_env + '''
    if args.served_model_name:
        os.environ["MLX_VLM_SERVED_MODEL_NAME"] = args.served_model_name'''

if old_env in cli_text:
    cli_text = cli_text.replace(old_env, new_env)
    print("  [PATCH] cli.py: added MLX_VLM_SERVED_MODEL_NAME env var")

with open(cli_file, "w") as f:
    f.write(cli_text)

# ============================================================
# 2. Patch app.py
# ============================================================
with open(app_file, "r") as f:
    app_text = f.read()

# Strip previous resolution patch
app_lines = app_text.split("\n")
cleaned = []
for line in app_lines:
    if "Resolve served-model-name alias" in line:
        continue
    if 'os.environ.get("MLX_VLM_SERVED_MODEL_NAME")' in line:
        continue
    if 'os.environ.get("MLX_VLM_PRELOAD_MODEL")' in line:
        continue
    if "model_path = _preload" in line:
        continue
    cleaned.append(line)
app_text = "\n".join(cleaned)

# Insert resolution at top of get_cached_model
old_sig = '''def get_cached_model(
    model_path: str,
    adapter_path=_INHERIT_ADAPTER,
    *,
    model_kind: str = "auto",
):'''

new_sig = old_sig + '''
    # Resolve served-model-name alias to actual preloaded path
    _served = os.environ.get("MLX_VLM_SERVED_MODEL_NAME")
    _preload = os.environ.get("MLX_VLM_PRELOAD_MODEL")
    if _served and _preload and model_path == _served:
        model_path = _preload'''

if old_sig in app_text:
    app_text = app_text.replace(old_sig, new_sig)
    print("  [PATCH] app.py: added model name resolution in get_cached_model")

with open(app_file, "w") as f:
    f.write(app_text)

print("[PATCH] Complete.")
PYEOF
