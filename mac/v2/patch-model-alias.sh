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
    cli_lines = f.readlines()

# Remove any previous patch attempts
cleaned = []
skip_block = False
for i, line in enumerate(cli_lines):
    stripped = line.strip()
    # Skip any previous --served-model-name arg block
    if '"--served-model-name"' in stripped or "'--served-model-name'" in stripped:
        skip_block = True
        continue
    if skip_block:
        if stripped == ")" or stripped.endswith(")"):
            skip_block = False
            continue
        if stripped.startswith("parser.add_argument(") or stripped == ")":
            skip_block = False
            continue
        continue
    # Remove previous env var references
    if "MLX_VLM_SERVED_MODEL_NAME" in stripped:
        continue
    # Remove previous args.served_model_name references
    if "args.served_model_name" in stripped:
        continue
    cleaned.append(line)

cli_lines = cleaned

# Find the position after --model argument block to insert --served-model-name
# The --model block looks like:
#     parser.add_argument(
#         "--model",
#         type=str,
#         default=None,
#         help="Pre-load a language model at startup...",
#     )
#
# Insert after the closing paren of the --model block
result = []
inserted = False
i = 0
while i < len(cli_lines):
    result.append(cli_lines[i])
    if not inserted and '"--model"' in cli_lines[i]:
        # Found the --model line, now find the closing ) of this parser.add_argument
        paren_depth = 0
        j = i
        while j < len(cli_lines):
            for ch in cli_lines[j]:
                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        # Insert after this line
                        served_arg = [
                            "\n",
                            '    parser.add_argument(\n',
                            '        "--served-model-name",\n',
                            '        type=str,\n',
                            '        default=None,\n',
                            '        help="Alias for the pre-loaded model used in API requests.",\n',
                            '    )\n',
                        ]
                        result.extend(served_arg)
                        inserted = True
                        break
            if inserted:
                break
            j += 1
    i += 1

if not inserted:
    print("  [WARNING] Could not find --model argument in cli.py to insert after")
else:
    print("  [PATCH] cli.py: added --served-model-name argument")

# Now add the env var assignment after the --model env block
cli_text = "".join(result)

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
else:
    print("  [WARNING] Could not find args.model env block in cli.py")

with open(cli_file, "w") as f:
    f.write(cli_text)

# ============================================================
# 2. Patch app.py: resolve served name in get_cached_model
# ============================================================
with open(app_file, "r") as f:
    app_text = f.read()

# Remove any previous resolution patch
app_lines = app_text.split("\n")
cleaned = []
for line in app_lines:
    if "Resolve served-model-name alias" in line:
        continue
    if line.strip().startswith("_served = os.environ.get("):
        continue
    if line.strip().startswith("_preload = os.environ.get("):
        continue
    if "_served and _preload and model_path == _served" in line:
        continue
    if "model_path = _preload" in line:
        # Only skip if it's the patch line, not some other assignment
        if "Resolve" in line or "_served" in line or "_preload" in line:
            continue
    cleaned.append(line)
app_text = "\n".join(cleaned)

# Insert resolution at the top of get_cached_model
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

if old_sig in app_text:
    app_text = app_text.replace(old_sig, new_sig)
    print("  [PATCH] app.py: added model name resolution in get_cached_model")
else:
    print("  [WARNING] Could not find get_cached_model signature in app.py")

with open(app_file, "w") as f:
    f.write(app_text)

print("[PATCH] Complete.")
PYEOF
