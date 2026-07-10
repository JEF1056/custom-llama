#!/usr/bin/env bash
# Patch mlx-vlm to add --served-model-name aliasing.
# Safe to run multiple times - strips previous attempts first, then re-applies.

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
    cli_lines = f.readlines()

# Strip ONLY our previous patch attempts (the --served-model-name arg block + env var)
cleaned = []
i = 0
while i < len(cli_lines):
    line = cli_lines[i]
    # Skip --served-model-name argument block
    if '"--served-model-name"' in line:
        depth = 0
        while i < len(cli_lines):
            for ch in cli_lines[i]:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            if depth == 0:
                break
        i += 1
        continue
    # Skip --served-model-name env var line
    if "MLX_VLM_SERVED_MODEL_NAME" in line:
        i += 1
        continue
    # Skip the if args.served_model_name: block
    if "if args.served_model_name:" in line:
        i += 1
        # Skip indented lines that are part of this block
        while i < len(cli_lines) and (cli_lines[i].startswith("        ") or cli_lines[i].strip() == ""):
            if cli_lines[i].strip() and not cli_lines[i].startswith("    if ") and not cli_lines[i].startswith("    elif ") and not cli_lines[i].startswith("    else"):
                i += 1
            else:
                break
        continue
    cleaned.append(line)
    i += 1

cli_text = "".join(cleaned)

# Insert --served-model-name after --model block
result_lines = []
inserted_arg = False
inserted_env = False

for i, line in enumerate(cli_text.split("\n")):
    result_lines.append(line)

    # Insert argument block after --model
    if not inserted_arg and '"--model"' in line and "served-model-name" not in line:
        depth = 0
        found_open = False
        # Go back to find parser.add_argument(
        for j in range(len(result_lines) - 1, -1, -1):
            if "parser.add_argument(" in result_lines[j]:
                found_open = True
                break
        if found_open:
            # Find the closing ) of this add_argument call
            start = j
            depth = 0
            for k in range(start, len(result_lines)):
                for ch in result_lines[k]:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            # Insert after line k
                            served_arg = [
                                '',
                                '    parser.add_argument(',
                                '        "--served-model-name",',
                                '        type=str,',
                                '        default=None,',
                                '        help="Alias for the pre-loaded model used in API requests.",',
                                '    )',
                            ]
                            result_lines = result_lines[:k+1] + served_arg + result_lines[k+1:]
                            inserted_arg = True
                            break
                if inserted_arg:
                    break

    # Insert env var after the --model env block
    if not inserted_env and 'MLX_VLM_PRELOAD_ADAPTER' in line:
        result_lines.append('    if args.served_model_name:')
        result_lines.append('        os.environ["MLX_VLM_SERVED_MODEL_NAME"] = args.served_model_name')
        inserted_env = True

if inserted_arg:
    print("  [PATCH] cli.py: added --served-model-name argument")
if inserted_env:
    print("  [PATCH] cli.py: added MLX_VLM_SERVED_MODEL_NAME env var")

with open(cli_file, "w") as f:
    f.write("\n".join(result_lines))

# ============================================================
# 2. Patch app.py: resolve served name in get_cached_model
# ============================================================
with open(app_file, "r") as f:
    app_text = f.read()

# Strip only our specific patch block (the 4 lines we added)
old_sig = '''def get_cached_model(
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

# If already patched, just confirm
if old_sig in app_text:
    print("  [PATCH] app.py: already patched, no change needed")
else:
    # Try to insert
    base_sig = '''def get_cached_model(
    model_path: str,
    adapter_path=_INHERIT_ADAPTER,
    *,
    model_kind: str = "auto",
):'''

    new_sig = base_sig + '''
    # Resolve served-model-name alias to actual preloaded path
    _served = os.environ.get("MLX_VLM_SERVED_MODEL_NAME")
    _preload = os.environ.get("MLX_VLM_PRELOAD_MODEL")
    if _served and _preload and model_path == _served:
        model_path = _preload'''

    if base_sig in app_text:
        app_text = app_text.replace(base_sig, new_sig)
        print("  [PATCH] app.py: added model name resolution in get_cached_model")
    else:
        print("  [WARNING] Could not find get_cached_model signature in app.py")

with open(app_file, "w") as f:
    f.write(app_text)

print("[PATCH] Complete.")
PYEOF
