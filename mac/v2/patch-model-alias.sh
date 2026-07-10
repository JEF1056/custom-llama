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

# Remove only our previous patch attempts, NOT the original --model block
cleaned = []
i = 0
while i < len(cli_lines):
    line = cli_lines[i]

    # Skip previous --served-model-name argument blocks
    if '"--served-model-name"' in line or "'--served-model-name'" in line:
        # Find the end of this parser.add_argument block by tracking parens
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

    # Skip previous env var lines
    if "MLX_VLM_SERVED_MODEL_NAME" in line or "args.served_model_name" in line:
        i += 1
        continue

    cleaned.append(line)
    i += 1

cli_text = "".join(cleaned)

# Now insert --served-model-name after the --model argument block
# Find the parser.add_argument( block that contains "--model"
result_lines = []
inserted = False
i = 0
while i < len(cli_text.split("\n")):
    line = cli_text.split("\n")[i]
    result_lines.append(line)

    if not inserted and '"--model"' in line:
        # Found --model line. Track paren depth to find end of this parser.add_argument block.
        # Go back to find the parser.add_argument( start
        j = len(result_lines) - 1
        while j >= 0 and "parser.add_argument(" not in result_lines[j]:
            j -= 1
        if j >= 0:
            # Count parens from that point forward
            depth = 0
            k = j
            while k < len(result_lines):
                for ch in result_lines[k]:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            # Insert after line k
                            insert_after = k
                            break
                k += 1

            # Insert the --served-model-name block
            served_block = [
                "",
                '    parser.add_argument(',
                '        "--served-model-name",',
                '        type=str,',
                '        default=None,',
                '        help="Alias for the pre-loaded model used in API requests.",',
                '    )',
            ]
            result_lines = result_lines[:insert_after + 1] + served_block + result_lines[insert_after + 1:]
            inserted = True
            break

    i += 1

if not inserted:
    print("  [WARNING] Could not find --model argument in cli.py to insert after")

cli_text = "\n".join(result_lines)
if inserted:
    print("  [PATCH] cli.py: added --served-model-name argument")

# Add env var assignment after the --model env block
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

# Remove only our previous resolution patch
app_lines = app_text.split("\n")
cleaned = []
for line in app_lines:
    if "Resolve served-model-name alias" in line:
        continue
    if 'os.environ.get("MLX_VLM_SERVED_MODEL_NAME")' in line and "app.py" not in line:
        continue
    if 'os.environ.get("MLX_VLM_PRELOAD_MODEL")' in line:
        continue
    if "model_path = _preload" in line:
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
