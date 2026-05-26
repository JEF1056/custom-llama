#!/usr/bin/env python3
"""Merge .env.default into .env, adding/updating non-secret variables only.

Supports multiline values (quoted across multiple lines).
Auto-generates token variables (LLAMA_API_KEY, MCP_API_KEY) if empty.

Usage:
    python sync-env.py              # use defaults, auto-generate empty tokens
    python sync-env.py --regenerate # force regenerate all token variables
    python sync-env.py --secrets CF_TUNNEL_TOKEN,CF_ACCESS_HOSTNAME,MY_SECRET
    python sync-env.py --secrets-file secrets.txt  # one var name per line
"""
import argparse
import os
import re
import secrets
import sys
from pathlib import Path

# ── Configurable: override via CLI or edit here ──────────────────────────
DEFAULT_SECRETS = [
    "CF_TUNNEL_TOKEN",
    "CF_ACCESS_HOSTNAME",
    "CF_ACCESS_GOOGLE_CLIENT_ID",
    "CF_ACCESS_GOOGLE_CLIENT_SECRET",
    "HF_TOKEN",
    "SEARCH_API_KEY",
    "MCP_API_KEY",
    "LLAMA_API_KEY",
]

# Variables that are auto-generated as random tokens.
# These are treated as secrets but get a generated value if empty.
AUTO_GENERATE_TOKENS = frozenset({"LLAMA_API_KEY", "MCP_API_KEY"})

# KEY=VALUE on a single line (value may be quoted, empty, or contain = signs)
VAR_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def parse_env(path: Path) -> dict[str, str]:
    """Return {KEY: RAW_VALUE} for every variable in the file.

    Handles multiline values — the raw value includes surrounding quotes and
    embedded newlines exactly as written in the file.
    """
    vars: dict[str, str] = {}
    if not path.exists():
        return vars

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = VAR_RE.match(line)
        if not m:
            i += 1
            continue

        key = m.group(1)
        value = m.group(2)

        # Check if the value starts a quoted multiline string
        quote_char = None
        if value and value[0] in ('"', "'"):
            # Count unescaped quotes — if odd, the string continues on next lines
            # Simple heuristic: strip escaped quotes then count remaining
            stripped = value.replace('\\' + value[0], '')
            if stripped.count(value[0]) % 2 == 1:
                # Odd number of quotes → multiline continuation
                quote_char = value[0]
                collected = [value]
                i += 1
                while i < len(lines):
                    cont = lines[i]
                    collected.append(cont)
                    i += 1
                    # Check if this line closes the quote
                    cont_stripped = cont.replace('\\' + quote_char, '')
                    if cont_stripped.count(quote_char) % 2 == 1:
                        break
                value = "\n".join(collected)

        vars[key] = value
        i += 1

    return vars


def merge(defaults: dict[str, str], current: dict[str, str], secrets: set[str]) -> dict[str, str]:
    """Start from current, add new keys from defaults, update non-secret values."""
    result = dict(current)
    for key, value in defaults.items():
        if key not in result:                       # new variable
            result[key] = value
        elif key not in secrets and result[key] != value:  # update non-secret if changed
            result[key] = value
    return result


def generate_token() -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(32)


def sync_env_pair(default_file: Path, env_file: Path, secrets: set[str],
                   regenerate: bool, dry_run: bool) -> bool:
    """Sync a single .env.default → .env pair. Returns True if changes were made."""
    defaults = parse_env(default_file)
    current = parse_env(env_file)

    if not defaults:
        print(f"  ⚠ No variables found in {default_file}")
        return False

    # Auto-generate or regenerate tokens
    generated = {}
    for key in AUTO_GENERATE_TOKENS:
        if key in defaults:
            current_val = current.get(key, "")
            if regenerate or not current_val:
                generated[key] = generate_token()

    # Merge defaults into current, then overlay generated tokens
    merged = merge(defaults, current, secrets)
    merged.update(generated)

    # Show diff
    added = [k for k in merged if k not in current]
    updated = [k for k in merged if k in current and k not in secrets and merged[k] != current[k]]
    regenerated = list(generated.keys())
    skipped = [k for k in defaults if k in current and k in secrets and defaults[k] != current[k]]

    prefix = f"  [{env_file}]"
    if added:
        print(f"{prefix} Variables to add: {', '.join(sorted(added))}")
    if updated:
        print(f"{prefix} Variables to update: {', '.join(sorted(updated))}")
    if regenerated:
        print(f"{prefix} Tokens generated: {', '.join(sorted(regenerated))}")
    if skipped:
        print(f"{prefix} Secrets preserved: {', '.join(sorted(skipped))}")
    if not added and not updated and not regenerated and not skipped:
        print(f"{prefix} No changes needed.")
        return False

    if dry_run:
        return True

    # Rebuild the .env file: only variable assignments, no comments
    out_lines: list[str] = []
    seen_keys: set[str] = set()

    if env_file.exists():
        raw_lines = env_file.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            m = VAR_RE.match(line)
            if not m:
                # Skip comments and blank lines
                i += 1
                continue

            key = m.group(1)
            value = m.group(2)
            seen_keys.add(key)

            # Detect multiline value so we skip the continuation lines
            is_multiline = False
            if value and value[0] in ('"', "'"):
                stripped = value.replace('\\' + value[0], '')
                if stripped.count(value[0]) % 2 == 1:
                    is_multiline = True

            if key in merged:
                out_lines.append(f"{key}={merged[key]}")
            else:
                # Variable removed from defaults — keep existing value
                out_lines.append(line)

            i += 1
            # Skip continuation lines for multiline values
            if is_multiline:
                i += 1
                while i < len(raw_lines):
                    cont = raw_lines[i]
                    cont_stripped = cont.replace('\\' + value[0], '')
                    if cont_stripped.count(value[0]) % 2 == 1:
                        break
                    i += 1

        # Append new variables at the end
        new_keys = sorted(merged.keys() - seen_keys)
        if new_keys:
            for key in new_keys:
                out_lines.append(f"{key}={merged[key]}")
    else:
        # No existing .env — write variables only, no comments
        out_lines = [f"{k}={v}" for k, v in merged.items()]

    env_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"{prefix} ✓ Wrote {env_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync .env.default → .env (preserves secrets)")
    parser.add_argument("--secrets", default=",".join(DEFAULT_SECRETS),
                        help="Comma-separated list of secret variable names")
    parser.add_argument("--secrets-file", help="File with one secret var name per line")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--default-file", default=".env.default")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate auto-generated token variables (LLAMA_API_KEY, MCP_API_KEY)")
    args = parser.parse_args()

    # Build secret set
    secrets = set(v.strip() for v in args.secrets.split(",") if v.strip())
    if args.secrets_file:
        secrets.update(line.strip() for line in Path(args.secrets_file).read_text(encoding="utf-8").splitlines() if line.strip())

    base_dir = Path(args.default_file).parent

    # ── Sync pairs: (default_file, env_file) ──────────────────────────────
    pairs = [
        (base_dir / args.default_file, base_dir / args.env_file),
        # MCP server has its own .env.default → .env
        (Path("mcp-search-server") / ".env.default", Path("mcp-search-server") / ".env"),
    ]

    changed = False
    for default_file, env_file in pairs:
        if not default_file.exists():
            print(f"  ⚠ Skipping {default_file} — not found")
            continue
        changed |= sync_env_pair(default_file, env_file, secrets, args.regenerate, args.dry_run)

    if not changed and not args.dry_run:
        print("No changes needed.")


if __name__ == "__main__":
    main()
