#!/usr/bin/env python3
"""Sync .env.default → all .env files in the monorepo.

Root .env.default is the single source of truth.  Running this script:
  - Writes / updates root .env       (all variables; consumed by docker-compose)
  - Writes / updates mcp-search-server/.env   (MCP/search vars; loaded by load_dotenv())
  - Writes / updates ui/.env.local   (VITE_* vars; loaded by Vite dev server)

In each target file:
  - New variables from .env.default are ADDED (with their default value).
  - Non-secret variables are UPDATED when .env.default changes.
  - Secret variables are PRESERVED (never overwritten after initial write).
  - Auto-generated tokens (SGLANG_API_KEY, MCP_API_KEY) are created on first run.

Usage:
    python sync-env.py              # sync all targets
    python sync-env.py --dry-run    # preview changes without writing
    python sync-env.py --regenerate # force-regenerate token variables
"""
import argparse
import os
import re
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Secret variables (never overwritten after initial write) ───────────────────

# Secrets shared across all targets
GLOBAL_SECRETS: set[str] = {
    "CF_TUNNEL_TOKEN",
    "CF_ACCESS_HOSTNAME",
    "CF_ACCESS_GOOGLE_CLIENT_ID",
    "CF_ACCESS_GOOGLE_CLIENT_SECRET",
    "HF_TOKEN",
}

# Auto-generated as random tokens on first run (re-generated with --regenerate)
AUTO_GENERATE_TOKENS: frozenset[str] = frozenset({"SGLANG_API_KEY", "MCP_API_KEY"})

# ── Projection definitions ─────────────────────────────────────────────────────
#
# Each projection writes a subset of root .env.default to a target file.
# "prefixes"     – include any key whose name starts with one of these strings
# "exact_keys"   – include these exact keys in addition to prefix matches
# "extra_secrets"– additional secrets specific to this projection

@dataclass
class Projection:
    target: Path
    prefixes: list[str] = field(default_factory=list)
    exact_keys: list[str] = field(default_factory=list)
    extra_secrets: set[str] = field(default_factory=set)
    # If True, ALL keys in this projection are treated as secrets
    all_secrets: bool = False

    def matches(self, key: str) -> bool:
        if any(key.startswith(p) for p in self.prefixes):
            return True
        return key in self.exact_keys


PROJECTIONS: list[Projection] = [
    # MCP server — loaded via python-dotenv's load_dotenv() in config.py
    # NOTE: MCP_API_KEY is NOT a secret here — it's auto-generated in root .env
    # and must always propagate to mcp-search-server/.env to keep them in sync.
    Projection(
        target=Path("mcp-search-server") / ".env",
        prefixes=["MCP_", "SEARCH_", "BROWSER_", "CACHE_", "MAX_"],
        exact_keys=["FILE_BASE_URL"],
        extra_secrets={"SEARCH_API_KEY", "GOOGLE_CSE_ID", "REDIS_PASSWORD"},
    ),
    # Vite dev server — loaded as VITE_* env vars by Vite
    Projection(
        target=Path("ui") / ".env.local",
        prefixes=["VITE_"],
        all_secrets=True,  # Firebase keys and UIDs — never auto-update
    ),
]

# ── Env file parser ────────────────────────────────────────────────────────────

# KEY=VALUE on a single line (value may be quoted, empty, or contain = signs)
_VAR_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def parse_env(path: Path) -> dict[str, str]:
    """Return {KEY: RAW_VALUE} for every variable in the file.

    Handles multiline quoted values (value includes surrounding quotes and
    embedded newlines exactly as written).
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _VAR_RE.match(line)
        if not m:
            i += 1
            continue

        key, value = m.group(1), m.group(2)

        # Detect multiline quoted strings
        if value and value[0] in ('"', "'"):
            q = value[0]
            stripped = value.replace("\\" + q, "")
            if stripped.count(q) % 2 == 1:   # odd quotes → multiline
                collected = [value]
                i += 1
                while i < len(lines):
                    cont = lines[i]
                    collected.append(cont)
                    i += 1
                    if cont.replace("\\" + q, "").count(q) % 2 == 1:
                        break
                value = "\n".join(collected)

        result[key] = value
        i += 1

    return result


# ── Merge logic ────────────────────────────────────────────────────────────────

def merge(
    defaults: dict[str, str],
    current: dict[str, str],
    effective_secrets: set[str],
) -> dict[str, str]:
    """Merge defaults into current, respecting secrets."""
    result = dict(current)
    for key, value in defaults.items():
        if key not in result:
            result[key] = value                              # new key → always add
        elif key not in effective_secrets and result[key] != value:
            result[key] = value                              # non-secret → update
        # secret with existing value → leave unchanged
    return result


# ── Token generation ───────────────────────────────────────────────────────────

def generate_token() -> str:
    return secrets.token_urlsafe(32)


# ── Writer ────────────────────────────────────────────────────────────────────

def write_env(path: Path, merged: dict[str, str]) -> None:
    """Write merged variables to path, preserving existing key order and appending new keys."""
    out: list[str] = []
    seen: set[str] = set()

    if path.exists():
        raw = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(raw):
            line = raw[i]
            m = _VAR_RE.match(line)
            if not m:
                i += 1
                continue

            key, value = m.group(1), m.group(2)
            seen.add(key)

            # Detect multiline so we skip continuation lines
            is_multiline = False
            if value and value[0] in ('"', "'"):
                q = value[0]
                if value.replace("\\" + q, "").count(q) % 2 == 1:
                    is_multiline = True

            out.append(f"{key}={merged[key]}" if key in merged else line)
            i += 1

            if is_multiline:
                i += 1
                while i < len(raw):
                    cont = raw[i]
                    if cont.replace("\\" + value[0], "").count(value[0]) % 2 == 1:
                        break
                    i += 1
    else:
        pass  # new file — will append everything below

    # Append keys that weren't in the existing file
    for key in sorted(merged.keys() - seen):
        out.append(f"{key}={merged[key]}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── Per-target sync ────────────────────────────────────────────────────────────

def sync_target(
    source_defaults: dict[str, str],
    target: Path,
    effective_secrets: set[str],
    auto_tokens: frozenset[str],
    regenerate: bool,
    dry_run: bool,
    label: str,
) -> bool:
    """Sync source_defaults → target. Returns True if changes were made."""
    current = parse_env(target)

    # Generate / regenerate tokens
    generated: dict[str, str] = {}
    for key in auto_tokens:
        if key in source_defaults:
            if regenerate or not current.get(key, ""):
                generated[key] = generate_token()

    merged = merge(source_defaults, current, effective_secrets)
    merged.update(generated)

    # Compute diff for reporting
    added     = [k for k in merged if k not in current]
    updated   = [k for k in merged if k in current and k not in effective_secrets and merged[k] != current[k]]
    regen     = list(generated.keys())
    preserved = [k for k in source_defaults if k in current and k in effective_secrets and source_defaults[k] != current[k]]

    prefix = f"  [{label}]"
    if added:     print(f"{prefix} Add:      {', '.join(sorted(added))}")
    if updated:   print(f"{prefix} Update:   {', '.join(sorted(updated))}")
    if regen:     print(f"{prefix} Generate: {', '.join(sorted(regen))}")
    if preserved: print(f"{prefix} Preserve: {', '.join(sorted(preserved))}")
    if not added and not updated and not regen:
        print(f"{prefix} No changes needed.")
        return False

    if dry_run:
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    write_env(target, merged)
    print(f"{prefix} ✓ Wrote {target}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync .env.default → all .env files (single source of truth)"
    )
    parser.add_argument("--dry-run",    action="store_true", help="Preview changes without writing")
    parser.add_argument("--regenerate", action="store_true", help="Force-regenerate auto-generated tokens")
    parser.add_argument("--default-file", default=".env.default", help="Path to the root .env.default")
    parser.add_argument("--env-file",     default=".env",         help="Path to the root .env output")
    args = parser.parse_args()

    default_path = Path(args.default_file)
    if not default_path.exists():
        print(f"Error: {default_path} not found.", file=sys.stderr)
        sys.exit(1)

    all_defaults = parse_env(default_path)
    if not all_defaults:
        print(f"Warning: no variables found in {default_path}", file=sys.stderr)

    changed = False

    # ── 1. Root .env (all variables) ──────────────────────────────────────────
    root_env_path = Path(args.env_file)
    root_secrets = GLOBAL_SECRETS | {"MCP_API_KEY", "SEARCH_API_KEY", "SGLANG_API_KEY"}
    changed |= sync_target(
        source_defaults=all_defaults,
        target=root_env_path,
        effective_secrets=root_secrets,
        auto_tokens=AUTO_GENERATE_TOKENS,
        regenerate=args.regenerate,
        dry_run=args.dry_run,
        label=args.env_file,
    )

    # ── 2. Projections ─────────────────────────────────────────────────────────
    # Source from the written root .env (not .env.default) so that auto-generated
    # tokens like MCP_API_KEY and SGLANG_API_KEY propagate to sub-project .env files.
    if not args.dry_run:
        projection_source = parse_env(root_env_path)
    else:
        # In dry-run mode the root .env wasn't written; simulate by merging defaults
        # with the current root .env (tokens may be stale, but that's fine for preview)
        projection_source = parse_env(root_env_path)
        projection_source.update({k: v for k, v in all_defaults.items() if k not in projection_source})

    for proj in PROJECTIONS:
        # Filter to only the keys this projection cares about
        proj_defaults = {k: v for k, v in projection_source.items() if proj.matches(k)}
        if not proj_defaults:
            print(f"  [{proj.target}] No matching keys — skipping")
            continue

        if proj.all_secrets:
            effective_secrets = GLOBAL_SECRETS | set(proj_defaults.keys())
        else:
            effective_secrets = GLOBAL_SECRETS | proj.extra_secrets

        # Projections never auto-generate tokens (tokens come from root .env)
        changed |= sync_target(
            source_defaults=proj_defaults,
            target=proj.target,
            effective_secrets=effective_secrets,
            auto_tokens=frozenset(),
            regenerate=False,
            dry_run=args.dry_run,
            label=str(proj.target),
        )

    if not changed and not args.dry_run:
        print("All .env files up to date.")


if __name__ == "__main__":
    main()
