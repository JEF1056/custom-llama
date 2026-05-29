#!/usr/bin/env python3
"""Sync .env.default → runtime and template env files in the monorepo.

Root .env.default is the single source of truth.  Running this script:
  - Writes / updates root .env           (all variables; consumed by docker-compose)
  - Writes / updates sub-project .env.default files (local-dev templates; committed to git)
      mcp-search-server/.env.default  — copy to .env to run the server outside Docker
      ui/.env.default                 — copy to .env to run the Vite dev server locally

Root .env behaviour:
  - New variables are ADDED (with their default value).
  - Non-secret variables are UPDATED when .env.default changes.
  - Secret variables are PRESERVED (never overwritten after initial write).
  - Stale variables (removed from .env.default) are DELETED.
  - Auto-generated tokens (SGLANG_API_KEY, MCP_API_KEY) are created on first run.

Sub-project .env.default behaviour:
  - Fully managed by this script — always reflects root .env.default defaults.
  - local_overrides swap production values for localhost equivalents.
  - No secrets, no tokens — safe to commit.

Usage:
    python sync-env.py              # sync all targets
    python sync-env.py --dry-run    # preview changes without writing
    python sync-env.py --regenerate # force-regenerate token variables
"""
import argparse
import re
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Secret variables (never overwritten after initial write in root .env) ──────

GLOBAL_SECRETS: set[str] = {
    "CF_TUNNEL_TOKEN",
    "CF_ACCESS_HOSTNAME",
    "CF_ACCESS_GOOGLE_CLIENT_ID",
    "CF_ACCESS_GOOGLE_CLIENT_SECRET",
    "HF_TOKEN",
}

# Auto-generated as random tokens on first run (re-generated with --regenerate)
AUTO_GENERATE_TOKENS: frozenset[str] = frozenset({"SGLANG_API_KEY", "MCP_API_KEY"})

# ── Sub-project template definitions ──────────────────────────────────────────
#
# Each entry writes a .env.default template to a sub-project directory.
# "prefixes"       – include any key whose name starts with one of these strings
# "exact_keys"     – include these exact keys in addition to prefix matches
# "local_overrides"– swap specific values for localhost-friendly defaults

@dataclass
class Projection:
    target: Path
    prefixes: list[str] = field(default_factory=list)
    exact_keys: list[str] = field(default_factory=list)
    local_overrides: dict[str, str] = field(default_factory=dict)

    def matches(self, key: str) -> bool:
        return key in self.exact_keys or any(key.startswith(p) for p in self.prefixes)


PROJECTIONS: list[Projection] = [
    # mcp-search-server — load_dotenv() reads .env for local-dev runs outside Docker.
    # Users: cp mcp-search-server/.env.default mcp-search-server/.env
    Projection(
        target=Path("mcp-search-server") / ".env.default",
        prefixes=["MCP_", "SEARCH_", "BROWSER_", "CACHE_", "MAX_"],
        exact_keys=["FILE_BASE_URL"],
        local_overrides={
            # Production URL → localhost when running outside Docker
            "FILE_BASE_URL": "http://localhost:3100",
        },
    ),
    # ui — Vite reads .env automatically for local dev (npm run dev).
    # Users: cp ui/.env.default ui/.env  then fill in Firebase credentials
    Projection(
        target=Path("ui") / ".env.default",
        prefixes=["VITE_"],
    ),
]

# ── Env file parser ────────────────────────────────────────────────────────────

_VAR_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def parse_env(path: Path) -> dict[str, str]:
    """Return {KEY: RAW_VALUE} for every variable in the file."""
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


# ── Merge logic (root .env only) ───────────────────────────────────────────────

def merge(
    defaults: dict[str, str],
    current: dict[str, str],
    effective_secrets: set[str],
) -> dict[str, str]:
    """Return only keys present in defaults, preserving secret values.

    Keys in current absent from defaults are omitted — they will be removed
    from the target file on the next write.
    """
    result: dict[str, str] = {}
    for key, value in defaults.items():
        if key not in current:
            result[key] = value                  # new key → add with default
        elif key in effective_secrets:
            result[key] = current[key]           # secret → keep existing value
        else:
            result[key] = value                  # non-secret → update to latest default
    return result


# ── Token generation ───────────────────────────────────────────────────────────

def generate_token() -> str:
    return secrets.token_urlsafe(32)


# ── Writers ───────────────────────────────────────────────────────────────────

def write_env(path: Path, merged: dict[str, str]) -> None:
    """Write merged variables to path, preserving existing key order, appending new keys."""
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

            if key in merged:
                out.append(f"{key}={merged[key]}")
            # key absent from merged → removed from .env.default; drop line
            i += 1

            if is_multiline:
                i += 1
                while i < len(raw):
                    cont = raw[i]
                    if cont.replace("\\" + value[0], "").count(value[0]) % 2 == 1:
                        break
                    i += 1

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

    generated: dict[str, str] = {}
    for key in auto_tokens:
        if key in source_defaults:
            if regenerate or not current.get(key, ""):
                generated[key] = generate_token()

    merged = merge(source_defaults, current, effective_secrets)
    merged.update(generated)

    added     = [k for k in merged if k not in current]
    updated   = [k for k in merged if k in current and k not in effective_secrets and merged[k] != current[k]]
    removed   = [k for k in current if k not in merged]
    regen     = list(generated.keys())
    preserved = [k for k in source_defaults if k in current and k in effective_secrets and source_defaults[k] != current[k]]

    prefix = f"  [{label}]"
    if added:     print(f"{prefix} Add:      {', '.join(sorted(added))}")
    if updated:   print(f"{prefix} Update:   {', '.join(sorted(updated))}")
    if removed:   print(f"{prefix} Remove:   {', '.join(sorted(removed))}")
    if regen:     print(f"{prefix} Generate: {', '.join(sorted(regen))}")
    if preserved: print(f"{prefix} Preserve: {', '.join(sorted(preserved))}")
    if not added and not updated and not removed and not regen:
        print(f"{prefix} No changes needed.")
        return False

    if dry_run:
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    write_env(target, merged)
    print(f"{prefix} ✓ Wrote {target}")
    return True


def sync_default_template(
    proj_defaults: dict[str, str],
    target: Path,
    dry_run: bool,
    label: str,
) -> bool:
    """Write a committed .env.default template for local development.

    Fully managed by sync-env.py — always reflects root .env.default with
    local_overrides applied.  No secret preservation, no token generation.
    """
    current = parse_env(target)

    added   = [k for k in proj_defaults if k not in current]
    updated = [k for k in proj_defaults if k in current and proj_defaults[k] != current[k]]
    removed = [k for k in current if k not in proj_defaults]

    prefix = f"  [{label}]"
    if added:   print(f"{prefix} Add:    {', '.join(sorted(added))}")
    if updated: print(f"{prefix} Update: {', '.join(sorted(updated))}")
    if removed: print(f"{prefix} Remove: {', '.join(sorted(removed))}")
    if not added and not updated and not removed:
        print(f"{prefix} No changes needed.")
        return False

    if dry_run:
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in proj_defaults.items()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{prefix} ✓ Wrote {target}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync .env.default → root .env and sub-project .env.default templates"
    )
    parser.add_argument("--dry-run",      action="store_true", help="Preview changes without writing")
    parser.add_argument("--regenerate",   action="store_true", help="Force-regenerate auto-generated tokens")
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

    # ── 1. Root .env (all variables; consumed by docker-compose) ──────────────
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

    # ── 2. Sub-project .env.default templates (local dev without Docker) ──────
    # Sourced from root .env.default — these are committed templates, not runtime
    # files. local_overrides swap production values for localhost equivalents.
    for proj in PROJECTIONS:
        proj_defaults = {
            k: proj.local_overrides.get(k, v)
            for k, v in all_defaults.items()
            if proj.matches(k)
        }
        if not proj_defaults:
            print(f"  [{proj.target}] No matching keys — skipping")
            continue

        changed |= sync_default_template(
            proj_defaults=proj_defaults,
            target=proj.target,
            dry_run=args.dry_run,
            label=str(proj.target),
        )

    if not changed and not args.dry_run:
        print("All env files up to date.")


if __name__ == "__main__":
    main()
